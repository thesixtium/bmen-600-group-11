"""Runtime watcher for a local benchmark run: progress, ETA, GPU state and running accuracy per model.

Usage (from experiments/v1.1/code/):
    ../../.venv/bin/python watch_run.py                 # one snapshot of the newest full run
    ../../.venv/bin/python watch_run.py --loop 60       # refresh every 60 s (Ctrl+C to stop)
    ../../.venv/bin/python watch_run.py --run 20261029_141407
"""
import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "rssns_for_mi_EEG" / "results" / "benchmark"
MODELS = ("ann", "snn", "rsnn", "rnn")  # the order main.py runs them in
N_FOLDS = 90
# Minutes per fold timed on this laptop (RTX 4060) with random data at the full settings, 2026-10-29.
# Used only for models that have no finished fold yet, scaled by how far the measured speed is from the estimate.
EST_MIN_PER_FOLD = {"ann": 0.7, "snn": 0.05, "rsnn": 7.4, "rnn": 3.7}
CHANCE_ACC = 0.2917  # binomial threshold, n = 288, p = 0.25, alpha = 0.05
LEAKAGE_ACC = 0.90   # project rule: a mean this high on 4-class 2a is a leakage / wrong-split signal


def fmt(seconds: float) -> str:
    seconds = int(seconds)
    h, m = divmod(seconds // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{seconds % 60:02d}s"


def newest_run() -> Path:
    runs = sorted(p for p in RESULTS.iterdir() if p.is_dir() and not p.name.startswith("quick_"))
    if not runs:
        raise SystemExit(f"no full run in {RESULTS}")
    return runs[-1]


def n_running() -> int:
    """Number of pipeline processes (shards) currently running."""
    out = subprocess.run(["pgrep", "-f", r"^\S*python\S* -u src/main\.py"], capture_output=True, text=True).stdout
    return len(out.split())


def shards(run: Path) -> dict[str, list[int]]:
    """Log file -> target subjects. Shard logs are run_log_t123.txt etc.; a plain run_log.txt covers subjects 1-9."""
    found = {log.name: [int(c) for c in log.stem.split("_t")[1]] for log in sorted(run.glob("run_log_t*.txt"))}
    return found or {"run_log.txt": list(range(1, 10))}


def gpu_line() -> str:
    q = "utilization.gpu,temperature.gpu,clocks.sm,power.draw,memory.used"
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        util, temp, clock, power, mem = (s.strip() for s in out.split(","))
        return f"GPU {util} busy, {temp} C, {clock}, {power}, {mem} used"
    except Exception as e:  # nvidia-smi missing or busy: the watcher should still print the rest
        return f"GPU state unavailable ({e})"


def last_log_line(log: Path) -> str:
    if not log.exists():
        return "(no run_log.txt)"
    lines = [l for l in log.read_text(errors="replace").splitlines()
             if l.strip() and "Warning" not in l and "F.conv2d" not in l]
    return lines[-1] if lines else "(empty log)"


def snapshot(run: Path) -> str:
    times = {m: [] for m in MODELS}
    ft = run / "fold_times.jsonl"
    if ft.exists():
        for line in ft.read_text().splitlines():
            r = json.loads(line)
            times.setdefault(r["model"], []).append(r["seconds"])
    # current speed: the last 10 folds per model (shards running side by side make each fold slower than alone)
    recent = {m: v[-10:] for m, v in times.items()}

    # how much slower or faster the real run is than the timing estimate, from the models already measured
    # The RNN is scaled by the RSNN (both step through 256 samples); the ANN slows down far more when sharing the GPU.
    ratio = {m: np.mean(recent[m]) / 60 / EST_MIN_PER_FOLD[m] for m in MODELS if recent[m] and m != "snn"}
    ratios = list(ratio.values())
    scale = float(ratio.get("rsnn", np.mean(ratios) if ratios else 1.0))
    per_fold = {m: (float(np.mean(recent[m])), "measured") if recent[m] else (EST_MIN_PER_FOLD[m] * 60 * scale, "estimate")
                for m in MODELS}

    done_sets, rows = {}, []
    for m in MODELS:
        files = sorted((run / "metrics" / m).glob("s*_r*.json")) if (run / "metrics" / m).exists() else []
        done_sets[m] = {f.stem for f in files}
        done = len(files)
        accs = [json.loads(f.read_text())["accuracy"] for f in files]
        acc = f"{np.mean(accs):.3f} (n={len(accs)})" if accs else "-"
        flag = ""
        if accs and np.mean(accs) >= LEAKAGE_ACC:
            flag = "  <- CHECK FOR LEAKAGE"
        elif accs and np.mean(accs) <= CHANCE_ACC:
            flag = "  <- at or below chance threshold"
        bar = "#" * (done * 20 // N_FOLDS) + "." * (20 - done * 20 // N_FOLDS)
        sec, src = per_fold[m]
        rows.append(f"  {m:<5} [{bar}] {done:>2}/{N_FOLDS}  {fmt(sec):>7}/fold ({src:<8})  mean acc {acc}{flag}")

    # each shard runs its subjects' folds model after model; the run ends when the slowest shard ends
    shard_rows, shard_left = [], []
    for log, subjects in shards(run).items():
        left = sum(per_fold[m][0] for m in MODELS for s in subjects for r in range(10) if f"s{s}_r{r}" not in done_sets[m])
        shard_left.append(left)
        shard_rows.append(f"  shard {','.join(map(str, subjects))}: ~{fmt(left)} left | {last_log_line(run / log)[:110]}")
    remaining = max(shard_left)

    running = n_running()
    all_done = all(len(done_sets[m]) == N_FOLDS for m in MODELS)
    if (run / "COMPLETE").exists() and all_done and not running:
        status = "COMPLETE"
    elif running:
        status = f"RUNNING ({running} process{'es' if running > 1 else ''})"
    elif all_done:
        status = "ALL FOLDS DONE - run the final analysis pass (see experiments/v1.1/code/README.md)"
    else:
        status = "STOPPED - resume with ./launch_shards.sh " + run.name
    finish = datetime.now() + timedelta(seconds=remaining)
    out = [
        f"Run {run.name}   status: {status}   ({datetime.now():%a %d %b %H:%M:%S})",
        *rows,
        *shard_rows,
        f"  time left ~{fmt(remaining)}  ->  finish ~{finish:%a %d %b %H:%M}"
        + (f"   (estimates scaled x{scale:.2f} to this run's measured speed)" if ratios else ""),
        f"  {gpu_line()}",
        "  Accuracies are preliminary (session-2 test, mean over finished folds) until the validation checklist passes.",
    ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="run id (default: newest non-quick run)")
    ap.add_argument("--loop", type=int, default=0, help="refresh every N seconds")
    args = ap.parse_args()
    run = RESULTS / args.run if args.run else newest_run()
    while True:
        text = snapshot(run)
        if args.loop:
            print("\033[2J\033[H" + text, flush=True)
            time.sleep(args.loop)
        else:
            print(text)
            break


if __name__ == "__main__":
    main()
