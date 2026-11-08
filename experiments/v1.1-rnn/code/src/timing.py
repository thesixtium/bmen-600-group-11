"""Wall-clock timing and ETA reporting for the benchmark pipeline.

What it gives you
  * stage timers      : `with timer.stage("preprocess"): ...` prints start/end and the running total.
  * per-fold tracker  : after every fold, prints its duration, the running average, the ETA of the current model and the
                        ETA of the whole benchmark (current model + models still to run).
  * per-subject lines : printed as soon as all folds of a target subject (in this process) are finished.
  * summary           : at the end (or on a crash), a table of stages, per-model fold times and per-subject totals.

Files written
  <run_dir>/fold_times.jsonl      one JSON line per finished fold (model, subject, repeat, seconds, phase breakdown)
  <run_dir>/timings.json          stage times, per-model and per-subject totals of this process
  results/benchmark/timing_history.json
                                  mean seconds per fold for each model, keyed by "full|cuda" etc. Used as a PRIOR for the
                                  ETA of models that have not started yet (e.g. the RSNN while the ANN is running). Without
                                  history those models are reported as "unknown" in the pipeline ETA.

Notes
  * Only folds run by THIS process are timed; folds skipped on a resume do not count (they are not re-run).
  * The pipeline ETA covers the three model benchmarks only. collect_results, plotting and compare_models are timed as
    stages but not predicted.
"""

import json
import os
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch

from benchmark_config import QUICK, RESULTS_DIR

HISTORY_PATH: Path = RESULTS_DIR / "timing_history.json"


def format_duration(seconds: float | None) -> str:
    if seconds is None or seconds != seconds or seconds == float("inf"):
        return "?"
    if seconds < 10:
        return f"{seconds:.1f}s"
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _finish_clock(seconds_from_now: float) -> str:
    return time.strftime("%a %d %b %H:%M", time.localtime(time.time() + seconds_from_now))


class PipelineTimer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.run_dir: Path | None = None
        self.models: tuple[str, ...] = ()
        self.plan: dict[str, int] = {}  # model -> number of folds this process will run
        self.stages: dict[str, float] = {}
        self.records: list[dict] = []  # one per finished fold of this process
        self.history_key = f"{'quick' if QUICK else 'full'}|unknown"
        self.history: dict[str, dict[str, float]] = self._load_history()
        # Optional wall-clock deadline (unix seconds), set by run_benchmark.slurm a safety margin before the job's time
        # limit. Once a model's next fold would not finish before it, no new folds are started (see out_of_time()).
        deadline = os.environ.get("BENCH_DEADLINE_EPOCH")
        self.deadline: float | None = float(deadline) if deadline else None
        self.stopped_for_time = False

    # ------------------------------------------------------------ setup / helpers
    @staticmethod
    def _load_history() -> dict:
        try:
            return json.loads(HISTORY_PATH.read_text())
        except (OSError, ValueError):
            return {}

    def bind(self, run_dir: Path, device: torch.device, models: tuple[str, ...], plan: dict[str, int]) -> None:
        """Call once the run directory, device and the number of pending folds per model are known."""
        self.run_dir, self.models, self.plan = run_dir, models, plan
        self.history_key = f"{'quick' if QUICK else 'full'}|{device.type}"
        prior = self.history.get(self.history_key, {})
        print(f"[time] pending folds per model: {plan} (device: {device.type})")
        if prior:
            print("[time] per-fold times from previous runs (used as ETA prior): "
                  + ", ".join(f"{m} {format_duration(s)}" for m, s in prior.items()))
            est = sum(plan.get(m, 0) * prior[m] for m in models if m in prior)
            missing = [m for m in models if plan.get(m, 0) and m not in prior]
            print(f"[time] initial benchmark estimate: {format_duration(est)}"
                  + (f" (+ unknown: {', '.join(missing)})" if missing else ""))
        else:
            print("[time] no timing history for this device/mode yet: ETAs for later models appear once they have run once.")

    def elapsed(self) -> float:
        return time.perf_counter() - self.t0

    def est_fold_seconds(self, model: str) -> float | None:
        """Mean seconds per fold: measured in this process if available, else from previous runs, else None."""
        done = [r["seconds"] for r in self.records if r["model"] == model]
        if done:
            return sum(done) / len(done)
        return self.history.get(self.history_key, {}).get(model)

    def out_of_time(self, model: str) -> bool:
        """True if no further fold of `model` should be started in this process (deadline too close). Sticky: once True,
        it stays True for every later model, which matters because the SNN needs the ANN checkpoints of its folds.
        Always False when BENCH_DEADLINE_EPOCH is not set (e.g. local runs)."""
        if self.deadline is None:
            return False
        if self.stopped_for_time:
            return True
        done = [r["seconds"] for r in self.records if r["model"] == model]
        if done:
            expected = 1.25 * max(done)  # slowest fold seen in this process, plus headroom
        else:
            prior = self.history.get(self.history_key, {}).get(model)
            expected = 1.5 * prior if prior else 0.0
        if time.time() + expected > self.deadline:
            self.stopped_for_time = True
            print(f"[time] deadline close (next {model} fold needs ~{format_duration(expected)}, "
                  f"{format_duration(self.deadline - time.time())} left): starting no more folds in this job.")
        return self.stopped_for_time

    def eta_pipeline(self, current: str, current_remaining: float) -> tuple[float, list[str]]:
        """Remaining benchmark time = rest of the current model + estimated later models. Returns (seconds, unknown models)."""
        total, unknown = current_remaining, []
        order = list(self.models)
        for m in order[order.index(current) + 1 :]:
            n = self.plan.get(m, 0)
            if n == 0:
                continue
            est = self.est_fold_seconds(m)
            if est is None:
                unknown.append(m)
            else:
                total += n * est
        return total, unknown

    # ------------------------------------------------------------ public API
    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        print(f"[time] >>> {name}")
        start = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - start
            self.stages[name] = self.stages.get(name, 0.0) + dt
            print(f"[time] <<< {name}: {format_duration(dt)} (pipeline elapsed {format_duration(self.elapsed())})")

    def track(self, model: str, folds: list) -> "FoldTracker":
        return FoldTracker(self, model, folds)

    def finish(self) -> None:
        """Print the summary and save timings.json; update the history used as ETA prior. Safe to call after a crash."""
        total = self.elapsed()
        per_model: dict[str, dict] = {}
        per_subject: dict[str, dict[str, float]] = defaultdict(dict)
        for m in self.models:
            secs = [r["seconds"] for r in self.records if r["model"] == m]
            if not secs:
                continue
            per_model[m] = {"n_folds": len(secs), "total_s": sum(secs), "mean_s": sum(secs) / len(secs),
                            "min_s": min(secs), "max_s": max(secs)}
            for r in self.records:
                if r["model"] == m:
                    per_subject[m][str(r["subject"])] = per_subject[m].get(str(r["subject"]), 0.0) + r["seconds"]

        print("\n[time] ================ TIMING SUMMARY ================")
        print(f"[time] total wall clock of this process: {format_duration(total)}")
        for name, s in self.stages.items():
            print(f"[time]   stage {name:<28s} {format_duration(s):>10s}  ({100 * s / total:4.1f}%)")
        for m, d in per_model.items():
            print(f"[time]   {m}: {d['n_folds']} folds, total {format_duration(d['total_s'])}, mean {format_duration(d['mean_s'])}"
                  f" (min {format_duration(d['min_s'])}, max {format_duration(d['max_s'])})")
            print("[time]     per subject: " + ", ".join(f"S{s} {format_duration(v)}" for s, v in sorted(per_subject[m].items(), key=lambda kv: int(kv[0]))))

        if self.run_dir is not None:
            try:
                (self.run_dir / "timings.json").write_text(json.dumps(
                    {"total_s": total, "stages_s": self.stages, "per_model": per_model, "per_model_subject_s": per_subject}, indent=1))
            except OSError as exc:
                print(f"[time] could not write timings.json: {exc}")
        if per_model:
            try:
                HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.history.setdefault(self.history_key, {}).update({m: d["mean_s"] for m, d in per_model.items()})
                HISTORY_PATH.write_text(json.dumps(self.history, indent=1))
            except OSError as exc:
                print(f"[time] could not update timing history: {exc}")


class FoldTracker:
    """Times the folds of one model and prints progress, per-subject totals and ETAs."""

    def __init__(self, pipeline: PipelineTimer, model: str, folds: list) -> None:
        self.p, self.model, self.folds = pipeline, model, folds
        self.total = len(folds)
        self.done = 0
        self.durations: list[float] = []
        self.subject_total = Counter(f.target for f in folds)
        self.subject_done: Counter = Counter()
        self.subject_seconds: dict[int, float] = defaultdict(float)
        self.subjects_finished = 0
        self.start = time.perf_counter()
        if not self.total:
            print(f"[time] [{model}] nothing to run (all folds already done)")
            return
        msg = f"[time] [{model}] {self.total} folds to run over {len(self.subject_total)} subject(s)"
        est = pipeline.est_fold_seconds(model)
        if est is not None:
            msg += f"; prior estimate {format_duration(self.total * est)} ({format_duration(est)}/fold)"
        print(msg)

    @contextmanager
    def fold(self, fold) -> Iterator[dict]:
        """Usage: `with tracker.fold(f) as t: ...; t["pretrain_s"] = ...` (extras are saved to fold_times.jsonl)."""
        idx = self.done + 1
        print(f"[time] [{self.model}] fold {idx}/{self.total} start (subject {fold.target}, repeat {fold.repeat})")
        extras: dict = {}
        start = time.perf_counter()
        yield extras
        dt = time.perf_counter() - start

        self.done += 1
        self.durations.append(dt)
        self.subject_done[fold.target] += 1
        self.subject_seconds[fold.target] += dt
        record = {"model": self.model, "subject": fold.target, "repeat": fold.repeat, "seconds": dt, **extras}
        self.p.records.append(record)
        if self.p.run_dir is not None:
            try:
                with open(self.p.run_dir / "fold_times.jsonl", "a") as fh:
                    fh.write(json.dumps(record) + "\n")
            except OSError:
                pass

        mean = sum(self.durations) / len(self.durations)
        remaining = (self.total - self.done) * mean
        pipe_total, unknown = self.p.eta_pipeline(self.model, remaining)
        phases = ", ".join(f"{k[:-2]} {format_duration(v)}" for k, v in extras.items() if k.endswith("_s"))
        line = (f"[time] [{self.model}] fold {self.done}/{self.total} done in {format_duration(dt)}"
                + (f" ({phases})" if phases else "")
                + f" | avg {format_duration(mean)}"
                + f" | {self.model} elapsed {format_duration(time.perf_counter() - self.start)}, ETA {format_duration(remaining)}"
                + f" | benchmark ETA {format_duration(pipe_total)}")
        if unknown:
            line += f" (excl. unknown: {', '.join(unknown)})"
        else:
            line += f" -> finish ~{_finish_clock(pipe_total)}"
        print(line)

        if self.subject_done[fold.target] == self.subject_total[fold.target]:
            self.subjects_finished += 1
            n = self.subject_total[fold.target]
            s = self.subject_seconds[fold.target]
            print(f"[time] [{self.model}] SUBJECT {fold.target} finished: {n} folds in {format_duration(s)} "
                  f"({format_duration(s / n)}/fold) | subjects done {self.subjects_finished}/{len(self.subject_total)}")
        if self.done == self.total:
            print(f"[time] [{self.model}] ALL {self.total} folds finished in {format_duration(time.perf_counter() - self.start)}")
