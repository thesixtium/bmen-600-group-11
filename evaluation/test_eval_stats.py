"""Quick checks for eval_stats.py. Run: python -m pytest -q test_eval_stats.py"""
import json

import numpy as np
import pytest

import eval_stats as es

Y = np.repeat([0, 1, 2, 3], 72)          # one balanced session-2 test set, 288 trials


def noisy(y, acc, rng):
    """Correct with probability acc, otherwise a random class."""
    return np.where(rng.random(y.size) < acc, y, rng.integers(0, 4, y.size))


def test_perfect_classifier():
    m = es.fold_metrics(Y, Y)
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0 and m["macro_f1"] == 1.0
    ch = es.subject_chance(Y, [Y] * 10, n_perm=2000)
    assert all(ch[k]["above_chance"] for k in es.METRICS)


def test_random_predictions_near_chance_not_significant():
    rng = np.random.default_rng(1)
    flagged = {k: 0 for k in es.METRICS}
    for i in range(20):                   # 20 guessing "subjects", 10 repeats each
        P = rng.integers(0, 4, size=(10, Y.size))
        ch = es.subject_chance(Y, P, n_perm=1000, seed=i)
        assert abs(ch["accuracy"]["observed"] - 0.25) < 0.05
        for k in es.METRICS:
            flagged[k] += ch[k]["above_chance"]
    assert max(flagged.values()) <= 3     # alpha = 0.05 -> about 1 in 20 expected


def test_identical_repeats_give_single_fold_null():
    """Ten copies of the same predictions carry no more evidence than one copy.
    With one shared shuffle per permutation the p-value must be exactly the same.
    (Independent shuffles per repeat would shrink the null and inflate significance.)"""
    rng = np.random.default_rng(5)
    pred = noisy(Y, 0.06, rng)
    one = es.subject_chance(Y, [pred], n_perm=2000, seed=7)
    ten = es.subject_chance(Y, [pred] * 10, n_perm=2000, seed=7)
    for k in es.METRICS:
        assert ten[k]["p_perm"] == pytest.approx(one[k]["p_perm"])
        assert ten[k]["null_sd"] == pytest.approx(one[k]["null_sd"])


def test_matches_sklearn_and_threshold():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    P = np.stack([noisy(Y, 0.5, rng) for _ in range(3)])
    for pred in P:
        m = es.fold_metrics(Y, pred)
        assert m["accuracy"] == pytest.approx(skm.accuracy_score(Y, pred))
        assert m["kappa"] == pytest.approx(skm.cohen_kappa_score(Y, pred))
        assert m["macro_f1"] == pytest.approx(skm.f1_score(Y, pred, average="macro"))
        assert (m["confusion"] == skm.confusion_matrix(Y, pred, labels=[0, 1, 2, 3])).all()
    ch = es.subject_chance(Y, P, n_perm=100)
    assert ch["kappa"]["observed"] == pytest.approx(np.mean([skm.cohen_kappa_score(Y, p) for p in P]))
    assert ch["macro_f1"]["observed"] == pytest.approx(np.mean([skm.f1_score(Y, p, average="macro") for p in P]))
    assert es.binomial_threshold(288) == pytest.approx(84 / 288)


def _write_fake_run(run_dir, rng):
    """3 models x 9 subjects x 10 repeats in the layout benchmark.py writes."""
    skill = {"ann": 0.55, "snn": 0.50, "rsnn": 0.45}
    folds = []
    for s in range(1, 10):
        for r in range(10):
            seed = 2026 + 100 * s + r
            folds.append({"target": s, "repeat": r, "seed": seed})
            for m, p in skill.items():
                for sub in ("predictions", "metrics"):
                    (run_dir / sub / m).mkdir(parents=True, exist_ok=True)
                np.savez_compressed(run_dir / "predictions" / m / f"s{s}_r{r}.npz",
                                    y_true=Y, y_pred=noisy(Y, p, rng), scores=np.zeros((Y.size, 4)))
                (run_dir / "metrics" / m / f"s{s}_r{r}.json").write_text(
                    json.dumps({"model": m, "subject": s, "repeat": r, "seed": seed}))
    (run_dir / "folds.json").write_text(json.dumps({"config": {}, "folds": folds}))


def test_end_to_end_on_fake_run_folder(tmp_path):
    _write_fake_run(tmp_path, np.random.default_rng(3))
    recs = es.load_run(tmp_path)
    assert len(recs) == 270 and {r["model"] for r in recs} == {"ANN", "SNN", "RSNN"}
    res = es.evaluate(recs, {"n_permutations": 200, "n_bootstrap": 2000})
    assert res["test_sets"][1] == {"n_trials": 288, "per_class": [72, 72, 72, 72]}
    assert res["chance_summary"]["ANN"]["accuracy"] == 9
    for metric, rows in res["comparisons"].items():
        assert [r["pair"] for r in rows] == ["RSNN-SNN", "RSNN-ANN", "ANN-SNN"]
        for r in rows:
            assert r["p_holm"] >= r["p"] and r["n_eff"] <= 9
    # every subject the same direction -> smallest exact two-sided p, 2 / 2**9
    assert res["comparisons"]["accuracy"][2]["p"] == pytest.approx(2 / 2**9)
    es.save_results(res, tmp_path / "results_smoke.json")


def test_check_folds_rejects_mismatched_test_sets():
    rng = np.random.default_rng(4)
    recs = [{"model": "ANN", "subject": 1, "repeat": r, "seed": r, "y_true": Y,
             "y_pred": noisy(Y, 0.5, rng)} for r in range(2)]
    recs[1]["y_true"] = np.roll(Y, 1)
    with pytest.raises(ValueError, match="different y_true"):
        es.check_folds(recs)


def test_holm_and_wilcoxon_zero_handling():
    assert np.allclose(es.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    x = np.arange(9) / 10
    y = x - np.array([0, .01, .02, .03, .04, .05, .06, .07, .08])
    w = es.wilcoxon_exact(x, y)
    assert w["n_zero"] == 1 and w["n_eff"] == 8
