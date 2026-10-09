"""Quick checks for eval_stats.py. Run: python -m pytest -q test_eval_stats.py"""
import numpy as np
import pytest

import eval_stats as es

Y = np.repeat([0, 1, 2, 3], 72)          # one balanced session-2 test set, 288 trials


def test_perfect_classifier():
    m = es.fold_metrics(Y, Y)
    assert m["accuracy"] == 1.0 and m["kappa"] == 1.0 and m["macro_f1"] == 1.0
    assert es.chance_assessment(Y, Y, n_perm=2000)["above_chance"]


def test_random_predictions_near_chance_not_significant():
    rng = np.random.default_rng(1)
    flagged = 0
    for i in range(20):
        pred = rng.integers(0, 4, size=Y.size)
        acc = es.fold_metrics(Y, pred)["accuracy"]
        assert abs(acc - 0.25) < 0.08
        flagged += es.chance_assessment(Y, pred, n_perm=2000, seed=i)["above_chance"]
    assert flagged <= 3                   # alpha = 0.05 -> about 1 in 20 expected


def test_matches_sklearn_and_threshold():
    skm = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(2)
    pred = np.where(rng.random(Y.size) < 0.5, Y, rng.integers(0, 4, Y.size))
    m = es.fold_metrics(Y, pred)
    assert m["accuracy"] == pytest.approx(skm.accuracy_score(Y, pred))
    assert m["kappa"] == pytest.approx(skm.cohen_kappa_score(Y, pred))
    assert m["macro_f1"] == pytest.approx(skm.f1_score(Y, pred, average="macro"))
    assert (m["confusion"] == skm.confusion_matrix(Y, pred, labels=[0, 1, 2, 3])).all()


def test_end_to_end_on_fake_predictions(tmp_path):
    """Smoke test of evaluate(): 3 models x 9 subjects x 10 repeats of fake predictions."""
    rng = np.random.default_rng(3)
    skill = {"ANN": 0.55, "SNN": 0.50, "RSNN": 0.45}
    recs = []
    for s in range(1, 10):
        for r in range(10):
            for m, p in skill.items():
                pred = np.where(rng.random(Y.size) < p, Y, rng.integers(0, 4, Y.size))
                recs.append({"model": m, "subject": s, "repeat": r, "seed": 100 * s + r,
                             "y_true": Y, "y_pred": pred})
    res = es.evaluate(recs, {"n_permutations": 200, "n_bootstrap": 2000})
    assert res["chance_summary"]["ANN"]["n_folds"] == 90
    assert es.binomial_threshold(288) == pytest.approx(84 / 288)
    for metric, rows in res["comparisons"].items():
        assert [r["pair"] for r in rows] == ["RSNN-SNN", "RSNN-ANN", "ANN-SNN"]
        for r in rows:
            assert r["p_holm"] >= r["p"] and r["n_eff"] <= 9
    # every subject the same direction -> smallest exact two-sided p, 2 / 2**9
    assert res["comparisons"]["accuracy"][2]["p"] == pytest.approx(2 / 2**9)
    es.save_results(res, tmp_path / "results_smoke.json")


def test_holm_and_wilcoxon_zero_handling():
    assert np.allclose(es.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    x = np.arange(9) / 10
    y = x - np.array([0, .01, .02, .03, .04, .05, .06, .07, .08])
    w = es.wilcoxon_exact(x, y)
    assert w["n_zero"] == 1 and w["n_eff"] == 8
