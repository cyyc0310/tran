"""Tests for transcif.evaluation.stats (Phase 5.2 significance toolkit).

TDD: these run against synthetic data with known answers to validate the
bootstrap CI, Wilcoxon/paired-t wrappers, Holm-Bonferroni, and the
Harvey-adjusted Diebold-Mariano test before they feed the paper's significance
claims.
"""
import numpy as np
import pytest

from transcif.evaluation.stats import (
    diebold_mariano,
    holm_bonferroni,
    paired_bootstrap_ci,
    paired_t_test,
    wilcoxon_test,
)


# ---------------------------------------------------------------------------
# paired_bootstrap_ci
# ---------------------------------------------------------------------------

def test_bootstrap_ci_brackets_true_median():
    rng = np.random.default_rng(0)
    deltas = rng.normal(loc=5.0, scale=2.0, size=2000)
    out = paired_bootstrap_ci(deltas, n_boot=5000, seed=0)
    # true median ~5; CI must bracket it
    assert out["ci_lo"] < 5.0 < out["ci_hi"]
    assert out["median"] == pytest.approx(5.0, abs=0.3)


def test_bootstrap_ci_ci_lo_le_ci_hi():
    deltas = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out = paired_bootstrap_ci(deltas, n_boot=1000, seed=1)
    assert out["ci_lo"] <= out["median"] <= out["ci_hi"]
    assert out["n"] == 5


# ---------------------------------------------------------------------------
# wilcoxon_test / paired_t_test
# ---------------------------------------------------------------------------

def test_wilcoxon_identical_arrays_high_p():
    x = np.linspace(0, 10, 50)
    out = wilcoxon_test(x, x)
    # identical -> cannot reject (p large / NaN handled)
    assert out["n"] == 50
    assert (out["p_value"] > 0.05) or np.isnan(out["p_value"])


def test_wilcoxon_clear_difference_low_p():
    rng = np.random.default_rng(0)
    x = rng.normal(10, 1, 100)
    y = rng.normal(13, 1, 100)  # y systematically larger
    out = wilcoxon_test(x, y)  # x - y negative
    assert out["p_value"] < 0.001
    assert out["median_delta"] < 0  # x - y


def test_paired_t_sign_and_significance():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 200)
    y = x + rng.normal(0.5, 0.1, 200)  # y slightly but consistently larger
    out = paired_t_test(x, y)
    assert out["mean_delta"] < 0  # x - y
    assert out["p_value"] < 0.001
    # cohen's d sign should match
    assert out["cohen_d"] < 0


def test_paired_t_identical_high_p():
    x = np.arange(100.0)
    out = paired_t_test(x, x)
    assert out["p_value"] > 0.99 or np.isnan(out["p_value"])


# ---------------------------------------------------------------------------
# holm_bonferroni
# ---------------------------------------------------------------------------

def test_holm_orders_and_corrects():
    pvals = [0.001, 0.04, 0.03, 0.20]
    res = holm_bonferroni(pvals, alpha=0.05)
    # corrected p-values are non-decreasing in sorted order
    corrected = [r["holm_p"] for r in res]
    for r in res:
        assert r["holm_p"] >= r["raw_p"] - 1e-12
    # smallest raw p rejected at 0.05
    assert res[0]["reject"] is True
    # largest raw p not rejected
    biggest = max(res, key=lambda r: r["raw_p"])
    assert biggest["reject"] is False


def test_holm_monotonic_in_sorted_order():
    rng = np.random.default_rng(2)
    pvals = list(rng.uniform(0, 1, 10))
    res = holm_bonferroni(pvals, alpha=0.05)
    order = sorted(range(len(res)), key=lambda i: res[i]["raw_p"])
    holm_in_order = [res[i]["holm_p"] for i in order]
    assert all(holm_in_order[k] <= holm_in_order[k + 1] + 1e-9 for k in range(len(holm_in_order) - 1))


# ---------------------------------------------------------------------------
# diebold_mariano (Harvey-adjusted)
# ---------------------------------------------------------------------------

def test_dm_identical_errors_not_significant():
    rng = np.random.default_rng(0)
    e = rng.normal(0, 5, 300)
    out = diebold_mariano(e, e, horizon=24)
    assert out["mean_loss_diff"] == pytest.approx(0.0, abs=1e-9)
    assert out["p_value"] > 0.05


def test_dm_clearly_better_forecast_significant():
    rng = np.random.default_rng(0)
    # e2 has smaller absolute error than e1 -> method 2 better -> d=|e1|-|e2|>0
    e1 = rng.normal(0, 10, 300)
    e2 = rng.normal(0, 3, 300)
    out = diebold_mariano(e1, e2, horizon=24)
    assert out["mean_loss_diff"] > 0  # method 1 worse
    assert out["p_value"] < 0.001


def test_dm_sign_reflects_which_is_better():
    rng = np.random.default_rng(3)
    e_good = rng.normal(0, 2, 300)
    e_bad = rng.normal(0, 8, 300)
    # e1=e_bad, e2=e_good -> positive stat (e1 worse)
    out1 = diebold_mariano(e_bad, e_good, horizon=24)
    # swapped -> negative stat
    out2 = diebold_mariano(e_good, e_bad, horizon=24)
    assert out1["dm_stat"] > 0
    assert out2["dm_stat"] < 0
    assert out1["dm_stat"] == pytest.approx(-out2["dm_stat"], rel=1e-6)


def test_dm_short_series_returns_finite():
    # n slightly above horizon -> must still return finite values, not crash
    rng = np.random.default_rng(4)
    e1 = rng.normal(0, 1, 30)
    e2 = rng.normal(0, 2, 30)
    out = diebold_mariano(e1, e2, horizon=24)
    assert np.isfinite(out["dm_stat"])
    assert np.isfinite(out["p_value"])
