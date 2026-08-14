"""Statistical significance toolkit for CIF forecast comparison (Phase 5.2).

Provides:
    - paired_bootstrap_ci : non-parametric median delta + bootstrap 95% CI
    - wilcoxon_test       : Wilcoxon signed-rank (non-parametric paired)
    - paired_t_test       : paired t-test + Cohen's d
    - holm_bonferroni     : family-wise error correction across comparisons
    - diebold_mariano     : Harvey (1997) h-step adjusted DM test on per-forecast
                            error series (the textbook temporal significance test)

These wrappers normalize scipy output into plain dicts so the significance
script and verdict can consume them uniformly.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as sp_stats

__all__ = [
    "paired_bootstrap_ci",
    "wilcoxon_test",
    "paired_t_test",
    "holm_bonferroni",
    "diebold_mariano",
]


def paired_bootstrap_ci(
    deltas: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
    ci: float = 0.95,
) -> dict:
    """Bootstrap CI for the median of a paired difference series.

    Args:
        deltas: 1-D array of per-pair differences (e.g. MAE_A - MAE_B).
        n_boot: number of bootstrap resamples.
        seed: RNG seed for reproducibility.
        ci: two-sided coverage (0.95 -> 2.5/97.5 percentiles).

    Returns:
        ``{median, mean, ci_lo, ci_hi, n, n_boot}``.
    """
    d = np.asarray(deltas, dtype=np.float64)
    n = len(d)
    rng = np.random.default_rng(seed)
    if n == 0:
        return {"median": float("nan"), "mean": float("nan"),
                "ci_lo": float("nan"), "ci_hi": float("nan"),
                "n": 0, "n_boot": n_boot}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_medians = np.median(d[idx], axis=1)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(boot_medians, [alpha, 1.0 - alpha])
    return {
        "median": float(np.median(d)),
        "mean": float(np.mean(d)),
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def wilcoxon_test(x: Sequence[float], y: Sequence[float]) -> dict:
    """Wilcoxon signed-rank test on paired samples.

    Tests H0: median(x - y) == 0. Reports the median delta and an
    rank-biserial style effect size r = |Z| / sqrt(N).

    Returns:
        ``{statistic, p_value, n, median_delta, effect_size_r}``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    delta = x - y
    n = len(delta)
    median_delta = float(np.median(delta))
    # scipy returns statistic=T and p; if all deltas are zero it raises /
    # returns nan — handle gracefully.
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return {"statistic": float("nan"), "p_value": 1.0, "n": int(n),
                "median_delta": median_delta, "effect_size_r": 0.0}
    try:
        res = sp_stats.wilcoxon(x, y, zero_method="wilcox",
                                alternative="two-sided")
        stat = float(res.statistic)
        p = float(res.pvalue)
        # Effect size r = |Z|/sqrt(N). Prefer scipy's normal-approx Z; if the
        # zstat attribute is unavailable, recover Z from the two-sided p-value
        # via the normal quantile function (valid for the large-n approx).
        z = 0.0
        try:
            res_z = sp_stats.wilcoxon(x, y, zero_method="wilcox",
                                      alternative="two-sided",
                                      method="approx")
            if hasattr(res_z, "zstat"):
                z = float(res_z.zstat)
        except Exception:
            pass
        if z == 0.0 and 0.0 < p < 1.0:
            from scipy.stats import norm
            z = float(norm.ppf(1.0 - p / 2.0))
            # restore sign from the median delta direction
            if median_delta < 0:
                z = -z
        r = abs(z) / np.sqrt(n) if n > 0 else 0.0
    except Exception:
        stat, p, r = float("nan"), float("nan"), 0.0
    return {
        "statistic": stat,
        "p_value": p,
        "n": int(n),
        "median_delta": median_delta,
        "effect_size_r": float(r),
    }


def paired_t_test(x: Sequence[float], y: Sequence[float]) -> dict:
    """Paired t-test on (x - y) with Cohen's d effect size.

    Returns:
        ``{statistic, p_value, n, mean_delta, cohen_d}``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    delta = x - y
    n = len(delta)
    mean_delta = float(np.mean(delta))
    res = sp_stats.ttest_rel(x, y)
    sd = float(np.std(delta, ddof=1)) if n > 1 else 0.0
    cohen_d = mean_delta / sd if sd > 0 else 0.0
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n": int(n),
        "mean_delta": mean_delta,
        "cohen_d": float(cohen_d),
    }


def holm_bonferroni(
    pvals: Sequence[float], alpha: float = 0.05
) -> list[dict]:
    """Holm-Bonferroni step-down correction.

    Args:
        pvals: raw per-comparison p-values.
        alpha: family-wise target significance level.

    Returns:
        List (same order as input) of
        ``{raw_p, holm_p, reject, rank}``.
    """
    pvals = list(pvals)
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    holm = [0.0] * m
    # Adjusted p-values: running max of (m - rank) * p_(rank), capped at 1.
    # Monotonic by construction (non-decreasing in sorted order).
    cumulative_max = 0.0
    for rank, idx in enumerate(order):
        corrected = (m - rank) * pvals[idx]
        cumulative_max = max(cumulative_max, corrected)
        holm[idx] = min(cumulative_max, 1.0)
    # Reject decisions: step-down — reject in sorted order while adjusted
    # p <= alpha; stop at the first failure (all subsequent also fail).
    reject = [False] * m
    for rank, idx in enumerate(order):
        if holm[idx] <= alpha:
            reject[idx] = True
        else:
            break
    rank_of = {idx: r for r, idx in enumerate(order)}
    return [
        {"raw_p": float(pvals[i]), "holm_p": float(holm[i]),
         "reject": bool(reject[i]), "rank": int(rank_of[i])}
        for i in range(m)
    ]


def diebold_mariano(
    e1: Sequence[float],
    e2: Sequence[float],
    horizon: int = 24,
    loss: str = "abs",
) -> dict:
    """Harvey (1997) h-step adjusted Diebold-Mariano test.

    Compares two forecast error series. Positive DM statistic means method 1
    has larger loss (method 2 is better); negative means method 1 is better.

    Args:
        e1, e2: 1-D arrays of per-forecast errors (signed pred - true),
                same length. The loss differential is ``L(e1) - L(e2)``.
        horizon: forecast horizon h; the Bartlett bandwidth is ``h - 1`` and
                 the Harvey small-sample adjustment uses h.
        loss: ``"abs"`` (default) or ``"sq"``.

    Returns:
        ``{dm_stat, p_value, n, mean_loss_diff, horizon}``.

    Notes:
        Uses the long-run variance estimator with Bartlett window of bandwidth
        ``h-1`` and Harvey's small-sample multiplier
        ``sqrt((n+1-2h+h(h-1)/n)/n)``; p-value from a t distribution with
        ``n-1`` dof (Harvey 1997), which is more conservative than N(0,1) for
        small n.
    """
    e1 = np.asarray(e1, dtype=np.float64)
    e2 = np.asarray(e2, dtype=np.float64)
    if e1.shape != e2.shape:
        raise ValueError(f"e1 {e1.shape} and e2 {e2.shape} must match")
    if loss == "abs":
        d = np.abs(e1) - np.abs(e2)
    elif loss == "sq":
        d = e1 ** 2 - e2 ** 2
    else:
        raise ValueError(f"loss must be 'abs' or 'sq', got {loss!r}")
    n = len(d)
    h = max(1, int(horizon))
    d_bar = float(d.mean())

    def gamma(k: int) -> float:
        if k >= n:
            return 0.0
        return float(np.mean((d[: n - k] - d_bar) * (d[k:] - d_bar)))

    # long-run variance: gamma(0) + 2 * sum_{k=1}^{h-1} (1 - k/h) gamma(k)
    var_lr = gamma(0)
    for k in range(1, h):
        if k >= n:
            break
        var_lr += 2.0 * (1.0 - k / h) * gamma(k)
    var_lr = max(var_lr, 1e-12)

    dm = d_bar / np.sqrt(var_lr / n)
    # Harvey small-sample adjustment (guard against negative underflow)
    adj_factor_num = n + 1 - 2 * h + h * (h - 1) / n
    adj_factor = np.sqrt(max(adj_factor_num, 0.0) / n)
    dm_adj = dm * adj_factor

    df = max(n - 1, 1)
    p = 2.0 * float(sp_stats.t.sf(abs(dm_adj), df=df))
    return {
        "dm_stat": float(dm_adj),
        "p_value": float(p),
        "n": int(n),
        "mean_loss_diff": d_bar,
        "horizon": h,
    }
