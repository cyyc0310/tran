"""V353: province-cluster uncertainty and influence for V352 interactions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from v342_china_sharp_group_hull import ROOT

V352 = ROOT / "results/v352-threshold-factorial-sweep-r1/paired_threshold_factorial.csv"
PROTOCOL = ROOT / "docs/V353_PROVINCE_CLUSTER_INTERACTION_BOOTSTRAP_PROTOCOL.md"
OUT = ROOT / "results/v353-province-cluster-interaction-bootstrap-r1"
SEED = 13_532
N_BOOT = 20_000


def sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    data = pd.read_csv(V352)
    province_tau = data.groupby(["province", "tau_g_kwh"]).complementarity_h.mean().unstack()
    taus = sorted(province_tau.columns.astype(int))
    province_tau.columns = province_tau.columns.astype(int)
    if province_tau.isna().any().any() or len(province_tau) < 20:
        raise ValueError("incomplete province-by-threshold panel or too few clusters")
    values = province_tau[taus].to_numpy(float)
    rng = np.random.default_rng(SEED)
    n_prov = len(province_tau)
    boot_means = np.empty((N_BOOT, len(taus)), dtype=float)
    for i in range(N_BOOT):
        ix = rng.integers(0, n_prov, size=n_prov)
        boot_means[i] = values[ix].mean(axis=0)
    summaries = []
    for j, tau in enumerate(taus):
        loo_means = (values[:, j].sum() - values[:, j]) / (n_prov - 1)
        summaries.append({"tau_g_kwh": tau, "provinces": n_prov,
                          "equal_province_mean_complementarity_h": float(values[:, j].mean()),
                          "cluster_bootstrap_95_low_h": float(np.quantile(boot_means[:, j], .025)),
                          "cluster_bootstrap_95_high_h": float(np.quantile(boot_means[:, j], .975)),
                          "leave_one_province_out_min_h": float(loo_means.min()),
                          "leave_one_province_out_max_h": float(loo_means.max()),
                          "positive_province_means": int((values[:, j] > 0).sum()),
                          "negative_province_means": int((values[:, j] < 0).sum())})
    contrast_summaries = []
    for left, right in ((100, 300), (300, 500), (100, 500)):
        if left not in taus or right not in taus:
            continue
        diff = values[:, taus.index(left)] - values[:, taus.index(right)]
        boot_diff = boot_means[:, taus.index(left)] - boot_means[:, taus.index(right)]
        contrast_summaries.append({"contrast": f"tau{left}_minus_tau{right}",
                                   "equal_province_mean_difference_h": float(diff.mean()),
                                   "cluster_bootstrap_95_low_h": float(np.quantile(boot_diff, .025)),
                                   "cluster_bootstrap_95_high_h": float(np.quantile(boot_diff, .975)),
                                   "provinces_positive_difference": int((diff > 0).sum())})
    OUT.mkdir(parents=True)
    pd.DataFrame(summaries).to_csv(OUT / "threshold_cluster_bootstrap.csv", index=False)
    pd.DataFrame(contrast_summaries).to_csv(OUT / "paired_threshold_contrasts.csv", index=False)
    summary = {"status": "COMPLETED_PROVINCE_CLUSTER_BOOTSTRAP_NOT_EXTERNAL_VALIDATION",
               "n_provinces": n_prov, "bootstrap_replicates": N_BOOT, "seed": SEED,
               "estimand": "equal-province average of each province's mean province-month interaction",
               "thresholds": summaries, "paired_threshold_contrasts": contrast_summaries,
               "limits": ["Bootstrap resamples provinces, not months, to preserve within-province dependence.",
                          "The 30 provinces are a finite cohort; percentile intervals describe cluster resampling sensitivity, not a universal population guarantee.",
                          "All underlying generation mix values are Ember estimates; no hourly truth or coverage validation."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V352, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(pd.DataFrame(summaries).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(pd.DataFrame(contrast_summaries).to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
