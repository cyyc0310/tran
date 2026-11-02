#!/usr/bin/env python3
"""Post-hoc diagnostic: does monthly mix forecast error identify hourly risk?"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
MIX_PATH = ROOT / "results/local-monthly-mix-accuracy-r1/monthly_mix_errors.csv"
METRICS_PATH = ROOT / "results/local-monthly-adaptation-pseudotarget-r1/monthly_metrics.csv"
OUT = ROOT / "results/local-monthly-risk-identifiability-r1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    mix = pd.read_csv(MIX_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    mix = mix.loc[mix["method"].eq("local_ridge")].copy()
    metrics = metrics.loc[
        metrics["adapter"].eq("local_ridge")
        & metrics["method"].eq("fixed_ef_recomposition")
    ].copy()
    data = mix.merge(
        metrics,
        left_on=["ba", "month"],
        right_on=["pseudo_target", "month"],
        how="inner",
    )

    records: list[dict[str, object]] = []
    for (track, ba), group in data.groupby(["track", "ba"], sort=True):
        for error_name in ("share_l1", "thermal_abs_error"):
            for risk_name in ("mae", "coverage90"):
                valid = group[[error_name, risk_name]].dropna()
                rho = float("nan")
                if len(valid) >= 6 and valid[error_name].nunique() > 1 and valid[risk_name].nunique() > 1:
                    rho = float(spearmanr(valid[error_name], valid[risk_name]).statistic)
                records.append(
                    {
                        "track": track,
                        "ba": ba,
                        "error": error_name,
                        "hourly_metric": risk_name,
                        "n_months": len(valid),
                        "spearman_rho": rho,
                    }
                )
    by_ba = pd.DataFrame.from_records(records)
    by_ba.to_csv(OUT / "within_ba_correlations.csv", index=False)

    summaries: list[dict[str, object]] = []
    for (track, error_name, risk_name), group in by_ba.groupby(
        ["track", "error", "hourly_metric"], sort=True
    ):
        values = group["spearman_rho"].dropna()
        summaries.append(
            {
                "track": track,
                "error": error_name,
                "hourly_metric": risk_name,
                "ba_count": len(values),
                "median_rho": float(values.median()) if len(values) else None,
                "min_rho": float(values.min()) if len(values) else None,
                "max_rho": float(values.max()) if len(values) else None,
                "positive_ba_count": int((values > 0).sum()),
                "negative_ba_count": int((values < 0).sum()),
            }
        )

    target_summary = {}
    for track, group in data.groupby("track", sort=True):
        target_summary[track] = {
            "n_ba_months": int(len(group)),
            "ba_count": int(group["ba"].nunique()),
            "miso_mean_share_l1": float(group.loc[group.ba.eq("MISO"), "share_l1"].mean()),
            "miso_mean_coverage90": float(group.loc[group.ba.eq("MISO"), "coverage90"].mean()),
            "nyis_mean_share_l1": float(group.loc[group.ba.eq("NYIS"), "share_l1"].mean()),
            "nyis_mean_coverage90": float(group.loc[group.ba.eq("NYIS"), "coverage90"].mean()),
            "bpat_mean_share_l1": float(group.loc[group.ba.eq("BPAT"), "share_l1"].mean()),
            "bpat_mean_coverage90": float(group.loc[group.ba.eq("BPAT"), "coverage90"].mean()),
        }

    result = {
        "status": "posthoc_descriptive_only",
        "n_joined_ba_months": int(len(data)),
        "tracks": sorted(data["track"].unique().tolist()),
        "within_ba_spearman": summaries,
        "selected_domain_contrasts": target_summary,
        "inputs": {
            str(MIX_PATH.relative_to(ROOT)): sha256(MIX_PATH),
            str(METRICS_PATH.relative_to(ROOT)): sha256(METRICS_PATH),
        },
        "interpretation_limit": (
            "Target-month realized mix is used only for this post-hoc diagnostic. "
            "These correlations are not a deployable risk predictor and do not establish causality."
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"n_joined_ba_months": len(data), "within_ba_spearman": summaries}, indent=2))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    main()
