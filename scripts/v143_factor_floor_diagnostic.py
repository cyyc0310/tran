"""V143: account for EF-box oracle width before evaluating transferable risk."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import v139_physical_reserve_calibration as v139
from v141_source_diversity_ablation import original_budget_frame

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V143_FACTOR_FLOOR_DIAGNOSTIC_PROTOCOL.md"
ORIGINAL = ROOT / "results/v140-monthly-conditional-risk-r1/monthly_feature_panel.csv"
ADDITIONS = ROOT / "results/v141-source-expansion-input-r1/additional_monthly_panel.csv"
V141 = ROOT / "results/v141-source-diversity-ablation-r1/outer_fold_rows.csv"
V142 = ROOT / "results/v142-pace-lgee-monthly-holdout-r1/target_rows.csv"
OUT = ROOT / "results/v143-factor-floor-diagnostic-r1"


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    original = original_budget_frame(pd.read_csv(ORIGINAL))
    additional = pd.read_csv(ADDITIONS)
    additional = additional.loc[additional.eligible_donor]
    columns = ["region", "month", "tau", "hours", "fuelshare_coal", "physical_lower", "physical_upper",
               "factor_box_K_lower", "factor_box_K_upper"]
    floor = pd.concat([original[columns], additional[columns]], ignore_index=True)
    keys = ["region", "month", "tau"]
    if floor.duplicated(keys).any():
        raise ValueError("duplicate oracle floor row")
    floor["factor_floor"] = floor.factor_box_K_upper - floor.factor_box_K_lower
    floor["factor_floor_fraction"] = floor.factor_floor / floor.hours
    floor["monthly_physical_width"] = floor.physical_upper - floor.physical_lower
    floor["physical_excess_above_factor_floor"] = floor.monthly_physical_width - floor.factor_floor
    if (floor.physical_excess_above_factor_floor < 0).any():
        raise ValueError("oracle factor floor exceeds physical monthly width")
    predictions = []
    for study, path in (("V141", V141), ("V142", V142)):
        frame = pd.read_csv(path)
        frame["study"] = study
        if "source_scenario" not in frame:
            frame["source_scenario"] = "expanded_PACE_LGEE_holdout"
        predictions.append(frame)
    predictions = pd.concat(predictions, ignore_index=True, sort=False)
    joined = predictions.merge(floor, on=keys, suffixes=("", "_oracle"), validate="many_to_one", how="left")
    if (joined.factor_floor.isna().any()
            or not joined.hours.eq(joined.hours_oracle).all()
            or not joined.factor_box_K_lower.eq(joined.factor_box_K_lower_oracle).all()
            or not joined.factor_box_K_upper.eq(joined.factor_box_K_upper_oracle).all()):
        raise ValueError("prediction/floor label join failed")
    covered = joined.lower.le(joined.factor_box_K_lower) & joined.upper.ge(joined.factor_box_K_upper)
    if not covered.eq(joined.covered).all():
        raise ValueError("saved coverage not reproduced")
    joined["too_narrow_to_cover"] = joined.width.lt(joined.factor_floor)
    joined["miss_with_sufficient_width"] = ~joined.covered & ~joined.too_narrow_to_cover
    joined["oracle_repair_lower"] = joined[["lower", "factor_box_K_lower"]].min(axis=1)
    joined["oracle_repair_upper"] = joined[["upper", "factor_box_K_upper"]].max(axis=1)
    joined["oracle_repair_width"] = joined.oracle_repair_upper - joined.oracle_repair_lower
    joined["oracle_repaired_excess"] = joined.oracle_repair_width - joined.factor_floor
    joined["covered_excess"] = (joined.width - joined.factor_floor).where(joined.covered)
    if (joined.loc[joined.too_narrow_to_cover, "covered"].any()
            or (joined.oracle_repaired_excess < 0).any()
            or (joined.oracle_repair_width < joined.width).any()):
        raise ValueError("oracle-width accounting invariant failed")
    if not joined[predictions.columns].equals(predictions):
        raise ValueError("original prediction columns changed")
    group_keys = ["study", "source_scenario", "seed", "tau", "method", "alpha", "region"]
    region = joined.groupby(group_keys).agg(
        rows=("month", "size"), coverage=("covered", "mean"), width=("width", "mean"),
        factor_floor=("factor_floor", "mean"), covered_excess=("covered_excess", "mean"),
        too_narrow_rate=("too_narrow_to_cover", "mean"), other_miss_rate=("miss_with_sufficient_width", "mean"),
        oracle_repair_width=("oracle_repair_width", "mean"), oracle_repaired_excess=("oracle_repaired_excess", "mean")).reset_index()
    metrics = [c for c in region.columns if c not in group_keys + ["rows"]]
    macro = region.groupby(group_keys[:-1])[metrics].mean().reset_index()
    inventory = floor.groupby(["region", "tau"]).agg(
        months=("month", "size"), mean_hours=("hours", "mean"), mean_coal_share=("fuelshare_coal", "mean"),
        factor_floor=("factor_floor", "mean"), factor_floor_fraction=("factor_floor_fraction", "mean"),
        monthly_physical_width=("monthly_physical_width", "mean"),
        physical_excess=("physical_excess_above_factor_floor", "mean")).reset_index()
    if v139.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during diagnostic")
    OUT.mkdir(parents=True)
    floor.to_csv(OUT / "oracle_factor_floor_panel.csv", index=False)
    joined.to_csv(OUT / "prediction_width_accounting.csv", index=False)
    inventory.to_csv(OUT / "region_factor_floor.csv", index=False)
    region.to_csv(OUT / "per_region_width_accounting.csv", index=False)
    macro.to_csv(OUT / "region_equal_width_accounting.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharex=True, sharey=True, constrained_layout=True)
    colors = {"SC": "#D17800", "PACE": "#0072B2", "LGEE": "#9B3B96"}
    for ax, tau in zip(axes, (600, 900)):
        subset = floor.loc[floor.tau.eq(tau)]
        other = subset.loc[~subset.region.isin(colors)]
        ax.scatter(other.fuelshare_coal, other.factor_floor_fraction, color="#A4ADB5", s=12, alpha=.35, label="Other source regions")
        for name, color in colors.items():
            target = subset.loc[subset.region.eq(name)]
            ax.scatter(target.fuelshare_coal, target.factor_floor_fraction, s=26, color=color, alpha=.8, label=name)
        ax.set_title(f"Threshold {tau} gCO2/kWh")
        ax.set_xlabel("Monthly coal generation share")
        ax.set_xlim(-.02, 1.02)
        ax.set_ylim(-.02, 1.02)
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Oracle EF-box tail-count width / observed hours")
    axes[1].legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(OUT / "factor_floor_by_coal_share.png", dpi=180)
    fig.savefig(OUT / "factor_floor_by_coal_share.pdf")
    plt.close(fig)
    payload = {"status": "ORACLE_FACTOR_CONTRACT_WIDTH_ACCOUNTING_NOT_A_PREDICTOR", "source_rows": len(floor),
               "region_months": len(floor[["region", "month"]].drop_duplicates()), "regions": int(floor.region.nunique()),
               "prediction_rows": len(joined), "assertion_failures": 0,
               "prediction_columns_exactly_preserved": True,
               "input_hashes": {str(p.relative_to(ROOT)): v139.sha(p) for p in (ORIGINAL, ADDITIONS, V141, V142, PROTOCOL)},
               "script_sha256": v139.sha(Path(__file__)),
               "limitations": ["The floor is conditional on the assumed factor-box support and observed proxy hourly allocation.",
                               "It is an evaluation oracle and must not enter target prediction or selection.",
                               "This is not measured Chinese factor uncertainty or a probabilistic lower bound without the stated contract."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(inventory.loc[inventory.region.isin(("PACE", "LGEE", "SC"))].to_string(index=False))
    print(region.loc[region.study.eq("V142") & region.alpha.eq(.1) & region.method.isin(("anchor_nearest12_sources", "direct_cqr_all"))].to_string(index=False))


if __name__ == "__main__":
    main()
