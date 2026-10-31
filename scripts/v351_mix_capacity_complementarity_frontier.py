"""V351: share-error sensitivity of the V350 mix-capacity interaction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from v342_china_sharp_group_hull import ROOT

V348 = ROOT / "results/v348-ember-share-error-frontier-r1/province_period_share_error_frontier.csv"
V350 = ROOT / "results/v350-fuel-mix-capacity-factorial-r1/paired_factorial_province_period.csv"
PROTOCOL = ROOT / "docs/V351_MIX_CAPACITY_COMPLEMENTARITY_FRONTIER_PROTOCOL.md"
OUT = ROOT / "results/v351-mix-capacity-complementarity-frontier-r1"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    mix = pd.read_csv(V348)
    outer = pd.read_csv(V350)
    rows = []
    for relerr in sorted(mix.relative_fuel_share_error.unique()):
        no_cap = mix.loc[(mix.relative_fuel_share_error.eq(relerr)) & mix.arm.eq("monthly_only"),
                         ["province", "period", "estimated_mix_width_h"]].rename(
                             columns={"estimated_mix_width_h": "mix_no_capacity_width_h"})
        with_cap = mix.loc[(mix.relative_fuel_share_error.eq(relerr)) & mix.arm.eq("lagged_2022_capacity"),
                           ["province", "period", "estimated_mix_width_h"]].rename(
                               columns={"estimated_mix_width_h": "mix_with_capacity_width_h"})
        paired = outer[["province", "period", "broad_no_capacity_width_h", "broad_with_capacity_width_h"]]
        paired = paired.merge(no_cap, on=["province", "period"], validate="one_to_one")
        paired = paired.merge(with_cap, on=["province", "period"], validate="one_to_one")
        if len(paired) != 328:
            raise ValueError(f"share-error band {relerr}: expected 328 matched province-periods, got {len(paired)}")
        paired["mix_gain_no_capacity_h"] = paired.broad_no_capacity_width_h-paired.mix_no_capacity_width_h
        paired["mix_gain_with_capacity_h"] = paired.broad_with_capacity_width_h-paired.mix_with_capacity_width_h
        paired["complementarity_h"] = paired.mix_gain_with_capacity_h-paired.mix_gain_no_capacity_h
        rows.append({"relative_share_error": float(relerr), "province_periods": len(paired),
                     "mean_mix_gain_no_capacity_h": float(paired.mix_gain_no_capacity_h.mean()),
                     "mean_mix_gain_with_capacity_h": float(paired.mix_gain_with_capacity_h.mean()),
                     "mean_complementarity_h": float(paired.complementarity_h.mean()),
                     "median_complementarity_h": float(paired.complementarity_h.median()),
                     "positive_complementarity_cells": int(paired.complementarity_h.gt(0).sum()),
                     "negative_complementarity_cells": int(paired.complementarity_h.lt(0).sum())})
    result = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "complementarity_frontier.csv", index=False)
    summary = {"status": "COMPLETED_SHARE_ERROR_SENSITIVITY_NOT_EMPIRICAL_VALIDATION",
               "frontier": result.to_dict("records"),
               "limits": ["Reuses V348 Ember share-error scenario sets and V350 collapsed-group widths.",
                          "Share-error bands are analyst-defined scenarios, not measured source error.",
                          "No hourly labels or interval coverage are available."],
               "sha256": {str(p.relative_to(ROOT)): sha(p) for p in (V348, V350, PROTOCOL, Path(__file__))}}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.2f}"))


if __name__ == "__main__":
    main()
