"""Audit the V121/V122 effect of excluding invalid source capacity contracts."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from v139_physical_reserve_calibration import ROOT, sha

SOURCE = ROOT / "results/v139-physical-reserve-calibration-r1/source_scores.csv"
OUT = ROOT / "results/v139-support-boundary-correction-r1"
TARGETS = {
    "V121": ROOT / "results/v121-nwmt-q3-frozen-support-fallback-r1/target_rows.csv",
    "V122": ROOT / "results/v122-october-mtd-frozen-rule-confirmation-r1/target_rows.csv",
}


def boundary(source: pd.DataFrame, region: str, tau: int, corrected: bool) -> tuple[float, int, int]:
    pool = source.loc[source.region.ne(region) & source.tau.eq(tau)].copy()
    if corrected:
        pool = pool.loc[~pool.capacity_unavailable]
        pool["support_gap"] = (pool.capacity_upper / pool.hours - pool.anchor_base_upper_fraction).clip(lower=0)
    maxima = pool.groupby("region").support_gap.max().sort_values()
    n = len(maxima)
    rank = math.ceil((n + 1) * .8)
    return float(maxima.iloc[rank - 1]), n, rank


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    source = pd.read_csv(SOURCE)
    outputs, frontiers = [], []
    for study, path in TARGETS.items():
        target = pd.read_csv(path)
        available_column = "capacity_feasible" if study == "V121" else "capacity_available"
        for row in target.itertuples(index=False):
            old, _, _ = boundary(source, row.region, row.tau, False)
            new, n, rank = boundary(source, row.region, row.tau, True)
            if not np.isclose(old, row.support_boundary, atol=1e-12, rtol=0):
                raise ValueError("archived cutoff cannot be reproduced")
            available = bool(getattr(row, available_column))
            lo, hi, covered = None, None, None
            fallback = None
            if available:
                fallback = row.tau == 900 and row.support_gap > new
                lo = row.capacity_lower if row.tau == 600 or fallback else row.intersection_lower
                hi = row.capacity_upper if row.tau == 600 or fallback else row.intersection_upper
                covered = lo <= row.factor_box_K_lower and hi >= row.factor_box_K_upper
            outputs.append({"study": study, "region": row.region, "month": row.month, "tau": row.tau,
                            "capacity_available": available, "old_boundary": old, "corrected_boundary": new,
                            "corrected_support_regions": n, "corrected_support_rank": rank,
                            "old_lower": row.frozen_lower, "old_upper": row.frozen_upper,
                            "old_covered": row.frozen_factor_box_covered,
                            "corrected_lower": lo, "corrected_upper": hi, "corrected_covered": covered,
                            "corrected_fallback": fallback,
                            "endpoints_changed": bool(available and (lo != row.frozen_lower or hi != row.frozen_upper))})
    result = pd.DataFrame(outputs)
    for (study, tau), group in result.groupby(["study", "tau"]):
        valid = group.loc[group.capacity_available].copy()
        valid["old_width"] = valid.old_upper - valid.old_lower
        valid["corrected_width"] = valid.corrected_upper - valid.corrected_lower
        metrics = valid.groupby("region")[["old_width", "corrected_width", "old_covered", "corrected_covered"]].mean().mean()
        frontiers.append({"study": study, "tau": tau, "eligible_rows": len(group), "capacity_rows": len(valid),
                          "changed_rows": int(valid.endpoints_changed.sum()), **metrics.to_dict()})
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "corrected_target_rows.csv", index=False)
    pd.DataFrame(frontiers).to_csv(OUT / "region_equal_effect.csv", index=False)
    payload = {"status": "POSTHOC_INPUT_AVAILABILITY_CORRECTION_AUDIT", "rows": len(result),
               "changed_endpoint_rows": int(result.endpoints_changed.sum()),
               "input_hashes": {str(p.relative_to(ROOT)): sha(p) for p in (SOURCE, *TARGETS.values())},
               "script_sha256": sha(Path(__file__)),
               "frontier": json.loads(pd.DataFrame(frontiers).to_json(orient="records")),
               "scope": "Recalibrates source support cutoffs on input-feasible months; reuses saved target endpoints, not a full raw-data rerun or new confirmation."}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(pd.DataFrame(frontiers).to_string(index=False))
    print(f"changed endpoint rows: {result.endpoints_changed.sum()}")


if __name__ == "__main__":
    main()
