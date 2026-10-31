"""V114: post-hoc intersect robust anchor and capacity sets on opened windows."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H1_ANCHOR = ROOT / "results/v112-worst-region-robust-anchor-r1/target_rows.csv"
H1_CAPACITY = ROOT / "results/v105-unseen-region-capacity-ablation-r1/paired_rows.csv"
Q3_ROWS = ROOT / "results/v113-2026q3-future-time-confirmation-r1/target_rows.csv"
OUT = ROOT / "results/v114-anchor-capacity-intersection-exploratory-r1"
PROTOCOL = ROOT / "docs/V114_ANCHOR_CAPACITY_INTERSECTION_EXPLORATORY_PROTOCOL.md"
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    for rate in RATES:
        required = [int((F(str(rate)) * int(hours)).__ceil__()) for hours in frame.hours]
        for method, lo_col, hi_col in (
            ("anchor", "anchor_lower", "anchor_upper"),
            ("capacity", "capacity_lower", "capacity_upper"),
            ("intersection", "intersection_lower", "intersection_upper"),
        ):
            pos, neg = [], []
            for lo, hi, req in zip(frame[lo_col], frame[hi_col], required):
                positive, negative = int(lo) >= req, int(hi) < req
                pos.append(positive); neg.append(negative)
            frame[f"{method}_positive_{rate:g}"] = pos
            frame[f"{method}_negative_{rate:g}"] = neg
            frame[f"{method}_abstain_{rate:g}"] = [not (a or b) for a, b in zip(pos, neg)]
    return frame


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    h1_anchor = pd.read_csv(H1_ANCHOR)
    h1 = h1_anchor.rename(columns={"anchor_robust_lower": "anchor_lower", "anchor_robust_upper": "anchor_upper",
                            "factor_box_K_lower": "factor_K_lower", "factor_box_K_upper": "factor_K_upper",
                            "capacity_robust_lower": "capacity_lower", "capacity_robust_upper": "capacity_upper"})
    h1["period"] = "2026H1"

    q3 = pd.read_csv(Q3_ROWS)
    q3 = q3.rename(columns={"anchor_robust_lower": "anchor_lower", "anchor_robust_upper": "anchor_upper",
                            "factor_box_K_lower": "factor_K_lower", "factor_box_K_upper": "factor_K_upper",
                            "capacity_robust_lower": "capacity_lower", "capacity_robust_upper": "capacity_upper"})
    q3["period"] = "2026Q3"
    cols = ["region", "month", "tau", "hours", "period", "anchor_lower", "anchor_upper",
            "capacity_lower", "capacity_upper", "factor_K_lower", "factor_K_upper"]
    detail = pd.concat([h1[cols], q3[cols]], ignore_index=True)
    detail["intersection_lower"] = detail[["anchor_lower", "capacity_lower"]].max(axis=1)
    detail["intersection_upper"] = detail[["anchor_upper", "capacity_upper"]].min(axis=1)
    if (detail.intersection_lower > detail.intersection_upper).any():
        raise ValueError("anchor and capacity intervals are disjoint")
    detail["anchor_factor_box_covered"] = ((detail.anchor_lower <= detail.factor_K_lower)
                                           & (detail.anchor_upper >= detail.factor_K_upper))
    detail["capacity_factor_box_covered"] = ((detail.capacity_lower <= detail.factor_K_lower)
                                             & (detail.capacity_upper >= detail.factor_K_upper))
    detail["intersection_factor_box_covered"] = ((detail.intersection_lower <= detail.factor_K_lower)
                                                 & (detail.intersection_upper >= detail.factor_K_upper))
    detail["anchor_width"] = detail.anchor_upper - detail.anchor_lower
    detail["capacity_width"] = detail.capacity_upper - detail.capacity_lower
    detail["intersection_width"] = detail.intersection_upper - detail.intersection_lower
    detail = add_decisions(detail)

    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "exploratory_rows.csv", index=False)
    metrics = ["anchor_width", "capacity_width", "intersection_width", "anchor_factor_box_covered",
               "capacity_factor_box_covered", "intersection_factor_box_covered"]
    for rate in RATES:
        metrics.extend(f"{method}_{state}_{rate:g}" for method in ("anchor", "capacity", "intersection")
                       for state in ("positive", "negative", "abstain"))
    per_region = detail.groupby(["period", "region", "tau"])[metrics].mean().reset_index()
    per_region.to_csv(OUT / "per_region_frontier.csv", index=False)
    macro = per_region.groupby(["period", "tau"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "frontier.csv", index=False)
    payload = {"status": "POSTHOC_EXPLORATORY_INTERSECTION_NOT_BLIND_CONFIRMATION",
               "periods": sorted(detail.period.unique().tolist()),
               "region_months_by_period": detail.groupby("period").apply(
                   lambda g: int(g[["region", "month"]].drop_duplicates().shape[0])).to_dict(),
               "rows": len(detail), "intersection_nonempty": True,
               "intersection_factor_box_coverage": bool(detail.intersection_factor_box_covered.all()),
               "frontier": macro.to_dict("records"), "h1_anchor_sha256": sha(H1_ANCHOR),
               "h1_capacity_sha256": sha(H1_CAPACITY), "q3_rows_sha256": sha(Q3_ROWS),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Uses two already opened windows; post-hoc exploratory result, not a holdout.",
                               "Empirical anchor interval is not a formal physical constraint.",
                               "No Chinese actual-CIF validation or month-ahead forecasting."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
