"""V105: paired information-budget ablation on the frozen V104 unseen regions."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
IN = ROOT / "results/v104-unseen-region-robust-confirmation-r1/region_month_scores.csv"
OUT = ROOT / "results/v105-unseen-region-capacity-ablation-r1"
PROTOCOL = ROOT / "docs/V105_UNSEEN_REGION_CAPACITY_ABLATION_PROTOCOL.md"
DELTA = 0.10


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    capacity_rows = pd.read_csv(IN)
    if set(capacity_rows.region.unique()) != set(REGIONS):
        raise ValueError("V104 roster mismatch")

    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + REGIONS))
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}

    margin_sets = {}
    old = dict(v90.EF)
    try:
        for region in REGIONS:
            frame = hourly[region].copy()
            times = pd.to_datetime(frame.time, utc=True)
            frame = frame.loc[times.ge("2026-01-01") & times.lt("2026-07-01")
                              & frame.valid.astype(bool)].copy()
            frame["time"] = pd.to_datetime(frame.time, utc=True)
            frame["month"] = frame.time.dt.strftime("%Y-%m-01")
            for month, group in frame.groupby("month", sort=True):
                months = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                          for fuel in v90.FUELS}
                actuals = []
                for _, obs in group.iterrows():
                    generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                    total = sum(generation, F(0))
                    carbon = sum((generation[i] * nominal[v90.FUELS[i]]
                                  for i in range(len(v90.FUELS))), F(0))
                    actuals.append((total, carbon))
                for tau in (600, 900):
                    observed_k = sum(carbon > F(tau) * total for total, carbon in actuals)
                    sets = []
                    for factors in (low, nominal, high):
                        v90.EF.clear()
                        v90.EF.update(factors)
                        sets.append(v90.v66_set(months, tau, len(group)))
                    v90.EF.clear()
                    v90.EF.update(old)
                    margin_sets[(region, month, tau)] = {
                        "hours": len(group), "actual": observed_k,
                        "lower": min(min(s) for s in sets), "upper": max(max(s) for s in sets),
                        "nominal_lower": min(sets[1]), "nominal_upper": max(sets[1]),
                    }
    finally:
        v90.EF.clear()
        v90.EF.update(old)

    rows = []
    for row in capacity_rows.itertuples(index=False):
        key = (row.region, str(row.month)[:10], int(row.tau))
        baseline = margin_sets[key]
        if not baseline["lower"] <= baseline["actual"] <= baseline["upper"]:
            raise ValueError(f"monthly-margin hull excludes proxy K: {key}")
        if not (baseline["lower"] <= row.robust_lower <= row.robust_upper <= baseline["upper"]):
            raise ValueError(f"capacity set is not nested in monthly-margin hull: {key}")
        if baseline["actual"] != row.actual_nominal_K:
            raise ValueError(f"V104 observed K mismatch: {key}")
        required = int((F(str(row.rate)) * int(row.hours)).__ceil__())
        bl, bu = baseline["lower"], baseline["upper"]
        ml, mu = int(row.robust_lower), int(row.robust_upper)
        rows.append({"region": row.region, "month": row.month, "tau": int(row.tau), "rate": row.rate,
                     "hours": int(row.hours), "required_hours": required,
                     "monthly_lower": bl, "monthly_upper": bu, "monthly_width": bu - bl,
                     "capacity_robust_lower": ml, "capacity_robust_upper": mu,
                     "capacity_robust_width": mu - ml, "width_reduction_hours": (bu - bl) - (mu - ml),
                     "monthly_positive": bl >= required, "monthly_negative": bu < required,
                     "monthly_abstain": not (bl >= required or bu < required),
                     "capacity_positive": ml >= required, "capacity_negative": mu < required,
                     "capacity_abstain": not (ml >= required or mu < required), "actual_covered_both": True})

    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "paired_rows.csv", index=False)
    per_region = detail.groupby(["region", "tau", "rate"]).agg(
        region_months=("month", "nunique"), monthly_width=("monthly_width", "mean"),
        capacity_width=("capacity_robust_width", "mean"), width_reduction_hours=("width_reduction_hours", "mean"),
        monthly_positive=("monthly_positive", "mean"), monthly_negative=("monthly_negative", "mean"),
        monthly_abstain=("monthly_abstain", "mean"), capacity_positive=("capacity_positive", "mean"),
        capacity_negative=("capacity_negative", "mean"), capacity_abstain=("capacity_abstain", "mean"),
        actual_coverage=("actual_covered_both", "mean"),
    ).reset_index()
    per_region.to_csv(OUT / "per_region.csv", index=False)
    macro = per_region.groupby(["tau", "rate"]).agg(
        regions=("region", "nunique"), mean_region_months=("region_months", "mean"),
        monthly_width=("monthly_width", "mean"), capacity_width=("capacity_width", "mean"),
        width_reduction_hours=("width_reduction_hours", "mean"),
        monthly_positive=("monthly_positive", "mean"), monthly_negative=("monthly_negative", "mean"),
        monthly_abstain=("monthly_abstain", "mean"), capacity_positive=("capacity_positive", "mean"),
        capacity_negative=("capacity_negative", "mean"), capacity_abstain=("capacity_abstain", "mean"),
        actual_coverage=("actual_coverage", "mean"),
    ).reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    payload = {"status": "POSTHOC_PAIRED_MECHANISM_ABLATION_ON_V104_FIXED_UNSEEN_REGIONS",
               "regions": list(REGIONS), "region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "paired_rows": len(detail), "region_equal_frontier": macro.to_dict("records"),
               "proxy_k_contained_both": bool(detail.actual_covered_both.all()),
               "monthly_margin_set_contains_capacity_set": True,
               "factor_box_radius": DELTA, "hourly_source_sha256": hourly_hashes,
               "source_audits": audits, "v104_rows_sha256": sha(IN),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Post-V104 fixed-cohort ablation, not a new blind confirmation.",
                               "Proxy-label set containment is not probability calibration or true CIF validation.",
                               "Monthly aggregates are target-month inputs; retrospective identification only."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
