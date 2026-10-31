"""V205: robust monthly fuel-margin tail-count endpoints over EF boxes."""
from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-monthly-ember-audit-r1/ember_china_province_monthly.csv"
PROTOCOL = ROOT / "docs/V205_MONTHLY_MARGIN_FACTOR_ROBUST_HULL_PROTOCOL.md"
OUT = ROOT / "results/v205-monthly-margin-factor-robust-hull-r2"
FUELS = ("Coal", "Gas", "Other fossil", "Nuclear", "Hydro", "Wind", "Solar", "Bioenergy")
NOMINAL = {"Coal": 1000, "Gas": 400, "Other fossil": 400, "Nuclear": 0,
           "Hydro": 0, "Wind": 0, "Solar": 0, "Bioenergy": 0}
BOXES = {
    "nominal": (NOMINAL, NOMINAL),
    "narrow": (
        {**NOMINAL, "Coal": 900, "Gas": 360, "Other fossil": 360},
        {**NOMINAL, "Coal": 1100, "Gas": 440, "Other fossil": 440},
    ),
    "crossing": (
        {**NOMINAL, "Coal": 900, "Gas": 360, "Other fossil": 200},
        {**NOMINAL, "Coal": 1100, "Gas": 440, "Other fossil": 600},
    ),
}
TAUS = (100, 300, 600, 900)
RATES = (0.01, 0.05, 0.10)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sharp_k_bounds(margins: dict[str, Fraction], ef: dict[str, int], tau: int, hours: int) -> tuple[int, int]:
    high = sum((mass for fuel, mass in margins.items() if ef[fuel] > tau), Fraction(0))
    low = sum((mass for fuel, mass in margins.items() if ef[fuel] < tau), Fraction(0))
    neutral = sum((mass for fuel, mass in margins.items() if ef[fuel] == tau), Fraction(0))
    if high == 0:
        return 0, 0
    excess = sum((mass * (ef[fuel] - tau) for fuel, mass in margins.items()), Fraction(0))
    if low:
        return (1, hours) if excess > 0 else (0, hours - 1)
    if neutral:
        return 1, hours
    return hours, hours


def category(lo: int, hi: int, required: int) -> str:
    if lo >= required:
        return "positive"
    if hi < required:
        return "negative"
    return "abstain"


def main() -> None:
    frame = pd.read_csv(INPUT, dtype={"Generation (TWh)": str}, low_memory=False)
    frame = frame.loc[~frame["Is aggregated source"].astype(str).str.lower().eq("true")]
    frame = frame.loc[frame["Electricity source"].isin(FUELS)]
    frame = frame.loc[frame["Date"].between("2016-01-01", "2024-12-31")]
    detail: list[dict[str, object]] = []
    for (province, date), group in frame.groupby(["Province", "Date"], sort=True):
        if province == "China (total)":
            continue
        indexed = group.set_index("Electricity source")
        if set(FUELS).difference(indexed.index) or indexed.index.duplicated().any():
            continue
        margins = {fuel: Fraction(str(indexed.loc[fuel, "Generation (TWh)"])) for fuel in FUELS}
        if any(value < 0 for value in margins.values()) or sum(margins.values()) <= 0:
            continue
        month = pd.Timestamp(date)
        hours = int(month.days_in_month * 24)
        for tau, (box, (ef_low, ef_high)) in product(TAUS, BOXES.items()):
            low_set = sharp_k_bounds(margins, ef_low, tau, hours)
            high_set = sharp_k_bounds(margins, ef_high, tau, hours)
            nominal_set = sharp_k_bounds(margins, NOMINAL, tau, hours)
            robust_lo, robust_hi = low_set[0], high_set[1]
            if not (low_set[0] <= nominal_set[0] <= high_set[0]
                    and low_set[1] <= nominal_set[1] <= high_set[1]
                    and robust_lo <= robust_hi):
                raise AssertionError(f"EF nesting failure: {province}/{date}/{tau}/{box}")
            width = robust_hi - robust_lo
            for rate in RATES:
                required = int((hours * Fraction(str(rate))).__ceil__())
                detail.append({
                    "province": province, "month": month.strftime("%Y-%m"), "hours": hours,
                    "tau_g_kwh": tau, "ef_box": box, "target_rate": rate,
                    "required_exceedance_hours": required,
                    "nominal_k_min": nominal_set[0], "nominal_k_max": nominal_set[1],
                    "robust_k_min": robust_lo, "robust_k_max": robust_hi,
                    "robust_width_hours": width, "robust_width_fraction": width / hours,
                    "decision": category(robust_lo, robust_hi, required),
                    "nominal_inside_robust": robust_lo <= nominal_set[0] <= nominal_set[1] <= robust_hi,
                })
    result = pd.DataFrame(detail)
    if result.empty:
        raise ValueError("No complete province-month fuel margins")
    # Give each province equal weight, regardless of the number of valid months.
    province_means = result.groupby(
        ["province", "tau_g_kwh", "ef_box", "target_rate"], as_index=False
    ).agg(mean_width_hours=("robust_width_hours", "mean"),
          mean_width_fraction=("robust_width_fraction", "mean"),
          positive_rate=("decision", lambda values: (values == "positive").mean()),
          negative_rate=("decision", lambda values: (values == "negative").mean()),
          abstain_rate=("decision", lambda values: (values == "abstain").mean()))
    macro = province_means.groupby(["tau_g_kwh", "ef_box", "target_rate"], as_index=False).agg(
        mean_width_hours=("mean_width_hours", "mean"),
        mean_width_fraction=("mean_width_fraction", "mean"),
        positive_rate=("positive_rate", "mean"),
        negative_rate=("negative_rate", "mean"),
        abstain_rate=("abstain_rate", "mean"),
        regions=("province", "nunique"),
    )
    OUT.mkdir(parents=True, exist_ok=False)
    result.to_csv(OUT / "province_month_bounds.csv", index=False)
    macro.to_csv(OUT / "macro_frontier.csv", index=False)
    payload = {
        "status": "COMPLETED_ROBUST_ENDPOINT_HULL_NOT_FORECAST_COVERAGE",
        "province_month_threshold_rate_rows": len(result),
        "province_months": int(result[["province", "month"]].drop_duplicates().shape[0]),
        "provinces": sorted(result.province.unique().tolist()),
        "period": [result.month.min(), result.month.max()],
        "thresholds": list(TAUS), "target_rates": list(RATES),
        "emission_factor_boxes": {name: {"low": low, "high": high}
                                  for name, (low, high) in BOXES.items()},
        "nesting_failures": int((~result.nominal_inside_robust).sum()),
        "input_sha256": sha256(INPUT), "protocol_sha256": sha256(PROTOCOL),
        "script_sha256": sha256(Path(__file__)),
        "limitations": [
            "Ember is a retrospective secondary estimate panel, not the user's raw table or a point-in-time release.",
            "Emission-factor boxes are sensitivity assumptions, not measured Chinese uncertainty distributions.",
            "No hourly capacity, ramping, demand, imports, or minimum-output constraints are used.",
            "The robust interval reports sharp endpoints; interior integer feasibility under one common factor vector is not asserted.",
            "This is partial identification, not a probabilistic prediction interval or coverage result.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(macro.to_string(index=False))
    print(f"rows={len(result)} province_months={payload['province_months']} "
          f"nesting_failures={payload['nesting_failures']} out={OUT}")


if __name__ == "__main__":
    main()
