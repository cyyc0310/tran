"""Prepare already examined donor histories for the V141 source-support ablation."""
from __future__ import annotations

import json
import math
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

import v120_support_gap_region_loo_cv as v120
import v139_physical_reserve_calibration as v139
import v90_coarse_capacity_identification as v90

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V141_SOURCE_DIVERSITY_ABLATION_PROTOCOL.md"
OUT = ROOT / "results/v141-source-expansion-input-r1"
OPENED = ROOT / "results/v118-region-loo-broad-cohort-diagnostic-r1/target_rows.csv"
ORIGINAL = ROOT / "results/v140-monthly-conditional-risk-r1/monthly_feature_panel.csv"
ADDITIONS = ("AVRN", "AZPS", "CISO", "CPLE", "DUK", "ERCO", "FPC", "IPCO", "LGEE", "PACE",
             "PNM", "PSCO", "SOCO", "SPA", "SRP", "TAL", "TEPC", "TVA", "WALC")
TAUS = (600, 900)


def excess_coefficients(ef: dict, tau: int) -> tuple[list[F], list[int]]:
    q = [F(str(float(ef[fuel]))) - tau for fuel in v90.FUELS]
    denominator = math.lcm(*(x.denominator for x in q))
    return q, [int(x * denominator) for x in q]


def rational_count(generation: np.ndarray, ef: dict, tau: int) -> int:
    q, _ = excess_coefficients(ef, tau)
    return sum(sum((F(str(float(value))) * coefficient for value, coefficient in zip(row, q)), F(0)) > 0
               for row in generation)


def exact_count(generation: np.ndarray, ef: dict, tau: int) -> tuple[int, int]:
    q, integer_q = excess_coefficients(ef, tau)
    integer_rows = np.equal(generation, np.floor(generation)).all(axis=1)
    if np.any(np.abs(generation[integer_rows]) > 2**53):
        raise ValueError("integer-valued float generation exceeds exact representation")
    if integer_rows.any():
        values = np.array([[int(value) for value in row] for row in generation[integer_rows]], dtype=object)
        count = int(np.sum(values @ np.array(integer_q, dtype=object) > 0))
    else:
        count = 0
    for row in generation[~integer_rows]:
        count += sum((F(str(float(value))) * coefficient for value, coefficient in zip(row, q)), F(0)) > 0
    return count, int((~integer_rows).sum())


def monthly_hull(margins: dict[str, F], ef: dict, tau: int, hours: int) -> tuple[int, int]:
    q, _ = excess_coefficients(ef, tau)
    positive = any(margins[f] > 0 and value > 0 for f, value in zip(v90.FUELS, q))
    negative = any(margins[f] > 0 and value < 0 for f, value in zip(v90.FUELS, q))
    neutral = any(margins[f] > 0 and value == 0 for f, value in zip(v90.FUELS, q))
    if not positive:
        return 0, 0
    excess = sum((margins[f] * value for f, value in zip(v90.FUELS, q)), F(0))
    if negative:
        return (1, hours) if excess > 0 else (0, hours - 1)
    if neutral:
        return 1, hours
    return hours, hours


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    protocol_hash = v139.sha(PROTOCOL)
    opened = set(pd.read_csv(OPENED).region)
    original = set(pd.read_csv(ORIGINAL).region)
    if not set(ADDITIONS).issubset(opened) or set(ADDITIONS) & original:
        raise ValueError("donor identities violate the already examined/disjoint roster")
    vectors = {name: value for name, value in v120.ef_box().items() if name in ("low", "high")}
    checks = []
    for name, ef in vectors.items():
        for tau in TAUS:
            # Includes zero, pure fuels, an exact low-factor boundary, and decimals.
            generation = np.zeros((6, len(v90.FUELS)))
            generation[1, 0] = 1
            generation[2, 1] = 1
            generation[3, :2] = (4, 5)
            generation[4, :2] = (8, 25)
            generation[5, :2] = (4.125, 5.25)
            fast, noninteger = exact_count(generation, ef, tau)
            direct = rational_count(generation, ef, tau)
            checks.append({"region": "synthetic", "month": "boundary", "factor": name, "tau": tau,
                           "fast_count": fast, "rational_count": direct, "noninteger_hours": noninteger,
                           "passed": fast == direct})
    data = v90.source_data
    data.ALL_REGIONS = ADDITIONS
    data.START, data.STOP = pd.Timestamp("2024-01-01T00:00:00Z"), pd.Timestamp("2026-01-01T00:00:00Z")
    hourly, _, audits, input_hashes = data.prepare_regions()
    records, availability = [], []
    for region in ADDITIONS:
        frame = hourly[region].copy()
        frame["month"] = pd.to_datetime(frame.time, utc=True).dt.strftime("%Y-%m-01")
        checked_first_month = False
        for month in pd.date_range("2024-01-01", "2025-12-01", freq="MS"):
            month_key = month.strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(month_key) & frame.valid.astype(bool)]
            hours, expected = len(group), month.days_in_month * 24
            eligible = hours / expected >= .80
            availability.append({"region": region, "month": month_key, "hours": hours,
                                 "expected_hours": expected, "coverage": hours / expected, "eligible": eligible})
            if not eligible:
                continue
            generation = group[[f"generation_{fuel}" for fuel in v90.FUELS]].to_numpy(float)
            if not np.isfinite(generation).all() or (generation < 0).any():
                raise ValueError("invalid generation after source proxy QA")
            margins = {fuel: sum((F(str(float(x))) for x in generation[:, j]), F(0))
                       for j, fuel in enumerate(v90.FUELS)}
            total = sum(margins.values(), F(0))
            if total <= 0:
                raise ValueError("nonpositive eligible monthly generation")
            base = {"region": region, "month": month_key, "hours": hours,
                    "log_total_mwh": math.log1p(float(total)),
                    "month_sin": math.sin(2 * math.pi * month.month / 12),
                    "month_cos": math.cos(2 * math.pi * month.month / 12)}
            base.update({f"fuelshare_{fuel}": float(margins[fuel] / total) for fuel in v90.FUELS})
            for tau in TAUS:
                observed, hulls, anchors = {}, {}, {}
                for name, ef in vectors.items():
                    observed[name], rational_hours = exact_count(generation, ef, tau)
                    hulls[name] = monthly_hull(margins, ef, tau, hours)
                    q, _ = excess_coefficients(ef, tau)
                    anchors[name] = int(sum((margins[f] * value for f, value in zip(v90.FUELS, q)), F(0)) > 0)
                    if not checked_first_month:
                        direct = rational_count(generation, ef, tau)
                        checks.append({"region": region, "month": month_key, "factor": name, "tau": tau,
                                       "fast_count": observed[name], "rational_count": direct,
                                       "noninteger_hours": rational_hours, "passed": observed[name] == direct})
                lower, upper = min(observed.values()), max(observed.values())
                physical_lower, physical_upper = min(x[0] for x in hulls.values()), max(x[1] for x in hulls.values())
                if not physical_lower <= lower <= upper <= physical_upper:
                    raise ValueError(f"monthly physical containment failed: {region}/{month_key}/{tau}")
                records.append({**base, "tau": tau, "factor_box_K_lower": lower, "factor_box_K_upper": upper,
                                "anchor_lower_fraction": min(anchors.values()),
                                "anchor_upper_fraction": max(anchors.values()),
                                "physical_lower": physical_lower, "physical_upper": physical_upper,
                                "physical_lower_fraction": physical_lower / hours,
                                "physical_upper_fraction": physical_upper / hours,
                                "physical_width_fraction": (physical_upper - physical_lower) / hours})
            checked_first_month = True
        print(f"donor panel {region}", flush=True)
    panel = pd.DataFrame(records)
    inventory = panel.loc[panel.tau.eq(900)].groupby("region").agg(
        eligible_months=("month", "size"), mean_coal_share=("fuelshare_coal", "mean"))
    inventory = inventory.reindex(ADDITIONS)
    inventory["eligible_months"] = inventory.eligible_months.fillna(0).astype(int)
    inventory["eligible_donor"] = inventory.eligible_months.ge(12)
    inventory["coal_heavy"] = inventory.mean_coal_share.ge(.40) & inventory.eligible_donor
    panel = panel.merge(inventory[["eligible_donor", "coal_heavy"]], on="region", validate="many_to_one")
    if not all(check["passed"] for check in checks):
        raise ValueError("strict-threshold exact arithmetic audit failed")
    if v139.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during data preparation")
    OUT.mkdir(parents=True)
    panel.to_csv(OUT / "additional_monthly_panel.csv", index=False)
    inventory.to_csv(OUT / "donor_inventory.csv")
    pd.DataFrame(availability).to_csv(OUT / "monthly_availability.csv", index=False)
    pd.DataFrame(checks).to_csv(OUT / "exact_count_audit.csv", index=False)
    payload = {"status": "V141_ALREADY_EXAMINED_SOURCE_HISTORY_INPUT", "candidate_regions": ADDITIONS,
               "eligible_donors": inventory.loc[inventory.eligible_donor].index.tolist(),
               "coal_heavy_donors": inventory.loc[inventory.coal_heavy].index.tolist(),
               "monthly_rows": len(panel), "region_months": int(panel[["region", "month"]].drop_duplicates().shape[0]),
               "exact_count_checks": len(checks), "exact_count_failures": sum(not r["passed"] for r in checks),
               "factor_values": vectors, "factor_exact_decimal_values": {name: {f: str(F(str(float(value))))
                    for f, value in ef.items()} for name, ef in vectors.items()},
               "raw_input_hashes": input_hashes, "source_audits": audits,
               "opened_roster_sha256": v139.sha(OPENED), "original_panel_sha256": v139.sha(ORIGINAL),
               "protocol_sha256": protocol_hash, "script_sha256": v139.sha(Path(__file__)),
               "panel_sha256": v139.sha(OUT / "additional_monthly_panel.csv"),
               "limitations": ["Source histories from region identities with already opened outcomes, not independent confirmation.",
                               "EF-box proxy labels and target-month aggregation, not observed Chinese CIF.",
                               "Floating factor-vector values are retained for compatibility with the frozen V120/V140 contract."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(inventory.to_string())
    print(f"checks: {len(checks)}, failures: 0")


if __name__ == "__main__":
    main()
