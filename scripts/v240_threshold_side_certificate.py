"""Audit the threshold-straddling EF criterion against V240's EIA cases."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402

OUT = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1"
BA_MONTHS = {
    "AZPS": (4, 7, 10), "CPLE": (4, 7, 10), "ERCO": (4, 7, 10),
    "PACE": (4, 7, 10), "SOCO": (4, 7, 10), "PSCO": (4,),
}
YEAR = 2025
TAU_STAR = 500.0001
EF_TOL = 0.10


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    milp = pd.read_csv(OUT / "risk_count_bounds.csv")
    rows = []
    for region, months in BA_MONTHS.items():
        for month in months:
            d = series[(region, YEAR, month)]
            margins = np.asarray(d["fuel_totals"], float)
            nominal_mean = float(margins @ partial.EF / margins.sum())
            low_mean, high_mean = nominal_mean * (1 - EF_TOL), nominal_mean * (1 + EF_TOL)
            critical_relative_ef_tolerance = abs(TAU_STAR / nominal_mean - 1.)
            n_hours = len(d["time"])
            low_all_below = low_mean < TAU_STAR
            high_all_above = high_mean >= TAU_STAR
            straddles = low_all_below and high_all_above

            # Proportional hourly allocation x[h,f] = G[h] * M[f]/M.
            # It preserves every month's source margin and every exact hourly
            # total. Under a constant-in-hours EF vector, every hour then has
            # the corresponding monthly weighted mean.
            hourly = np.asarray(d["fuel_profile"], float) @ margins
            allocation = hourly[:, None] * (margins / margins.sum())[None, :]
            margin_error = float(np.max(np.abs(allocation.sum(axis=0) - margins)))
            total_error = float(np.max(np.abs(allocation.sum(axis=1) - hourly)))
            if margin_error > 1e-6 or total_error > 1e-8:
                raise RuntimeError(f"proportional witness does not conserve totals at {(region, month)}")

            obs = milp[(milp.region == region) & (milp.month == month)]
            daily = obs[obs.regime == "daily_total_only"].iloc[0]
            hour = obs[obs.regime == "hourly_total"].iloc[0]
            full_range = (daily.min_bound == 0 and daily.max_bound == n_hours
                          and hour.min_bound == 0 and hour.max_bound == n_hours)
            rows.append({"region": region, "year": YEAR, "month": month, "n_hours": n_hours,
                         "nominal_monthly_mean_cif": nominal_mean,
                         "ef_box_monthly_mean_low": low_mean, "ef_box_monthly_mean_high": high_mean,
                         "critical_relative_ef_tolerance_to_reach_threshold": critical_relative_ef_tolerance,
                         "threshold_star": TAU_STAR,
                         "low_factor_all_hours_below_possible": low_all_below,
                         "high_factor_all_hours_above_possible": high_all_above,
                         "threshold_straddled": straddles,
                         "daily_outer_set_min": int(daily.min_bound), "daily_outer_set_max": int(daily.max_bound),
                         "hourly_outer_set_min": int(hour.min_bound), "hourly_outer_set_max": int(hour.max_bound),
                         "full_0_to_month_hours_range": bool(full_range),
                         "proportional_witness_margin_error": margin_error,
                         "proportional_witness_hourly_total_error": total_error})
    cert = pd.DataFrame(rows)
    if not (cert.threshold_straddled == cert.full_0_to_month_hours_range).all():
        raise RuntimeError("threshold-side certificate disagrees with the observed V240 outer sets")
    path = OUT / "threshold_side_certificate.csv"
    cert.to_csv(path, index=False)
    metadata = {"criterion": "EF uncertainty box straddles tau*: low proportional-mix mean < tau*, high proportional-mix mean >= tau*",
                "sample_count": int(len(cert)), "threshold_straddled_cases": int(cert.threshold_straddled.sum()),
                "straddled_cases_with_full_risk_range": int((cert.threshold_straddled & cert.full_0_to_month_hours_range).sum()),
                "non_straddled_cases_with_narrowed_hourly_range": int((~cert.threshold_straddled & (cert.hourly_outer_set_max - cert.hourly_outer_set_min < cert.n_hours)).sum()),
                "max_proportional_witness_margin_error": float(cert.proportional_witness_margin_error.max()),
                "max_proportional_witness_hourly_total_error": float(cert.proportional_witness_hourly_total_error.max()),
                "source_v240_bounds_sha256": sha256(OUT / "risk_count_bounds.csv"),
                "certificate_sha256": sha256(path), "input_hashes": input_hashes,
                "interpretation": "analytic proportional-mix obstruction with dataset-specific witness audit; not a universal sufficiency criterion for narrowing"}
    (OUT / "threshold_side_certificate.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
