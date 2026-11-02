#!/usr/bin/env python3
"""V401: sharp hourly tail-count bounds from monthly generation margins."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from v369_crossfit_risk_intervals import REGIONS, THRESHOLDS, load_hourly


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v401-monthly-margin-tail-identification-r7"
THERMAL = ["generation_coal", "generation_gas", "generation_oil"]
HYDRO = ["generation_hydro"]
NUCLEAR = ["generation_nuclear"]
WIND = ["generation_wind"]
SOLAR = ["generation_solar"]
OTHER = ["generation_other"]
GROUPS = [THERMAL, HYDRO, NUCLEAR, WIND, SOLAR, OTHER]
GROUP_NAMES = ["thermal", "hydro", "nuclear", "wind", "solar", "other"]
OBSERVED_FACTOR = 700.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solve_tail_count(
    hourly_totals: np.ndarray,
    monthly_thermal_margin: float,
    daily_ids: np.ndarray | None,
    daily_thermal_margins: np.ndarray | None,
    thermal_ef: float,
    threshold: float,
    maximize: bool,
) -> tuple[int, float]:
    """Optimize the count above threshold over all feasible hourly allocations."""
    hours = len(hourly_totals)
    n_vars = 2 * hours
    z_offset = hours

    # One thermal variable plus one binary crossing indicator per hour.
    n_days = 0 if daily_thermal_margins is None else len(daily_thermal_margins)
    n_margin_rows = n_days if n_days else 1
    matrix = lil_matrix((2 * hours + n_margin_rows, n_vars), dtype=float)
    lower = np.full(2 * hours + n_margin_rows, -np.inf, dtype=float)
    upper = np.full(2 * hours + n_margin_rows, np.inf, dtype=float)

    row = 0
    if n_days:
        assert daily_ids is not None and daily_thermal_margins is not None
        for day in range(n_days):
            indices = np.flatnonzero(daily_ids == day)
            matrix[row, indices] = 1.0
            lower[row] = upper[row] = float(daily_thermal_margins[day])
            row += 1
    else:
        matrix[row, :hours] = 1.0
        lower[row] = upper[row] = float(monthly_thermal_margin)
        row += 1

    crossing_epsilon = 1e-5
    big_m_scale = max(abs(thermal_ef - threshold), abs(threshold))
    big_m = np.maximum(big_m_scale * hourly_totals, 1e-6)
    for hour in range(hours):
        # D_h = (EF - threshold) * thermal_h - threshold * nonthermal_h,
        # with nonthermal_h = hourly_total_h - thermal_h.
        matrix[row, hour] = thermal_ef
        matrix[row, z_offset + hour] = -big_m[hour]
        lower[row] = crossing_epsilon - big_m[hour] + threshold * hourly_totals[hour]
        upper[row] = np.inf
        row += 1

        matrix[row, hour] = thermal_ef
        matrix[row, z_offset + hour] = -big_m[hour]
        lower[row] = -np.inf
        upper[row] = threshold * hourly_totals[hour]
        row += 1

    # For each hour, z=1 iff D>0 (crossing); the big-M relaxations are valid
    # because the hourly total bounds D between -threshold*T and (EF-threshold)*T.

    objective = np.zeros(n_vars, dtype=float)
    objective[z_offset:] = -1.0 if maximize else 1.0
    variable_lower = np.zeros(n_vars, dtype=float)
    variable_upper = np.ones(n_vars, dtype=float)
    variable_upper[:hours] = hourly_totals
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[z_offset:] = 1

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(variable_lower, variable_upper),
        constraints=LinearConstraint(matrix.tocsr(), lower, upper),
        options={"time_limit": 30.0, "mip_rel_gap": 1e-8},
    )
    if result.status != 0 or result.x is None:
        raise RuntimeError(
            f"MILP not proven optimal (status={result.status}, message={result.message})"
        )
    count = int(np.rint(result.x[z_offset:].sum()))
    return count, float(getattr(result, "mip_gap", 0.0) or 0.0)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")

    hourly = load_hourly()
    rows: list[dict[str, object]] = []
    source_path = ROOT / "results/research/data"
    source_hashes = {
        f"{region}_hourly.csv": sha256(source_path / f"{region}_hourly.csv")
        for region in REGIONS
    }

    for (region, year, month), block in hourly.groupby(["region", "year", "month"], sort=True):
        if int(year) != 2024:
            continue
        block = block.sort_values("time")
        grouped = np.column_stack(
            [block.loc[:, columns].sum(axis=1).to_numpy(float) for columns in GROUPS]
        )
        hourly_totals = grouped.sum(axis=1)
        monthly_margins = grouped.sum(axis=0)
        if np.any(hourly_totals <= 0):
            raise RuntimeError(f"nonpositive hourly total in {region} {year}-{month}")
        if not np.isclose(monthly_margins.sum(), hourly_totals.sum(), rtol=1e-9, atol=1e-6):
            raise RuntimeError(f"monthly margin does not balance for {region} {year}-{month}")

        hours = len(block)
        daily_ids, _ = pd.factorize(block["time"].dt.floor("D"), sort=True)
        daily_thermal_margins = np.bincount(daily_ids, weights=grouped[:, 0])
        thermal_share = grouped[:, 0] / hourly_totals
        observed_proxy = OBSERVED_FACTOR * thermal_share
        monthly_mean_proxy = OBSERVED_FACTOR * monthly_margins[0] / monthly_margins.sum()
        for threshold in THRESHOLDS:
            has_thermal = monthly_margins[0] > 1e-9
            has_nonthermal = monthly_margins[1:].sum() > 1e-9
            has_both_classes = has_thermal and has_nonthermal
            if threshold >= OBSERVED_FACTOR:
                m_only_lower = m_only_upper = 0
            elif not has_both_classes:
                m_only_lower = m_only_upper = (
                    hours if has_thermal and OBSERVED_FACTOR > threshold else 0
                )
            elif monthly_mean_proxy > threshold:
                m_only_lower, m_only_upper = 1, hours
            else:
                m_only_lower, m_only_upper = 0, hours - 1
            m_t_lower, m_t_lower_gap = solve_tail_count(
                hourly_totals,
                float(monthly_margins[0]),
                None,
                None,
                OBSERVED_FACTOR,
                float(threshold),
                False,
            )
            m_t_upper, m_t_upper_gap = solve_tail_count(
                hourly_totals,
                float(monthly_margins[0]),
                None,
                None,
                OBSERVED_FACTOR,
                float(threshold),
                True,
            )
            daily_lower, daily_lower_gap = solve_tail_count(
                hourly_totals,
                float(monthly_margins[0]),
                daily_ids,
                daily_thermal_margins,
                OBSERVED_FACTOR,
                float(threshold),
                False,
            )
            daily_upper, daily_upper_gap = solve_tail_count(
                hourly_totals,
                float(monthly_margins[0]),
                daily_ids,
                daily_thermal_margins,
                OBSERVED_FACTOR,
                float(threshold),
                True,
            )
            actual_count = int(np.sum(observed_proxy > threshold))
            rows.append(
                {
                    "region": region,
                    "year": int(year),
                    "month": int(month),
                    "threshold_g_kwh": float(threshold),
                    "hours": hours,
                    "observed_proxy_factor_g_kwh": OBSERVED_FACTOR,
                    "thermal_margin_mwh": float(monthly_margins[0]),
                    "total_generation_mwh": float(monthly_margins.sum()),
                    "thermal_share_month": float(monthly_margins[0] / monthly_margins.sum()),
                    "monthly_energy_weighted_proxy_mean_g_kwh": monthly_mean_proxy,
                    "observed_proxy_tail_count": actual_count,
                    "m_only_lower_count": m_only_lower,
                    "m_only_upper_count": m_only_upper,
                    "m_only_width_hours": m_only_upper - m_only_lower,
                    "m_only_width_fraction": (m_only_upper - m_only_lower) / hours,
                    "hourly_total_oracle_lower_count": m_t_lower,
                    "hourly_total_oracle_upper_count": m_t_upper,
                    "hourly_total_oracle_width_hours": m_t_upper - m_t_lower,
                    "hourly_total_oracle_width_fraction": (m_t_upper - m_t_lower) / hours,
                    "hourly_total_oracle_covers_proxy": m_t_lower <= actual_count <= m_t_upper,
                    "daily_thermal_oracle_lower_count": daily_lower,
                    "daily_thermal_oracle_upper_count": daily_upper,
                    "daily_thermal_oracle_width_hours": daily_upper - daily_lower,
                    "daily_thermal_oracle_width_fraction": (daily_upper - daily_lower) / hours,
                    "daily_thermal_oracle_covers_proxy": daily_lower <= actual_count <= daily_upper,
                    "hourly_total_lower_solver_gap": m_t_lower_gap,
                    "hourly_total_upper_solver_gap": m_t_upper_gap,
                    "daily_thermal_lower_solver_gap": daily_lower_gap,
                    "daily_thermal_upper_solver_gap": daily_upper_gap,
                    "daily_thermal_margin_count": len(daily_thermal_margins),
                }
            )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise RuntimeError("no 2024 region-month observations found")
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_bounds.csv", index=False)
    by_region = detail.groupby(["region", "threshold_g_kwh"], as_index=False).agg(
        months=("month", "nunique"),
        mean_observed_tail_count=("observed_proxy_tail_count", "mean"),
        mean_oracle_width_hours=("hourly_total_oracle_width_hours", "mean"),
        mean_oracle_width_fraction=("hourly_total_oracle_width_fraction", "mean"),
        proxy_count_coverage=("hourly_total_oracle_covers_proxy", "mean"),
        mean_daily_oracle_width_hours=("daily_thermal_oracle_width_hours", "mean"),
        mean_daily_oracle_width_fraction=("daily_thermal_oracle_width_fraction", "mean"),
        daily_oracle_proxy_count_coverage=("daily_thermal_oracle_covers_proxy", "mean"),
        max_solver_gap=("hourly_total_upper_solver_gap", "max"),
        max_daily_solver_gap=("daily_thermal_upper_solver_gap", "max"),
    )
    by_region.to_csv(OUT / "region_summary.csv", index=False)
    m_only_coverage = detail.assign(
        m_only_covers_proxy=(
            detail.m_only_lower_count.le(detail.observed_proxy_tail_count)
            & detail.m_only_upper_count.ge(detail.observed_proxy_tail_count)
        )
    ).groupby(["region", "threshold_g_kwh"], as_index=False).agg(
        m_only_proxy_count_coverage=("m_only_covers_proxy", "mean")
    )
    m_only_region = detail.groupby(["region", "threshold_g_kwh"], as_index=False).agg(
        mean_m_only_width_fraction=("m_only_width_fraction", "mean"),
        mean_m_only_width_hours=("m_only_width_hours", "mean"),
    ).merge(
        m_only_coverage, on=["region", "threshold_g_kwh"], validate="one_to_one"
    )
    summary = by_region.merge(
        m_only_region, on=["region", "threshold_g_kwh"], validate="one_to_one"
    ).groupby("threshold_g_kwh", as_index=False).agg(
        regions=("region", "nunique"),
        region_equal_mean_observed_tail_count=("mean_observed_tail_count", "mean"),
        region_equal_m_only_width_hours=("mean_m_only_width_hours", "mean"),
        region_equal_m_only_width_fraction=("mean_m_only_width_fraction", "mean"),
        region_equal_mean_oracle_width_hours=("mean_oracle_width_hours", "mean"),
        region_equal_mean_oracle_width_fraction=("mean_oracle_width_fraction", "mean"),
        region_equal_m_only_proxy_count_coverage=("m_only_proxy_count_coverage", "mean"),
        region_equal_proxy_count_coverage=("proxy_count_coverage", "mean"),
        worst_region_proxy_count_coverage=("proxy_count_coverage", "min"),
        worst_region_mean_width_fraction=("mean_oracle_width_fraction", "max"),
        max_solver_gap=("max_solver_gap", "max"),
        region_equal_daily_oracle_width_hours=("mean_daily_oracle_width_hours", "mean"),
        region_equal_daily_oracle_width_fraction=("mean_daily_oracle_width_fraction", "mean"),
        region_equal_daily_oracle_coverage=("daily_oracle_proxy_count_coverage", "mean"),
        worst_region_daily_oracle_width_fraction=("mean_daily_oracle_width_fraction", "max"),
        max_daily_solver_gap=("max_daily_solver_gap", "max"),
    )
    summary["m_only_reduction_fraction"] = np.where(
        summary.region_equal_m_only_width_fraction > 0,
        (
            summary.region_equal_m_only_width_fraction
            - summary.region_equal_mean_oracle_width_fraction
        )
        / summary.region_equal_m_only_width_fraction,
        np.nan,
    )
    summary["daily_reduction_vs_m_only_fraction"] = np.where(
        summary.region_equal_m_only_width_fraction > 0,
        (
            summary.region_equal_m_only_width_fraction
            - summary.region_equal_daily_oracle_width_fraction
        )
        / summary.region_equal_m_only_width_fraction,
        np.nan,
    )
    summary["daily_incremental_reduction_vs_hourly_total_fraction"] = np.where(
        summary.region_equal_mean_oracle_width_fraction > 0,
        (
            summary.region_equal_mean_oracle_width_fraction
            - summary.region_equal_daily_oracle_width_fraction
        )
        / summary.region_equal_mean_oracle_width_fraction,
        np.nan,
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    metadata = {
        "experiment": "V401 monthly-margin tail-count identification with hourly-total and daily-thermal oracles",
        "target": "700 g/kWh fixed thermal-factor generation proxy; not measured or official CIF",
        "target_year": 2024,
        "regions": list(REGIONS),
        "groups": GROUP_NAMES,
        "thermal_group": THERMAL,
        "nonthermal_grouping": "hydro, nuclear, wind, solar, and the source-only other residual; all zero fixed-factor proxy intensity",
        "scored_proxy_factor_g_kwh": OBSERVED_FACTOR,
        "thresholds_g_kwh": [float(value) for value in THRESHOLDS],
        "information_sets": {
            "M": "monthly five-class margins only; analytical sharp count bounds from the monthly energy-weighted proxy mean and y in [0,700]",
            "M+T": "same margins plus true hourly total generation, fuel allocation hidden; oracle only",
            "M+T+D": "M+T plus true UTC-day thermal generation margins, hourly allocation remains unknown; oracle only",
        },
        "optimization": "MILP with nonnegative hourly group allocations, exact hourly totals, exact monthly and UTC-day thermal margins, binary indicators for strict crossing above threshold (1e-5 g/kWh*MWh tolerance)",
        "limitations": [
            "all target regions and 2024 outcomes were previously opened; development identification analysis only",
            "hourly total generation is an oracle and is not in the currently available China data contract",
            "fixed-factor proxy is not measured emissions, dispatch-aware marginal CIF, or consumption-side CIF",
            "bounds are conditional on the fixed 700 g/kWh thermal factor and five-class mapping",
        ],
        "source_data_sha256": source_hashes,
        "runner_sha256": sha256(Path(__file__)),
        "summary_rows": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
        "detail_sha256": sha256(OUT / "region_month_bounds.csv"),
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
