"""V359: sharp CIF-hour K hull with uncertain cross-source technology margins."""
from __future__ import annotations

import csv
import json
from fractions import Fraction as F
from itertools import product
from pathlib import Path

import pandas as pd
from scipy.optimize import linprog

from v342_china_sharp_group_hull import sharp_group_set

ROOT = Path(__file__).resolve().parents[1]
NBS = ROOT / "results/v320-oreo-official-monthly-generation-audit-r1/cn_prov_monthly_generation_by_tech.csv"
NEA = ROOT / "data/china/henan_nea_monthly_generation_2024.csv"
CAPACITY = ROOT / "data/china/cec_province_installed_capacity_2022_2023.csv"
OUT = ROOT / "results/v359-signed-multisource-margin-hull-r2"
TECHS = ("thermal", "hydro", "nuclear", "wind", "solar")
SOURCE_TECHS = ("thermal", "hydro", "wind", "solar")
TAUS = (100, 300)
THERMAL_EFS = (400, 1000)
BOX_EXPANSIONS = (0.0, 0.25, 0.50)


def ef_corners(thermal_ef: int) -> dict[str, int]:
    return {"thermal": thermal_ef, "hydro": 0, "nuclear": 0, "wind": 0, "solar": 0, "unresolved": 0}


def lp_k_set(
    lower: dict[str, F], upper: dict[str, F], total: F, caps: dict[str, F],
    ef: dict[str, int], tau: int, hours: int,
) -> tuple[int, ...] | None:
    """Exact set characterization; LP numerics are used only for strict signs."""
    if hours < 1 or total <= 0:
        raise ValueError("hours and total generation must be positive")
    if any(lower[g] < 0 or lower[g] > upper[g] for g in TECHS):
        raise ValueError("invalid source interval")
    if any(lower[g] > hours * caps[g] for g in TECHS):
        return None

    q = [F(ef[g] - tau) for g in TECHS]
    q_float = [float(x) for x in q]
    lo = [float(lower[g]) for g in TECHS]
    hi = [min(float(upper[g]), float(hours * caps[g])) for g in TECHS]
    if sum(lo) > float(total) or sum(hi) < float(total):
        return None
    equality = [[1.0] * len(TECHS)]
    equality_value = [float(total)]
    out: list[int] = []

    min_mean = linprog(q_float, A_eq=equality, b_eq=equality_value, bounds=list(zip(lo, hi)), method="highs")
    if min_mean.success and min_mean.fun <= 1e-7:
        out.append(0)
    max_mean = linprog([-x for x in q_float], A_eq=equality, b_eq=equality_value,
                       bounds=list(zip(lo, hi)), method="highs")
    if max_mean.success and -max_mean.fun > 1e-7:
        out.append(hours)

    for k in range(1, hours):
        # Variables are x[0:n] (total mass by technology), then e[0:n]
        # (mass assigned to the K exceedance hours).
        n = len(TECHS)
        # Require positive event excess; without this strict condition a zero
        # output complement could make an arbitrary subset look like K events.
        objective = q_float + [-x for x in q_float]
        a_ub: list[list[float]] = []
        b_ub: list[float] = []
        for i, g in enumerate(TECHS):
            cap = float(caps[g])
            row = [0.0] * (2 * n); row[i] = -1.0; row[n + i] = 1.0
            a_ub.append(row); b_ub.append(0.0)  # e_g <= x_g
            row = [0.0] * (2 * n); row[n + i] = 1.0
            a_ub.append(row); b_ub.append(k * cap)  # e_g <= K cap_g
            row = [0.0] * (2 * n); row[i] = 1.0; row[n + i] = -1.0
            a_ub.append(row); b_ub.append((hours - k) * cap)  # x_g-e_g <= (H-K) cap_g
        # Complement emissions <= threshold; event emissions are strictly >.
        complement_row = q_float + [-x for x in q_float]
        event_row = [0.0] * n + [-x for x in q_float]
        a_ub.append(complement_row); b_ub.append(0.0)
        a_ub.append(event_row); b_ub.append(-1e-7)
        bounds = list(zip(lo, hi)) + [(0.0, None)] * n
        result = linprog(objective, A_ub=a_ub, b_ub=b_ub,
                         A_eq=[equality[0] + [0.0] * n], b_eq=equality_value,
                         bounds=bounds, method="highs")
        if result.success and result.fun < -1e-7:
            out.append(k)
            continue
        # If the best complement is exactly at threshold, V342 admits K only
        # when the complement still carries positive generation.
        result_equal = linprog(
            [-1.0] * n + [1.0] * n,
            A_ub=[*a_ub[:3 * n], event_row], b_ub=[*b_ub[:3 * n], -1e-7],
            A_eq=[equality[0] + [0.0] * n, complement_row],
            b_eq=[*equality_value, 0.0], bounds=bounds, method="highs",
        )
        if result_equal.success and -result_equal.fun > 1e-7:
            out.append(k)
    return tuple(sorted(out))


def periods() -> list[tuple[str, int, int, list[int]]]:
    return [("Jan-Feb", 2, 1440, [1, 2])] + [
        (str(m), m, pd.Timestamp(2024, m, 1).days_in_month * 24, [m]) for m in range(3, 13)
    ]


def check_fixed_margin_reduction() -> int:
    """Cross-check the joint LP against V342 when the source box collapses."""
    cases = [
        ({"thermal": F(100), "hydro": F(20), "nuclear": F(0), "wind": F(30), "solar": F(10)},
         {"thermal": F(10), "hydro": F(5), "nuclear": F(1), "wind": F(5), "solar": F(5)}),
        ({"thermal": F(100), "hydro": F(0), "nuclear": F(0), "wind": F(0), "solar": F(0)},
         {"thermal": F(10), "hydro": F(5), "nuclear": F(1), "wind": F(5), "solar": F(5)}),
        ({"thermal": F(200), "hydro": F(10), "nuclear": F(0), "wind": F(20), "solar": F(20)},
         {"thermal": F(10), "hydro": F(5), "nuclear": F(1), "wind": F(5), "solar": F(5)}),
    ]
    comparisons = 0
    for margins, caps in cases:
        for tau, thermal_ef in product(TAUS, THERMAL_EFS):
            ef = ef_corners(thermal_ef)
            fixed = sharp_group_set(margins | {"unresolved": F(0)}, caps | {"unresolved": F(1)}, ef, tau, 24)
            lp = lp_k_set(margins, margins, sum(margins.values(), F(0)), caps, ef, tau, 24)
            if fixed is None or lp != fixed:
                raise AssertionError(f"collapsed-source LP mismatch: tau={tau}, EF={thermal_ef}, LP={lp}, V342={fixed}")
            comparisons += 1
    return comparisons


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    fixed_margin_checks = check_fixed_margin_reduction()
    nbs = pd.read_csv(NBS)
    nea = pd.read_csv(NEA)
    capdata = pd.read_csv(CAPACITY)
    caprow = capdata.loc[capdata.province.eq("Henan") & capdata.year.eq(2023)].iloc[0]
    caps = {g: F(str(caprow[g])) / 100 for g in SOURCE_TECHS}
    caps["nuclear"] = F(0)
    caps["unresolved"] = F(str(caprow.total)) / 100

    rows: list[dict[str, object]] = []
    for label, month, hours, report_months in periods():
        nbs_rows = nbs.loc[nbs.province.eq("Henan") & nbs.year.eq(2024) & nbs.month.eq(month)]
        nbs_margins = {g: F(0) for g in TECHS}
        for rec in nbs_rows.itertuples():
            if rec.technology in TECHS:
                nbs_margins[rec.technology] = F(str(rec.generation_gwh))
        ext = nea.loc[nea.month.isin(report_months)]
        total = sum((F(str(x)) * 100 for x in ext.total_100m_kwh), F(0))
        nea_margins = {g: sum((F(str(x)) * 100 for x in ext[f"{g}_100m_kwh"]), F(0))
                       for g in SOURCE_TECHS}
        nea_margins["nuclear"] = F(0)
        source_lower = {g: min(nbs_margins[g], nea_margins[g]) for g in TECHS}
        source_upper = {g: max(nbs_margins[g], nea_margins[g]) for g in TECHS}
        if sum(source_lower.values(), F(0)) > total or sum(source_upper.values(), F(0)) < total:
            raise ValueError(f"source interval cannot reconcile total for {label}")
        for expansion in BOX_EXPANSIONS:
            lower = {g: max(F(0), source_lower[g] * F(str(1 - expansion))) for g in TECHS}
            upper = {g: source_upper[g] * F(str(1 + expansion)) for g in TECHS}
            for tau, thermal_ef in product(TAUS, THERMAL_EFS):
                reconciled = lp_k_set(lower, upper, total, caps, ef_corners(thermal_ef), tau, hours)
                nbs_set = sharp_group_set(nbs_margins | {"unresolved": F(0)}, caps,
                                          ef_corners(thermal_ef), tau, hours)
                nea_set = sharp_group_set(nea_margins | {"unresolved": F(0)}, caps,
                                          ef_corners(thermal_ef), tau, hours)
                residual = total - sum(nbs_margins.values(), F(0))
                residual_set = sharp_group_set(nbs_margins | {"unresolved": residual}, caps,
                                               {"thermal": thermal_ef, "hydro": 0, "nuclear": 0,
                                                "wind": 0, "solar": 0, "unresolved": 0}, tau, hours)
                if any(x is None for x in (reconciled, nbs_set, nea_set, residual_set)):
                    raise ValueError(f"infeasible arm: {label}, tau={tau}, EF={thermal_ef}, expand={expansion}")
                rows.append({
                    "period": label, "month_index": month, "hours": hours,
                    "source_box_expansion": expansion,
                    "tau_g_kwh": tau, "thermal_ef_g_kwh": thermal_ef,
                    "total_gwh": float(total), "nbs_classified_gwh": float(sum(nbs_margins.values(), F(0))),
                    "source_residual_gwh": float(residual),
                    "reconciled_K_min": min(reconciled), "reconciled_K_max": max(reconciled),
                    "reconciled_K_width": max(reconciled) - min(reconciled),
                    "nbs_K_min": min(nbs_set), "nbs_K_max": max(nbs_set), "nbs_K_width": max(nbs_set) - min(nbs_set),
                    "nea_K_min": min(nea_set), "nea_K_max": max(nea_set), "nea_K_width": max(nea_set) - min(nea_set),
                    "net_residual_K_min": min(residual_set), "net_residual_K_max": max(residual_set),
                    "net_residual_K_width": max(residual_set) - min(residual_set),
                    "reconciled_K_values": ";".join(map(str, reconciled)),
                })
    out = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    out.to_csv(OUT / "period_factor_k_sets.csv", index=False)
    metrics = []
    for expansion, tau in product(BOX_EXPANSIONS, TAUS):
        sub = out.loc[out.tau_g_kwh.eq(tau) & out.source_box_expansion.eq(expansion)]
        period_summary = sub.groupby("period", sort=False).agg(
            hours=("hours", "first"),
            reconciled_K_min=("reconciled_K_min", "min"),
            reconciled_K_max=("reconciled_K_max", "max"),
            nbs_K_min=("nbs_K_min", "min"),
            nbs_K_max=("nbs_K_max", "max"),
            nea_K_min=("nea_K_min", "min"),
            nea_K_max=("nea_K_max", "max"),
            net_residual_K_min=("net_residual_K_min", "min"),
            net_residual_K_max=("net_residual_K_max", "max"),
        ).reset_index()
        period_summary["reconciled_K_width"] = period_summary.reconciled_K_max - period_summary.reconciled_K_min
        period_summary["nbs_K_width"] = period_summary.nbs_K_max - period_summary.nbs_K_min
        period_summary["nea_K_width"] = period_summary.nea_K_max - period_summary.nea_K_min
        period_summary["net_residual_K_width"] = period_summary.net_residual_K_max - period_summary.net_residual_K_min
        htotal = int(period_summary.hours.sum())
        metrics.append({
            "source_box_expansion": expansion, "tau_g_kwh": tau,
            "mean_reconciled_width_h": float(period_summary.reconciled_K_width.mean()),
            "mean_nbs_width_h": float(period_summary.nbs_K_width.mean()),
            "mean_nea_width_h": float(period_summary.nea_K_width.mean()),
            "mean_net_residual_width_h": float(period_summary.net_residual_K_width.mean()),
            "periods_reconciled_narrower_than_nbs": int((period_summary.reconciled_K_width < period_summary.nbs_K_width).sum()),
            "periods_reconciled_narrower_than_nea": int((period_summary.reconciled_K_width < period_summary.nea_K_width).sum()),
            "hours_weighted_reconciled_width_h": float((period_summary.reconciled_K_width * period_summary.hours).sum() / htotal),
            "period_hours": htotal,
        })
    summary = {"method": "joint LP over source interval margins and fixed total; K feasibility requires event excess > 0 and complement excess <= 0",
               "n_periods": int(out.period.nunique()), "solver": "scipy.optimize.linprog(method=highs)",
               "strict_event_tolerance": 1e-7, "fixed_margin_crosscheck_cases": fixed_margin_checks,
               "source_box_expansions": list(BOX_EXPANSIONS),
               "total_source_residual_twh": float(out.drop_duplicates('period').source_residual_gwh.sum() / 1000),
               "metrics": metrics}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
