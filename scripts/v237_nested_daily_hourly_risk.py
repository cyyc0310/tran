"""V237: compare nested monthly, daily-total, and hourly-total CIF risk sets."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

import v235_china_load_proxy_information as v235

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V237_NESTED_DAILY_HOURLY_RISK_INFORMATION_PROTOCOL.md"
OUT = ROOT / "results/v237-nested-daily-hourly-risk-r1"
PROVINCES = ("Inner Mongolia", "Yunnan", "Jiangsu", "Guangdong", "Sichuan", "Xinjiang")
MONTHS = (1, 7)
YEAR = 2024
TAUS = (500., 900.)
THRESHOLD_MARGIN = 1e-4
TIME_LIMIT_SECONDS = 30.


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def grouped_margins(province: str, month: int, margins_by_key) -> np.ndarray:
    values = margins_by_key[(province, YEAR, month)]
    return np.array([values[0], values[2], values[1] + values[3], values[4:].sum()], dtype=float)


def monthly_only_bounds(margins: np.ndarray, ef: np.ndarray, tau: float, n_hours: int) -> tuple[int, int]:
    delta = ef - tau
    active = margins > 1e-12
    pos = bool(np.any(delta[active] > 0.))
    neg = bool(np.any(delta[active] < 0.))
    neutral = bool(np.any(delta[active] == 0.))
    if not pos:
        return 0, 0
    excess = float(delta @ margins)
    if neg:
        return (1, n_hours) if excess > 0. else (0, n_hours - 1)
    if neutral:
        return 1, n_hours
    return n_hours, n_hours


def solve_bounds(hourly_proxy: np.ndarray, daily_totals: np.ndarray, margins: np.ndarray,
                 ef: np.ndarray, tau: float, hourly_known: bool) -> list[dict]:
    n_hours, n_fuels = len(hourly_proxy), len(margins)
    n_days = len(daily_totals)
    x_count = n_hours * n_fuels
    n_vars = x_count + n_hours
    aef = ef - (tau + THRESHOLD_MARGIN)
    min_hour = float(margins.sum()) / n_hours * 1e-6
    total_rows = n_fuels + n_days + (n_hours if hourly_known else 0) + n_hours + 2 * n_hours
    row_lower = np.full(total_rows, -np.inf)
    row_upper = np.full(total_rows, np.inf)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row = 0

    # Month fuel-group margins.
    for k, amount in enumerate(margins):
        for h in range(n_hours):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = row_upper[row] = float(amount)
        row += 1

    # Daily totals; these remain present when hourly totals are known.
    for day, amount in enumerate(daily_totals):
        start, end = day * 24, (day + 1) * 24
        for h in range(start, end):
            for k in range(n_fuels):
                rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = row_upper[row] = float(amount)
        row += 1

    # Optional exact hourly totals.
    if hourly_known:
        for h, amount in enumerate(hourly_proxy):
            for k in range(n_fuels):
                rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
            row_lower[row] = row_upper[row] = float(amount)
            row += 1

    # Positive generation in every hour so CIF is defined.
    for h in range(n_hours):
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = min_hour
        row += 1

    # Tail event z=1 implies excess emissions >=0; z=0 implies <=0.
    for h in range(n_hours):
        day_cap = float(daily_totals[h // 24])
        cap = float(hourly_proxy[h]) if hourly_known else day_cap
        s_min, s_max = float(aef.min() * cap), float(aef.max() * cap)
        z = x_count + h
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(float(aef[k]))
        rows.append(row); cols.append(z); vals.append(s_min)
        row_lower[row] = s_min
        row += 1
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(float(aef[k]))
        rows.append(row); cols.append(z); vals.append(-s_max)
        row_upper[row] = 0.
        row += 1
    if row != total_rows:
        raise RuntimeError(f"constraint row mismatch: {row} != {total_rows}")

    matrix = coo_matrix((vals, (rows, cols)), shape=(total_rows, n_vars)).tocsr()
    constraints = LinearConstraint(matrix, row_lower, row_upper)
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[x_count:] = 1
    bounds = Bounds(np.zeros(n_vars), np.concatenate((np.full(x_count, np.inf), np.ones(n_hours))))
    solved = []
    for endpoint in ("min", "max"):
        objective = np.zeros(n_vars)
        objective[x_count:] = 1. if endpoint == "min" else -1.
        result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                      options={"time_limit": TIME_LIMIT_SECONDS, "mip_rel_gap": 0.})
        count = int(np.rint(result.x[x_count:].sum())) if result.x is not None else None
        solved.append({"endpoint": endpoint, "bound": count, "status": int(result.status),
                       "message": str(result.message), "optimal": bool(result.status == 0 and result.x is not None),
                       "mip_gap": float(getattr(result, "mip_gap", np.nan))})
    return solved


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    load_profiles = v235.read_load_profiles()
    margins_by_key = v235.read_ember()
    ef = np.array([1000., 800., 400., 0.])
    detail = []
    for province in PROVINCES:
        for month in MONTHS:
            key = (province, YEAR, month)
            if key not in load_profiles or key not in margins_by_key:
                raise RuntimeError(f"frozen query input missing: {key}")
            hourly_raw, _ = load_profiles[key]
            margins = grouped_margins(province, month, margins_by_key)
            month_total = float(margins.sum())
            hourly = month_total * hourly_raw / float(hourly_raw.sum())
            daily = hourly.reshape(-1, 24).sum(axis=1)
            if not np.allclose(daily.sum(), month_total, rtol=1e-12, atol=1e-10):
                raise RuntimeError(f"daily totals do not conserve month: {key}")
            for tau in TAUS:
                month_bounds = monthly_only_bounds(margins, ef, tau + THRESHOLD_MARGIN, len(hourly))
                for regime, hourly_known in (("daily_total_only", False), ("hourly_total", True)):
                    endpoints = solve_bounds(hourly, daily, margins, ef, tau, hourly_known)
                    for endpoint in endpoints:
                        detail.append({"province": province, "year": YEAR, "month": month, "tau": tau,
                                       "regime": regime, "n_hours": len(hourly),
                                       "monthly_min": month_bounds[0], "monthly_max": month_bounds[1],
                                       "monthly_width": month_bounds[1] - month_bounds[0],
                                       "endpoint": endpoint["endpoint"], "bound": endpoint["bound"],
                                       "status": endpoint["status"], "message": endpoint["message"],
                                       "optimal": endpoint["optimal"], "mip_gap": endpoint["mip_gap"]})
    endpoints = pd.DataFrame(detail)
    cases = endpoints.groupby(["province", "month", "tau", "regime"], as_index=False).agg(
        min_bound=("bound", "first"), max_bound=("bound", "last"), both_optimal=("optimal", "all"),
        monthly_min=("monthly_min", "first"), monthly_max=("monthly_max", "first"),
        monthly_width=("monthly_width", "first"))
    cases["identified_width"] = cases.max_bound - cases.min_bound
    cases["width_reduction"] = cases.monthly_width - cases.identified_width
    complete = cases.pivot(index=["province", "month", "tau"], columns="regime",
                           values=["min_bound", "max_bound", "both_optimal"])
    pair_complete = (complete[("both_optimal", "daily_total_only")]
                     & complete[("both_optimal", "hourly_total")])
    for idx in complete.index[pair_complete]:
        d_lo = complete.loc[idx, ("min_bound", "daily_total_only")]
        d_hi = complete.loc[idx, ("max_bound", "daily_total_only")]
        h_lo = complete.loc[idx, ("min_bound", "hourly_total")]
        h_hi = complete.loc[idx, ("max_bound", "hourly_total")]
        m_lo = complete.loc[idx, ("min_bound", "daily_total_only")]
        m_hi = complete.loc[idx, ("max_bound", "daily_total_only")]
        if h_lo < d_lo or h_hi > d_hi:
            raise RuntimeError(f"nested interval invariant failed at {idx}: daily=({d_lo},{d_hi}) hourly=({h_lo},{h_hi})")

    OUT.mkdir(parents=True)
    endpoints.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "nested_risk_count_bounds.csv", index=False)
    meta = {"status": "nested_information_set_geometry_not_forecast_validation",
            "protocol_sha256": protocol_hash, "load_sha256": sha256(v235.LOAD),
            "ember_sha256": sha256(v235.EMBER), "script_sha256": sha256(Path(__file__)),
            "scipy_version": scipy.__version__, "endpoint_count": int(len(endpoints)),
            "optimal_endpoint_count": int(endpoints.optimal.sum()),
            "optimal_endpoint_fraction": float(endpoints.optimal.mean()),
            "complete_nested_cases": int(pair_complete.sum()), "candidate_cases": int(len(pair_complete)),
            "provinces": list(PROVINCES), "months": list(MONTHS), "year": YEAR,
            "thresholds": list(TAUS), "threshold_margin_g_per_kwh": THRESHOLD_MARGIN,
            "time_limit_seconds_per_endpoint": TIME_LIMIT_SECONDS,
            "interpretation": "daily/hourly totals are reconstructed load profiles scaled to Ember monthly generation, not measured generation or CIF"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(cases.to_string(index=False))
    print(f"optimal={meta['optimal_endpoint_count']}/{meta['endpoint_count']} nested_cases={meta['complete_nested_cases']}/{meta['candidate_cases']}")


if __name__ == "__main__":
    main()
