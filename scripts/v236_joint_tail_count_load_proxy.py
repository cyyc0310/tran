"""V236: MILP sharp sets for monthly CIF threshold-hour counts."""
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
PROTOCOL = ROOT / "docs/V236_JOINT_TAIL_COUNT_LOAD_PROXY_PROTOCOL.md"
OUT = ROOT / "results/v236-joint-tail-count-load-proxy-r3"
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
    # Order is coal, other fossil, gas+bioenergy, non-fossil.
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


def milp_count_bounds(hourly_total: np.ndarray, margins: np.ndarray, ef: np.ndarray, tau: float,
                      time_limit: float) -> list[dict]:
    h_count, k_count = len(hourly_total), len(margins)
    x_count = h_count * k_count
    n_vars = x_count + h_count
    aef = ef - tau
    epsilon = THRESHOLD_MARGIN * hourly_total
    rows, cols, vals = [], [], []
    lower = np.full(h_count + k_count + 2 * h_count, -np.inf)
    upper = np.full(h_count + k_count + 2 * h_count, np.inf)

    # Hourly total equalities.
    for h, total in enumerate(hourly_total):
        row = h
        for k in range(k_count):
            rows.append(row); cols.append(h * k_count + k); vals.append(1.)
        lower[row] = upper[row] = float(total)

    # Monthly fuel-group margin equalities.
    for k, amount in enumerate(margins):
        row = h_count + k
        for h in range(h_count):
            rows.append(row); cols.append(h * k_count + k); vals.append(1.)
        lower[row] = upper[row] = float(amount)

    # For each hour, binary z=1 means CIF >= tau + epsilon-intensity.
    for h, total in enumerate(hourly_total):
        s_min = float(aef.min() * total)
        s_max = float(aef.max() * total)
        z_idx = x_count + h
        row_lo = h_count + k_count + h
        for k in range(k_count):
            rows.append(row_lo); cols.append(h * k_count + k); vals.append(float(aef[k]))
        rows.append(row_lo); cols.append(z_idx); vals.append(-(float(epsilon[h]) - s_min))
        lower[row_lo] = s_min

        row_hi = h_count + k_count + h_count + h
        for k in range(k_count):
            rows.append(row_hi); cols.append(h * k_count + k); vals.append(float(aef[k]))
        rows.append(row_hi); cols.append(z_idx); vals.append(-(s_max - float(epsilon[h])))
        upper[row_hi] = float(epsilon[h])

    matrix = coo_matrix((vals, (rows, cols)), shape=(len(lower), n_vars)).tocsr()
    constraints = LinearConstraint(matrix, lower, upper)
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[x_count:] = 1
    lb = np.zeros(n_vars, dtype=float)
    ub = np.concatenate((np.full(x_count, np.inf), np.ones(h_count)))
    bounds = Bounds(lb, ub)
    results = []
    for direction in ("min", "max"):
        objective = np.zeros(n_vars, dtype=float)
        objective[x_count:] = 1. if direction == "min" else -1.
        result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                      options={"time_limit": time_limit, "mip_rel_gap": 0.})
        count = None
        if result.x is not None:
            count = int(np.rint(result.x[x_count:].sum()))
        results.append({
            "direction": direction, "status": int(result.status), "message": str(result.message),
            "count": count, "objective": float(result.fun) if result.fun is not None else None,
            "mip_gap": float(getattr(result, "mip_gap", np.nan)),
            "dual_bound": float(getattr(result, "mip_node_count", np.nan)) if False else None,
            "success": bool(result.status == 0 and result.x is not None),
        })
    return results


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    load_profiles = v235.read_load_profiles()
    margins_by_key = v235.read_ember()
    rows = []
    grouped_ef = np.array([1000., 800., 400., 0.])
    for province in PROVINCES:
        for month in MONTHS:
            key = (province, YEAR, month)
            if key not in margins_by_key or key not in load_profiles:
                raise RuntimeError(f"frozen query input missing: {key}")
            margins8 = margins_by_key[key]
            margins4 = grouped_margins(province, month, margins_by_key)
            load_hourly, load_daily = load_profiles[key]
            scale = float(margins8.sum())
            regimes = {
                "daily_flat": scale * load_daily / float(load_daily.sum()),
                "hourly_profile": scale * load_hourly / float(load_hourly.sum()),
            }
            n_hours = len(load_hourly)
            for tau in TAUS:
                only = monthly_only_bounds(margins4, grouped_ef, tau + THRESHOLD_MARGIN, n_hours)
                for regime, totals in regimes.items():
                    endpoints = milp_count_bounds(totals, margins4, grouped_ef, tau, TIME_LIMIT_SECONDS)
                    for endpoint in endpoints:
                        rows.append({
                            "province": province, "year": YEAR, "month": month, "tau": tau,
                            "regime": regime, "monthly_only_min": only[0], "monthly_only_max": only[1],
                            "monthly_only_width": only[1] - only[0], "n_hours": n_hours,
                            "margin_coal_twh": margins4[0], "margin_other_fossil_twh": margins4[1],
                            "margin_gas_bio_twh": margins4[2], "margin_nonfossil_twh": margins4[3],
                            "endpoint": endpoint["direction"], "bound": endpoint["count"],
                            "milp_status": endpoint["status"], "milp_message": endpoint["message"],
                            "milp_gap": endpoint["mip_gap"], "optimal": endpoint["success"],
                        })
    detail = pd.DataFrame(rows)
    cases = detail.groupby(["province", "month", "tau", "regime"], as_index=False).agg(
        min_bound=("bound", lambda s: s.iloc[0] if len(s) else np.nan),
        max_bound=("bound", lambda s: s.iloc[1] if len(s) > 1 else np.nan),
        both_optimal=("optimal", "all"), monthly_only_min=("monthly_only_min", "first"),
        monthly_only_max=("monthly_only_max", "first"), monthly_only_width=("monthly_only_width", "first"),
    )
    cases["constrained_width"] = cases.max_bound - cases.min_bound
    cases["width_reduction"] = cases.monthly_only_width - cases.constrained_width
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "risk_count_bounds.csv", index=False)
    meta = {
        "status": "risk_count_identification_not_forecast_validation",
        "protocol_sha256": protocol_hash, "load_sha256": sha256(v235.LOAD), "ember_sha256": sha256(v235.EMBER),
        "script_sha256": sha256(Path(__file__)), "scipy_version": scipy.__version__,
        "provinces": list(PROVINCES), "months": list(MONTHS), "year": YEAR,
        "thresholds": list(TAUS), "threshold_margin_g_per_kwh": THRESHOLD_MARGIN,
        "time_limit_seconds_per_endpoint": TIME_LIMIT_SECONDS,
        "endpoint_count": int(len(detail)), "optimal_endpoint_count": int(detail.optimal.sum()),
        "optimal_endpoint_fraction": float(detail.optimal.mean()),
        "queries": int(cases.shape[0]),
        "interpretation": "all results conditional on estimated hourly demand normalized to monthly Ember generation total; not observed generation/CIF",
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(cases.to_string(index=False))
    print(f"optimal={meta['optimal_endpoint_count']}/{meta['endpoint_count']} output={OUT}")


if __name__ == "__main__":
    main()
