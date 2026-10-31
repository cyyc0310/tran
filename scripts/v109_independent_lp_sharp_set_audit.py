"""V109: independently verify complete grouped-capacity K sets with LPs."""
from __future__ import annotations

import hashlib
import json
import sys
from fractions import Fraction as F
from pathlib import Path

import pandas as pd
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402
from scripts.v92_sharp_coarse_capacity_identification import sharp_group_set  # noqa: E402

REGIONS = ("CPLE", "FPC", "IPCO", "PNM", "SOCO")
V104_ROWS = ROOT / "results/v104-unseen-region-robust-confirmation-r1/region_month_scores.csv"
OUT = ROOT / "results/v109-independent-lp-sharp-set-audit-r1"
PROTOCOL = ROOT / "docs/V109_INDEPENDENT_LP_SHARP_SET_AUDIT_PROTOCOL.md"
TAUS = (600, 900)
DELTA = 0.10
REL_TOL = 1e-8


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def capacity_polytope(margins, caps, k: int, hours: int):
    fuels = list(v90.FUELS)
    bounds = [(0.0, float(margins[fuel])) for fuel in fuels]
    a_ub, b_ub = [], []
    for group, members in v90.GROUPS.items():
        indexes = [fuels.index(fuel) for fuel in members]
        amount = sum(float(margins[fuel]) for fuel in members)
        lo = max(0.0, amount - (hours - k) * float(caps[group]))
        hi = min(amount, k * float(caps[group]))
        upper = [0.0] * len(fuels)
        lower = [0.0] * len(fuels)
        for index in indexes:
            upper[index] = 1.0
            lower[index] = -1.0
        a_ub.extend((upper, lower))
        b_ub.extend((hi, -lo))
    return fuels, bounds, a_ub, b_ub


def lp_k_set(margins, caps, tau: int, hours: int, factors):
    fuels = list(v90.FUELS)
    amounts = [float(margins[fuel]) for fuel in fuels]
    q = [float(factors[fuel] - tau) for fuel in fuels]
    total = sum(amounts)
    whole_excess = sum(q[i] * amounts[i] for i in range(len(fuels)))
    qscale = max(1.0, abs(whole_excess), sum(abs(q[i] * amounts[i]) for i in range(len(fuels))))
    eps_q = REL_TOL * qscale
    eps_g = REL_TOL * max(1.0, total)
    identified = []
    near = []

    # At the two endpoints, every hour has the same feasible monthly mean mix.
    if whole_excess <= eps_q:
        identified.append(0)
    if whole_excess > eps_q:
        identified.append(hours)

    for k in range(1, hours):
        fuels, bounds, a_ub, b_ub = capacity_polytope(margins, caps, k, hours)
        result = linprog(c=[-value for value in q], A_ub=a_ub, b_ub=b_ub,
                         bounds=bounds, method="highs")
        if result.status == 2:
            continue
        if not result.success:
            raise RuntimeError(f"LP failed for k={k}: {result.message}")
        maximum = -float(result.fun)
        threshold = max(0.0, whole_excess)
        if abs(maximum - threshold) <= 10 * eps_q:
            near.append({"k": k, "event_excess_max": maximum, "monthly_excess": whole_excess})
        if maximum <= eps_q or maximum < whole_excess - eps_q:
            continue
        if maximum > whole_excess + eps_q:
            identified.append(k)
            continue

        # On the equality face, require positive complement generation.
        q_floor = [(-value) for value in q]
        a_eq_face = list(a_ub) + [q_floor]
        b_eq_face = list(b_ub) + [-whole_excess + eps_q]
        face = linprog(c=[1.0] * len(fuels), A_ub=a_eq_face, b_ub=b_eq_face,
                       bounds=bounds, method="highs")
        if not face.success:
            continue
        complement_generation = total - sum(float(value) for value in face.x)
        if complement_generation > eps_g:
            identified.append(k)
    return tuple(identified), near


def sharp_set(margins, caps, tau, hours, factors):
    old = dict(v90.EF)
    try:
        v90.EF.clear()
        v90.EF.update(factors)
        return sharp_group_set(margins, caps, tau, hours)
    finally:
        v90.EF.clear()
        v90.EF.update(old)


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    v104 = pd.read_csv(V104_ROWS)
    if set(v104.region.unique()) != set(REGIONS):
        raise ValueError("V104 cohort roster mismatch")
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + REGIONS))
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()
    capacities, capacity_hash = v90.capacities(2025)
    nominal = dict(v90.EF)
    low = {fuel: ef * (1 - DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
           for fuel, ef in nominal.items()}
    high = {fuel: ef * (1 + DELTA) if fuel in ("coal", "gas", "oil", "other") else ef
            for fuel, ef in nominal.items()}
    factors = {"low": low, "nominal": nominal, "high": high}
    rows, near_rows = [], []

    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-01-01") & times.lt("2026-07-01")
                          & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month_key, group in frame.groupby("month", sort=True):
            hours = len(group)
            margins = {fuel: F(str(float(x))) for fuel in v90.FUELS
                       for x in [group[f"generation_{fuel}"].sum()]}
            caprow = capacities.loc[region] if region in capacities.index else pd.Series(dtype=float)
            fine = {fuel: float(caprow.get(fuel, 0)) for fuel in
                    ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            group_caps = {name: sum(fine[fuel] for fuel in members)
                          for name, members in v90.GROUPS.items()}
            reference = v104.loc[v104.region.eq(region)
                                 & v104.month.astype(str).str[:10].eq(month_key)]
            for tau in TAUS:
                for factor_name, ef in factors.items():
                    lp_set, near = lp_k_set(margins, group_caps, tau, hours, ef)
                    exact = sharp_set(margins, group_caps, tau, hours, ef)
                    if lp_set != exact:
                        raise ValueError(f"full K-set disagreement {region}/{month_key}/{tau}/{factor_name}: "
                                         f"LP={lp_set}, exact={exact}")
                    for item in near:
                        near_rows.append({"region": region, "month": month_key, "tau": tau,
                                          "factor_vector": factor_name, **item})
                    v104_tau = reference.loc[reference.tau.eq(tau)]
                    if len(v104_tau) != 3:
                        raise ValueError(f"missing V104 risk-rate rows {region}/{month_key}/{tau}")
                    saved = v104_tau.iloc[0]
                    rows.append({"region": region, "month": month_key, "tau": tau,
                                 "factor_vector": factor_name, "hours": hours,
                                 "lp_set_size": len(lp_set), "exact_set_size": len(exact),
                                 "lp_lower": min(lp_set), "lp_upper": max(lp_set),
                                 "exact_lower": min(exact), "exact_upper": max(exact),
                                 "complete_set_equal": True,
                                 "v104_nominal_endpoints_match": bool(
                                     factor_name != "nominal" or
                                     (min(lp_set) == int(saved.nominal_lower)
                                      and max(lp_set) == int(saved.nominal_upper))),
                                 "v104_robust_hull_contains_endpoints": bool(
                                     factor_name == "nominal" or
                                     (min(lp_set) >= int(saved.robust_lower)
                                      and max(lp_set) <= int(saved.robust_upper))),
                                 "near_boundary_k_count": len(near)})
        print(f"LP audited region {region}", flush=True)

    audit = pd.DataFrame(rows)
    if not audit.complete_set_equal.all():
        raise ValueError("complete-set comparison failed")
    if not audit.v104_nominal_endpoints_match.all():
        raise ValueError("nominal V104 endpoint discrepancy")
    if not audit.v104_robust_hull_contains_endpoints.all():
        raise ValueError("V104 robust hull discrepancy")
    OUT.mkdir(parents=True, exist_ok=False)
    audit.to_csv(OUT / "complete_set_lp_comparison.csv", index=False)
    pd.DataFrame(near_rows).to_csv(OUT / "near_boundary_cases.csv", index=False)
    summary = {"status": "INDEPENDENT_FLOATING_POINT_LP_COMPLETE_SET_AUDIT",
               "regions": list(REGIONS), "region_months": int(audit[["region", "month"]].drop_duplicates().shape[0]),
               "set_comparisons": len(audit), "complete_K_sets_equal": bool(audit.complete_set_equal.all()),
               "v104_nominal_endpoints_match": bool(audit.v104_nominal_endpoints_match.all()),
               "v104_robust_hull_contains_low_high_sets": bool(audit.v104_robust_hull_contains_endpoints.all()),
               "near_boundary_cases": int(len(near_rows)), "lp_relative_tolerance": REL_TOL,
               "factor_vectors": {"low": low, "nominal": nominal, "high": high},
               "hourly_source_sha256": hourly_hashes, "capacity_source_sha256": capacity_hash,
               "source_audits": audits, "v104_rows_sha256": sha(V104_ROWS),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Numerical LP cross-check is not an analytic proof or independent software package implementation.",
                               "Near-boundary cases use scale-aware floating tolerances and require inspection.",
                               "Checks fixed-EF EIA proxy identified sets, not actual CIF coverage or predictive calibration."]}
    (OUT / "summary.json").write_text(json.dumps(json_safe(summary), indent=2, ensure_ascii=False,
                                                   allow_nan=False) + "\n")
    print(json.dumps({key: summary[key] for key in
                      ("region_months", "set_comparisons", "complete_K_sets_equal",
                       "near_boundary_cases")}, indent=2))


if __name__ == "__main__":
    main()
