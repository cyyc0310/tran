"""V106: test coarse monthly technology margins and lagged capacity out of region."""
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
OUT = ROOT / "results/v106-unseen-china-style-coarse-info-r1"
PROTOCOL = ROOT / "docs/V106_UNSEEN_CHINA_STYLE_COARSE_INFO_PROTOCOL.md"
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")
TAUS = (600, 900)
RATES = (0.01, 0.05, 0.10)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coarse_k_set(margins, caps, tau: int, hours: int) -> tuple[int, ...]:
    """Sharp K set for group totals with interval thermal EF and zero-EF clean mix."""
    mt = sum((margins[f] for f in THERMAL), F(0))
    mc = sum((margins[f] for f in CLEAN), F(0))
    total = mt + mc
    if total <= 0:
        return (0,)
    low_q, high_q, t = F(400 - tau), F(1000 - tau), F(tau)
    result = []
    if low_q * mt - t * mc <= 0:
        result.append(0)
    if high_q * mt - t * mc > 0:
        result.append(hours)

    for k in range(1, hours):
        intervals = {}
        for group in ("thermal",) + CLEAN:
            members = THERMAL if group == "thermal" else (group,)
            amount = sum((margins[f] for f in members), F(0))
            lower = max(F(0), amount - (hours - k) * caps[group])
            upper = min(amount, k * caps[group])
            if lower > upper:
                intervals = {}
                break
            intervals[group] = (lower, upper)
        if not intervals:
            continue
        xlo, xhi = intervals["thermal"]
        ylo = sum((intervals[g][0] for g in CLEAN), F(0))
        yhi = sum((intervals[g][1] for g in CLEAN), F(0))

        # Feasible rectangle clipped by the complement non-event half-plane.
        rhs = t * mc - low_q * mt
        points = {(xlo, ylo), (xlo, yhi), (xhi, ylo), (xhi, yhi)}
        if t != 0:
            for x in (xlo, xhi):
                y = (rhs + low_q * x) / t
                if ylo <= y <= yhi:
                    points.add((x, y))
        if low_q != 0:
            for y in (ylo, yhi):
                x = (t * y - rhs) / low_q
                if xlo <= x <= xhi:
                    points.add((x, y))
        feasible = [(x, y) for x, y in points if -low_q * x + t * y <= rhs]
        if not feasible:
            continue
        event_excess = max(high_q * x - t * y for x, y in feasible)
        minimum_event_total = min(x + y for x, y in feasible)
        if event_excess > 0 and minimum_event_total < total:
            result.append(k)
    return tuple(sorted(set(result)))


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {OUT}")
    v104 = pd.read_csv(IN)
    if set(v104.region.unique()) != set(REGIONS):
        raise ValueError("V104 cohort roster mismatch")
    v90.REGIONS = REGIONS
    v90.source_data.ALL_REGIONS = tuple(dict.fromkeys(v90.source_data.ALL_REGIONS + REGIONS))
    hourly, _, audits, hourly_hashes = v90.source_data.prepare_regions()
    capacity, capacity_hash = v90.capacities(2025)
    rows, availability = [], []

    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2026-01-01") & times.lt("2026-07-01")
                          & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month in pd.date_range("2026-01-01", "2026-06-01", freq="MS", tz="UTC"):
            month_key = month.strftime("%Y-%m-01")
            group = frame.loc[frame.month.eq(month_key)]
            hours = len(group)
            expected = month.days_in_month * 24
            margins = {fuel: sum((F(str(float(x))) for x in group[f"generation_{fuel}"]), F(0))
                       for fuel in v90.FUELS}
            caprow = capacity.loc[region] if region in capacity.index else pd.Series(dtype=float)
            grouped_caps = {"thermal": sum((F(str(float(caprow.get(fuel, 0)))) for fuel in THERMAL), F(0))}
            grouped_caps.update({fuel: F(str(float(caprow.get(fuel, 0)))) for fuel in CLEAN})
            energy_ok = hours > 0 and all(
                sum((margins[f] for f in (THERMAL if g == "thermal" else (g,))), F(0))
                <= hours * grouped_caps[g] for g in ("thermal",) + CLEAN
            )
            eligible = hours / expected >= 0.80 and energy_ok
            availability.append({"region": region, "month": month_key, "hours": hours,
                                 "coverage": hours / expected, "group_energy_feasible": bool(energy_ok),
                                 "scored": bool(eligible)})
            if not eligible:
                continue
            no_capacity = {"thermal": sum((margins[f] for f in THERMAL), F(0)),
                           **{fuel: margins[fuel] for fuel in CLEAN}}
            observed = []
            for _, obs in group.iterrows():
                generation = [F(str(float(obs[f"generation_{fuel}"]))) for fuel in v90.FUELS]
                total = sum(generation, F(0))
                emissions = sum((generation[i] * v90.EF[v90.FUELS[i]]
                                 for i in range(len(v90.FUELS))), F(0))
                observed.append((total, emissions))
            for tau in TAUS:
                actual = sum(emissions > F(tau) * total for total, emissions in observed)
                no_set = coarse_k_set(margins, no_capacity, tau, hours)
                cap_set = coarse_k_set(margins, grouped_caps, tau, hours)
                if actual not in no_set or actual not in cap_set:
                    raise ValueError(f"coarse information set excludes proxy K: {region}/{month_key}/{tau}/{actual}")
                if not set(cap_set).issubset(no_set):
                    raise ValueError(f"capacity set fails nesting: {region}/{month_key}/{tau}")
                fine = v104.loc[v104.region.eq(region) & v104.month.astype(str).str[:10].eq(month_key)
                                & v104.tau.eq(tau)].iloc[0]
                for rate in RATES:
                    required = int((F(str(rate)) * hours).__ceil__())
                    nlo, nhi, clo, chi = min(no_set), max(no_set), min(cap_set), max(cap_set)
                    rows.append({"region": region, "month": month_key, "tau": tau, "rate": rate,
                                 "hours": hours, "required_hours": required, "actual_K": actual,
                                 "coarse_month_only_lower": nlo, "coarse_month_only_upper": nhi,
                                 "coarse_month_only_width": nhi - nlo,
                                 "coarse_month_capacity_lower": clo, "coarse_month_capacity_upper": chi,
                                 "coarse_month_capacity_width": chi - clo,
                                 "width_reduction_hours": (nhi - nlo) - (chi - clo),
                                 "coarse_month_only_positive": nlo >= required,
                                 "coarse_month_only_negative": nhi < required,
                                 "coarse_month_only_abstain": not (nlo >= required or nhi < required),
                                 "coarse_month_capacity_positive": clo >= required,
                                 "coarse_month_capacity_negative": chi < required,
                                 "coarse_month_capacity_abstain": not (clo >= required or chi < required),
                                 "detailed_fuel_robust_capacity_width": int(fine.robust_width),
                                 "proxy_K_covered_both": True})

    detail = pd.DataFrame(rows)
    avail = pd.DataFrame(availability)
    if detail.empty:
        raise ValueError("No eligible rows")
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "paired_rows.csv", index=False)
    avail.to_csv(OUT / "availability.csv", index=False)
    metrics = ["coarse_month_only_width", "coarse_month_capacity_width", "width_reduction_hours",
               "coarse_month_only_positive", "coarse_month_only_negative", "coarse_month_only_abstain",
               "coarse_month_capacity_positive", "coarse_month_capacity_negative",
               "coarse_month_capacity_abstain", "detailed_fuel_robust_capacity_width", "proxy_K_covered_both"]
    per_region = detail.groupby(["region", "tau", "rate", "month"])[metrics].mean().reset_index()
    per_region = per_region.groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    per_region["region_months"] = detail.groupby(["region", "tau", "rate"])["month"].nunique().to_numpy()
    per_region.to_csv(OUT / "per_region_frontier.csv", index=False)
    macro = per_region.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_equal_frontier.csv", index=False)
    pooled = detail.groupby(["tau", "rate"])[metrics].mean().reset_index()
    pooled.to_csv(OUT / "pooled_frontier.csv", index=False)
    payload = {"status": "UNSEEN_REGION_COARSE_MONTHLY_BUDGET_RETROSPECTIVE_PROXY",
               "regions": list(REGIONS), "candidate_region_months": int(len(avail)),
               "scored_region_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
               "unavailable_region_months": int((~avail.scored).sum()), "scored_rows": len(detail),
               "energy_feasibility_by_region": avail.groupby("region").agg(
                   months=("month", "nunique"), scored=("scored", "sum"),
                   group_energy_feasible=("group_energy_feasible", "sum"),
                   mean_coverage=("coverage", "mean")).reset_index().to_dict("records"),
               "region_equal_frontier": macro.to_dict("records"), "pooled_frontier": pooled.to_dict("records"),
               "thermal_EF_support": [400, 1000], "capacity_vintage": 2025,
               "capacity_source_sha256": capacity_hash, "hourly_source_sha256": hourly_hashes,
               "source_audits": audits, "v104_rows_sha256": sha(IN),
               "protocol_sha256": sha(PROTOCOL), "script_sha256": sha(Path(__file__)),
               "limitations": ["Post-hoc information-budget audit on V104 fixed region roster; not a new blind test.",
                               "Monthly group margins are oracle-coarsened from EIA hourly data, not ingested Chinese reports.",
                               "Thermal EF support [400,1000] is a proxy sensitivity range, not an empirical confidence set.",
                               "Target-month monthly totals make this retrospective identification, not a month-ahead forecast.",
                               "Fixed-factor EIA labels are not measured actual CIF; lagged US capacity is not verified China input."]}
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print("availability", payload["energy_feasibility_by_region"])
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
