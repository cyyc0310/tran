"""V297: value of one local monthly CIF anchor for hourly tail-count sets.

The anchor is oracle-computed from EIA proxy hours. This is a measurement-value
experiment, not a deployment or Chinese validation result.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v90_coarse_capacity_identification as v90  # noqa: E402
import v93_china_budget_coarsening as v93  # noqa: E402
import v96_threshold_side_energy as v96  # noqa: E402

PROTOCOL = ROOT / "docs/V297_MONTHLY_CIF_ANCHOR_PROTOCOL.md"
OUT = ROOT / "results/v297-monthly-cif-anchor-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
FUELS = tuple(v90.FUELS)
THERMAL = ("coal", "gas", "oil", "other")
CLEAN = ("nuclear", "hydro", "wind", "solar")
EF = dict(zip(FUELS, (1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 400.0)))
TAUS = (600, 900)
RATES = (.01, .05, .10)
ANCHOR_ERRORS = (0.0, 5.0, 10.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def capacities(year: int):
    data = ROOT / "data/research-v87"
    z = zipfile.ZipFile(data / f"eia860{year}.zip")
    with z.open(f"2___Plant_Y{year}.xlsx") as f:
        plants = pd.read_excel(f, sheet_name="Plant", header=1)
    with z.open(f"3_1_Generator_Y{year}.xlsx") as f:
        gens = pd.read_excel(f, sheet_name="Operable", header=1)
    plants = plants[["Plant Code", "Balancing Authority Code"]].dropna().drop_duplicates("Plant Code")
    gens = gens[gens.Status.isin(["OP", "OS", "SB"])].merge(plants, on="Plant Code", how="left")
    gens = gens[gens["Balancing Authority Code"].isin(v90.REGIONS)]
    fuels_by_code = {
        **{x: "coal" for x in {"ANT", "BIT", "LIG", "SUB", "WC", "RC"}},
        **{x: "gas" for x in {"NG", "OG", "PG", "BFG", "SG", "SGC"}},
        **{x: "oil" for x in {"DFO", "RFO", "JF", "KER", "WO"}},
        "NUC": "nuclear", "WAT": "hydro", "MWH": "hydro", "WND": "wind", "SUN": "solar",
    }
    cap_cols = ("Nameplate Capacity (MW)", "Summer Capacity (MW)", "Winter Capacity (MW)")
    fuel_cols = tuple(f"Energy Source {i}" for i in range(1, 7))
    rows = []
    for row in gens.to_dict("records"):
        vals = [pd.to_numeric(row.get(c), errors="coerce") for c in cap_cols]
        vals = [x for x in vals if pd.notna(x) and x > 0]
        if not vals:
            continue
        active = {fuels_by_code.get(str(row.get(c, "")).strip().upper(), "other") for c in fuel_cols if pd.notna(row.get(c))}
        if not active:
            active = {"other"}
        for fuel in active:
            rows.append((row["Balancing Authority Code"], fuel, max(vals)))
    d = pd.DataFrame(rows, columns=["region", "fuel", "mw"]).groupby(["region", "fuel"]).mw.sum().unstack(fill_value=0)
    return d, sha(data / f"eia860{year}.zip")


def anchor_set(margins, caps, tau, hours, anchor, error):
    """Exact two-block K set for hidden thermal mix plus a monthly-CIF interval.

    The thermal factor may be any convex mixture of the declared fixed fuel
    factors. Clean-source monthly margins stay observed separately. For a
    candidate event count, the high-side thermal excess is optimized over the
    block-capacity interval; the objective is piecewise linear, so endpoints
    and its single kink suffice.
    """
    mt = sum(float(margins[f]) for f in THERMAL)
    clean = sum(float(margins[f]) for f in CLEAN)
    total = mt + clean
    if total <= 0:
        return (0,)
    qvals = [EF[f] - tau for f in THERMAL]
    qlo, qhi = min(qvals), max(qvals)
    # S is total monthly threshold-excess carbon, in g/kWh·MWh units.
    physical_lo = qlo * mt - tau * clean
    physical_hi = qhi * mt - tau * clean
    band_lo = (anchor - error - tau) * total
    band_hi = (anchor + error - tau) * total
    slo, shi = max(physical_lo, band_lo), min(physical_hi, band_hi)
    if slo > shi + 1e-6:
        return ()
    if anchor is None:
        slo, shi = physical_lo, physical_hi

    thermal_cap = float(caps["thermal"])
    found = []
    if slo <= 1e-8:
        found.append(0)
    if shi > 1e-8:
        found.append(hours)
    for k in range(1, hours):
        xlo = max(0.0, mt - (hours-k) * thermal_cap)
        xhi = min(mt, k * thermal_cap)
        if xlo > xhi + 1e-8:
            continue
        # Minimum clean MWh that must lie in the event block to fit the
        # complement capacity. It minimizes the event threshold requirement.
        ylo = sum(max(0.0, float(margins[f]) - (hours-k) * float(caps[f])) for f in CLEAN)
        if any(float(margins[f]) > hours * float(caps[f]) + 1e-6 for f in CLEAN):
            continue
        candidates_s = {slo, shi}
        if slo <= 0 <= shi:
            candidates_s.add(0.0)
        # The inner maximum changes slope only when its kink enters/leaves
        # [xlo,xhi]. Include those breakpoints for exact interval optimization.
        for x in (xlo, xhi):
            sb = qlo * mt - tau * clean + (qhi-qlo) * x
            if slo <= sb <= shi:
                candidates_s.add(sb)
        feasible = False
        for s in candidates_s:
            thermal_s = s + tau * clean
            kink = (thermal_s - qlo * mt) / (qhi-qlo) if qhi != qlo else xlo
            xs = {xlo, xhi, min(max(kink, xlo), xhi)}
            max_event_thermal = max(min(qhi*x, thermal_s - qlo*(mt-x)) for x in xs)
            required = tau*ylo + max(0.0, s)
            if max_event_thermal > required + 1e-7 * max(1.0, total):
                feasible = True
                break
        if feasible:
            found.append(k)
    return tuple(found)


def main():
    protocol_hash = sha(PROTOCOL)
    prior = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(r.region), str(r.month)[:10]) for r in prior.itertuples(index=False)}
    data = v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    cap_data, cap_hashes = {}, {}
    for year in (2023, 2024):
        cap_data[year], cap_hashes[str(year)] = capacities(year)
    rows = []
    for region in v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, g in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            year = int(month[:4]); T = len(g)
            margins = {f: float(g[f"generation_{f}"].sum()) for f in FUELS}
            raw = cap_data[year-1].loc[region] if region in cap_data[year-1].index else pd.Series(dtype=float)
            cap_f = {f: float(raw.get(f, 0.0)) for f in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            caps = {"thermal": sum(cap_f[f] for f in THERMAL), **{f: cap_f[f] for f in CLEAN}}
            # Exclude no additional rows; cohort is fixed. A lagged capacity
            # infeasibility is recorded as a hard failure instead of post-hoc drop.
            for group, members in (("thermal", THERMAL), ("nuclear", ("nuclear",)), ("hydro", ("hydro",)), ("wind", ("wind",)), ("solar", ("solar",))):
                if sum(margins[f] for f in members) > T * caps[group] + 1e-5:
                    raise ValueError(f"lagged group capacity infeasible: {region} {month} {group}")
            actual_excess, actual_generation = [], []
            for _, r in g.iterrows():
                gen = sum(float(r[f"generation_{f}"]) for f in FUELS)
                excess = sum((EF[f] - 0.0) * float(r[f"generation_{f}"]) for f in FUELS)
                actual_generation.append(gen); actual_excess.append(excess)
            month_gen = sum(margins.values())
            tau_rows = []
            for tau in TAUS:
                # Local monthly mean CIF is derived from the same hourly proxy,
                # deliberately treated as one hypothetical added observation.
                s_obs = sum((EF[f] - tau) * margins[f] for f in FUELS)
                anchor = tau + s_obs / month_gen
                actual_k = sum(ex > 0 for ex in (sum((EF[f]-tau)*float(r[f"generation_{f}"]) for f in FUELS) for _, r in g.iterrows()))
                side = margins["coal"] + margins["oil"] if tau == 600 else margins["coal"]
                margins_f = {f: v90.F(str(margins[f])) for f in FUELS}
                caps_f = {f: v90.F(str(caps[f])) for f in caps}
                coarse_caps = {"thermal": caps_f["thermal"], **{f: caps_f[f] for f in CLEAN}}
                if tau == 600:
                    high_range, low_range = (800, 1000), (400, 400)
                else:
                    high_range, low_range = (1000, 1000), (400, 800)
                detailed_set = v90.capacity_group_set(margins_f, coarse_caps, tau, T, v90.GROUPS)
                if detailed_set is None:
                    raise ValueError(f"detailed capacity set infeasible: {region} {month} tau={tau}")
                sets = {
                    "coarse": v93.coarse_budget_set(margins_f, coarse_caps, tau, T),
                    "mean_anchor_0": anchor_set(margins, caps, tau, T, anchor, 0),
                    "mean_anchor_5": anchor_set(margins, caps, tau, T, anchor, 5),
                    "mean_anchor_10": anchor_set(margins, caps, tau, T, anchor, 10),
                    "threshold_side": v96.threshold_side_set(
                        margins_f, v90.F(str(side)), high_range, low_range, coarse_caps, tau, T
                    ),
                    "detailed": detailed_set,
                }
                for name, ks in sets.items():
                    if not ks or actual_k not in ks:
                        raise ValueError(f"empty/excluding set {name}: {region} {month} tau={tau} K={actual_k} set={ks}")
                for rich, poor in (("mean_anchor_0", "coarse"), ("mean_anchor_5", "coarse"), ("mean_anchor_10", "coarse"), ("threshold_side", "coarse"), ("detailed", "mean_anchor_0"), ("detailed", "threshold_side")):
                    if not set(sets[rich]).issubset(sets[poor]):
                        raise ValueError(f"nesting failure {rich} !<= {poor}: {region} {month} tau={tau}")
                for rate in RATES:
                    req = int(np.ceil(rate * T))
                    record = {"region": region, "month": month, "tau": tau, "rate": rate, "hours": T,
                              "actual_K": actual_k, "monthly_cif_anchor": anchor, "side_mwh": side}
                    for name, ks in sets.items():
                        lo, hi = min(ks), max(ks)
                        record.update({f"{name}_lower": lo, f"{name}_upper": hi,
                                       f"{name}_width": hi-lo,
                                       f"{name}_positive": lo >= req,
                                       f"{name}_negative": hi < req,
                                       f"{name}_abstain": not (lo >= req or hi < req)})
                    rows.append(record)
    OUT.mkdir(parents=True, exist_ok=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "monthly_cif_anchor_frontier.csv", index=False)
    metrics = [c for c in d if c.endswith(("_width", "_positive", "_negative", "_abstain"))]
    rm = d.groupby(["region", "tau", "rate", "month"])[metrics].mean().reset_index().groupby(["region", "tau", "rate"])[metrics].mean().reset_index()
    macro = rm.groupby(["tau", "rate"])[metrics].mean().reset_index()
    macro.to_csv(OUT / "region_macro_frontier.csv", index=False)
    paired = d.drop_duplicates(["region", "month", "tau"])
    widths = ["coarse_width", "mean_anchor_0_width", "mean_anchor_5_width", "mean_anchor_10_width", "threshold_side_width", "detailed_width"]
    wide = paired.groupby(["tau"])[widths].mean().reset_index()
    wide.to_csv(OUT / "mean_width_comparison.csv", index=False)
    summary = {"status": "ORACLE_MONTHLY_CIF_ANCHOR_INFORMATION_VALUE_NOT_FORECAST",
               "region_months": int(d[["region", "month"]].drop_duplicates().shape[0]), "scored_rows": len(d),
               "mean_width_by_tau": wide.to_dict("records"), "frontier": macro.to_dict("records"),
               "hourly_source_sha256": input_hashes, "capacity_source_sha256": cap_hashes,
               "source_audits": audits, "protocol_sha256": protocol_hash, "script_sha256": sha(Path(__file__)),
               "limitations": ["The monthly CIF anchor is oracle-computed from the same EIA hourly fuel proxy; not an independent or presently verified China field.",
                               "Fixed EIA emission-factor proxy and retrospectively screened V97 cohort; no actual CIF or untouched region.",
                               "The set is sharp only under the declared two-block allocation, shared group-capacity, exact margin and strict threshold contract."]}
    if sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed while running")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(wide.to_string(index=False))


if __name__ == "__main__":
    main()
