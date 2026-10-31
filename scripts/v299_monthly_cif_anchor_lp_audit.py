"""V299: independent two-block LP audit of V297's analytic K sets."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

import v297_monthly_cif_anchor_value as v297

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V299_MONTHLY_CIF_ANCHOR_LP_AUDIT_PROTOCOL.md"
OUT = ROOT / "results/v299-monthly-cif-anchor-lp-audit-r1"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
ERRORS = (0.0, 0.25)


def lp_set(margins, caps, tau, hours, anchor, error):
    fuels = v297.FUELS
    n = len(fuels)
    total = sum(margins.values())
    # Normalize MWh by total generation and emission excess by 1000 g/kWh;
    # otherwise HiGHS' feasibility tolerance can create false tiny events when
    # the observed anchor is exactly on the minimum-factor boundary.
    q = np.array([(v297.EF[f]-tau)/1000.0 for f in fuels], dtype=float)
    m_norm = {f: margins[f]/total for f in fuels}
    c_norm = {f: caps[f]/total for f in caps}
    mt = sum(m_norm[f] for f in v297.THERMAL)
    s_lo = (anchor-error-tau)/1000.0
    s_hi = (anchor+error-tau)/1000.0
    ae, be = [], []
    row = np.zeros(2*n)
    for i, fuel in enumerate(fuels):
        if fuel in v297.THERMAL:
            row[i] = row[n+i] = 1
    ae.append(row); be.append(mt)
    for fuel in v297.CLEAN:
        row = np.zeros(2*n); i = fuels.index(fuel); row[i] = row[n+i] = 1
        ae.append(row); be.append(m_norm[fuel])

    # Total threshold excess from the monthly CIF anchor is an interval row.
    a_ub0, b_ub0 = [], []
    row = np.r_[q, q]
    a_ub0.extend([row, -row]); b_ub0.extend([s_hi, -s_lo])
    groups = (("thermal", v297.THERMAL), ("nuclear", ("nuclear",)),
              ("hydro", ("hydro",)), ("wind", ("wind",)), ("solar", ("solar",)))
    found = []
    for k in range(hours+1):
        a_ub = list(a_ub0); b_ub = list(b_ub0)
        # Complement block is non-event. Event is strictly above threshold.
        comp = np.zeros(2*n); comp[n:] = q
        a_ub.append(comp); b_ub.append(0.0)
        if k > 0:
            event = np.zeros(2*n); event[:n] = -q
            a_ub.append(event); b_ub.append(-1e-8)
        for group, members in groups:
            cap = float(c_norm[group])
            ev = np.zeros(2*n); co = np.zeros(2*n)
            for fuel in members:
                i = fuels.index(fuel); ev[i] = 1; co[n+i] = 1
            a_ub.extend([ev, co]); b_ub.extend([k*cap, (hours-k)*cap])
        result = linprog(np.zeros(2*n), A_ub=np.asarray(a_ub), b_ub=np.asarray(b_ub),
                         A_eq=np.asarray(ae), b_eq=np.asarray(be), bounds=(0, None), method="highs",
                         options={"primal_feasibility_tolerance": 1e-9,
                                  "dual_feasibility_tolerance": 1e-9})
        if result.success:
            found.append(k)
    return tuple(found)


def main():
    protocol_hash = v297.sha(PROTOCOL)
    prior = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(r.region), str(r.month)[:10]) for r in prior.itertuples(index=False)}
    eligible_regions = {region for region, _ in cohort}
    selected_regions = tuple(r for r in v297.v90.REGIONS if r in eligible_regions)[:10]
    data = v297.v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v297.v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    cap_data, cap_hashes = {}, {}
    for year in (2023, 2024):
        cap_data[year], cap_hashes[str(year)] = v297.capacities(year)
    rows = []
    for region in selected_regions:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        chosen = None
        for month, g in frame.groupby("month", sort=True):
            if (region, month) in cohort:
                chosen = (month, g); break
        if chosen is None:
            raise ValueError(f"no frozen eligible month for {region}")
        month, g = chosen; T = len(g); year = int(month[:4])
        margins = {f: float(g[f"generation_{f}"].sum()) for f in v297.FUELS}
        raw = cap_data[year-1].loc[region] if region in cap_data[year-1].index else pd.Series(dtype=float)
        cap_f = {f: float(raw.get(f, 0.0)) for f in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
        caps = {"thermal": sum(cap_f[f] for f in v297.THERMAL), **{f: cap_f[f] for f in v297.CLEAN}}
        for tau in v297.TAUS:
            total = sum(margins.values())
            excess = sum((v297.EF[f]-tau)*margins[f] for f in v297.FUELS)
            anchor = tau + excess/total
            for error in ERRORS:
                analytic = v297.anchor_set(margins, caps, tau, T, anchor, error)
                lp = lp_set(margins, caps, tau, T, anchor, error)
                if set(analytic) != set(lp):
                    mismatch = sorted(set(analytic)^set(lp))
                    raise ValueError(f"complete-set mismatch: {region} {month} tau={tau} err={error}; "
                                     f"diff={mismatch[:20]} analytic_n={len(analytic)} lp_n={len(lp)} "
                                     f"analytic_head={analytic[:20]} lp_head={lp[:20]} "
                                     f"analytic_tail={analytic[-10:]} lp_tail={lp[-10:]} "
                                     f"anchor={anchor} margins={margins} caps={caps}")
                rows.append({"region": region, "month": month, "tau": tau, "error_g_per_kwh": error,
                             "hours": T, "n_K": len(lp), "lower": min(lp), "upper": max(lp),
                             "complete_set_equal": True})
    OUT.mkdir(parents=True, exist_ok=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "lp_audit_rows.csv", index=False)
    summary = {"status": "INDEPENDENT_LP_COMPLETE_SET_MATCH",
               "selected_regions": list(selected_regions), "audited_ba_months": int(d[["region", "month"]].drop_duplicates().shape[0]),
               "complete_sets_compared": len(d), "all_complete_sets_equal": bool(d.complete_set_equal.all()),
               "hourly_source_sha256": input_hashes, "capacity_source_sha256": cap_hashes,
               "source_audits": audits, "protocol_sha256": protocol_hash,
               "script_sha256": v297.sha(Path(__file__)),
               "limitations": ["Only the first eligible month per selected region was checked.",
                               "Fixed EIA proxy factors, retrospective cohort, and oracle monthly anchor.",
                               "LP agreement verifies declared set math only; not target CIF validity or forecast coverage."]}
    if v297.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed while running")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    print(d.to_string(index=False))


if __name__ == "__main__":
    main()
