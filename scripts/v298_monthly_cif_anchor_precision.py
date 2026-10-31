"""V298: information-value sweep for monthly CIF anchor precision."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import v297_monthly_cif_anchor_value as v297

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V298_MONTHLY_CIF_ANCHOR_PRECISION_PROTOCOL.md"
OUT = ROOT / "results/v298-monthly-cif-anchor-precision-r2"
COHORT = ROOT / "results/v90-coarse-capacity-identification-r5/coarse_capacity_frontier.csv"
ERRORS = (0.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)


def main():
    protocol_hash = v297.sha(PROTOCOL)
    prior = pd.read_csv(COHORT).drop_duplicates(["region", "month"])
    cohort = {(str(r.region), str(r.month)[:10]) for r in prior.itertuples(index=False)}
    data = v297.v90.source_data
    data.ALL_REGIONS = tuple(dict.fromkeys(data.ALL_REGIONS + v297.v90.REGIONS))
    hourly, _, audits, input_hashes = data.prepare_regions()
    caps, cap_hashes = {}, {}
    for year in (2023, 2024):
        caps[year], cap_hashes[str(year)] = v297.capacities(year)
    rows = []
    for region in v297.v90.REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge("2024-01-01") & times.lt("2026-01-01") & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["month"] = frame.time.dt.strftime("%Y-%m-01")
        for month, g in frame.groupby("month", sort=True):
            if (region, month) not in cohort:
                continue
            year = int(month[:4]); T = len(g)
            margins = {f: float(g[f"generation_{f}"].sum()) for f in v297.FUELS}
            raw = caps[year-1].loc[region] if region in caps[year-1].index else pd.Series(dtype=float)
            cap_f = {f: float(raw.get(f, 0.0)) for f in ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")}
            cap_group = {"thermal": sum(cap_f[f] for f in v297.THERMAL),
                         **{f: cap_f[f] for f in v297.CLEAN}}
            total_gen = sum(margins.values())
            for tau in v297.TAUS:
                s_obs = sum((v297.EF[f]-tau)*margins[f] for f in v297.FUELS)
                anchor = tau + s_obs/total_gen
                side = margins["coal"]+margins["oil"] if tau == 600 else margins["coal"]
                high_range, low_range = ((800, 1000), (400, 400)) if tau == 600 else ((1000, 1000), (400, 800))
                margins_f = {f: v297.v90.F(str(margins[f])) for f in v297.FUELS}
                cap_group_f = {f: v297.v90.F(str(x)) for f, x in cap_group.items()}
                detail = v297.v90.capacity_group_set(margins_f, cap_group_f, tau, T, v297.v90.GROUPS)
                side_set = v297.v96.threshold_side_set(margins_f, v297.v90.F(str(side)), high_range, low_range, cap_group_f, tau, T)
                exact_set = v297.anchor_set(margins, cap_group, tau, T, anchor, 0)
                if not detail or not side_set or not exact_set:
                    raise ValueError(f"empty reference set at {region} {month} tau={tau}")
                if not set(detail).issubset(exact_set) or not set(detail).issubset(side_set):
                    raise ValueError(f"reference nesting failure at {region} {month} tau={tau}")
                previous_set = exact_set
                for error in ERRORS:
                    ks = v297.anchor_set(margins, cap_group, tau, T, anchor, error)
                    if not ks or not set(previous_set).issubset(ks):
                        raise ValueError(f"error-band nesting failure at {region} {month} tau={tau} error={error}")
                    previous_set = ks
                    # Anchor and threshold-side contracts are not always
                    # ordered; record crossing rather than impose a false
                    # set relation. Fuel-detail must remain contained.
                    if not set(detail).issubset(ks):
                        raise ValueError(f"anchor excludes detailed set at {region} {month} tau={tau} error={error}")
                    for rate in v297.RATES:
                        req = int((v297.v90.F(str(rate))*T).__ceil__())
                        lo, hi = min(ks), max(ks)
                        rows.append({"region": region, "month": month, "tau": tau, "error_g_per_kwh": error,
                                     "rate": rate, "hours": T, "anchor": anchor,
                                     "lower": lo, "upper": hi, "width": hi-lo,
                                     "detail_width": max(detail)-min(detail),
                                     "side_width": max(side_set)-min(side_set),
                                     "positive": lo >= req, "negative": hi < req,
                                     "abstain": not (lo >= req or hi < req),
                                     "detail_negative": max(detail) < req,
                                     "side_negative": max(side_set) < req})
    OUT.mkdir(parents=True, exist_ok=False)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "anchor_precision_frontier.csv", index=False)
    month = d.drop_duplicates(["region", "month", "tau", "error_g_per_kwh"])
    width = month.groupby(["tau", "error_g_per_kwh"]).agg(
        width_mean=("width", "mean"), width_median=("width", "median"),
        detail_width_mean=("detail_width", "mean"), detail_width_median=("detail_width", "median"),
        side_width_mean=("side_width", "mean"), side_width_median=("side_width", "median"),
    ).reset_index()
    width.to_csv(OUT / "width_precision_curve.csv", index=False)
    metrics = d.groupby(["tau", "error_g_per_kwh", "rate"])[["positive", "negative", "abstain"]].mean().reset_index()
    metrics.to_csv(OUT / "selective_precision_frontier.csv", index=False)
    summary = {"status": "ORACLE_MONTHLY_CIF_ANCHOR_PRECISION_SWEEP_NOT_FORECAST",
               "region_months": int(month[["region", "month"]].drop_duplicates().shape[0]),
               "scored_rows": len(d), "errors_g_per_kwh": list(ERRORS),
               "width_precision_curve": width.to_dict("records"),
               "selective_frontier": metrics.to_dict("records"),
               "share_anchor_no_wider_than_detail": month.groupby(["tau", "error_g_per_kwh"]).apply(lambda x: float((x.width <= x.detail_width).mean()), include_groups=False).reset_index(name="share").to_dict("records"),
               "share_anchor_no_wider_than_side": month.groupby(["tau", "error_g_per_kwh"]).apply(lambda x: float((x.width <= x.side_width).mean()), include_groups=False).reset_index(name="share").to_dict("records"),
               "hourly_source_sha256": input_hashes, "capacity_source_sha256": cap_hashes,
               "source_audits": audits, "protocol_sha256": protocol_hash,
               "script_sha256": v297.sha(Path(__file__)),
               "limitations": ["Monthly anchor and reference sets are oracle-derived from EIA proxy labels.",
                               "Absolute error bands are sensitivity values, not empirical China measurement error.",
                               "Retrospective 358 BA-month cohort; no China or untouched-region validation."]}
    if v297.sha(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed while running")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(width.to_string(index=False))
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
