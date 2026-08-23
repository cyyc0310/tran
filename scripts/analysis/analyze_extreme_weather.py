#!/usr/bin/env python
"""Extreme-weather attribution for 2023 CIF anomalies (29 regions).

Question: are the extreme CIF events (jumps / volatility spikes / level
breaks) caused by weather — cyclones/hurricanes (gust+pressure signature),
heavy rain (precipitation), freeze/ice, heat domes, or wind droughts?

Pipeline per region:
    1. load CIF + fuel shares + ERA5 channels (temp/swrad/wind100,
       gusts/MSL pressure, precipitation) joined on UTC hours
    2. hourly anomaly indices: |dCIF|, gust z-score, 6 h pressure
       tendency, temp z-score, 6 h precip accumulation, csi (cloud proxy)
    3. daily table -> Spearman corr(CIF volatility, weather severity)
    4. top-K extreme days per region, annotated with weather type
    5. known-2023-event windows scored against the region baseline

Outputs:
    results/extreme_weather_analysis.json  (all tables)
    results/extreme_weather_analysis.md    (human report)
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from transcif.config.region_meta import REGION_META, get_region_meta
from transcif.data.fuel import (
    CANONICAL_FUELS,
    attach_fuel_and_exog,
    jurisdiction_of,
    load_fuel_shares,
    load_pressure_winds,
    load_raw_weather,
)
from transcif.data.loaders import all_region_configs, load_region_data

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
RESULTS = Path(__file__).resolve().parent.parent.parent / "results"

# Verified 2023 extreme-weather calendar (region -> list of event windows).
# UTC month-day windows; sources: BOM (Jasper), NWS/FPL (Idalia), NWS
# (Mara), NOAA/CW3E (Hilary), Met Office/EUMETNET (UK named storms).
KNOWN_EVENTS = {
    "US_ERCO": [("Winter Storm Mara (freeze/ice)", (1, 30), (2, 3)),
                ("July heat dome", (7, 27), (8, 2)),
                ("August heat dome", (8, 15), (8, 25))],
    "US_FPL": [("Hurricane Idalia (Cat 3 landfall)", (8, 29), (9, 1))],
    "US_CISO": [("Hurricane Hilary remnant TS", (8, 19), (8, 22)),
                ("July heat wave", (7, 10), (7, 20))],
    "US_MISO": [("July heat dome", (7, 27), (8, 2))],
    "US_PJM": [("Feb cold snap", (2, 2), (2, 5)),
               ("July heat dome", (7, 27), (8, 2))],
    "QLD1": [("TC Jasper (Cat 2 landfall + 2.2 m rain)", (12, 13), (12, 18))],
    "NSW1": [("June cold snap", (6, 5), (6, 12))],
    "VIC1": [("June cold snap", (6, 5), (6, 12)),
             ("September wind drought", (9, 8), (9, 16))],
    "SA1": [("September wind drought", (9, 8), (9, 16))],
}
# UK storms apply to all 17 UK regions; per-region impact differs.
UK_STORMS = [
    ("Storm Antoni (wind+rain)", (8, 4), (8, 6)),
    ("UK September heatwave", (9, 5), (9, 10)),
    ("Storm Agnes (wind+rain)", (9, 27), (9, 29)),
    ("Storm Babet (extreme rain)", (10, 18), (10, 22)),
    ("Storm Ciarán (extreme gusts)", (11, 1), (11, 3)),
    ("Storm Debi (wind)", (11, 13), (11, 15)),
    ("Storm Elin/Fergus (wind+rain)", (12, 9), (12, 12)),
    ("Storm Gerrit (wind/rain/snow)", (12, 27), (12, 29)),
]


def load_precip(region_name, all_configs):
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = DATA / "weather3" / f"{stem}_precip_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"]).sort_values("hour")
    return pd.DatetimeIndex(df["hour"]), df["precipitation"].values.astype(np.float32)


def zscore(v):
    v = np.asarray(v, dtype=np.float64)
    s = np.nanstd(v)
    if not np.isfinite(s) or s < 1e-9:
        return np.zeros_like(v)
    return np.nan_to_num((v - np.nanmean(v)) / s)


def build_region_table(name, cfgs):
    data = load_region_data(name, cfgs)
    hours = pd.DatetimeIndex(data["hours"])
    if jurisdiction_of(name) == "au":
        lat, lon, tz = get_region_meta(name)
        dst = name in {"NSW1", "VIC1", "SA1"} and np.isin(
            np.asarray(hours.month), [10, 11, 12, 1, 2, 3])
        off = tz + (dst.astype(float) if name in {"NSW1", "VIC1", "SA1"} else 0.0)
        hours = hours - pd.to_timedelta(off, unit="h")
    df = pd.DataFrame({"cif": data["cif"].astype(float)}, index=hours)

    fh, fs = load_fuel_shares(name, cfgs)
    if fh is not None:
        fdf = pd.DataFrame(fs, index=pd.DatetimeIndex(fh))
        fdf = fdf[~fdf.index.duplicated(keep="first")]
        fdf = fdf.reindex(hours).fillna(0.0)
        i_wind = CANONICAL_FUELS.index("wind")
        i_solar = CANONICAL_FUELS.index("solar")
        df["wind_share"] = fdf.values[:, i_wind]
        df["solar_share"] = fdf.values[:, i_solar]
    else:
        df["wind_share"] = np.nan
        df["solar_share"] = np.nan

    wh, wx = load_raw_weather(name, cfgs)
    if wh is not None:
        wdf = pd.DataFrame(wx, columns=["temp", "swrad", "wind100"],
                           index=pd.DatetimeIndex(wh))
        wdf = wdf[~wdf.index.duplicated(keep="first")]
        joined = wdf.reindex(hours)
        df["temp"] = joined["temp"].values
        df["wind100"] = joined["wind100"].values
    ph, pw = load_pressure_winds(name, cfgs)
    if ph is not None:
        pdf = pd.DataFrame(pw, columns=["gust", "pres"], index=pd.DatetimeIndex(ph))
        pdf = pdf[~pdf.index.duplicated(keep="first")]
        joined = pdf.reindex(hours)
        df["gust"] = joined["gust"].values
        df["pres"] = joined["pres"].values + 1013.0
    rh, pr = load_precip(name, cfgs)
    if rh is not None:
        rdf = pd.DataFrame({"precip": pr}, index=pd.DatetimeIndex(rh))
        rdf = rdf[~rdf.index.duplicated(keep="first")]
        df["precip"] = rdf.reindex(hours)["precip"].values

    df = df.dropna(subset=["cif"]).sort_index()
    return df


def hourly_indices(df):
    out = df.copy()
    out["dcif"] = out["cif"].diff().abs()
    out["gust_z"] = zscore(out["gust"])
    out["temp_z"] = zscore(out["temp"])
    out["pres_tend6"] = out["pres"] - out["pres"].shift(6)
    out["precip6"] = out["precip"].rolling(6, min_periods=1).sum()
    out["wind100_z"] = zscore(out["wind100"])
    out["wind_drop6"] = out["wind_share"] - out["wind_share"].shift(6)
    return out


def daily_table(h):
    g = h.resample("1D")
    d = pd.DataFrame({
        "cif_std": g["cif"].std(),
        "dcif_max": g["dcif"].max(),
        "cif_range": g["cif"].max() - g["cif"].min(),
        "gust_max": g["gust"].max(),
        "pres_min": g["pres"].min(),
        "precip_sum": g["precip"].sum(),
        "wind100_mean": g["wind100"].mean(),
        "temp_max": g["temp"].max(),
        "wind_share_mean": g["wind_share"].mean(),
    }).dropna(subset=["cif_std"])
    return d


def classify_weather(h, df):
    """Dominant weather type during an event window (hh = hourly slice)."""
    gmax = np.nanmax(h["gust_z"]) if h["gust_z"].notna().any() else 0.0
    pmin = np.nanmin(h["pres_tend6"]) if h["pres_tend6"].notna().any() else 0.0
    p99 = np.nanpercentile(df["precip6"].dropna(), 99) if df["precip6"].notna().any() else 0.0
    p6 = np.nanmax(h["precip6"]) if h["precip6"].notna().any() else 0.0
    tmin = np.nanmin(h["temp_z"]) if h["temp_z"].notna().any() else 0.0
    tmax = np.nanmax(h["temp_z"]) if h["temp_z"].notna().any() else 0.0
    wmin = np.nanmin(h["wind100_z"]) if h["wind100_z"].notna().any() else 0.0
    tags = []
    if gmax >= 1.5 and pmin <= -4:
        tags.append("cyclone/gust")
    elif gmax >= 2.5:
        tags.append("gust")
    if p6 >= max(p99, 5.0):
        tags.append("heavy-rain")
    if tmin <= -2.0:
        tags.append("freeze/cold")
    if tmax >= 2.5:
        tags.append("heat")
    if wmin <= -1.5:
        tags.append("wind-lull")
    return "+".join(tags) if tags else "calm", {
        "gust_z_max": round(float(gmax), 2),
        "pres_tend6_min": round(float(pmin), 1),
        "precip6_max_mm": round(float(p6), 1),
        "temp_z_min": round(float(tmin), 2),
        "temp_z_max": round(float(tmax), 2),
        "wind100_z_min": round(float(wmin), 2),
    }


def main():
    cfgs = all_region_configs()
    names = [n for n in cfgs if n in REGION_META]
    report = {"regions": {}, "daily_corr": {}, "top_days": {}, "known_events": {}}

    for name in names:
        df = build_region_table(name, cfgs)
        if len(df) < 24 * 30:
            continue
        h = hourly_indices(df)
        d = daily_table(h)

        # --- daily volatility-vs-weather correlations (Spearman) ---
        def sp(a, b):
            m = d[a].notna() & d[b].notna()
            if m.sum() < 30:
                return np.nan
            from scipy.stats import spearmanr
            return float(spearmanr(d[a][m], d[b][m]).statistic)

        report["daily_corr"][name] = {
            "cif_std~gust_max": round(sp("cif_std", "gust_max"), 3),
            "cif_std~precip_sum": round(sp("cif_std", "precip_sum"), 3),
            "cif_std~wind100_mean": round(sp("cif_std", "wind100_mean"), 3),
            "cif_std~temp_max": round(sp("cif_std", "temp_max"), 3),
            "dcif_max~gust_max": round(sp("dcif_max", "gust_max"), 3),
        }

        # --- top extreme days by CIF volatility ---
        top = d.sort_values("dcif_max", ascending=False).head(5)
        rows = []
        for day, r in top.iterrows():
            hday = h[(h.index >= day) & (h.index < day + pd.Timedelta(days=1))]
            tag, stats = classify_weather(hday, h)
            rows.append({
                "date": str(day.date()), "dcif_max": round(float(r["dcif_max"]), 1),
                "cif_range": round(float(r["cif_range"]), 1),
                "gust_max_ms": round(float(r["gust_max"]), 1),
                "precip_sum_mm": round(float(r["precip_sum"]), 1),
                "type": tag, **stats,
            })
        report["top_days"][name] = rows

        # --- known-event windows ---
        events = list(KNOWN_EVENTS.get(name, []))
        if name.startswith("UK_"):
            events = UK_STORMS
        ev_rows = []
        base_std = float(np.nanmedian(d["cif_std"]))
        for label, (m0, d0), (m1, d1) in events:
            t0 = pd.Timestamp(2023, m0, d0)
            t1 = pd.Timestamp(2023, m1, d1)
            w = h[(h.index >= t0) & (h.index <= t1 + pd.Timedelta(hours=23))]
            if len(w) < 12:
                continue
            wtag, wstats = classify_weather(w, h)
            ev_std = float(np.nanstd(w["cif"]))
            ev_rows.append({
                "event": label, "type": wstats and wtag, "stats": wstats,
                "cif_std": round(ev_std, 1),
                "cif_std_ratio": round(ev_std / base_std, 2) if base_std > 0 else None,
                "wind_share_mean": (round(float(np.nanmean(w["wind_share"])), 3)
                                    if w["wind_share"].notna().any() else None),
            })
        if ev_rows:
            report["known_events"][name] = ev_rows
        report["regions"][name] = {"hours": int(len(df))}

    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "extreme_weather_analysis.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"[extreme] wrote {RESULTS / 'extreme_weather_analysis.json'}")
    print(f"[extreme] regions: {len(report['regions'])}")


if __name__ == "__main__":
    main()
