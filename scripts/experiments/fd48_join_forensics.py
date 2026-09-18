#!/usr/bin/env python
"""FD-48: join forensics for the FD-47 degradation regions.

FD-47 verdict attributed the residual degradations (VIC1 +2.8, NSW1 +1.8,
BPAT +3.8 ensemble; GFS-only +5.8/+4.6 on VIC1/NSW1) to "AU DST seam and
BPAT hour-offset join artifacts".  This script tests that attribution
EMPIRICALLY before any fix — no training, pure data alignment forensics.

Pre-registered diagnostics (per region, per model gfs/icon):
  D1  NWP hole map on the join axis: hours where the NWP archive is NaN
      after reindexing onto hours_utc; how many fall in the TEST zone
      (positions >= split, i.e. reachable by window horizons).  A missing
      horizon hour is written as weather zeros (0 m/s wind, 0 C temp, 0
      csi) — a large synthetic shock if it lands inside a horizon.
  D2  Lag scan: corr(nwp[t+lag], era5[t]) for lag in [-3..3] on the join
      axis, for w100 and swr.  A join misalignment shows as best-lag != 0
      with a clear corr gain; genuine forecast error keeps best-lag = 0.
  D3  Diurnal phase: local-hour argmax of mean swr and mean w100 for NWP
      vs ERA5 — a timeline offset shifts the solar clock visibly.
  D4  DST calendar exactness (AU): month-rule offset vs true
      zoneinfo offset per hour; count differing hours inside the test
      zone.  If zero, the month rule cannot explain horizon artifacts.
  D5  Farmblend isolation (VIC1/SA1): the ERA5 baseline uses the
      capacity-weighted farm-fleet blend while the NWP arms use the
      centroid grid.  Correlate centroid-ERA5 vs farmblend-ERA5 vs NWP
      separately — splits "forecast error" from "grid mismatch".

Usage:
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd48_join_forensics.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))

from fd47_nwp_fut_weather import (  # noqa: E402
    DATA_DIR, RESULTS, _safe_prepare_fd_region, load_nwp_table,
    region_utc_axis,
)
from transcif.config import TRAIN_FRACTION  # noqa: E402
from transcif.data.loaders import all_region_configs  # noqa: E402

PANEL = [
    "NSW1", "QLD1", "VIC1", "SA1",            # AU (QLD1 = DST-free control)
    "US_BPAT", "US_CISO", "US_NYIS",          # US (CISO/NYIS controls)
    "UK_16_Scotland", "UK_11_South_West_England",  # UK no-conversion control
]
LAGS = list(range(-3, 4))


def load_centroid_era5(region_name, all_configs, data_dir):
    """Centroid-grid ERA5 (T-axis source arrays) bypassing the farmblend
    override in load_raw_weather — isolates grid mismatch (D5)."""
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = data_dir / "weather" / f"{stem}_weather_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"])
    df = (df.drop_duplicates("hour", keep="first")
            .sort_values("hour").reset_index(drop=True))
    cols = ["temperature_c", "shortwave_radiation", "wind_speed_100m"]
    arr = df[cols].values.astype(np.float32)
    arr[:, 2] = arr[:, 2] / 3.6  # km/h -> m/s (FD-17 unit fix)
    return pd.DatetimeIndex(df["hour"]), arr


def lag_corr(nwp_vals, era5_vals, lag):
    """corr(nwp[t+lag], era5[t]) over common valid positions."""
    n = min(len(nwp_vals), len(era5_vals))
    nwp_vals = nwp_vals[:n]
    era5_vals = era5_vals[:n]
    if lag >= 0:
        a, b = nwp_vals[lag:], era5_vals[:n - lag] if lag else era5_vals[:]
    else:
        a, b = nwp_vals[:n + lag], era5_vals[-lag:]
    m = min(len(a), len(b))
    a, b = a[:m].astype(float), b[:m].astype(float)
    ok = np.isfinite(a) & np.isfinite(b) & (a != 0.0) & (b != 0.0)
    if ok.sum() < 200:
        return None
    a, b = a[ok], b[ok]
    if a.std() < 1e-6 or b.std() < 1e-6:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def scan(region, cfgs, out):
    data = _safe_prepare_fd_region(region, cfgs, DATA_DIR)
    hours_utc = region_utc_axis(data, region)
    T = len(data["rs"])
    split = int(T * TRAIN_FRACTION)
    pos = np.arange(T)

    # joined exog weather (the FD pipeline's own ERA5 axis values)
    # NOTE: exog["weather"] is ALREADY the pipeline's reindexed join result
    # — same length as hours_utc (8761 for AU, duplicates included).  Do
    # NOT dedup here: deduping breaks length parity with the NWP reindex
    # (the 2023-09-30 forensic crash was exactly this asymmetry).
    wx = data["exog"]["weather"]          # (T, 10): temp, swr, w100, ...
    era5_joined = pd.DataFrame(
        {"temp": wx[:, 0], "swr": wx[:, 1], "w100": wx[:, 2]},
        index=hours_utc)

    # farmblend detection (D5): does a farmblend file exist?
    info = cfgs.get(region)
    stem = info["file"].replace("_2023_hourly.csv", "")
    farm = sorted((DATA_DIR / "weather").glob(
        f"{stem}_farmblend_weather_*_hourly.csv"))
    has_farmblend = bool(farm)

    # centroid ERA5 (D5) — different source grid, same axis convention
    c_hours, c_arr = load_centroid_era5(region, cfgs, DATA_DIR)

    df_nwp = load_nwp_table(region, cfgs, DATA_DIR)
    if df_nwp is None:
        out[region] = {"error": "no nwp archive"}
        return

    rec = {"T": T, "split": split, "has_farmblend": has_farmblend,
           "n_axis_hours": len(hours_utc),
           "axis_dup_hours": int(len(hours_utc) - len(set(hours_utc)))}

    # D4: DST calendar exactness for AU
    if region in ("NSW1", "VIC1", "SA1", "QLD1"):
        try:
            from zoneinfo import ZoneInfo
            zmap = {"NSW1": "Australia/Sydney", "VIC1": "Australia/Melbourne",
                    "SA1": "Australia/Adelaide", "QLD1": "Australia/Brisbane"}
            local_hours = pd.DatetimeIndex(data["hours"])
            true_off = np.array([
                int(local_hours[i].tz_localize(
                    ZoneInfo(zmap[region])).utcoffset().total_seconds() // 3600)
                for i in range(len(local_hours))])
            month_off = (10 + np.isin(local_hours.month,
                                      [10, 11, 12, 1, 2, 3]).astype(int)
                         if region != "QLD1" else
                         np.full(len(local_hours), 10))
            diff = true_off != month_off
            rec["dst_rule_mismatch_hours"] = int(diff.sum())
            rec["dst_rule_mismatch_test_zone"] = int(diff[split:].sum())
            if diff.any():
                mm = pd.Series(local_hours[diff]).dt.strftime("%m-%d %H")
                rec["dst_rule_mismatch_examples"] = (
                    sorted(set(mm.dt.strftime("%m-%d")))[:6])
        except Exception as e:  # noqa: BLE001
            rec["dst_rule_error"] = str(e)[:80]

    models = {}
    for m in ("gfs", "icon"):
        if m == "ensemble":
            continue
        cols = {"temp": f"{m}_temp", "swr": f"{m}_swr",
                "w100": f"{m}_wind100"}
        j = pd.DataFrame({k: df_nwp[v].values for k, v in cols.items()},
                         index=pd.DatetimeIndex(df_nwp["hour"]))
        j = j[~j.index.duplicated(keep="first")].reindex(hours_utc)

        # D1: hole map
        holes = j.index[j["w100"].isna().values]
        hole_pos = pos[np.asarray(j["w100"].isna().values)]
        mrec = {
            "holes_total": int(len(holes)),
            "holes_test_zone": int((hole_pos >= split).sum()),
            "hole_positions_test_zone": [int(x) for x in
                                         hole_pos[hole_pos >= split]][:24],
            "hole_hours_test_zone": [str(h) for h in
                                     holes[hole_pos >= split]][:24],
        }

        # D2: lag scans vs the pipeline's joined ERA5 (farmblend when
        # present) and vs centroid ERA5 (grid-mismatch isolation)
        scans = {}
        for label, src in (("era5_pipeline", era5_joined), ):
            for col in ("w100", "swr"):
                r = {lag: lag_corr(j[col].values, src[col].values, lag)
                     for lag in LAGS}
                r = {k: (round(v, 4) if v is not None else None)
                     for k, v in r.items()}
                valid = {k: v for k, v in r.items() if v is not None}
                best = max(valid, key=valid.get) if valid else None
                scans[f"{label}_{col}"] = {
                    "corr_by_lag": {str(k): v for k, v in r.items()},
                    "best_lag": int(best) if best is not None else None,
                    "best_corr": valid.get(best) if best is not None else None,
                    "corr_lag0": r.get(0),
                }
        if c_arr is not None:
            cj = pd.DataFrame(
                {"temp": c_arr[:, 0], "swr": c_arr[:, 1], "w100": c_arr[:, 2]},
                index=c_hours)
            cj = cj[~cj.index.duplicated(keep="first")].reindex(hours_utc)
            for col in ("w100",):
                r = {lag: lag_corr(j[col].values, cj[col].values, lag)
                     for lag in LAGS}
                r = {k: (round(v, 4) if v is not None else None)
                     for k, v in r.items()}
                valid = {k: v for k, v in r.items() if v is not None}
                best = max(valid, key=valid.get) if valid else None
                scans[f"era5_centroid_{col}"] = {
                    "corr_by_lag": {str(k): v for k, v in r.items()},
                    "best_lag": int(best) if best is not None else None,
                    "best_corr": valid.get(best) if best is not None else None,
                    "corr_lag0": r.get(0),
                }
            # grid-mismatch baseline: centroid vs pipeline(farmblend) ERA5
            for col in ("w100",):
                a = cj[col].values
                b = era5_joined[col].values
                ok = np.isfinite(a) & np.isfinite(b) & (a != 0) & (b != 0)
                if ok.sum() > 200:
                    scans[f"era5_gridmismatch_{col}"] = round(float(
                        np.corrcoef(a[ok], b[ok])[0, 1]), 4)
        mrec["lag_scans"] = scans

        # D3: diurnal phase on local clock
        tz = 10.0 if region in ("NSW1", "QLD1", "VIC1", "SA1") else (
            {"US_BPAT": -8.0, "US_CISO": -8.0, "US_NYIS": -5.0}.get(region, 0.0))
        lh = ((hours_utc.hour.values + tz) % 24).astype(int)
        ph = {}
        for label, src in (("nwp", j), ("era5_pipeline", era5_joined)):
            swr = np.nan_to_num(src["swr"].values)
            w = np.nan_to_num(src["w100"].values)
            swr_diurnal = np.array([swr[lh == h].mean() for h in range(24)])
            w_diurnal = np.array([w[lh == h].mean() for h in range(24)])
            ph[f"{label}_swr_peak_local_hour"] = int(np.argmax(swr_diurnal))
            ph[f"{label}_w100_peak_local_hour"] = int(np.argmax(w_diurnal))
        mrec["diurnal_phase"] = ph
        models[m] = mrec
    rec["models"] = models
    out[region] = rec
    print(f"[fd48] {region}: farmblend={has_farmblend}, "
          f"gfs_holes_test={models['gfs']['holes_test_zone']}, "
          f"gfs_best_lag={models['gfs']['lag_scans']['era5_pipeline_w100']['best_lag']}, "
          f"icon_best_lag={models['icon']['lag_scans']['era5_pipeline_w100']['best_lag']}")


def main():
    out = {}
    cfgs = all_region_configs()
    for region in PANEL:
        try:
            scan(region, cfgs, out)
        except Exception as e:  # noqa: BLE001
            out[region] = {"error": repr(e)[:200]}
            print(f"[fd48][ERR] {region}: {e}")

    path = RESULTS / "fd48_join_forensics.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[fd48] -> {path}")

    # compact verdict-style summary
    md = ["# FD-48 join 取证（FD-47 残余恶化的对齐诊断）", "",
          "| region | farmblend | gfs holes(test) | icon holes(test) | "
          "gfs best lag w100 | icon best lag w100 | swr peak nwp/era5 |",
          "|---|---|---|---|---|---|---|"]
    for r, rec in out.items():
        if "error" in rec:
            md.append(f"| {r} | ERR | | | | | |")
            continue
        g = rec["models"]["gfs"]
        i = rec["models"]["icon"]
        gl = g["lag_scans"]["era5_pipeline_w100"]
        il = i["lag_scans"]["era5_pipeline_w100"]
        gp = g["diurnal_phase"]
        ep = g["diurnal_phase"]
        md.append(
            f"| {r} | {rec['has_farmblend']} "
            f"| {g['holes_test_zone']} | {i['holes_test_zone']} "
            f"| {gl['best_lag']} (r={gl['best_corr']}) "
            f"| {il['best_lag']} (r={il['best_corr']}) "
            f"| {gp['nwp_swr_peak_local_hour']}h/{ep['era5_pipeline_swr_peak_local_hour']}h |")
    for r, rec in out.items():
        if "error" in rec:
            continue
        if "dst_rule_mismatch_hours" in rec:
            md.append(f"\n- {r} DST 月规则 vs 真实日历："
                      f"{rec['dst_rule_mismatch_hours']} 小时不符，"
                      f"测试区 {rec['dst_rule_mismatch_test_zone']} 小时"
                      f"（例 {rec.get('dst_rule_mismatch_examples', [])}）")
    md_path = RESULTS / "fd48_join_forensics.md"
    md_path.write_text("\n".join(md) + "\n")
    print(f"[fd48] -> {md_path}")


if __name__ == "__main__":
    main()
