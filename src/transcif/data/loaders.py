"""Data loading for TransCIF: region discovery and per-region timeseries loading.

Region data lives in ``data_2023/`` as hourly CSVs with a uniform schema
(``renew_share``, ``cif_real_gco2_per_kwh``, ...).  This module provides the
entry points used by every experiment script:

    discover_uk_regions : scan the data dir and populate UK region configs
    load_region_data    : load one region's rs / cif arrays + scalar config
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.config import (
    DATA_DIR,
    TRAIN_FRACTION,
    AU_REGIONS,
    US_REGIONS,
    UK_REGIONS,
    get_fuel_shares,
    get_fuel_order,
)


def discover_uk_regions(data_dir=None, train_fraction=TRAIN_FRACTION):
    """Populate UK region configs by scanning the data directory for UK CSVs.

    Each discovered region gets an estimated non-renewable emission factor
    ``ef_nr`` from its own **training-period** data (median of CIF / (1 - rs)
    over valid hours).  Only the first ``train_fraction`` of each region's
    series is used so that the estimate does not leak test-period information
    into downstream zero-shot config vectors.

    Returns the populated ``UK_REGIONS`` dict.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    discovered = {}
    for f in sorted(glob.glob(str(data_dir / "UK_*_2023_hourly.csv"))):
        name = Path(f).stem.replace("_2023_hourly", "")
        df = pd.read_csv(f)
        rs = df["renew_share"].values
        cif = df["cif_real_gco2_per_kwh"].values
        # Restrict ef_nr estimation to the training split to avoid leaking
        # test-period CIF into the zero-shot config vector.
        split = int(len(rs) * train_fraction)
        rs_tr, cif_tr = rs[:split], cif[:split]
        mask = (rs_tr < 0.95) & (rs_tr > 0.05) & (cif_tr > 0)
        if mask.sum() > 500:
            ef_nr_est = float(np.median(cif_tr[mask] / (1 - rs_tr[mask])))
            if 100 < ef_nr_est < 2000:
                discovered[name] = {
                    "file": Path(f).name, "ef_r": 0.0, "ef_nr": ef_nr_est}
    UK_REGIONS.clear()
    UK_REGIONS.update(discovered)
    return UK_REGIONS


def load_region_data(region_name: str, all_configs: dict,
                     data_dir=None, train_fraction=TRAIN_FRACTION,
                     multi_year=False) -> dict:
    """Load a single region's rs / cif timeseries and scalar config.

    Args:
        region_name : key in ``all_configs`` (e.g. ``"QLD1"``, ``"US_CISO"``)
        all_configs : mapping name -> {"file", "ef_r", "ef_nr"}
        data_dir    : override for the data directory (defaults to DATA_DIR)
        train_fraction : fraction of the series used to derive the scalar
                      ``mean_rs`` / ``config`` statistics.  Only the first
                      ``train_fraction`` of the cleaned series contributes, so
                      the returned config does not leak test-period renewable
                      share into downstream zero-shot weighting or model input.
                      The full ``rs`` / ``cif`` arrays are still returned so
                      callers retain control over train/test splitting.

    Returns a dict with keys:
        rs, cif     : float32 arrays (cleaned to finite, non-negative CIF),
                     full length — callers slice as needed
        hours       : pd.DatetimeIndex (UTC) aligned to the cleaned rs/cif
                     arrays — added for the fuel-decomposed architecture so
                     per-fuel / weather / astronomy joins are exact
        mean_rs     : mean renewable share over the training split only
        ef_r, ef_nr : emission factors (tCO2/MWh)
        config      : np.array([train_mean_rs, ef_nr/1000], float32) — model
                     input built from the training split only
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs[region_name]
    # Discover optional additional years only when explicitly requested.  The
    # default remains the original 2023 protocol even if extra files have
    # been downloaded, so cached baselines stay reproducible.
    stem = info["file"].replace("_2023_hourly.csv", "")
    paths = sorted(data_dir.glob(f"{stem}_*_hourly.csv")) if multi_year else []
    if not paths:
        paths = [data_dir / info["file"]]
    ef_r, ef_nr = info["ef_r"], info["ef_nr"]
    frames = [pd.read_csv(path, parse_dates=["hour"]) for path in paths
              if path.exists()]
    if not frames:
        raise FileNotFoundError(f"no hourly files found for {region_name}")
    df = (pd.concat(frames, ignore_index=True)
            .drop_duplicates("hour", keep="last")
            .sort_values("hour").reset_index(drop=True))
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)
    valid = np.isfinite(rs) & np.isfinite(cif) & (cif >= 0)
    rs, cif = rs[valid], cif[valid]
    hours = pd.DatetimeIndex(df["hour"].values[valid])
    # Derive scalar config statistics from the training split only so the
    # zero-shot config vector does not embed test-period information.
    split = int(len(rs) * train_fraction)
    train_mean_rs = float(rs[:split].mean())
    # Multi-fuel config (Stage A): extend the 2-D [mean_rs, ef_nr/1000] vector
    # with per-fuel annual shares when the region has fuel breakdown data.
    # Regions without fuel data (e.g. AU) keep the legacy 2-D vector, so
    # downstream code that reads config[:2] is unaffected.
    fuel_shares = get_fuel_shares(region_name)
    base_cfg = [train_mean_rs, ef_nr / 1000.0]
    if fuel_shares:
        fuel_order = get_fuel_order()
        for f in fuel_order:
            base_cfg.append(float(fuel_shares.get(f, 0.0)))
    return {
        "rs": rs, "cif": cif, "hours": hours,
        "mean_rs": train_mean_rs,
        "ef_r": ef_r, "ef_nr": ef_nr,
        "config": np.array(base_cfg, dtype=np.float32),
        "fuel_shares": fuel_shares,
        "weather": _load_weather_aligned(data_dir, info["file"], len(rs),
                                          multi_year=multi_year),
    }


def _load_weather_aligned(data_dir, region_file, target_len, multi_year=False):
    """Load optional per-hour weather, aligned to the rs series length.

    Returns ``(target_len, 3)`` float32 array (temperature_c,
    shortwave_radiation, wind_speed_100m) padded/trimmed to match ``rs``, or
    ``None`` if no weather CSV exists.  Alignment guarantees downstream
    windowing (build_windows) produces matching rs/weather window counts.
    """
    region_stem = region_file.replace("_2023_hourly.csv", "")
    paths = sorted((data_dir / "weather").glob(
        f"{region_stem}_weather_*_hourly.csv"))
    if not multi_year:
        paths = [p for p in paths if p.name.endswith("_2023_hourly.csv")]
    if not paths:
        return None
    try:
        wdf = (pd.concat([pd.read_csv(p, parse_dates=["hour"]) for p in paths],
                         ignore_index=True)
                 .drop_duplicates("hour", keep="last")
                 .sort_values("hour").reset_index(drop=True))
        cols = ["temperature_c", "shortwave_radiation", "wind_speed_100m"]
        for c in cols:
            if c not in wdf.columns:
                return None
        w = wdf[cols].values.astype(np.float32)
        # Robust normalise each channel to zero mean / unit std so the model
        # sees comparable scales across regions (e.g. tropics vs temperate).
        std = w.std(axis=0, keepdims=True)
        std[std < 1e-3] = 1.0
        w = (w - w.mean(axis=0, keepdims=True)) / std
        # Align to rs length: trim if longer, pad with edge values if shorter.
        if len(w) >= target_len:
            return w[:target_len]
        pad = np.tile(w[-1:], (target_len - len(w), 1))
        return np.concatenate([w, pad], axis=0)
    except Exception:
        return None


def _load_weather(data_dir, region_file):
    """Load optional per-hour weather (temp, radiation, wind) aligned to rs.

    Returns a ``(T, 3)`` float32 array (temperature_c, shortwave_radiation,
    wind_speed_100m) if a matching ``weather/{REGION}_weather_2023_hourly.csv``
    exists, else ``None``.  Weather is a dynamic side input, not part of the
    static config vector.
    """
    region_stem = region_file.replace("_2023_hourly.csv", "")
    wpath = data_dir / "weather" / f"{region_stem}_weather_2023_hourly.csv"
    if not wpath.exists():
        return None
    try:
        wdf = pd.read_csv(wpath, parse_dates=["hour"])
        wdf = wdf.sort_values("hour").reset_index(drop=True)
        cols = ["temperature_c", "shortwave_radiation", "wind_speed_100m"]
        for c in cols:
            if c not in wdf.columns:
                return None
        w = wdf[cols].values.astype(np.float32)
        # Robust normalise each channel to zero mean / unit std so the model
        # sees comparable scales across regions (e.g. tropics vs temperate).
        std = w.std(axis=0, keepdims=True)
        std[std < 1e-3] = 1.0
        w = (w - w.mean(axis=0, keepdims=True)) / std
        return w
    except Exception:
        return None


def all_region_configs() -> dict:
    """Return the combined AU + US + UK region config mapping."""
    discover_uk_regions()
    return {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
