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

from transcif.config import DATA_DIR, AU_REGIONS, US_REGIONS, UK_REGIONS


def discover_uk_regions(data_dir=None):
    """Populate UK region configs by scanning the data directory for UK CSVs.

    Each discovered region gets an estimated non-renewable emission factor
    ``ef_nr`` from its own data (median of CIF / (1 - rs) over valid hours).
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
        mask = (rs < 0.95) & (rs > 0.05) & (cif > 0)
        if mask.sum() > 500:
            ef_nr_est = float(np.median(cif[mask] / (1 - rs[mask])))
            if 100 < ef_nr_est < 2000:
                discovered[name] = {
                    "file": Path(f).name, "ef_r": 0.0, "ef_nr": ef_nr_est}
    UK_REGIONS.clear()
    UK_REGIONS.update(discovered)
    return UK_REGIONS


def load_region_data(region_name: str, all_configs: dict,
                     data_dir=None) -> dict:
    """Load a single region's rs / cif timeseries and scalar config.

    Args:
        region_name : key in ``all_configs`` (e.g. ``"QLD1"``, ``"US_CISO"``)
        all_configs : mapping name -> {"file", "ef_r", "ef_nr"}
        data_dir    : override for the data directory (defaults to DATA_DIR)

    Returns a dict with keys:
        rs, cif     : float32 arrays (cleaned to finite, non-negative CIF)
        mean_rs     : mean renewable share
        ef_r, ef_nr : emission factors (tCO2/MWh)
        config      : np.array([mean_rs, ef_nr/1000], float32) — model input
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs[region_name]
    path = data_dir / info["file"]
    ef_r, ef_nr = info["ef_r"], info["ef_nr"]
    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)
    valid = np.isfinite(rs) & np.isfinite(cif) & (cif >= 0)
    rs, cif = rs[valid], cif[valid]
    return {
        "rs": rs, "cif": cif,
        "mean_rs": float(rs.mean()),
        "ef_r": ef_r, "ef_nr": ef_nr,
        "config": np.array([rs.mean(), ef_nr / 1000.0], dtype=np.float32),
    }


def all_region_configs() -> dict:
    """Return the combined AU + US + UK region config mapping."""
    discover_uk_regions()
    return {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
