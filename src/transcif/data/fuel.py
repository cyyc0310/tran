"""Fuel-decomposed data layer for TransCIF-FD.

Loads per-fuel hourly share series (US: ``gen_*`` MWh columns / UK:
``perc_*`` percent columns from ``data_2023/fuel/``), aligns them to the
cleaned rs/cif series by timestamp, and builds the exogenous feature stack
(weather, astronomy, local-time calendar) that a telemetry-free target
region can still supply at deployment.

Canonical fuel axis (union of US and UK jurisdictions; AU regions have no
fuel telemetry and carry zeros + ``has_fuel=False``)::

    coal, gas, petroleum, nuclear, hydro, solar, wind, biomass, imports, other
"""

import numpy as np
import pandas as pd

from transcif.config import (
    DATA_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE,
    get_fuel_emission_factors,
)
from transcif.config.region_meta import get_region_meta
from transcif.physics.astro import sin_solar_elevation, clearsky_ghi, wind_capacity_factor
from transcif.data.calendar import calendar_features

# Canonical per-fuel axis used by every FD tensor.  Order is fixed so that
# model weights and per-fuel EF vectors are index-aligned across regions.
CANONICAL_FUELS = [
    "coal", "gas", "petroleum", "nuclear", "hydro",
    "solar", "wind", "biomass", "imports", "other",
]
FUEL_INDEX = {f: i for i, f in enumerate(CANONICAL_FUELS)}

THERMAL_FUELS = ("coal", "gas", "petroleum")
BASELOAD_FUELS = ("nuclear", "hydro", "biomass", "imports", "other")

# Fallback per-fuel emission factors (gCO2/kWh); the fuel_shares_*.json
# ``_emission_factors`` block is the source of truth and overrides these.
DEFAULT_FUEL_EFS = {
    "coal": 980.0, "gas": 410.0, "petroleum": 650.0,
    "nuclear": 0.0, "hydro": 0.0, "solar": 0.0, "wind": 0.0,
    "biomass": 0.0, "imports": 250.0, "other": 500.0,
}


def canonical_fuel_efs():
    """Canonical EF vector (len(CANONICAL_FUELS),) merged with the JSON EFs."""
    efs = dict(DEFAULT_FUEL_EFS)
    efs.update({k: float(v) for k, v in get_fuel_emission_factors().items()})
    return np.array([efs[f] for f in CANONICAL_FUELS], dtype=np.float64)


def load_fuel_shares(region_name, all_configs, data_dir=None):
    """Load a region's per-fuel hourly share matrix.

    Returns ``(hours, shares)`` — a UTC DatetimeIndex and an (T, F) float32
    matrix in canonical fuel order — or ``(None, None)`` when the region has
    no fuel telemetry (all AU regions).

    US fuel CSVs expose ``gen_{fuel}`` MWh columns (share = gen /
    total_gen); UK CSVs expose ``perc_{fuel}`` percent columns (share =
    perc / 100).  Shares are clipped to [0, 1] but not renormalised so the
    model sees the raw composition including its measurement noise.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = data_dir / "fuel" / f"{stem}_fuel_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    hours = pd.DatetimeIndex(df["hour"])
    T = len(df)
    shares = np.zeros((T, len(CANONICAL_FUELS)), dtype=np.float32)
    if "gen_coal" in df.columns or "gen_gas" in df.columns:
        # US schema: per-fuel generation over total generation.
        total = df["total_gen"].values.astype(np.float64)
        total = np.where(total > 1e-6, total, np.nan)
        for f in CANONICAL_FUELS:
            col = f"gen_{f}"
            if col in df.columns:
                shares[:, FUEL_INDEX[f]] = (df[col].values / total).astype(np.float32)
    else:
        # UK schema: perc_* columns already sum to ~100.
        for f in CANONICAL_FUELS:
            col = f"perc_{f}"
            if col in df.columns:
                shares[:, FUEL_INDEX[f]] = (df[col].values / 100.0).astype(np.float32)
    shares = np.nan_to_num(np.clip(shares, 0.0, 1.0), nan=0.0)
    return hours, shares


def load_raw_weather(region_name, all_configs, data_dir=None):
    """Load RAW (un-normalised) per-hour weather (T, 3) by timestamp.

    Columns: temperature_c, shortwave_radiation (W/m^2), wind_speed_100m
    (m/s).  Raw scales matter here: the physics transforms (clear-sky
    index, wind power curve) need physical units, unlike the z-scored
    side-channel in ``loaders._load_weather_aligned``.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = data_dir / "weather" / f"{stem}_weather_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    cols = ["temperature_c", "shortwave_radiation", "wind_speed_100m"]
    return pd.DatetimeIndex(df["hour"]), df[cols].values.astype(np.float32)


def attach_fuel_and_exog(data, region_name, all_configs, data_dir=None):
    """Enrich a ``load_region_data`` dict with fuel shares + exog features.

    Adds keys (all aligned to ``len(data['rs'])``):

        fuel_shares : (T, F) float32 — zeros when the region has no fuel
                      telemetry, with ``has_fuel=False``
        has_fuel    : bool
        ef_vec      : (F,) float64 — per-fuel EFs, thermal trio rescaled so
                      the config-weighted non-renewable average matches the
                      region's authoritative ``ef_nr``
        exog        : dict with raw ``weather`` (T, 3), ``astro`` (T, 2),
                      ``wind_cf`` (T,), ``clearsky_index`` (T,) and
                      ``calendar`` (T, 6)

    Alignment is by exact timestamp join so the historical 1-hour weather
    offset for UK regions cannot contaminate window construction.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    hours = data.get("hours")
    if hours is None:
        raise ValueError("data dict lacks 'hours'; use load_region_data from "
                         "transcif.data.loaders (updated 2026-08-15)")
    T = len(data["rs"])

    fuel_hours, fuel_shares = load_fuel_shares(region_name, all_configs, data_dir)
    has_fuel = fuel_hours is not None
    if has_fuel:
        # Timestamp join onto the cleaned rs series; missing rows -> zeros.
        fdf = pd.DataFrame(fuel_shares, index=fuel_hours)
        joined = fdf.reindex(hours).fillna(0.0)
        # Guard against duplicate timestamps in either index.
        joined = joined[~joined.index.duplicated(keep="first")]
        fuel_shares = joined.values.astype(np.float32)
    else:
        fuel_shares = np.zeros((T, len(CANONICAL_FUELS)), dtype=np.float32)

    lat, lon, tz = get_region_meta(region_name)
    sin_elev = sin_solar_elevation(hours, lat, lon)
    astro = np.stack([sin_elev, clearsky_ghi(sin_elev)], axis=1).astype(np.float32)
    clearsky = np.maximum(astro[:, 1], 1.0)
    cal = calendar_features(hours, tz_offset=tz)

    w_hours, w_raw = load_raw_weather(region_name, all_configs, data_dir)
    if w_hours is not None:
        wdf = pd.DataFrame(w_raw, index=w_hours)
        wdf = wdf[~wdf.index.duplicated(keep="first")]
        w_joined = wdf.reindex(hours)
        weather = np.nan_to_num(w_joined.values, nan=0.0).astype(np.float32)
    else:
        weather = np.zeros((T, 3), dtype=np.float32)

    wind_cf = wind_capacity_factor(weather[:, 2]).astype(np.float32)
    csi = np.clip(weather[:, 1] / clearsky, 0.0, 1.3).astype(np.float32)
    # 5-channel weather-exog matrix: physical raw values + physics transforms
    # so models never need to re-learn the turbine curve / clear-sky ratio.
    wx = np.concatenate([weather, wind_cf[:, None], csi[:, None]], axis=1).astype(np.float32)

    data["fuel_shares"] = fuel_shares
    data["has_fuel"] = has_fuel
    data["ef_vec"] = region_fuel_efs(data, region_name)
    data["exog"] = {
        "weather": wx, "astro": astro, "calendar": cal,
        "wind_cf": wind_cf, "clearsky_index": csi,
    }
    return data


def region_fuel_efs(data, region_name):
    """Per-fuel EF vector with the thermal trio rescaled to the region ef_nr.

    The canonical per-fuel EFs carry the coal/gas distinction (which drives
    CIF dynamics), while the region's ``ef_nr`` is the authoritative
    non-renewable level (hard-coded for AU/US, estimated from the training
    split for UK).  We scale coal/gas/petroleum EFs by a single factor so
    that ``Σ_{non-renewable f} ef_f · config_share_f == ef_nr ·
    Σ config_share_f`` for regions with fuel configs.  Regions without
    fuel configs (AU) get the thermal trio set to ``ef_nr`` directly —
    the coal/gas split then correctly has zero CIF effect.
    """
    efs = canonical_fuel_efs().copy()
    ef_nr = float(data["ef_nr"])
    fs = data.get("fuel_shares")
    if fs is None or not data.get("has_fuel", False):
        for f in THERMAL_FUELS:
            efs[FUEL_INDEX[f]] = ef_nr
        return efs
    # Config (annual) shares from the fuel series restricted to the
    # training split — no test leakage into an input feature.
    split = int(len(fs) * 0.8)
    annual = fs[:split].mean(axis=0)
    # Per-jurisdiction non-renewable set: US counts nuclear as a zero-EF
    # non-renewable; UK counts nuclear + biomass as renewable.
    juris = "us" if region_name.startswith("US_") else "uk"
    renewable = jurisdiction_renewable_fuels().get(juris, set())
    nonren_idx = [FUEL_INDEX[f] for f in CANONICAL_FUELS if f not in renewable]
    thermal_idx = [FUEL_INDEX[f] for f in THERMAL_FUELS]
    nonren_mass = annual[nonren_idx].sum()
    thermal_mass = annual[thermal_idx].sum()
    if thermal_mass < 1e-3 or nonren_mass < 1e-3:
        for f in THERMAL_FUELS:
            efs[FUEL_INDEX[f]] = ef_nr
        return efs
    # target thermal contribution = ef_nr·nonren_mass − (imports+other) part
    other_idx = [i for i in nonren_idx if i not in thermal_idx]
    other_part = float(np.dot(efs[other_idx], annual[other_idx])) if other_idx else 0.0
    target_thermal = ef_nr * nonren_mass - other_part
    cur_thermal = float(np.dot(efs[thermal_idx], annual[thermal_idx]))
    if cur_thermal < 1e-6:
        for f in THERMAL_FUELS:
            efs[FUEL_INDEX[f]] = ef_nr
        return efs
    scale = float(np.clip(target_thermal / cur_thermal, 0.2, 3.0))
    efs[thermal_idx] *= scale
    return efs


def get_renewable_fuels():
    """Renewable fuel keys per jurisdiction JSON ``_renewable_fuels``.

    US counts [hydro, solar, wind] as renewable (nuclear is a zero-EF
    non-renewable); UK additionally counts biomass and nuclear.  The union
    drives share-consistency checks; per-jurisdiction semantics live in
    ``jurisdiction_renewable_fuels``.
    """
    union = set()
    for fuels in jurisdiction_renewable_fuels().values():
        union.update(fuels)
    return union


def jurisdiction_renewable_fuels():
    """{jurisdiction: set(renewable fuel keys)} from the fuel JSONs."""
    import json  # noqa: PLC0415
    from transcif.config import FUEL_DIR  # noqa: PLC0415
    out = {}
    for juris, name in (("us", "fuel_shares_us.json"), ("uk", "fuel_shares_uk.json")):
        path = FUEL_DIR / name
        if not path.exists():
            continue
        with open(path) as f:
            doc = json.load(f)
        out[juris] = set(doc.get("_renewable_fuels", []))
    return out


def fuel_cif(shares, ef_vec):
    """Vectorised CIF from a share matrix: (..., F) @ (F,) -> (...)."""
    return shares @ ef_vec


FD_CONFIG_FIELDS = (
    ["mean_rs", "ef_nr_scaled"]
    + list(CANONICAL_FUELS)          # 10 per-fuel annual shares (0 for AU)
    + ["ann_wind_cf", "ann_clearsky_index", "has_fuel", "abs_lat_scaled"]
)


def build_fd_config(data, region_name):
    """FD config vector (len(FD_CONFIG_FIELDS) == 16, all deployment-public).

    = [mean_rs, ef_nr/1000, 10 fuel shares, annual weather-capacity means
    (wind CF, clear-sky index), has_fuel flag, |lat|/60].  Every entry is
    derivable from public statistics only: monthly fuel-mix tables, IPCC
    emission factors and reanalysis weather climatology — the exact
    interface a telemetry-free target region (e.g. a Chinese province)
    can supply.
    """
    lat, lon, tz = get_region_meta(region_name)
    fuel_cfg = data["fuel_shares"]
    split = int(len(fuel_cfg) * 0.8)
    annual = fuel_cfg[:split].mean(axis=0) if data.get("has_fuel") else np.zeros(
        len(CANONICAL_FUELS))
    ex = data["exog"]
    w_split = ex["wind_cf"][:split]
    csi_split = ex["clearsky_index"][:split]
    ann_windcf = float(w_split.mean())
    # Daytime-only clear-sky index mean (night values are 0 by construction).
    day = ex["astro"][:split, 0] > 0
    ann_csi = float(csi_split[day].mean()) if day.any() else 0.0
    vec = [
        float(data["mean_rs"]), float(data["ef_nr"]) / 1000.0,
        *[float(s) for s in annual],
        ann_windcf, ann_csi,
        1.0 if data.get("has_fuel") else 0.0,
        abs(lat) / 60.0,
    ]
    return np.array(vec, dtype=np.float32)


def build_fd_windows(data, seq_len=SEQ_LEN, horizon=HORIZON, stride=TRAIN_STRIDE,
                     max_windows=None, rng=None, starts=None):
    """Build fuel-decomposed training/eval windows for one region.

    ``starts`` optionally supplies explicit local start positions (e.g. a
    shared absolute-origin grid so windows from different regions cover the
    same calendar period and can be mixed pairwise — see
    ``training.synthetic``); otherwise positions come from ``stride``.

    Returns a dict of float32 arrays (empty trailing dims when the series
    is too short):

        x_rs        (n, L)      past renewable-share history
        x_fuel      (n, L, F)   past per-fuel shares (zeros for AU)
        y_fuel      (n, H, F)   future per-fuel shares (zeros for AU)
        y_rs        (n, H)      future renewable-share ground truth
        y_cif       (n, H)      future CIF ground truth
        x_weather   (n, L, 5)   past weather-exog: temp, shortwave, wind
                                speed, wind capacity factor, clear-sky index
        fut_weather (n, H, 5)   future weather-exog (24 h reanalysis proxy)
        fut_exog    (n, H, K)   future exog: sin_elev, clearsky, wind_cf,
                                clearsky_index + 6 calendar channels (K=10)
        origin_hours (n,)       pd.DatetimeIndex of each window's origin

    ``fut_*`` channels are legitimately available at deployment: astronomy
    is exact for any future date and day-ahead weather is a standard input
    (reanalysis serves as its proxy, mirroring EnsembleCI's protocol).
    """
    rs = data["rs"]
    cif = data["cif"]
    fuel = data["fuel_shares"]
    ex = data["exog"]
    weather, astro, cal = ex["weather"], ex["astro"], ex["calendar"]
    wind_cf, csi = ex["wind_cf"], ex["clearsky_index"]
    fut_exog_full = np.concatenate([astro, wind_cf[:, None], csi[:, None], cal],
                                   axis=1).astype(np.float32)
    hours = data["hours"]

    window = seq_len + horizon
    if starts is not None:
        starts = [int(s) for s in starts
                  if 0 <= s and s + window <= len(rs)]
    else:
        starts = list(range(0, len(rs) - window + 1, stride))
    if max_windows is not None and len(starts) > max_windows:
        if rng is None:
            rng = np.random.default_rng(0)
        idx = rng.choice(len(starts), size=max_windows, replace=False)
        starts = [starts[i] for i in sorted(idx)]

    def _out(n):
        return {
            "x_rs": np.empty((n, seq_len), np.float32),
            "x_fuel": np.empty((n, seq_len, len(CANONICAL_FUELS)), np.float32),
            "y_fuel": np.empty((n, horizon, len(CANONICAL_FUELS)), np.float32),
            "y_rs": np.empty((n, horizon), np.float32),
            "y_cif": np.empty((n, horizon), np.float32),
            "x_weather": np.empty((n, seq_len, 5), np.float32),
            "fut_weather": np.empty((n, horizon, 5), np.float32),
            "fut_exog": np.empty((n, horizon, fut_exog_full.shape[1]), np.float32),
        }

    if not starts:
        empty = _out(0)
        empty["origin_hours"] = pd.DatetimeIndex([])
        return empty
    out = _out(len(starts))
    for i, s in enumerate(starts):
        h0, h1 = s + seq_len, s + window
        out["x_rs"][i] = rs[s:h0]
        out["x_fuel"][i] = fuel[s:h0]
        out["y_fuel"][i] = fuel[h0:h1]
        out["y_rs"][i] = rs[h0:h1]
        out["y_cif"][i] = cif[h0:h1]
        out["x_weather"][i] = weather[s:h0]
        out["fut_weather"][i] = weather[h0:h1]
        out["fut_exog"][i] = fut_exog_full[h0:h1]
    out["origin_hours"] = pd.DatetimeIndex([hours[s + seq_len] for s in starts])
    return out
