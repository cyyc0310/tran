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
    paths = sorted((data_dir / "fuel").glob(f"{stem}_fuel_*_hourly.csv"))
    if not paths:
        return None, None
    df = (pd.concat([pd.read_csv(p, parse_dates=["hour"]) for p in paths],
                    ignore_index=True)
            .drop_duplicates("hour", keep="last")
            .sort_values("hour").reset_index(drop=True))
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


def load_raw_weather(region_name, all_configs, data_dir=None, multi_year=False):
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
    # Wind-farm blend override (FD-17): when a capacity-weighted
    # farm-fleet weather file exists (VIC1/SA1 — centroid cells badly
    # misrepresent clustered wind fleets), prefer it.  Same schema and
    # units as the centroid file; the wind column is the blended 100 m
    # speed, temperature/shortwave stay centroid (demand channels).
    farm_paths = sorted((data_dir / "weather").glob(
        f"{stem}_farmblend_weather_*_hourly.csv"))
    if not farm_paths and region_name.startswith("UK_"):
        # GB synoptic wind is nationwide-coherent (FD-26): UK regions
        # without their own farm table share the national fleet blend.
        farm_paths = sorted((data_dir / "weather").glob(
            "UK_18_GB_farmblend_weather_*_hourly.csv"))
    paths = farm_paths or sorted((data_dir / "weather").glob(
        f"{stem}_weather_*_hourly.csv"))
    if not multi_year:
        paths = [p for p in paths if p.name.endswith("_2023_hourly.csv")]
    if not paths:
        return None, None
    df = (pd.concat([pd.read_csv(p, parse_dates=["hour"]) for p in paths],
                    ignore_index=True)
            .drop_duplicates("hour", keep="last")
            .sort_values("hour").reset_index(drop=True))
    cols = ["temperature_c", "shortwave_radiation", "wind_speed_100m"]
    arr = df[cols].values.astype(np.float32)
    # UNIT FIX (FD-17): the Open-Meteo archive serves wind speed in km/h
    # (its default) — 39-42% of hours sat above the IEC cut-out when fed
    # to the m/s power curve, reading good wind as zero output.  Convert
    # here so every consumer (wind CF channel, regime features, annual
    # climatology config, weather-noise track) sees m/s.
    arr[:, 2] = arr[:, 2] / 3.6
    return pd.DatetimeIndex(df["hour"]), arr


def load_pressure_winds(region_name, all_configs, data_dir=None):
    """Load optional gust + synoptic-pressure tracks (T, 2) by timestamp.

    From ``data_2023/weather2/{REGION}_wind2_2023_hourly.csv`` (Open-Meteo
    ERA5: wind_gusts_10m km/h -> m/s, pressure_msl -> hPa anomaly from
    1013).  Returns ``(hours, array)`` or ``(None, None)`` when absent —
    the FD layer degrades gracefully to zeros.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = data_dir / "weather2" / f"{stem}_wind2_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    gust = df["wind_gusts_10m"].values / 3.6                  # km/h -> m/s
    pres = df["pressure_msl"].values - 1013.0                  # hPa anomaly
    w = np.stack([gust, pres], axis=1)
    return pd.DatetimeIndex(df["hour"]), w.astype(np.float32)


def load_demand(region_name, all_configs, data_dir=None):
    """Load optional hourly demand (T, 2) by timestamp: actual + forecast.

    From ``data_2023/demand/{REGION}_demand_2023_hourly.csv`` (EIA-930;
    the forecast column is the balancing authority's own DAY-AHEAD load
    forecast — deployment-legal day-ahead input).  Returns
    ``(hours, array)`` or ``(None, None)`` for non-US regions.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs.get(region_name)
    if info is None:
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    paths = sorted((data_dir / "demand").glob(f"{stem}_demand_*_hourly.csv"))
    if not paths:
        return None, None
    # Multi-year demand files concat on the timestamp grid (the 2023-only
    # file kept the original single-year protocol reproducible).
    frames = []
    for path in paths:
        d = pd.read_csv(path, parse_dates=["hour"])
        d = d.drop_duplicates("hour", keep="last")
        frames.append(d)
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("hour", keep="last")
          .sort_values("hour").reset_index(drop=True))
    return pd.DatetimeIndex(df["hour"]), df[
        ["demand_actual_mw", "demand_forecast_mw"]].values.astype(np.float32)


def load_regional_state(region_name, all_configs, data_dir=None):
    """Load optional public AU hourly system-state output.

    This is a causal deployment feature: the past channel is observed
    regional sent-out generation, while the future channel is constructed in
    ``attach_fuel_and_exog`` from a train-only month/hour climatology.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    info = all_configs.get(region_name)
    if info is None or not region_name.endswith(("1", "2", "3", "4", "5")):
        return None, None
    stem = info["file"].replace("_2023_hourly.csv", "")
    path = data_dir / "state" / f"{stem}_state_2023_hourly.csv"
    if not path.exists():
        return None, None
    df = pd.read_csv(path, parse_dates=["hour"]).sort_values("hour")
    return pd.DatetimeIndex(df["hour"]), df[
        ["generation_sent_out_mw"]].values.astype(np.float32)


def load_prices_2023(data_dir=None):
    """Monthly fuel prices (World Bank pink sheet + FRED), z-scored.

    Returns {jurisdiction: (coal_z (12,), gas_z (12,))} or None.  AU gas
    proxies to Japan LNG (east-coast export parity); UK to Europe TTF; US
    to the FRED daily Henry-Hub monthly mean (fallback: pink-sheet US).
    """
    if data_dir is None:
        data_dir = DATA_DIR
    path = data_dir / "prices" / "prices_2023.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)

    def z(v):
        v = np.asarray(v, dtype=np.float64)
        s = np.nanstd(v)
        return np.nan_to_num((v - np.nanmean(v)) / (s if s > 1e-9 else 1.0))

    gas_us = df["gas_us_daily"].fillna(df["gas_us"])
    return {
        "au": (z(df["coal_newc"]), z(df["gas_jp"])),
        "uk": (z(df["coal_newc"]), z(df["gas_eu"])),
        "us": (z(df["coal_newc"]), z(gas_us)),
    }


def jurisdiction_of(region_name):
    return "us" if region_name.startswith("US_") else (
        "uk" if region_name.startswith("UK_") else "au")


def attach_fuel_and_exog(data, region_name, all_configs, data_dir=None,
                         use_au_state=False):
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
        # Dedup the SOURCE index first (DST repeated hours), then reindex to
        # ``hours`` so the result is exactly aligned to len(rs).
        fdf = pd.DataFrame(fuel_shares, index=fuel_hours)
        fdf = fdf[~fdf.index.duplicated(keep="first")]
        joined = fdf.reindex(hours).fillna(0.0)
        fuel_shares = joined.values.astype(np.float32)
    else:
        fuel_shares = np.zeros((T, len(CANONICAL_FUELS)), dtype=np.float32)

    lat, lon, tz = get_region_meta(region_name)
    # Timeline normalisation: US/UK dataset timestamps are UTC, but AU NEM
    # series are LOCAL (NEM time, UTC+10) — astronomy and the UTC-stamped
    # weather joins must therefore run on a converted index for AU.
    # (QLD is DST-free so the fixed offset is exact; NSW/VIC/SA carry a
    # 1 h summer kink, documented.)
    if jurisdiction_of(region_name) == "au":
        # NEM local -> UTC with DST correction: NSW/VIC/SA observe
        # summer time (Oct-Apr, +1 h); QLD does not.
        dst_regions = {"NSW1", "VIC1", "SA1"}
        in_dst_months = np.isin(
            np.asarray(hours.month), [10, 11, 12, 1, 2, 3])
        off = tz + (in_dst_months.astype(float)
                    if region_name in dst_regions else 0.0)
        hours_utc = hours - pd.to_timedelta(off, unit="h")
    else:
        hours_utc = hours
    sin_elev = sin_solar_elevation(hours_utc, lat, lon)
    astro = np.stack([sin_elev, clearsky_ghi(sin_elev)], axis=1).astype(np.float32)
    clearsky = np.maximum(astro[:, 1], 1.0)
    cal = calendar_features(hours_utc, tz_offset=tz)

    w_hours, w_raw = load_raw_weather(
        region_name, all_configs, data_dir,
        multi_year=bool(len(data.get("hours", [])) > 9000))
    if w_hours is not None:
        wdf = pd.DataFrame(w_raw, index=w_hours)
        wdf = wdf[~wdf.index.duplicated(keep="first")]
        w_joined = wdf.reindex(hours_utc)
        weather = np.nan_to_num(w_joined.values, nan=0.0).astype(np.float32)
    else:
        weather = np.zeros((T, 3), dtype=np.float32)

    wind_cf = wind_capacity_factor(weather[:, 2]).astype(np.float32)
    csi = np.clip(weather[:, 1] / clearsky, 0.0, 1.3).astype(np.float32)
    # Degree-hour channels (roadmap C-class): heating/cooling degree hours
    # drive the thermal dispatch response to load (the duck-curve evening
    # ramp) — HDH = max(0, 15.5 - T), CDH = max(0, T - 22).
    hdh = np.clip(15.5 - weather[:, 0], 0.0, None).astype(np.float32)
    cdh = np.clip(weather[:, 0] - 22.0, 0.0, None).astype(np.float32)
    # Gust + synoptic pressure (roadmap B-class): ramp-event proxy and
    # frontal signal, timestamp-joined; graceful zeros when absent.
    p_hours, p_winds = load_pressure_winds(region_name, all_configs, data_dir)
    if p_hours is not None:
        pdf = pd.DataFrame(p_winds, index=p_hours)
        pdf = pdf[~pdf.index.duplicated(keep="first")]
        p_joined = pdf.reindex(hours_utc)
        winds2 = np.nan_to_num(p_joined.values, nan=0.0).astype(np.float32)
    else:
        winds2 = np.zeros((T, 2), dtype=np.float32)
    # 7-channel weather-exog matrix: raw surface weather + physics
    # transforms + pressure-level winds.
    wx = np.concatenate([weather, wind_cf[:, None], csi[:, None], winds2],
                        axis=1).astype(np.float32)

    # Demand channels (roadmap E/C-class, FD-15): EIA-930 actual (past
    # windows) + day-ahead forecast (horizon), z-scored on the train split.
    d_hours, d_raw = load_demand(region_name, all_configs, data_dir)
    demand = np.zeros((T, 2), dtype=np.float32)
    if d_hours is not None:
        ddf = pd.DataFrame(d_raw, index=d_hours)
        ddf = ddf[~ddf.index.duplicated(keep="first")]
        dj = ddf.reindex(hours_utc)   # US timelines are UTC already
        vals = np.nan_to_num(dj.values, nan=0.0)
        split = int(T * 0.8)
        mu = vals[:split].mean(axis=0)
        sd = vals[:split].std(axis=0)
        sd[sd < 1e-6] = 1.0
        demand = ((vals - mu) / sd).astype(np.float32)
    elif use_au_state:
        # AU has no EIA-930 feed in this project.  Reuse the existing demand
        # feature slots for public NEM regional state when available.
        s_hours, s_raw = load_regional_state(region_name, all_configs, data_dir)
        if s_hours is not None:
            sdf = pd.DataFrame(s_raw, index=s_hours)
            sdf = sdf[~sdf.index.duplicated(keep="first")]
            sj = sdf.reindex(hours_utc)
            vals = np.nan_to_num(sj.values, nan=0.0)
            split = int(T * 0.8)
            mu, sd = vals[:split].mean(axis=0), vals[:split].std(axis=0)
            sd[sd < 1e-6] = 1.0
            actual = ((vals - mu) / sd).astype(np.float32)[:, 0]
            # Train-only seasonal forecast; no target test values enter it.
            train_hours = hours_utc[:split]
            train_z = actual[:split]
            key = np.array([(h.month, h.hour) for h in train_hours])
            table = {(m, h): float(train_z[(key[:, 0] == m) &
                                             (key[:, 1] == h)].mean())
                     for m in range(1, 13) for h in range(24)
                     if np.any((key[:, 0] == m) & (key[:, 1] == h))}
            forecast = np.array([table.get((h.month, h.hour), 0.0)
                                 for h in hours_utc], dtype=np.float32)
            demand = np.stack([actual, forecast], axis=1)
    wx = np.concatenate([wx, demand[:, :1]], axis=1).astype(np.float32)
    dem_fut = demand[:, 1]

    # Wind-regime channels (extreme-weather attribution, FD-16): the
    # attribution study showed CIF volatility peaks in the wind-share
    # TRANSITION band (drought onset/exit), not during storms.  Both
    # channels are strictly causal (only past hours) and weather-derived,
    # so they are deployment-legal at every information tier:
    #   idx 8  wind_regime24 : trailing 24 h mean of the normalised wind
    #                         CF — drought persistence (1 = regime normal)
    #   idx 9  wind_tend6    : 6 h change of the regime — onset/exit ramps
    regime24 = pd.Series(wind_cf).rolling(24, min_periods=1).mean().values
    tend6 = np.zeros_like(regime24)
    tend6[6:] = regime24[6:] - regime24[:-6]
    wx = np.concatenate([wx, regime24[:, None].astype(np.float32),
                         tend6[:, None].astype(np.float32)], axis=1)

    # Fuel-price channels (roadmap E-class): jurisdiction-mapped z-scored
    # monthly coal/gas prices broadcast to hours with a 1-month publication
    # lag (window in month m uses month m-1; January wraps to December of
    # the same table — a documented approximation at monthly granularity).
    prices = load_prices_2023(data_dir)
    coal_z = np.zeros(T, dtype=np.float32)
    gas_z = np.zeros(T, dtype=np.float32)
    if prices is not None:
        cz, gz = prices[jurisdiction_of(region_name)]
        lagged = np.roll(np.stack([cz, gz], axis=1), 1, axis=0)  # (12, 2)
        month_idx = (hours.month.values - 1).astype(int)
        coal_z = lagged[month_idx, 0].astype(np.float32)
        gas_z = lagged[month_idx, 1].astype(np.float32)

    data["fuel_shares"] = fuel_shares
    data["has_fuel"] = has_fuel
    data["ef_vec"] = calibrated_fuel_efs(data, region_name)
    data["exog"] = {
        "weather": wx, "astro": astro, "calendar": cal,
        "wind_cf": wind_cf, "clearsky_index": csi,
        "wind_regime24": regime24.astype(np.float32),
        "wind_tend6": tend6.astype(np.float32),
        "hdh": hdh, "cdh": cdh, "coal_z": coal_z, "gas_z": gas_z,
        "demand_fut": dem_fut,
    }
    return data


def calibrated_fuel_efs(data, region_name):
    """Effective per-fuel EFs calibrated on the target's TRAIN split.

    The canonical/IPCC EFs (rescaled to ef_nr) misrepresent how each
    data source actually maps its own fuel mix to a reported CIF — the
    UK API's interconnector accounting, DUID classification drift, EIA
    methodology.  A ridge regression of the reported CIF on the observed
    per-fuel shares (train split only, shrunken toward the canonical
    vector) recovers each source's EFFECTIVE emission factors:

        argmin ||X @ ef - y||^2 + lam ||ef - ef_canonical||^2

    True-share residual floor collapses for the label-accounting family
    (test-period, true shares): UK_14 59.6 -> 27.1, UK_13 34.5 -> 19.2,
    UK_12 35.1 -> 20.5, UK_09 14.0 -> 8.0, US_NYIS 19.3 -> 2.5,
    US_PJM 5.6 -> 1.3.  Regions without fuel telemetry keep the
    canonical vector (the regression is unidentifiable).
    """
    ef_c = region_fuel_efs(data, region_name)
    if not data.get("has_fuel"):
        return ef_c
    fs = np.asarray(data["fuel_shares"], dtype=np.float64)
    y = np.asarray(data["cif"], dtype=np.float64)
    split = int(len(y) * 0.8)
    X, yt = fs[:split], y[:split]
    scale = float(np.linalg.norm(X, axis=0).mean())
    import os
    lam = float(os.environ.get("CALIB_LAMBDA", 15.0)) * split / 1000.0 * max(scale, 1e-6)
    A = X.T @ X + lam * np.eye(X.shape[1])
    b = X.T @ yt + lam * ef_c
    ef_k = np.linalg.solve(A, b)
    return np.clip(ef_k, 0.0, 1400.0)


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
    juris = jurisdiction_of(region_name)
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
    for juris, name in (("us", "fuel_shares_us.json"), ("uk", "fuel_shares_uk.json"),
                        ("au", "fuel_shares_au.json")):
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


def build_monthly_config_table(data, region_name, shrink=None,
                               history_only=False):
    """Month-indexed FD config table (12, D) — the monthly-interface variant.

    Built from MONTHLY fuel-mix statistics (the exact input a Chinese
    province publishes: per-month generation by fuel).  Each row is the
    16-dim FD config with that month's fuel shares; ``mean_rs`` is derived
    from the fuel shares (per the jurisdiction renewable definition), NOT
    from rs telemetry — so the whole table is deployment-legal for I_cfg.
    Weather climatology entries stay annual (reanalysis-derived).

    Returns None for regions without fuel telemetry (AU fallback = annual).
    """
    if not data.get("has_fuel"):
        return None
    lat, lon, tz = get_region_meta(region_name)
    hours = data["hours"]
    import os
    if shrink is None:
        shrink = float(os.environ.get("MONTHLY_SHRINK", 0.5))
    fuel = data["fuel_shares"]
    ex = data["exog"]
    split = int(len(fuel) * 0.8)
    annual_shares = fuel[:split].mean(axis=0)
    juris = jurisdiction_of(region_name)
    renewable = jurisdiction_renewable_fuels().get(juris, set())
    renew_mask = np.array([f in renewable for f in CANONICAL_FUELS])
    w_split = ex["wind_cf"][:split]
    day = ex["astro"][:split, 0] > 0
    ann_windcf = float(w_split.mean())
    ann_csi = float(ex["clearsky_index"][:split][day].mean()) if day.any() else 0.0
    months = hours.month.values
    years = hours.year.values
    table = np.zeros((12, len(FD_CONFIG_FIELDS)), dtype=np.float32)
    for m in range(1, 13):
        # Row m is only ever read by origins in month m+1 or later (the
        # 1-month publication lag of official statistics — exactly the
        # Chinese province interface).  By then month m is complete and
        # its aggregate is public, so the FULL month m is legal input;
        # using only the train split here would simulate a deployment
        # with no access to published monthly statistics.
        sel = months == m
        if history_only:
            # Strict deployment protocol: only statistics available before
            # the global train/test boundary may define the month table.
            # This matters for multi-year holdouts, where otherwise the
            # target's test-year fuel mix would leak into the input.
            sel &= np.arange(len(months)) < split
        elif np.unique(years).size > 1:
            # Multi-year series: official statistics are published per
            # YEAR, so month m means the LATEST year's month m (FD-29b) —
            # averaging 2022+2023+2024 Octobers dilutes the 2024-specific
            # structural drift the monthly interface exists to track.
            latest = years[sel].max()
            sel &= (years == latest)
        monthly = fuel[sel].mean(axis=0) if sel.any() else None
        if monthly is None:
            shares = annual_shares
        else:
            # Shrinkage toward the annual mean (FD-20): raw single-month
            # telemetry means carry sampling noise that hurts grids whose
            # annual level was already right (NYIS/PJM-class) — keep half
            # of the seasonal deviation, which is where the signal lives
            # (CISO-class cold-mode bias).
            shares = (1.0 - shrink) * annual_shares + shrink * monthly
        mean_rs_m = float(shares[renew_mask].sum())
        table[m - 1] = np.array([
            mean_rs_m, float(data["ef_nr"]) / 1000.0,
            *[float(s) for s in shares],
            ann_windcf, ann_csi, 1.0, abs(lat) / 60.0,
        ], dtype=np.float32)
    return table


def monthly_config_at(table, origin_hours, lag_months=1):
    """Per-window config lookup with publication lag.

    For a window in month m, return the table row of month ``m - lag``
    (deployment realism: monthly statistics publish with a ~1-month lag;
    lag=0 uses the same month's structure).  January with lag 1 wraps to
    December of the previous year (same table — monthly climatology of the
    publication year).
    """
    m = origin_hours.month.values - 1 - lag_months
    m = np.mod(m, 12)
    return table[m]


def build_fd_windows(data, seq_len=SEQ_LEN, horizon=HORIZON, stride=TRAIN_STRIDE,
                     max_windows=None, rng=None, starts=None,
                     monthly_table=None, lag_months=1):
    """Build fuel-decomposed training/eval windows for one region.

    ``starts`` optionally supplies explicit local start positions (e.g. a
    shared absolute-origin grid so windows from different regions cover the
    same calendar period and can be mixed pairwise — see
    ``training.synthetic``); otherwise positions come from ``stride``.

    ``monthly_table`` (12, D) optionally supplies per-month FD configs; each
    window then carries its lagged monthly config in the extra ``config``
    output (n, D) — the deployment interface for regions publishing
    monthly fuel-mix statistics.

    Returns a dict of float32 arrays (empty trailing dims when the series
    is too short):

        x_rs        (n, L)      past renewable-share history
        x_fuel      (n, L, F)   past per-fuel shares (zeros for AU)
        y_fuel      (n, H, F)   future per-fuel shares (zeros for AU)
        y_rs        (n, H)      future renewable-share ground truth
        y_cif       (n, H)      future CIF ground truth
        x_weather   (n, L, W)   past weather-exog: temp, shortwave, wind
                                speed, wind CF, clear-sky index, gusts, MSL
                                pressure, demand actual (z), wind regime
                                24 h mean, regime 6 h tendency
        fut_weather (n, H, W)   future weather-exog (24 h reanalysis proxy)
        fut_exog    (n, H, K)   future exog: sin_elev, clearsky, wind_cf,
                                clearsky_index + 6 calendar + HDH/CDH +
                                coal/gas price z, day-ahead demand
                                forecast z (K=15)
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
    hdh, cdh = ex["hdh"], ex["cdh"]
    coal_z, gas_z = ex["coal_z"], ex["gas_z"]
    dem_fut = ex["demand_fut"]
    parts = [astro, wind_cf[:, None], csi[:, None], cal,
             hdh[:, None], cdh[:, None], coal_z[:, None], gas_z[:, None],
             dem_fut[:, None]]
    regime24 = ex.get("wind_regime24")
    tend6 = ex.get("wind_tend6")
    if regime24 is not None and tend6 is not None:
        parts += [regime24[:, None], tend6[:, None]]
    fut_exog_full = np.concatenate(parts, axis=1).astype(np.float32)
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
            "x_weather": np.empty((n, seq_len, weather.shape[1]), np.float32),
            "fut_weather": np.empty((n, horizon, weather.shape[1]), np.float32),
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
    # Demand channel: per-window DEMEANED deviation (shape-only) — the
    # annual z-score leaked seasonal level into the shape pathway and
    # perturbed the calibrated levels (FD-15 ablation: Spearman +0.05 but
    # MAE +1-4).  Subtract the trailing-week mean of the observed demand.
    if fut_exog_full.shape[1] >= 15 and (weather[:, 7] != 0).any():
        for i, s in enumerate(starts):
            ref = weather[max(0, s + seq_len - 168):s + seq_len, 7].mean()
            out["fut_exog"][i, :, 14] -= ref
    out["origin_hours"] = pd.DatetimeIndex([hours[s + seq_len] for s in starts])
    if monthly_table is not None:
        out["config"] = monthly_config_at(monthly_table, out["origin_hours"],
                                          lag_months=lag_months)
    return out
