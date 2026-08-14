"""Astronomy and weather-physics features for TransCIF-FD.

Deterministic, region-transferable exogenous features computed from
(timestamp, lat, lon) and reanalysis weather:

    sin_solar_elevation  — solar geometry (Cooper declination + hour angle)
    clearsky_ghi         — Haurwitz clear-sky envelope (W/m^2)
    wind_capacity_factor — IEC-style turbine power curve transform

These features are deployable anywhere: solar geometry needs only the
calendar and a coordinate, and reanalysis weather (Open-Meteo) is globally
available — the exact interface a telemetry-free target region (e.g. a
Chinese province with monthly fuel-mix data only) can supply.
"""

import numpy as np
import pandas as pd


def solar_declination(dayofyear):
    """Cooper (1969) declination in radians for each day of year."""
    doy = np.asarray(dayofyear, dtype=np.float64)
    return np.deg2rad(23.45) * np.sin(2 * np.pi * (284.0 + doy) / 365.0)


def sin_solar_elevation(hours, lat, lon):
    """Sine of solar elevation for UTC timestamps at (lat, lon).

    Args:
        hours : pd.DatetimeIndex / array of pd.Timestamp in UTC
        lat   : latitude in degrees
        lon   : longitude in degrees

    Returns float64 array in [-1, 1].  Nighttime values <= 0.
    """
    if not isinstance(hours, pd.DatetimeIndex):
        hours = pd.DatetimeIndex(hours)
    doy = hours.dayofyear.values.astype(np.float64)
    dec = solar_declination(doy)
    # Decimal UTC hour
    h_utc = (hours.hour.values + hours.minute.values / 60.0)
    # Solar hour angle: local solar time = UTC + lon/15 hours
    hour_angle = np.deg2rad(15.0 * (h_utc + lon / 15.0 - 12.0))
    lat_r = np.deg2rad(lat)
    return (np.sin(lat_r) * np.sin(dec)
            + np.cos(lat_r) * np.cos(dec) * np.cos(hour_angle))


def clearsky_ghi(sin_elev):
    """Haurwitz clear-sky global horizontal irradiance (W/m^2).

    GHI_cs = 1098 * sin(elev) * exp(-0.057 / sin(elev)) for sin(elev) > 0,
    else 0.  A smooth, parameter-free envelope used to normalise observed
    shortwave radiation into a clear-sky index.
    """
    s = np.asarray(sin_elev, dtype=np.float64)
    out = np.zeros_like(s)
    pos = s > 1e-3
    out[pos] = 1098.0 * s[pos] * np.exp(-0.059 / s[pos])
    return out


def wind_capacity_factor(wind_speed_ms, v_in=3.0, v_rated=12.0, v_out=25.0):
    """IEC-style turbine capacity factor from 100 m wind speed.

    Piecewise: 0 below cut-in, cubic ramp ((v^3 - v_in^3) / (v_rated^3 -
    v_in^3)) between cut-in and rated, 1.0 between rated and cut-out, 0
    above cut-out.  Returns values in [0, 1].
    """
    v = np.asarray(wind_speed_ms, dtype=np.float64)
    cf = np.zeros_like(v)
    ramp = (v >= v_in) & (v < v_rated)
    cf[ramp] = (v[ramp] ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3)
    cf[(v >= v_rated) & (v <= v_out)] = 1.0
    return np.clip(cf, 0.0, 1.0)


def astro_features(hours, lat, lon):
    """Stacked (T, 2) astronomy matrix: [sin_elev, clearsky_ghi]."""
    s = sin_solar_elevation(hours, lat, lon)
    return np.stack([s, clearsky_ghi(s)], axis=1)
