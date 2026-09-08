"""China deployment layer for TransCIF-FD (Phase A, telemetry-free).

Everything in this module consumes only deployment-legal public inputs:

* monthly generation by fuel (province statistical bulletins, ~1-month lag)
* monthly inter-provincial exchange with sender identity (power-exchange
  centre monthly results, e.g. Beijing/Guangzhou power exchange centres)
* reanalysis / day-ahead NWP weather at a coordinate
* official holiday & heating-season calendars (State Council notices)

No hourly telemetry of any kind enters this module.  It provides:

1. ``CN_HOLIDAYS`` / ``cn_calendar_features`` — the local-time calendar with
   an INPUT-SPACE remap: Spring-Festival / Golden-Week holidays map onto the
   weekend day-of-week channels and 调休 make-up workdays onto the weekday
   channels, so a model trained on Western grids (no CNY) at least sees
   "holiday-shaped" days through channels it already knows.  The remap is an
   approximation and is always reported through advisory flags — never
   silent.  ``heating_season_mask`` flags the northern CHP season.
2. ``import_ef_from_senders`` — sender-weighted MONTHLY imports EF (the
   deployment form of the carbon-flow-aware imports-EF idea): the imports
   axis carries the flow-weighted average CIF of the sender provinces'
   published monthly mixes (dry-season Sichuan hydro -> higher imports EF,
   flood season -> lower), not a constant 250.
3. ``cn_monthly_table`` — monthly generation -> the (12, 16) FD-config
   table + annual config + per-fuel EF vector under the CN renewable
   definition (hydro/solar/wind/biomass renewable; nuclear and imports are
   non-renewable, imports at the sender-weighted EF).
4. ``proxy_anchor_trust`` — the anchor-trust gate for provinces with NO CIF
   observations: audit the monthly table against an independent public
   proxy (e.g. a monthly thermal-generation index / daily coal-burn
   aggregate).  Without any proxy it returns 0.0 — "cannot verify", the
   conservative annual-config fallback — instead of guessing.
5. ``build_cn_windows`` — assembles a full deployment window (18-channel
   fut_exog, 10-channel weather stack) from raw weather + the CN calendar
   at the province coordinate.

Caveats kept explicit (see docs/CN_DEPLOYMENT.md): the 2023 Western source
domain never saw CNY load collapse, northern CHP winter shape or curtailment
constraints, so holiday/heating effects are flagged as advisories with lower
shape confidence rather than silently trusted; Shanxi-class sender provinces
are approximated by their GENERATION structure (export separation needs the
Phase-B spot-market interface).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from transcif.data.calendar import calendar_features
from transcif.data.fuel import CANONICAL_FUELS, FUEL_INDEX, monthly_config_at
from transcif.physics.astro import astro_features, wind_capacity_factor

# ---------------------------------------------------------------------------
# 1. Official CN calendars (国务院办公厅 annual holiday notices; heating
#    windows from city heating regulations — Taiyuan 11/1-3/31 verified
#    2026-09, Beijing 11/15-3/15).  Dates are (month, day) tuples in LOCAL
#    time.  ``holiday`` = 放假日 (incl. 调休拼出的连休), ``workday`` = 调休
#    上班日 (make-up workdays on weekends).
# ---------------------------------------------------------------------------

CN_HOLIDAYS = {
    2022: {"holiday": [(12, 31)], "workday": []},
    2023: {
        "holiday": [(1, 1), (1, 2), (1, 21), (1, 22), (1, 23), (1, 24),
                    (1, 25), (1, 26), (1, 27), (4, 5), (4, 29), (4, 30),
                    (5, 1), (5, 2), (5, 3), (6, 22), (6, 23), (6, 24),
                    (9, 29), (9, 30), (10, 1), (10, 2), (10, 3), (10, 4),
                    (10, 5), (10, 6)],
        "workday": [(1, 28), (1, 29), (4, 23), (5, 6), (6, 25), (10, 7),
                    (10, 8)],
    },
    2024: {
        "holiday": [(1, 1), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14),
                    (2, 15), (2, 16), (2, 17), (4, 4), (4, 5), (4, 6),
                    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (6, 10),
                    (9, 15), (9, 16), (9, 17), (10, 1), (10, 2), (10, 3),
                    (10, 4), (10, 5), (10, 6), (10, 7)],
        "workday": [(2, 4), (2, 18), (4, 28), (5, 11), (9, 14), (9, 29),
                    (10, 12)],
    },
    2025: {
        "holiday": [(1, 1), (1, 28), (1, 29), (1, 30), (1, 31), (2, 1),
                    (2, 2), (2, 3), (2, 4), (4, 4), (4, 5), (4, 6),
                    (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 31),
                    (6, 1), (6, 2), (10, 1), (10, 2), (10, 3), (10, 4),
                    (10, 5), (10, 6), (10, 7), (10, 8)],
        "workday": [(1, 26), (2, 8), (4, 27), (5, 10), (9, 28), (10, 11)],
    },
    2026: {  # 国办发明电〔2025〕7号
        "holiday": [(1, 1), (1, 2), (1, 3), (2, 15), (2, 16), (2, 17),
                    (2, 18), (2, 19), (2, 20), (2, 21), (2, 22), (2, 23),
                    (4, 4), (4, 5), (4, 6), (5, 1), (5, 2), (5, 3), (5, 4),
                    (5, 5), (6, 19), (6, 20), (6, 21), (9, 25), (9, 26),
                    (9, 27), (10, 1), (10, 2), (10, 3), (10, 4), (10, 5),
                    (10, 6), (10, 7)],
        "workday": [(1, 4), (2, 14), (2, 28), (5, 9), (9, 20), (10, 10)],
    },
}

# Northern heating windows ((start_m, start_d), (end_m, end_d)), local time.
# Taiyuan: 每年11月1日至次年3月31日 (太原市供热管理条例, verified 2026-09).
# Beijing/Tianjin: 11/15 - 3/15.  Missing keys fall back to the northern
# default; southern provinces pass province=None for "no district heating".
HEATING_WINDOWS = {
    "default": ((11, 1), (3, 31)),
    "shanxi": ((11, 1), (3, 31)),
    "beijing": ((11, 15), (3, 15)),
    "tianjin": ((11, 15), (3, 15)),
    "hebei": ((11, 15), (3, 15)),
    "shandong": ((11, 15), (3, 15)),
}

# CN renewable definition (生态环境部 accounting): hydro/solar/wind/biomass
# are renewable; nuclear and inter-provincial imports are NON-renewable
# (imports carry their sender-weighted average CIF on the imports axis,
# matching the UK interconnector convention calibrated in FD-23).
CN_RENEWABLES = {"hydro", "solar", "wind", "biomass"}

# Default thermal EFs (gCO2/kWh, IPCC-style direct values; province fuel
# quality / unit age can override via ``thermal_efs``).
CN_THERMAL_DEFAULT = {"coal": 950.0, "gas": 400.0, "petroleum": 650.0}

_SUN_DOW = np.array([np.sin(2 * np.pi * 6 / 7), np.cos(2 * np.pi * 6 / 7)],
                    dtype=np.float32)
_MON_DOW = np.array([0.0, 1.0], dtype=np.float32)


def _holiday_lookup(year: int):
    entry = CN_HOLIDAYS.get(year)
    if entry is None:
        return set(), set()
    return set(entry["holiday"]), set(entry["workday"])


def cn_calendar_features(hours, tz_offset=8.0):
    """Local-time calendar channels with the CN holiday input-space remap.

    Returns ``(calendar (T, 6) float32, flags (T, 2) float32)``:

    * calendar: the standard [sin/cos hour, sin/cos dow, sin/cos doy]
      channels, except that hours on official HOLIDAY dates that fall on
      Mon-Fri carry the Sunday day-of-week channels (holiday week days
      behave like weekends for industrial load), and 调休 make-up WORKDAY
      dates that fall on Sat/Sun carry the Monday channels.
    * flags: ``[is_holiday, is_special_workday]`` per hour — the advisory
      channel; the caller must surface it, never swallow it.
    """
    cal = calendar_features(hours, tz_offset).copy()
    if not isinstance(hours, pd.DatetimeIndex):
        hours = pd.DatetimeIndex(hours)
    local = hours + pd.to_timedelta(tz_offset, unit="h")
    years = local.year.values
    months = local.month.values
    days = local.day.values
    dows = local.dayofweek.values
    n = len(hours)
    is_hol = np.zeros(n, dtype=bool)
    is_spec = np.zeros(n, dtype=bool)
    to_weekend = np.zeros(n, dtype=bool)
    to_weekday = np.zeros(n, dtype=bool)
    cache = {}
    for i in range(n):
        y = int(years[i])
        if y not in cache:
            cache[y] = _holiday_lookup(y)
        hol, spec = cache[y]
        key = (int(months[i]), int(days[i]))
        if key in hol:
            is_hol[i] = True
            if dows[i] < 5:            # a weekday acting as a holiday
                to_weekend[i] = True
        elif key in spec:
            is_spec[i] = True
            if dows[i] >= 5:           # a weekend acting as a workday
                to_weekday[i] = True
    cal[to_weekend, 2:4] = _SUN_DOW
    cal[to_weekday, 2:4] = _MON_DOW
    flags = np.stack([is_hol, is_spec], axis=1).astype(np.float32)
    return cal.astype(np.float32), flags


def heating_season_mask(hours, tz_offset=8.0, province="default"):
    """Boolean (T,) mask: local date inside the province's heating window.

    ``province=None`` (southern provinces, no district heating) -> all
    False.  Unknown keys fall back to the northern default 11/1-3/31.
    """
    if province is None:
        return np.zeros(len(hours), dtype=bool)
    if not isinstance(hours, pd.DatetimeIndex):
        hours = pd.DatetimeIndex(hours)
    local = hours + pd.to_timedelta(tz_offset, unit="h")
    md = np.array(list(zip(local.month.values, local.day.values)))
    start, end = HEATING_WINDOWS.get(province, HEATING_WINDOWS["default"])
    start, end = np.array(start), np.array(end)
    if tuple(start) <= tuple(end):
        inside = (md >= start).all(1) & (md <= end).all(1)
    else:  # window crosses the year boundary (Nov -> Mar)
        after_start = (md >= start).all(1)
        before_end = (md <= end).all(1)
        inside = after_start | before_end
    return inside


# ---------------------------------------------------------------------------
# 2. Sender-weighted monthly imports EF (carbon-flow-aware, FD-18 部署化).
# ---------------------------------------------------------------------------

def monthly_cif_from_gen(monthly_gen_gwh, thermal_efs=None):
    """(12,) average CIF of a sender province from ITS monthly generation.

    ``monthly_gen_gwh`` : {fuel: 12 values in GWh} — the sender's own
    published monthly statistics.  Renewable fuels (CN definition) carry
    zero EF; the thermal trio carries ``thermal_efs``.  The returned CIF is
    the flow-weighted emission intensity of energy the sender injects.
    """
    thermal_efs = {**CN_THERMAL_DEFAULT, **(thermal_efs or {})}
    gen = np.zeros((12, len(CANONICAL_FUELS)))
    for f, vals in monthly_gen_gwh.items():
        gen[:, FUEL_INDEX[f]] = vals
    total = gen.sum(axis=1, keepdims=True)
    shares = np.divide(gen, np.clip(total, 1e-9, None))
    renew_mask = np.array([f in CN_RENEWABLES for f in CANONICAL_FUELS])
    ef_vec = np.zeros(len(CANONICAL_FUELS))
    for f, ef in thermal_efs.items():
        ef_vec[FUEL_INDEX[f]] = ef
    ef_vec[FUEL_INDEX["other"]] = 500.0
    nonren = ~renew_mask
    nonren_mass = shares[:, nonren].sum(axis=1)
    thermal_mass = shares[:, [FUEL_INDEX[f] for f in thermal_efs]].sum(axis=1)
    cif = np.zeros(12)
    for m in range(12):
        if nonren_mass[m] < 1e-6:
            cif[m] = 0.0
            continue
        # nonren average EF with the remaining (imports/other) at 500
        rest_mass = max(nonren_mass[m] - thermal_mass[m], 0.0)
        thermal_ef_avg = (np.dot(shares[m, [FUEL_INDEX[f] for f in thermal_efs]],
                                 [thermal_efs[f] for f in thermal_efs])
                          / max(thermal_mass[m], 1e-9)) if thermal_mass[m] > 1e-9 else 0.0
        ef_nr_m = (thermal_ef_avg * thermal_mass[m] + 500.0 * rest_mass) \
            / nonren_mass[m]
        cif[m] = nonren_mass[m] * ef_nr_m
    return cif


def import_ef_from_senders(sender_cif_monthly, flow_shares):
    """Flow-weighted monthly imports EF.

    ``sender_cif_monthly`` : {sender_name: (12,) monthly average CIF} —
        from ``monthly_cif_from_gen`` on each sender's published stats.
    ``flow_shares``       : {sender_name: constant share OR (12,) monthly
        share of the received energy} — from power-exchange monthly results.

    Returns (12,) gCO2/kWh.  Dry-season hydro senders raise the imports EF
    (their own thermal share rises), flood season lowers it — the seasonal
    level signal a constant 250 cannot carry.
    """
    names = list(sender_cif_monthly)
    if not names:
        return np.full(12, 250.0)
    flows = np.zeros((12, len(names)))
    for j, name in enumerate(names):
        w = np.asarray(flow_shares[name], dtype=float)
        if w.ndim == 0:
            w = np.full(12, float(w))
        flows[:, j] = w
    flows = flows / np.clip(flows.sum(axis=1, keepdims=True), 1e-9, None)
    cifs = np.stack([np.asarray(sender_cif_monthly[n], float) for n in names],
                    axis=1)
    return (flows * cifs).sum(axis=1)


# ---------------------------------------------------------------------------
# 3. CN province config builders (monthly statistics -> FD interface).
# ---------------------------------------------------------------------------

def cn_monthly_table(monthly_gen_gwh, thermal_efs=None, imports_gwh=None,
                     sender_cif_monthly=None, flow_shares=None,
                     ann_windcf=0.25, ann_csi=0.55, lat=35.0):
    """Monthly generation by fuel -> the (12, 16) FD-config monthly table.

    Args:
      monthly_gen_gwh : {fuel: 12 GWh values} — LOCAL generation only
                        (imports passed separately).
      thermal_efs     : optional per-province thermal EF overrides.
      imports_gwh     : (12,) monthly received energy (GWh) or None.
      sender_cif_monthly / flow_shares : sender structure for the imports
                        EF (see ``import_ef_from_senders``); without them a
                        constant 250 fallback is used (documented).
      ann_windcf / ann_csi : reanalysis climatology means (deployment
                        supplies them from the province cell).
      lat             : province centroid latitude (|lat|/60 config slot).

    Returns a dict with keys:
      table (12,16), fd_config (16,), ef_vec (10,), mean_rs (float),
      ef_nr (float, annual non-renewable average EF), ef_nr_monthly (12,),
      imports_ef_monthly (12,) or None, shares (12,10)
    """
    thermal_efs = {**CN_THERMAL_DEFAULT, **(thermal_efs or {})}
    D = 16
    gen = np.zeros((12, len(CANONICAL_FUELS)))
    for f, vals in monthly_gen_gwh.items():
        gen[:, FUEL_INDEX[f]] = vals
    imp = (np.zeros(12) if imports_gwh is None
           else np.asarray(imports_gwh, dtype=float))
    total = gen.sum(axis=1) + imp
    safe = np.clip(total, 1e-9, None)
    shares = gen / safe[:, None]
    shares[:, FUEL_INDEX["imports"]] = imp / safe

    if imp.max() > 1e-9:
        if sender_cif_monthly and flow_shares:
            imports_ef = import_ef_from_senders(sender_cif_monthly,
                                                flow_shares)
        else:
            imports_ef = np.full(12, 250.0)  # documented constant fallback
    else:
        imports_ef = None

    ef_vec = np.zeros(len(CANONICAL_FUELS))
    for f, ef in thermal_efs.items():
        ef_vec[FUEL_INDEX[f]] = ef
    ef_vec[FUEL_INDEX["imports"]] = (float(imports_ef.mean())
                                     if imports_ef is not None else 250.0)
    ef_vec[FUEL_INDEX["other"]] = 500.0

    renew_mask = np.array([f in CN_RENEWABLES for f in CANONICAL_FUELS])
    nonren_mask = ~renew_mask
    mean_rs_m = shares[:, renew_mask].sum(axis=1)
    nonren_mass = shares[:, nonren_mask].sum(axis=1)
    ef_nr_monthly = (shares[:, nonren_mask] @ ef_vec[nonren_mask]) \
        / np.clip(nonren_mass, 1e-9, None)

    # Energy-weighted annual shares (not the mean of monthly shares).
    annual_total = np.zeros(len(CANONICAL_FUELS))
    annual_total[:FUEL_INDEX["imports"]] = gen.sum(axis=0)[:FUEL_INDEX["imports"]]
    annual_total[FUEL_INDEX["imports"]] = imp.sum()
    annual_total[FUEL_INDEX["other"]] = gen.sum(axis=0)[FUEL_INDEX["other"]]
    annual_shares = annual_total / max(total.sum(), 1e-9)
    mean_rs = float(annual_shares[renew_mask].sum())
    nm = float(annual_shares[nonren_mask].sum())
    ef_nr = float(annual_shares[nonren_mask] @ ef_vec[nonren_mask]) \
        / max(nm, 1e-9)

    def _row(shares_m, mean_rs_i, ef_nr_i):
        return np.array([mean_rs_i, ef_nr_i / 1000.0, *shares_m,
                         ann_windcf, ann_csi, 1.0, abs(lat) / 60.0],
                        dtype=np.float32)

    table = np.stack([_row(shares[m], float(mean_rs_m[m]),
                           float(ef_nr_monthly[m])) for m in range(12)])
    fd_config = _row(annual_shares.astype(np.float32),
                     mean_rs, ef_nr).astype(np.float32)
    return {
        "table": table,
        "fd_config": fd_config,
        "ef_vec": ef_vec,
        "mean_rs": mean_rs,
        "ef_nr": ef_nr,
        "ef_nr_monthly": ef_nr_monthly,
        "imports_ef_monthly": imports_ef,
        "shares": shares.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# 4. Proxy anchor-trust for provinces with NO CIF observations.
# ---------------------------------------------------------------------------

def proxy_anchor_trust(table, thermal_proxy=None, proxy_cif_annual=None):
    """Anchor-trust score in [0, 1] without any CIF observation.

    The benchmark gate (``anchor_trust``) correlates the monthly table's
    implied level against the target's own observed monthly CIF —
    impossible for a telemetry-free province.  This proxy variant audits
    the table against independent PUBLIC statistics instead:

    ``thermal_proxy``   : (12,) monthly thermal-generation index — any
        public aggregate that tracks the true thermal level (monthly 火电
        电量, 电煤日耗 monthly mean, 火电利用小时).  Correlation of the
        table's implied level ``(1-rs_m) * ef_nr_m`` against the proxy is
        the shape gate.
    ``proxy_cif_annual``: optional annual average CIF anchor (e.g. an
        official grid EF factor) for the exponential bias penalty.

    No proxy at all -> 0.0: "cannot verify", and the deployment blends the
    monthly table all the way back to the annual config (the documented
    conservative fallback) rather than trusting an unauditable table.
    """
    if thermal_proxy is None:
        return 0.0
    proxy = np.asarray(thermal_proxy, dtype=float)
    implied = (1.0 - table[:, 0].astype(float)) * table[:, 1].astype(float) * 1000.0
    valid = np.isfinite(proxy) & np.isfinite(implied)
    if valid.sum() < 6:
        return 0.0
    x, y = implied[valid], proxy[valid]
    if np.std(x) < 1e-6 or np.std(y) < 1e-6:
        return 0.0
    r = float(np.corrcoef(x, y)[0, 1])
    penalty = 1.0
    if proxy_cif_annual is not None:
        bias = float(x.mean() - float(proxy_cif_annual))
        penalty = float(np.exp(-abs(bias) / max(np.std(x), 1e-6)))
    return float(max(r, 0.0) * penalty)


def blend_monthly_table(table, fd_config, trust):
    """FD-39 blend: annual config + trust * (monthly - annual), per month."""
    t = float(np.clip(trust, 0.0, 1.0))
    return (fd_config[None, :] + t * (table - fd_config[None, :])) \
        .astype(np.float32)


# ---------------------------------------------------------------------------
# 5. Deployment window builder (raw weather + CN calendar -> model inputs).
# ---------------------------------------------------------------------------

def build_cn_windows(hours_utc, raw_weather, gust_pressure=None,
                     lat=38.0, lon=115.0, tz_offset=8.0, table=None,
                     fd_config=None, lag_months=2, interchange_monthly=None,
                     province="default", seq_len=336, horizon=24):
    """Assemble one deployment window at the province coordinate.

    Args:
      hours_utc       : pd.DatetimeIndex (UTC) of length >= seq_len+horizon.
      raw_weather     : (T, 3) [temperature_c, shortwave W/m^2, wind100 m/s]
                        — reanalysis history + day-ahead NWP horizon proxy.
      gust_pressure   : optional (T, 2) [gust m/s, MSL pressure anomaly];
                        None -> zeros (no synoptic channels at deployment).
      table           : (12, 16) monthly FD-config table; per-window config
                        rows follow the publication calendar (lag_months=2:
                        at day-ahead time D-1 the latest published monthly
                        row covers month m-2 — a row covering m-1 is only
                        published mid-month m, after D-1).
      interchange_monthly : (12,) import share broadcast to hours (same
                        publication-lag alignment, m-2 row); defaults
                        to the table's imports column.
      province        : heating-window key for the advisory flag.

    Returns ``(windows, flags, heating)`` where ``windows`` matches the
    ``predict_fuel_windows`` contract (18-channel fut_exog, 10-channel
    weather) and flags/heating are the advisory channels for the horizon.
    """
    if not isinstance(hours_utc, pd.DatetimeIndex):
        hours_utc = pd.DatetimeIndex(hours_utc)
    T = seq_len + horizon
    if len(hours_utc) < T:
        raise ValueError(f"need >= {T} hours, got {len(hours_utc)}")
    hours_utc = hours_utc[:T]
    wx = np.asarray(raw_weather, dtype=np.float32)[:T]
    if wx.shape[0] < T:
        raise ValueError("raw_weather shorter than the window")

    astro = astro_features(hours_utc, lat, lon).astype(np.float32)
    clearsky = np.maximum(astro[:, 1], 1.0)
    wind_cf = wind_capacity_factor(wx[:, 2]).astype(np.float32)
    csi = np.clip(wx[:, 1] / clearsky, 0.0, 1.3).astype(np.float32)
    hdh = np.clip(15.5 - wx[:, 0], 0.0, None).astype(np.float32)
    cdh = np.clip(wx[:, 0] - 22.0, 0.0, None).astype(np.float32)
    cal, flags = cn_calendar_features(hours_utc, tz_offset)

    regime24 = pd.Series(wind_cf).rolling(24, min_periods=1).mean() \
        .values.astype(np.float32)
    tend6 = np.zeros_like(regime24)
    tend6[6:] = regime24[6:] - regime24[:-6]

    gp = (np.zeros((T, 2), np.float32) if gust_pressure is None
          else np.asarray(gust_pressure, np.float32)[:T])
    demand_z = np.zeros(T, np.float32)  # no demand telemetry at I_cfg

    weather = np.concatenate(
        [wx, wind_cf[:, None], csi[:, None], gp, demand_z[:, None],
         regime24[:, None], tend6[:, None]], axis=1)  # (T, 10)

    if interchange_monthly is None and table is not None:
        interchange_monthly = table[:, 2 + FUEL_INDEX["imports"]]
    inter = np.zeros(T, np.float32)
    if interchange_monthly is not None:
        local = hours_utc + pd.to_timedelta(tz_offset, unit="h")
        # 发布滞后对齐: 日前时点 (D-1) 当月及上月统计尚未发布 (月度
        # 统计 ~1 月滞后) → 受电份额取 m-2 月行 (1/2 月回卷上年末),
        # 与 config 通道 lag_months=2 同一发布日历语义。
        m_idx = (local.month.values - 3) % 12
        inter = np.asarray(interchange_monthly, np.float32)[m_idx]

    fut_exog = np.concatenate(
        [astro, wind_cf[:, None], csi[:, None], cal,
         hdh[:, None], cdh[:, None],
         np.zeros((T, 2), np.float32),          # coal_z / gas_z placeholders
         demand_z[:, None], regime24[:, None], tend6[:, None],
         inter[:, None]], axis=1)               # (T, 18)

    s, e = seq_len, seq_len + horizon
    windows = {
        "x_rs": np.zeros((1, seq_len), np.float32),
        "x_fuel": np.zeros((1, seq_len, len(CANONICAL_FUELS)), np.float32),
        "x_weather": weather[None, :s],
        "fut_weather": weather[None, s:e],
        "fut_exog": fut_exog[None, s:e],
    }
    if table is not None:
        origin = pd.DatetimeIndex([hours_utc[s]])
        # lag 2 = 发布日历口径: D-1 (m-1 月中旬) 时, m-1 月统计行尚未
        # 发布 (次月中旬发布) → 最近已发布行 = m-2 月。docstring 的
        # "~1-month lag" 指统计覆盖月与发布月的关系, 此处取「日前可
        # 得」的保守对齐 (每月表行覆盖当月, 发布于次月中旬)。
        windows["config"] = monthly_config_at(table, origin,
                                              lag_months=lag_months)
    heating = heating_season_mask(hours_utc, tz_offset, province)
    return windows, flags[s:e], heating[s:e]
