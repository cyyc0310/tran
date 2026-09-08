#!/usr/bin/env python
"""Demo: deployment path for a telemetry-free Chinese province (I_cfg).

Input — exactly what a province publishes: monthly generation by fuel for
the past year (火力/水力/风力/光伏/核电发电量, GWh), inter-provincial
received energy with sender identity (power-exchange monthly results), a
coordinate (lat/lon) and raw weather (reanalysis history + day-ahead NWP
horizon).  Weather climatology is borrowed from a real region's stack so
the demo runs offline.

Two scenarios (see docs/CN_DEPLOYMENT.md):

  shanxi  — a northern coal-heavy sender province: expected the BEST case
            (coal-steady benchmark analogs PJM/NYIS/FPL reach 17-20 MAE).
  shanghai — an eastern receiving province: 45.1% imported energy (2023
            official: 834/1849 亿kWh) from southwest UHVDC hydro, Anhui
            coal and Qinshan nuclear — the STRUCTURALLY HARDEST case
            (ISNE/UK_14 analogs; the imports EF path and its caveats are
            printed).

Pipeline:
    12×F monthly generation + monthly sender exchange
        -> CN monthly FD-config table (CN renewable definition)
        -> per-month sender-weighted imports EF (FD-18 部署化)
        -> proxy anchor-trust gate (no CIF observation needed)
        -> FuelDecompNet trained on the 29 source regions
           (province supplies ONLY its config)
        -> I_cfg CIF trajectory + per-fuel decomposition + advisories

Usage:
    .venv-nemed/bin/python scripts/experiments/demo_cn_province.py            # shanxi
    .venv-nemed/bin/python scripts/experiments/demo_cn_province.py shanghai   # receiving case
    .venv-nemed/bin/python scripts/experiments/demo_cn_province.py both       # both
"""

import sys

import numpy as np
import pandas as pd
import torch

from transcif.config import SEQ_LEN, HORIZON
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import FUEL_INDEX
from transcif.data.cn_deploy import (
    cn_monthly_table, proxy_anchor_trust, blend_monthly_table,
    build_cn_windows, CN_HOLIDAYS,
)
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, train_fuel_zero_shot, predict_fuel_windows,
)


def prepare_sources(cfgs):
    """Prepare the 29 benchmark sources with the AU monthly_table=None fix.

    ``prepare_fd_region`` computes ``anchor_trust`` on
    ``data["monthly_table"]`` which is None for AU regions (no fuel
    telemetry) — a latent crash for any caller passing all regions.  This
    wrapper keeps the benchmark module untouched and patches the call for
    the deployment path.
    """
    from transcif.data.fuel import attach_fuel_and_exog, build_fd_config, \
        build_monthly_config_table
    from transcif.data.loaders import load_region_data
    from transcif.models.zeroshot.fuel import anchor_trust
    import os

    out = {}
    for name in cfgs:
        data = load_region_data(name, cfgs)
        attach_fuel_and_exog(data, name, cfgs)
        data["fd_config"] = build_fd_config(data, name)
        table = build_monthly_config_table(data, name)
        data["monthly_table"] = table
        data["anchor_trust"] = (anchor_trust(data, table)
                                if table is not None else 0.0)
        mode = os.environ.get("ANCHOR_TRUST", "0")
        if mode in ("1", "2") and table is not None:
            t = data["anchor_trust"]
            blended = (data["fd_config"][None, :]
                       + t * (table - data["fd_config"][None, :]))
            if mode == "1":
                data["monthly_table"] = blended
            else:
                data["monthly_table_target"] = blended
        out[name] = data
    return out

# ---------------------------------------------------------------------------
# 1. Scenario inputs (hypothetical provinces, realistic public statistics).
# ---------------------------------------------------------------------------

# Shanxi-class northern coal sender: coal-heavy with wind/solar buildout.
SHANXI_GEN = {
    "coal":    [34, 30, 32, 28, 26, 24, 26, 25, 27, 30, 33, 35],
    "gas":     [4.0, 3.6, 3.8, 3.4, 3.2, 3.6, 4.2, 4.2, 3.4, 3.2, 3.6, 4.2],
    "petroleum": [0.1] * 12,
    "nuclear": [0.0] * 12,
    "hydro":   [0.6, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.5, 1.2, 0.9, 0.7, 0.6],
    "solar":   [1.6, 2.0, 2.6, 3.2, 3.7, 3.9, 3.7, 3.4, 2.9, 2.3, 1.7, 1.4],
    "wind":    [2.8, 3.0, 3.4, 3.6, 3.2, 2.6, 2.2, 2.2, 2.8, 3.4, 3.6, 3.4],
    "biomass": [0.3] * 12,
}
SHANXI_META = {"lat": 37.87, "lon": 112.55, "tz": 8.0, "province": "shanxi",
               "label": "Shanxi-class coal sender (best case)"}

# Shanghai-class eastern receiver (2023 official, 沪经信运〔2024〕306号):
# local generation thermal-heavy (coal ~68% + gas ~22% of local gen),
# received energy 45.1% of consumption (834 vs 1849 亿kWh) from
# southwest UHVDC hydro (~479亿, 葛南/宜华/林枫/复奉) + Anhui coal
# (~240亿, 皖电东送) + Qinshan nuclear & green trades (~118亿).
SHANGHAI_GEN = {
    "coal":    [65, 59, 61, 56, 53, 55, 57, 56, 54, 56, 60, 64],
    "gas":     [19, 17, 18, 17, 16, 20, 22, 22, 19, 17, 18, 20],
    "petroleum": [0.4] * 12,
    "nuclear": [0.0] * 12,
    "hydro":   [0.0] * 12,
    "solar":   [0.9, 1.2, 1.9, 2.6, 3.1, 3.4, 3.3, 3.0, 2.3, 1.7, 1.1, 0.9],
    "wind":    [1.9, 2.0, 2.2, 2.2, 2.0, 1.8, 1.6, 1.6, 1.9, 2.1, 2.1, 1.9],
    "biomass": [3.4] * 12,
}
# Received energy (GWh/month): annual 834亿 kWh anchored to the official
# KPI report; summer peak (max reception 2023-07, 1812万kW scale) and
# winter cold-surge secondary peak; spring/autumn troughs.
SHANGHAI_IMPORTS_GWH = [74, 65, 60, 58, 62, 70, 80, 78, 72, 68, 72, 75]
SHANGHAI_SENDER_CIF = {
    "XN_hydro":   [25, 25, 25, 20, 15, 10, 8, 8, 12, 18, 22, 25],
    "AH_coal":    [880, 875, 870, 865, 860, 855, 850, 850, 860, 870, 875, 880],
    "QN_nuclear": [15] * 12,
}
SHANGHAI_FLOW_SHARES = {  # rows sum to 1: hydro 479 / AH coal ~240 / nuc+green ~118
    "XN_hydro":   [0.50, 0.48, 0.49, 0.53, 0.58, 0.63, 0.67, 0.67, 0.64, 0.59, 0.53, 0.50],
    "AH_coal":    [0.33, 0.34, 0.34, 0.31, 0.28, 0.25, 0.22, 0.22, 0.24, 0.28, 0.32, 0.34],
    "QN_nuclear": [0.17, 0.18, 0.17, 0.16, 0.14, 0.12, 0.11, 0.11, 0.12, 0.13, 0.15, 0.16],
}
SHANGHAI_META = {"lat": 31.23, "lon": 121.47, "tz": 8.0, "province": None,
                 "label": "Shanghai-class eastern receiver (hardest case)"}


def scenario(name):
    """Assemble one scenario dict -> (label, cfgd, months_pub_lag)."""
    if name == "shanxi":
        cfgd = cn_monthly_table(
            SHANXI_GEN, lat=SHANXI_META["lat"], ann_windcf=0.25,
            ann_csi=0.55)
        return SHANXI_META["label"], cfgd, 1
    if name == "shanghai":
        cfgd = cn_monthly_table(
            SHANGHAI_GEN, imports_gwh=SHANGHAI_IMPORTS_GWH,
            sender_cif_monthly=SHANGHAI_SENDER_CIF,
            flow_shares=SHANGHAI_FLOW_SHARES,
            lat=SHANGHAI_META["lat"], ann_windcf=0.22, ann_csi=0.50)
        return SHANGHAI_META["label"], cfgd, 1
    raise ValueError(f"unknown scenario {name}")


def main():
    scenario_name = sys.argv[1] if len(sys.argv) > 1 else "shanxi"
    if scenario_name == "both":
        for s in ("shanxi", "shanghai"):
            run_scenario(s)
        return
    run_scenario(scenario_name)


def synthetic_demo_weather(hours_utc, lat, lon, tz_offset=8.0):
    """Physically self-consistent demo weather at the province coordinate.

    The benchmark weather stack is absent on this machine (data_2023/
    weather/ not synced), so the offline demo synthesises a July diurnal
    cycle instead of borrowing an all-zero donor stack: shortwave peaks at
    the province's solar noon, temperature lags ~2 h, wind carries a mild
    day-night swing plus synoptic noise.  Demo-only; real deployment feeds
    reanalysis history + day-ahead NWP.
    """
    T = len(hours_utc)
    if not isinstance(hours_utc, pd.DatetimeIndex):
        hours_utc = pd.DatetimeIndex(hours_utc)
    utc_h = hours_utc.hour.values + hours_utc.minute.values / 60.0
    solar_noon = 12.0 - lon / 15.0              # UTC hour of local noon
    day_phase = np.cos(2 * np.pi * (utc_h - solar_noon) / 24.0)
    day_frac = np.clip(day_phase, 0.0, None)
    shortwave = 850.0 * np.sin(np.pi * np.clip(
        (utc_h - (solar_noon - 6.5)) / 13.0, 0.0, 1.0)) * (0.75 + 0.25 * day_frac)
    rng = np.random.default_rng(7)
    temp = 24.0 + 7.0 * np.cos(2 * np.pi * (utc_h - solar_noon - 2.0) / 24.0) \
        + rng.normal(0, 1.2, T)
    wind = np.clip(5.5 + 2.0 * np.cos(2 * np.pi * (utc_h - solar_noon + 3.0)
                                      / 24.0) + rng.normal(0, 1.5, T),
                   0.5, None)
    return np.stack([temp, shortwave, wind], axis=1).astype(np.float32)


def run_scenario(name):
    label, cfgd, _ = scenario(name)
    meta = SHANXI_META if name == "shanxi" else SHANGHAI_META
    print(f"\n{'=' * 70}\n[demo:{name}] {label}\n{'=' * 70}")
    torch.manual_seed(0)

    # -- 1. monthly table + proxy gate --------------------------------
    table, fd_config = cfgd["table"], cfgd["fd_config"]
    trust = proxy_anchor_trust(table, thermal_proxy=None)
    print(f"[gate] proxy anchor-trust: {trust:.2f} "
          f"(no public proxy supplied -> conservative annual config)")
    table_used = blend_monthly_table(table, fd_config, trust)
    table_used = cfgd["table"] if trust > 0 else table_used
    print(f"[gate] {('blended at trust ' + format(trust, '.2f')) if trust > 0 else 'annual fallback'}")

    # -- 2. train the shared model on the 29 benchmark sources --------
    print(f"[demo] training FuelDecompNet on 29 source regions "
          f"(province supplies config only) ...")
    cfgs = all_region_configs()
    fd = prepare_sources(cfgs)
    # Pseudo-region as the LORO target: province supplies ONLY its config,
    # the shared model trains on the 29 benchmark sources.
    fd["CN_DEMO"] = {
        "mean_rs": float(cfgd["mean_rs"]), "ef_r": 0.0,
        "ef_nr": float(cfgd["ef_nr"]),
        "fd_config": cfgd["fd_config"],
        "monthly_table": table_used,
        "has_fuel": True, "ef_vec": cfgd["ef_vec"],
    }
    model = train_fuel_zero_shot(fd, target_name="CN_DEMO", seed=0,
                                 epochs=600, use_monthly=True, device="mps")

    # -- 3. sample week prediction -------------------------------------
    hours = pd.date_range("2023-07-10", periods=SEQ_LEN + HORIZON, freq="h",
                          tz="UTC")
    raw_wx = synthetic_demo_weather(hours, meta["lat"], meta["lon"])
    windows, flags, heating = build_cn_windows(
        hours, raw_wx, lat=meta["lat"], lon=meta["lon"],
        tz_offset=meta["tz"], table=table_used,
        province=meta["province"])
    cif, sh, _ = predict_fuel_windows(
        model, windows, cfgd["fd_config"], cfgd["ef_vec"].astype(np.float32),
        cold=True, device="mps")

    # -- 4. report -----------------------------------------------------
    hours_f = hours[SEQ_LEN:] + pd.Timedelta(hours=meta["tz"])
    day = pd.DataFrame({
        "local_hour": pd.DatetimeIndex(hours_f.tz_localize(None))
            .strftime("%m-%d %H:%M"),
        "CIF": cif[0].round(1),
        "solar": sh[0, :, FUEL_INDEX["solar"]].round(3),
        "wind": sh[0, :, FUEL_INDEX["wind"]].round(3),
        "coal": sh[0, :, FUEL_INDEX["coal"]].round(3),
        "gas": sh[0, :, FUEL_INDEX["gas"]].round(3),
    })
    print("\n[province] monthly fuel shares (from published statistics):")
    df = pd.DataFrame(cfgd["shares"], index=["Jan", "Feb", "Mar", "Apr",
                                             "May", "Jun", "Jul", "Aug",
                                             "Sep", "Oct", "Nov", "Dec"],
                      columns=[f.capitalize() for f in
                               ["coal", "gas", "petroleum", "nuclear",
                                "hydro", "solar", "wind", "biomass",
                                "imports", "other"]])
    print((df * 100).round(1).to_string())
    print(f"\n[province] annual mean_rs {cfgd['mean_rs']:.3f} | "
          f"ef_nr {cfgd['ef_nr']:.0f} gCO2/kWh"
          + (f" | imports EF {cfgd['imports_ef_monthly'].mean():.0f} "
             f"(dry {cfgd['imports_ef_monthly'][0]:.0f} / wet "
             f"{cfgd['imports_ef_monthly'][6]:.0f})"
             if cfgd["imports_ef_monthly"] is not None else ""))
    print("\n[demo] I_cfg prediction, sample July week (no telemetry):")
    print(day.iloc[::3].to_string(index=False))
    print(f"\n[demo] week mean CIF {cif.mean():.1f} | min {cif.min():.1f} "
          f"@h{int(cif[0].argmin())} | max {cif.max():.1f} "
          f"@h{int(cif[0].argmax())} | swing {cif.max() - cif.min():.1f}")
    print("[demo] solar peak hour",
          int(sh[0, :, FUEL_INDEX['solar']].argmax()),
          "-> CIF trough", int(cif[0].argmin()),
          "(carbon-aware scheduling signal)")
    if flags[:, 0].any():
        print(f"[advisory] holiday hours in horizon: "
              f"{int(flags[:, 0].sum())} — shape confidence LOW "
              f"(CNY collapse untrained)")
    if heating.any():
        print(f"[advisory] heating-season hours: {int(heating.sum())} — "
              f"CHP baseload shape untrained; treat level with caution")
    if name == "shanghai":
        print("[advisory] receiving province: imports EF is a monthly "
              "average — intra-day import structure unobservable at "
              "I_cfg; consumption-side CIF inherits this uncertainty")


if __name__ == "__main__":
    main()
