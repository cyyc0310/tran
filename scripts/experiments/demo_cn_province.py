#!/usr/bin/env python
"""Demo: deployment path for a telemetry-free Chinese province (I_cfg).

Input  — exactly what a province publishes: monthly generation by fuel for
          the past year (火力/水力/风力/光伏/核电发电量, GWh), plus a
          coordinate (lat/lon).  Weather climatology comes from reanalysis
          (Open-Meteo; here we borrow a real region's stack as a stand-in
          so the demo runs offline).

Pipeline:
    12×F monthly generation
        -> monthly fuel-share table            (share_f = gen_f / Σ gen)
        -> per-month ef_nr (thermal-mix EF)    (IPCC: coal 980 / gas 410 / oil 650)
        -> 12×16 monthly FD-config table       (build_monthly_config_table layout)
        -> FuelDecompNet trained on the 25 fuel-telemetry + 4 AU regions
           (no target data — the province only supplies its config)
        -> I_cfg CIF trajectory + per-fuel decomposition for a sample week

Usage:
    .venv/bin/python scripts/experiments/demo_cn_province.py
"""

import numpy as np
import pandas as pd
import torch

from transcif.config import SEQ_LEN, HORIZON
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import CANONICAL_FUELS, FUEL_INDEX
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, train_fuel_zero_shot, predict_fuel_windows,
)
from transcif.physics.astro import astro_features
from transcif.data.calendar import calendar_features

# ---------------------------------------------------------------------------
# 1. Hypothetical province: monthly generation by fuel (GWh).
#    Shape: a northern coal-heavy province with strong wind/solar buildout
#    (wind peaks in spring, solar in summer, coal backs off mid-year).
# ---------------------------------------------------------------------------
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GEN_GWH = {
    "coal":    [34, 30, 32, 28, 26, 24, 26, 25, 27, 30, 33, 35],
    "gas":     [4.0, 3.6, 3.8, 3.4, 3.2, 3.6, 4.2, 4.2, 3.4, 3.2, 3.6, 4.2],
    "petroleum": [0.1] * 12,
    "nuclear": [2.8] * 12,
    "hydro":   [0.6, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.5, 1.2, 0.9, 0.7, 0.6],
    "solar":   [1.6, 2.0, 2.6, 3.2, 3.7, 3.9, 3.7, 3.4, 2.9, 2.3, 1.7, 1.4],
    "wind":    [2.8, 3.0, 3.4, 3.6, 3.2, 2.6, 2.2, 2.2, 2.8, 3.4, 3.6, 3.4],
    "biomass": [0.3] * 12,
    "imports": [0.0] * 12,
    "other":   [0.0] * 12,
}
LAT, LON, TZ = 38.0, 115.0, 8.0

THERMAL_EFS = {"coal": 980.0, "gas": 410.0, "petroleum": 650.0}
RENEWABLES = {"hydro", "solar", "wind", "biomass"}


def monthly_table_from_generation():
    """Monthly generation (GWh) -> (12, 16) monthly FD-config table."""
    gen = np.zeros((12, len(CANONICAL_FUELS)))
    for f, vals in GEN_GWH.items():
        gen[:, FUEL_INDEX[f]] = vals
    total = gen.sum(axis=1, keepdims=True)
    shares = gen / np.clip(total, 1e-9, None)          # (12, F)
    thermal_mass = shares[:, [FUEL_INDEX[f] for f in THERMAL_EFS]].sum(axis=1)
    ef_nr = np.zeros(12)
    for f, ef in THERMAL_EFS.items():
        ef_nr += shares[:, FUEL_INDEX[f]] * ef
    ef_nr = ef_nr / np.clip(thermal_mass, 1e-9, None) * thermal_mass \
        + shares[:, FUEL_INDEX["imports"]] * 250.0 \
        + shares[:, FUEL_INDEX["other"]] * 500.0
    renew_mask = np.array([f in RENEWABLES for f in CANONICAL_FUELS])
    mean_rs = shares[:, renew_mask].sum(axis=1)
    return gen, shares, ef_nr, mean_rs


def build_pseudo_region(shares, ef_nr, mean_rs, donor_exog, donor_ef_vec):
    """Assemble the pseudo-target dict the trainer expects (config-only)."""
    D = 16
    table = np.zeros((12, D), dtype=np.float32)
    ex = donor_exog  # weather climatology stand-in (reanalysis offline)
    split = int(len(ex["wind_cf"]) * 0.8)
    day = ex["astro"][:split, 0] > 0
    ann_windcf = float(ex["wind_cf"][:split].mean())
    ann_csi = float(ex["clearsky_index"][:split][day].mean())
    for m in range(12):
        table[m] = np.array([
            mean_rs[m], ef_nr[m] / 1000.0, *shares[m],
            ann_windcf, ann_csi, 1.0, abs(LAT) / 60.0], dtype=np.float32)
    annual = shares.mean(axis=0)
    fd_config = np.array([
        float(mean_rs.mean()), float(ef_nr.mean() / 1000.0), *annual,
        ann_windcf, ann_csi, 1.0, abs(LAT) / 60.0], dtype=np.float32)
    # Per-fuel EF vector: thermal trio rescaled to the province ef_nr.
    ef_vec = donor_ef_vec.copy()
    thermal_idx = [FUEL_INDEX[f] for f in THERMAL_EFS]
    thermal_cfg = annual[thermal_idx].sum()
    if thermal_cfg > 1e-3:
        cur = float(np.dot(ef_vec[thermal_idx], annual[thermal_idx]))
        ef_vec[thermal_idx] *= float(np.clip(
            (ef_nr.mean() * thermal_cfg) / max(cur, 1e-6), 0.2, 3.0))
    else:
        ef_vec[thermal_idx] = ef_nr.mean()
    pseudo = {
        "mean_rs": float(mean_rs.mean()), "ef_r": 0.0,
        "ef_nr": float(ef_nr.mean()),
        "fd_config": fd_config, "monthly_table": table,
        "has_fuel": True, "ef_vec": ef_vec,
    }
    return pseudo, table


def main():
    torch.manual_seed(0)
    cfgs = all_region_configs()
    print("[demo] preparing 28 source regions ...")
    fd = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    gen, shares, ef_nr, mean_rs = monthly_table_from_generation()
    df = pd.DataFrame(shares, index=MONTHS, columns=CANONICAL_FUELS)
    print("\n[province] monthly fuel shares (derived from monthly generation):")
    print((df * 100).round(1).to_string())
    print(f"\n[province] annual mean_rs {mean_rs.mean():.3f} | "
          f"ef_nr {ef_nr.mean():.0f} gCO2/kWh")

    # Weather stand-in: borrow a real region's exog stack (offline demo).
    donor = fd["US_CISO"]
    pseudo, table = build_pseudo_region(shares, ef_nr, mean_rs,
                                        donor["exog"], donor["ef_vec"])
    fd["CN_DEMO"] = pseudo

    print("\n[demo] training FuelDecompNet on all 28 regions "
          "(province supplies config only) ...")
    model = train_fuel_zero_shot(fd, "CN_DEMO", seed=0, epochs=600,
                                 use_monthly=True, device="mps")

    # Sample week: mid-July, exog = astronomy at (LAT, LON) + donor weather.
    hours = pd.date_range("2023-07-10", periods=SEQ_LEN + HORIZON, freq="h")
    astro = astro_features(hours, LAT, LON).astype(np.float32)
    cal = calendar_features(hours, tz_offset=TZ)
    wx = donor["exog"]["weather"][:SEQ_LEN + HORIZON].copy()
    wcf = donor["exog"]["wind_cf"][:SEQ_LEN + HORIZON, None]
    csi = donor["exog"]["clearsky_index"][:SEQ_LEN + HORIZON, None]
    fut_exog = np.concatenate(
        [astro, wcf, csi, cal], axis=1)[SEQ_LEN:].astype(np.float32)[None]
    windows = {
        "x_rs": np.zeros((1, SEQ_LEN), np.float32),
        "x_fuel": np.zeros((1, SEQ_LEN, len(CANONICAL_FUELS)), np.float32),
        "x_weather": wx[None, :SEQ_LEN],
        "fut_weather": wx[None, SEQ_LEN:],
        "fut_exog": fut_exog,
        "config": table[6][None],  # July -> June row (1-month publication lag)
    }
    cif, sh, _ = predict_fuel_windows(
        model, windows, pseudo["fd_config"], pseudo["ef_vec"].astype(np.float32),
        cold=True, device="mps")

    print("\n[demo] I_cfg prediction for a sample July week (no telemetry):")
    hours_f = hours[SEQ_LEN:]
    day = pd.DataFrame({
        "hour": hours_f.strftime("%m-%d %H:%M"),
        "CIF_pred": cif[0].round(1),
        "solar": sh[0, :, FUEL_INDEX["solar"]].round(3),
        "wind": sh[0, :, FUEL_INDEX["wind"]].round(3),
        "coal": sh[0, :, FUEL_INDEX["coal"]].round(3),
        "gas": sh[0, :, FUEL_INDEX["gas"]].round(3),
    })
    print(day.iloc[::3].to_string(index=False))
    print(f"\n[demo] week: mean CIF {cif.mean():.1f} gCO2/kWh | "
          f"min {cif.min():.1f} (hour {int(cif[0].argmin())}) | "
          f"max {cif.max():.1f} (hour {int(cif[0].argmax())}) | "
          f"diurnal swing {cif.max() - cif.min():.1f}")
    print("[demo] solar share peaks at hour",
          int(sh[0, :, FUEL_INDEX['solar']].argmax()),
          "| CIF trough follows the solar peak (carbon-aware scheduling signal)")


if __name__ == "__main__":
    main()
