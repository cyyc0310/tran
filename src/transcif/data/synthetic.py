"""Small synthetic integration fixture, not an empirical carbon benchmark."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.data.dataset import (FUELS, DISPATCHABLE, CAPACITY_FIELDS, MIX_FIELDS,
                                    EF_FIELDS, COST_FIELDS, WEATHER_FIELDS)
from transcif.physics.astro import wind_capacity_factor


def make_bundle(directory, days=90, seed=7):
    if days < 12:
        raise ValueError("at least twelve days required for time splits")
    root = Path(directory)
    if root.exists() and any(root.iterdir()):
        raise ValueError("synthetic data directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01", tz="UTC")
    origins = pd.date_range(start, periods=days, freq="D")
    capacities = {
        "source_coal": [700, 200, 80, 100, 180, 140],
        "source_gas": [180, 650, 150, 80, 120, 250],
        "source_wind": [180, 350, 80, 200, 650, 120],
        "target_mixed": [400, 420, 100, 150, 350, 220]}
    ef = np.array([980., 410., 0., 0., 0., 0.])
    regions = []
    for name, values in capacities.items():
        path = root / name
        path.mkdir(exist_ok=True)
        cap = np.array(values, dtype=float)
        costs = np.array([35., 55., 15.])
        prior = cap * np.array([.55, .35, .85, .4, .3, .2])
        prior /= prior.sum()
        release = {"period_end": start - pd.Timedelta(days=15),
                   "available_at": start - pd.Timedelta(days=1)}
        for columns, array in ((CAPACITY_FIELDS, cap), (MIX_FIELDS, prior),
                               (EF_FIELDS, ef), (COST_FIELDS, costs)):
            release.update(zip(columns, array.tolist()))
        pd.DataFrame([release]).to_csv(path / "releases.csv", index=False)
        forecasts, labels = [], []
        for origin in origins:
            hour = np.arange(24)
            innovation = rng.normal(0, 1.2, 24)
            wind = np.empty(24)
            wind[0] = rng.uniform(5, 10)
            for t in range(1, 24):
                wind[t] = max(0, .9 * wind[t - 1] + .1 * 8 + innovation[t])
            solar = np.maximum(np.sin(np.pi * (hour - 6) / 12), 0) * rng.uniform(550, 950)
            temp = 15 + 6 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 2)
            weather = np.stack([temp, wind, solar], -1)
            demand = cap.sum() * .4 * (1 + .15 * np.sin(2 * np.pi * (hour - 9) / 24))
            demand *= rng.uniform(.92, 1.08)
            available = cap * rng.uniform(.75, .97, 6)
            available[3] *= rng.uniform(.35, .9)
            gen = np.zeros((24, 6))
            gen[:, 2] = available[2]
            gen[:, 4] = available[4] * wind_capacity_factor(wind)
            gen[:, 5] = available[5] * np.clip(.85 * solar / 1000, 0, 1)
            nondisp = gen.sum(-1)
            gen *= np.minimum(1, demand / np.maximum(nondisp, 1e-9))[:, None]
            remaining = np.maximum(demand - gen.sum(-1), 0)
            # Hard merit order differs from the model's smooth quadratic dispatch.
            actual_cost = costs + rng.normal(0, [8, 14, 3])
            for order in np.argsort(actual_cost):
                fuel = DISPATCHABLE[order]
                gen[:, fuel] = np.minimum(remaining, available[fuel])
                remaining -= gen[:, fuel]
            cif = (gen @ ef) / np.maximum(gen.sum(-1), 1e-9)
            for t in range(24):
                valid = origin + pd.Timedelta(hours=t)
                labels.append({"valid_at": valid, "available_at": valid + pd.Timedelta(hours=1),
                    "cif_g_per_kwh": cif[t], **dict(zip(
                        [f"generation_{f}_mw" for f in FUELS], gen[t]))})
            shared_error = rng.normal(0, [1.2, 1., 60.], (1, 3))
            forecast_mean = weather + shared_error
            for member in range(4):
                predicted = forecast_mean + rng.normal(0, [1., .8, 40.], (24, 3))
                predicted[:, 1:] = np.maximum(predicted[:, 1:], 0)
                predicted[solar <= 0, 2] = 0
                load = demand * (1 + rng.normal(0, .025) + rng.normal(0, .01, 24))
                for t in range(24):
                    forecasts.append({"issued_at": origin - pd.Timedelta(hours=6),
                        "available_at": origin - pd.Timedelta(hours=5),
                        "valid_at": origin + pd.Timedelta(hours=t), "member": member,
                        **dict(zip(WEATHER_FIELDS, predicted[t])), "demand_forecast_mw": load[t]})
        pd.DataFrame(forecasts).to_csv(path / "forecasts.csv", index=False)
        pd.DataFrame(labels).to_csv(path / "labels.csv", index=False)
        regions.append({"name": name, "jurisdiction": name.split("_")[-1], "coverage": [name],
            "timezone": "UTC", "members": [0, 1, 2, 3], "config_kind": "synthetic",
            "forecast_kind": "synthetic", "label_kind": "synthetic",
            "config_source": "synthetic ex-ante parameters, independent of hourly labels",
            "forecast_source": "synthetic issued ensemble",
            "releases": f"{name}/releases.csv", "forecasts": f"{name}/forecasts.csv",
            "labels": f"{name}/labels.csv"})
    doc = {"schema_version": 1, "track": "synthetic", "accounting": "production_direct_co2",
           "synthetic_generator": {"seed": seed, "days": days, "configuration_family": "four_fixed_grids"},
           "cost_currency": "synthetic_currency",
           "target": "target_mixed", "horizon": 24,
           "origins": [str(o) for o in origins],
           "train_until": str(origins[int(days * .6)]),
           "validation_until": str(origins[int(days * .8)]), "regions": regions}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps(doc, indent=2) + "\n")
    return manifest
