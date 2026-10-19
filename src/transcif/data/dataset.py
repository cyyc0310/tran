"""Public model inputs and separately loaded source/scoring labels."""

from dataclasses import dataclass, fields
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch

from transcif.data.asof import forecast_asof, public_config_asof, utc
from transcif.data.calendar import calendar_features

FUELS = ("coal", "gas", "nuclear", "hydro", "wind", "solar")
DISPATCHABLE = (0, 1, 3)
CAPACITY_FIELDS = [f"capacity_{f}_mw" for f in FUELS]
MIX_FIELDS = [f"mix_{f}" for f in FUELS]
EF_FIELDS = [f"ef_{f}_g_per_kwh" for f in FUELS]
COST_FIELDS = [f"cost_{FUELS[i]}_per_mwh" for i in DISPATCHABLE]
CONFIG_FIELDS = CAPACITY_FIELDS + MIX_FIELDS + EF_FIELDS + COST_FIELDS
WEATHER_FIELDS = ["temperature_c", "wind_speed_ms", "shortwave_w_per_m2"]
FORECAST_FIELDS = WEATHER_FIELDS + ["demand_forecast_mw"]


@dataclass
class ModelInputs:
    capacity: torch.Tensor       # B,F, MW
    mix: torch.Tensor            # B,F, published energy-share prior
    ef: torch.Tensor             # B,F, gCO2/kWh
    cost: torch.Tensor           # B,3, currency/MWh in a consistent currency
    weather: torch.Tensor        # B,M,H,3
    demand: torch.Tensor         # B,M,H, MW, issued forecast
    calendar: torch.Tensor       # B,H,6

    def to(self, device):
        return ModelInputs(**{f.name: getattr(self, f.name).to(device) for f in fields(self)})


@dataclass(frozen=True)
class PublicExample:
    region: str
    origin: pd.Timestamp
    values: dict
    provenance: dict


@dataclass
class GridData:
    name: str
    jurisdiction: str
    coverage: frozenset
    timezone: str
    members: tuple
    releases: pd.DataFrame
    forecasts: pd.DataFrame
    labels_path: Path | None
    track: str
    config_source: str
    forecast_source: str
    label_kind: str

    def example(self, origin, horizon=24):
        origin = utc([origin], "origin")[0]
        if origin != origin.floor("h") or horizon < 1:
            raise ValueError("hour-aligned origin and positive horizon required")
        hours = pd.date_range(origin, periods=horizon, freq="h")
        cfg = public_config_asof(self.releases, origin, CONFIG_FIELDS)
        run = forecast_asof(self.forecasts, origin, hours, self.members, FORECAST_FIELDS)
        values = {"capacity": cfg[CAPACITY_FIELDS].to_numpy(dtype=np.float32),
                  "mix": cfg[MIX_FIELDS].to_numpy(dtype=np.float32),
                  "ef": cfg[EF_FIELDS].to_numpy(dtype=np.float32),
                  "cost": cfg[COST_FIELDS].to_numpy(dtype=np.float32),
                  "weather": run[WEATHER_FIELDS].to_numpy(dtype=np.float32).reshape(
                      len(self.members), horizon, 3),
                  "demand": run.demand_forecast_mw.to_numpy(dtype=np.float32).reshape(
                      len(self.members), horizon),
                  "calendar": calendar_features(hours, self.timezone)}
        validate_values(values)
        return PublicExample(self.name, origin, values, {
            "config_period_end": str(cfg.period_end), "config_available_at": str(cfg.available_at),
            "forecast_issued_at": str(run.issued_at.iloc[0]),
            "forecast_available_at": str(run.available_at.max()),
            "config_source": self.config_source, "forecast_source": self.forecast_source,
            "track": self.track})

    def labels(self, example, available_by=None):
        """Used only for source supervision or after inference for scoring."""
        if self.labels_path is None:
            raise ValueError(f"no labels for {self.name}")
        frame = pd.read_csv(self.labels_path)
        frame["valid_at"] = utc(frame.valid_at, "label valid_at")
        frame["available_at"] = utc(frame.available_at, "label available_at")
        if (frame.available_at < frame.valid_at + pd.Timedelta(hours=1)).any():
            raise ValueError("hourly labels must be available after their interval ends")
        if frame.duplicated(["valid_at", "available_at"]).any():
            raise ValueError("ambiguous label revisions")
        if available_by is not None:
            cutoff = utc([available_by], "label cutoff")[0]
            frame = frame[frame.available_at <= cutoff]
        hours = pd.date_range(example.origin, periods=example.values["demand"].shape[-1], freq="h")
        frame = (frame.sort_values("available_at").drop_duplicates("valid_at", keep="last")
                 .set_index("valid_at").reindex(hours))
        columns = ["cif_g_per_kwh"] + [f"generation_{f}_mw" for f in FUELS]
        if not np.isfinite(frame[columns].to_numpy(dtype=float)).all():
            raise ValueError(f"missing, late or nonfinite labels for {self.name} at {example.origin}")
        cif = frame.cif_g_per_kwh.to_numpy(dtype=np.float32)
        generation = frame[columns[1:]].to_numpy(dtype=np.float32)
        if (cif < 0).any() or (generation < 0).any() or (generation.sum(-1) <= 0).any():
            raise ValueError("nonnegative production labels and positive generation required")
        return {"cif": cif, "generation": generation}


def validate_values(values):
    expected = {f.name for f in fields(ModelInputs)}
    if set(values) != expected:
        raise ValueError(f"public input fields must be exactly {sorted(expected)}")
    for name, array in values.items():
        if not np.isfinite(array).all():
            raise ValueError(f"nonfinite {name}")
    for name in ("capacity", "mix", "ef"):
        if values[name].shape != (len(FUELS),) or (values[name] < 0).any():
            raise ValueError(f"{name} must be a nonnegative six-fuel vector")
    if values["capacity"].sum() <= 0 or not np.isclose(values["mix"].sum(), 1, atol=1e-4):
        raise ValueError("positive capacity and a unit-sum mix prior required")
    weather, demand = values["weather"], values["demand"]
    if (weather.ndim != 3 or weather.shape[-1] != 3 or min(weather.shape[:2]) < 1
            or demand.shape != weather.shape[:2] or (demand <= 0).any()
            or (weather[..., 1:] < 0).any()):
        raise ValueError("weather (M,H,3), positive forecast demand (M,H) required")
    if values["cost"].shape != (3,) or values["calendar"].shape != (weather.shape[1], 6):
        raise ValueError("cost (3,) and calendar (H,6) required")


def stack_inputs(examples, device="cpu"):
    if not examples:
        raise ValueError("no examples")
    for example in examples:
        validate_values(example.values)
    return ModelInputs(**{f.name: torch.as_tensor(np.stack(
        [e.values[f.name] for e in examples]), dtype=torch.float32, device=device)
        for f in fields(ModelInputs)})


def load_bundle(manifest_path):
    path = Path(manifest_path)
    doc = json.loads(path.read_text())
    if doc.get("schema_version") != 1 or doc.get("accounting") != "production_direct_co2":
        raise ValueError("schema_version=1 and accounting=production_direct_co2 required")
    track = doc.get("track")
    if track not in ("issued_public", "synthetic", "development_proxy_oracle"):
        raise ValueError("declare an explicit experiment track")
    if not isinstance(doc.get("cost_currency"), str) or not doc["cost_currency"].strip():
        raise ValueError("declare one consistent cost_currency for all regions")
    grids = {}
    for region in doc["regions"]:
        name = region["name"]
        if (name in grids or not name or not region["coverage"] or not region["members"]
                or len(set(region["members"])) != len(region["members"])):
            raise ValueError("unique regions, nonempty coverage and unique members required")
        if not region.get("config_source") or not region.get("forecast_source"):
            raise ValueError("public input provenance must be provided")
        if track == "issued_public" and (region.get("config_kind") != "published"
                                         or region.get("forecast_kind") != "issued"):
            raise ValueError("issued_public requires published configs and issued forecasts")
        if region.get("label_kind") not in ("observed", "reconstructed", "synthetic"):
            raise ValueError("declare observed/reconstructed/synthetic label_kind")
        if track == "issued_public" and region["label_kind"] == "synthetic":
            raise ValueError("synthetic labels cannot support an issued_public experiment")
        grids[name] = GridData(name, region["jurisdiction"], frozenset(region["coverage"]),
            region["timezone"], tuple(region["members"]),
            pd.read_csv(path.parent / region["releases"]),
            pd.read_csv(path.parent / region["forecasts"]),
            path.parent / region["labels"] if region.get("labels") else None,
            track, region["config_source"], region["forecast_source"], region["label_kind"])
    if not grids:
        raise ValueError("at least one region required")
    return doc, grids


def source_names(grids, target, leave_jurisdiction_out=False):
    held = grids[target]
    names = [name for name, grid in grids.items()
             if name != target and not (held.coverage & grid.coverage)
             and (not leave_jurisdiction_out or grid.jurisdiction != held.jurisdiction)]
    if not names:
        raise ValueError("no independent source grids remain")
    return names
