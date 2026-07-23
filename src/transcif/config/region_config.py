"""`RegionConfig` / `DeploymentConfig`: declarative descriptions of a deployment region and
of a full source->target adaptation run, loadable from JSON (zero-dependency) or, if PyYAML
happens to be installed, YAML. These replace the module-level constants hardcoded in
`scripts/sa1_ablation.py` (DATA_DIR, SOURCE_REGIONS, REGION_TO_FACTOR_CODE, SEQ_LEN, ...)
with data a deployer can edit without touching Python.

Emission factors are the key externalization: a `RegionConfig` may carry its own
(renewable, nonrenewable) gCO2/kWh factors inline via `emission_renewable` /
`emission_nonrenewable`. When present these take precedence over the `EMISSION_FACTOR_TABLES`
lookup in `physics/cif.py`, so onboarding a brand-new grid (with its own measured factors)
is a config edit, never a code edit -- and never silently reuses the placeholder EU_*/US_*
rows that `cif.py` explicitly warns against."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from typing import Any

from transcif.physics.cif import EMISSION_FACTOR_TABLES


@dataclass
class RegionConfig:
    """One grid region. `name` is a human label; `hourly_csv` is the real AEMO/NEMED-style
    hourly export; `temperature_csv` is an optional weather covariate source.

    Emission factors resolve in this order:
      1. inline `emission_renewable` / `emission_nonrenewable` (both must be set), else
      2. `EMISSION_FACTOR_TABLES[factor_code]` from `physics/cif.py`.
    A region with neither an inline pair nor a resolvable `factor_code` raises on lookup."""

    name: str
    hourly_csv: str
    factor_code: str | None = None
    temperature_csv: str | None = None
    emission_renewable: float | None = None
    emission_nonrenewable: float | None = None

    def resolve_emission_factors(self) -> tuple[float, float]:
        """Return (renewable_factor, nonrenewable_factor) in gCO2/kWh, preferring inline
        config values over the `EMISSION_FACTOR_TABLES` lookup. Raises ValueError with an
        actionable message rather than silently defaulting when neither is available."""
        inline = (self.emission_renewable, self.emission_nonrenewable)
        if all(v is not None for v in inline):
            return float(self.emission_renewable), float(self.emission_nonrenewable)
        if any(v is not None for v in inline):
            raise ValueError(
                f"region '{self.name}' sets only one of emission_renewable/emission_nonrenewable; "
                "set both to override, or neither to fall back to factor_code."
            )
        if self.factor_code is None:
            raise ValueError(
                f"region '{self.name}' has no inline emission factors and no factor_code; "
                "cannot resolve emission factors."
            )
        if self.factor_code not in EMISSION_FACTOR_TABLES:
            raise ValueError(
                f"region '{self.name}' factor_code '{self.factor_code}' is not in EMISSION_FACTOR_TABLES; "
                "either add inline emission_renewable/emission_nonrenewable, or use a known factor_code "
                f"({sorted(EMISSION_FACTOR_TABLES)})."
            )
        table = EMISSION_FACTOR_TABLES[self.factor_code]
        return float(table["renewable"]), float(table["nonrenewable"])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RegionConfig:
        return cls(**_only_known_fields(cls, data))


@dataclass
class DeploymentConfig:
    """A full source->target adaptation run. `sources` are the labeled source regions
    (>= 2 to enable MLDG meta-learning; a single source falls back to plain source-domain
    training in `deploy_region`); `target` is the deployment region.

    The remaining fields externalize what were module-level constants in the experiment
    scripts. Channel switches (`include_generation` / `include_temperature`) and the D/E
    domain-adaptation toggles (`fine_tune` / `coral`) map 1:1 onto the ablation variants, so
    an ablation is now a set of config files rather than an edited script."""

    target: RegionConfig
    sources: list[RegionConfig] = field(default_factory=list)

    seq_len: int = 48
    horizon: int = 12
    stride: int = 6
    calib_fraction: float = 0.7

    include_generation: bool = False
    include_temperature: bool = False
    gate_conditioning: bool = False
    mldg_weighted: bool = True

    lt_feature_dim: int = 16
    cv_feature_dim: int = 8
    mldg_epochs: int = 80

    fine_tune: bool = False
    fine_tune_epochs_per_stage: int = 15
    fine_tune_lr: float = 5e-4
    coral: bool = False
    coral_weight: float = 0.1

    seed: int = 42

    @property
    def num_channels(self) -> int:
        """Input channel count implied by the switches: RenewShare + LoadNorm (2), plus
        RenewOutNorm/NonRenewOutNorm when generation is on, plus TempAnomaly when temp is
        on. Must match `DomainInvariantEncoder(num_variables=...)`."""
        channels = 2
        if self.include_generation:
            channels += 2
        if self.include_temperature:
            channels += 1
        return channels

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeploymentConfig:
        payload = dict(data)
        target = payload.pop("target")
        sources = payload.pop("sources", [])
        return cls(
            target=RegionConfig.from_dict(target),
            sources=[RegionConfig.from_dict(s) for s in sources],
            **_only_known_fields(cls, payload, exclude={"target", "sources"}),
        )

    @classmethod
    def from_json(cls, path: str) -> DeploymentConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_yaml(cls, path: str) -> DeploymentConfig:
        """Optional convenience: only usable if PyYAML is installed. JSON is the primary,
        zero-dependency format; this raises an actionable error when PyYAML is absent rather
        than adding a hard dependency the rest of the project doesn't need."""
        try:
            import yaml  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ImportError(
                "DeploymentConfig.from_yaml requires PyYAML, which is not installed; "
                "use DeploymentConfig.from_json instead, or `pip install pyyaml`."
            ) from exc
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))


def _only_known_fields(cls, data: dict[str, Any], exclude: set[str] | None = None) -> dict[str, Any]:
    """Filter a dict down to a dataclass's declared fields, raising on unknown keys so a
    typo in a config file (e.g. `calibration_fraction` vs `calib_fraction`) fails loudly
    at load time instead of being silently ignored. Keys beginning with `_` are treated as
    inline documentation (JSON has no comment syntax) and skipped."""
    exclude = exclude or set()
    data = {k: v for k, v in data.items() if not k.startswith("_")}
    known = {f.name for f in fields(cls)} - exclude
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__} got unknown config keys: {sorted(unknown)} (known: {sorted(known)})")
    return {k: v for k, v in data.items() if k in known}
