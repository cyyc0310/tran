"""Config-driven deployment layer: describe a new deployment region entirely in a config
file (data paths, emission factors, channel switches, windowing) and hand it to
`deploy_region`, rather than editing hardcoded constants in the experiment scripts.

This is the code-level support for the paper's central claim -- that adapting TransCIF to a
new grid is a *configuration* change, not a code change. `RegionConfig` in particular lets a
new region carry its own measured emission factors inline, so onboarding a region never
requires editing the `EMISSION_FACTOR_TABLES` dict in `physics/cif.py`."""

from transcif.config.region_config import DeploymentConfig, RegionConfig

__all__ = ["RegionConfig", "DeploymentConfig"]
