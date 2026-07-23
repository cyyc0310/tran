"""End-to-end smoke test for `deploy_region`: drives the full Stage 1->3 pipeline through a
`DeploymentConfig` on the real SA1 fixture, verifying the config layer wires the existing
training/physics/calibration code together and returns a well-formed result dict.

This is a plumbing test, not a scientific result: the SA1 fixture is used as its own single
source region (so the single-source training path is exercised without needing the full
QLD1/NSW1/VIC1 export), epochs are tiny, and the sample is only ~300 rows -- so the metric
values here are not the paper's numbers and are only asserted to be finite and well-typed."""

import math
from pathlib import Path

import numpy as np

from transcif.config.deploy import deploy_region
from transcif.config.region_config import DeploymentConfig, RegionConfig

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SA1_CSV = str(FIXTURES / "real_aemo_sample_sa1.csv")


def _fixture_config(**overrides) -> DeploymentConfig:
    defaults = dict(
        target=RegionConfig(name="SA1", hourly_csv=SA1_CSV, factor_code="AU_SA"),
        sources=[RegionConfig(name="SA1_src", hourly_csv=SA1_CSV, factor_code="AU_SA")],
        seq_len=36,
        horizon=6,
        stride=6,
        calib_fraction=0.7,
        include_generation=True,
        mldg_epochs=3,
        seed=0,
    )
    defaults.update(overrides)
    return DeploymentConfig(**defaults)


def test_deploy_region_end_to_end_with_ground_truth():
    result = deploy_region(_fixture_config())

    assert result["target"] == "SA1"
    assert result["num_channels"] == 4
    assert result["emission_factors"] == (0.0, 490.43)
    assert result["has_ground_truth"] is True

    for key in ("corrected_mae", "physics_only_mae", "persistence_mae", "conformal_halfwidth"):
        assert math.isfinite(result[key]), f"{key} should be finite"
    assert 0.0 <= result["empirical_coverage"] <= 1.0

    assert result["ci_pred_corrected_eval"].shape == result["ci_pred_physics_eval"].shape
    assert np.all(np.isfinite(result["ci_pred_corrected_eval"]))


def test_deploy_region_inline_factors_override_table():
    """Deploying with inline emission factors uses them verbatim (the new-region path)."""
    config = _fixture_config(
        target=RegionConfig(
            name="SA1", hourly_csv=SA1_CSV, emission_renewable=5.0, emission_nonrenewable=500.0
        )
    )
    result = deploy_region(config)
    assert result["emission_factors"] == (5.0, 500.0)
