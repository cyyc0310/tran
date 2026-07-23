"""Unit tests for the config-driven deployment layer's declarative pieces: emission-factor
resolution precedence, config parsing/validation, and the channel-count derivation. These
are pure (no torch, no data) so they pin down the config contract independently of the
training pipeline exercised in test_deploy.py."""

import json

import pytest

from transcif.config.region_config import DeploymentConfig, RegionConfig


def test_inline_emission_factors_take_precedence_over_table():
    region = RegionConfig(
        name="SA1",
        hourly_csv="x.csv",
        factor_code="AU_SA",
        emission_renewable=1.0,
        emission_nonrenewable=999.0,
    )
    assert region.resolve_emission_factors() == (1.0, 999.0)


def test_factor_code_fallback_matches_table():
    region = RegionConfig(name="SA1", hourly_csv="x.csv", factor_code="AU_SA")
    assert region.resolve_emission_factors() == (0.0, 490.43)


def test_partial_inline_factors_raise():
    region = RegionConfig(name="X", hourly_csv="x.csv", emission_renewable=1.0)
    with pytest.raises(ValueError, match="only one of"):
        region.resolve_emission_factors()


def test_missing_factors_and_code_raise():
    region = RegionConfig(name="X", hourly_csv="x.csv")
    with pytest.raises(ValueError, match="no inline emission factors and no factor_code"):
        region.resolve_emission_factors()


def test_unknown_factor_code_raises():
    region = RegionConfig(name="X", hourly_csv="x.csv", factor_code="ZZ_UNKNOWN")
    with pytest.raises(ValueError, match="not in EMISSION_FACTOR_TABLES"):
        region.resolve_emission_factors()


def test_inline_factors_bypass_missing_table_entry():
    """A brand-new region with only inline factors resolves without any table entry."""
    region = RegionConfig(
        name="NEW", hourly_csv="x.csv", emission_renewable=12.0, emission_nonrenewable=430.0
    )
    assert region.resolve_emission_factors() == (12.0, 430.0)


def test_num_channels_tracks_switches():
    base = dict(target={"name": "T", "hourly_csv": "t.csv", "factor_code": "AU_SA"})
    assert DeploymentConfig.from_dict(base).num_channels == 2
    assert DeploymentConfig.from_dict({**base, "include_generation": True}).num_channels == 4
    assert DeploymentConfig.from_dict({**base, "include_temperature": True}).num_channels == 3
    assert (
        DeploymentConfig.from_dict(
            {**base, "include_generation": True, "include_temperature": True}
        ).num_channels
        == 5
    )


def test_unknown_config_key_fails_loudly():
    with pytest.raises(ValueError, match="unknown config keys"):
        DeploymentConfig.from_dict(
            {"target": {"name": "T", "hourly_csv": "t.csv", "factor_code": "AU_SA"}, "calibration_fraction": 0.5}
        )


def test_underscore_keys_treated_as_comments():
    config = DeploymentConfig.from_dict(
        {
            "_comment": "human note ignored by the loader",
            "target": {"_note": "also ignored", "name": "T", "hourly_csv": "t.csv", "factor_code": "AU_SA"},
        }
    )
    assert config.target.name == "T"


def test_from_json_round_trip(tmp_path):
    payload = {
        "target": {"name": "SA1", "hourly_csv": "sa1.csv", "factor_code": "AU_SA"},
        "sources": [{"name": "QLD1", "hourly_csv": "qld1.csv", "factor_code": "AU_QLD"}],
        "include_generation": True,
        "mldg_epochs": 5,
    }
    path = tmp_path / "deploy.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    config = DeploymentConfig.from_json(str(path))
    assert config.target.name == "SA1"
    assert [s.name for s in config.sources] == ["QLD1"]
    assert config.include_generation is True
    assert config.mldg_epochs == 5
    assert config.num_channels == 4
