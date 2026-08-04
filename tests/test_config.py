"""Tests for transcif.config constants and region configs."""

from transcif.config import (
    SEQ_LEN, HORIZON, AU_REGIONS, US_REGIONS, get_region_config,
)


def test_constants():
    assert SEQ_LEN == 336
    assert HORIZON == 24


def test_region_configs_present():
    assert len(AU_REGIONS) == 4
    assert len(US_REGIONS) == 8


def test_get_region_config():
    cfg = get_region_config("US_CISO")
    assert cfg["file"] == "US_CISO_2023_hourly.csv"
    assert cfg["ef_nr"] == 342.8
