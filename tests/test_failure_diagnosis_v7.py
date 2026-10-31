"""Numerical diagnostic identities; no target-dependent fitting is exercised."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest


SPEC = importlib.util.spec_from_file_location("diagnose_v7", Path(__file__).resolve().parents[1] / "scripts/diagnose_failure_v7.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def test_crps_matches_explicit_pairwise_identity_and_degenerate_absolute_error():
    rng = np.random.default_rng(17)
    truth = rng.normal(size=(3, 24))
    samples = rng.normal(size=(3, 5, 24))
    expected = (np.abs(samples - truth[:, None]).mean(axis=1)
                - .5 * np.abs(samples[:, :, None] - samples[:, None, :]).mean(axis=(1, 2))).mean(axis=1)
    assert np.allclose(D.ensemble_metrics(samples, truth)["crps"], expected)
    assert np.allclose(D.ensemble_metrics(samples[:, :1], truth)["crps"], np.abs(samples[:, 0] - truth).mean(axis=1))


def test_empirical_interval_and_crps_match_actual_v6_distribution_convention():
    from transcif.research.weighted_metrics import PreparedDistribution
    rng = np.random.default_rng(19)
    samples = rng.normal(size=(3, 128, 24))
    truth = rng.normal(size=(3, 24))
    result = D.ensemble_metrics(samples, truth)
    for day in range(3):
        distribution = PreparedDistribution(samples[day], np.ones(128))
        score = distribution.score(truth[day])
        assert result["crps"][day] == pytest.approx(score["crps"])
        assert result["coverage90"][day] == score["coverage90"]
        assert result["width90"][day] == pytest.approx(score["width90"])


def test_illegal_oracles_replace_exactly_one_component_without_clipping():
    rng = np.random.default_rng(18)
    samples = rng.normal(size=(4, 8, 24)) * 500
    truth = rng.normal(size=(4, 24)) * 200
    level = D.oracle_replace(samples, truth, "level")
    shape = D.oracle_replace(samples, truth, "shape")
    assert np.allclose(D.centered(level), D.centered(samples))
    assert np.allclose(level.mean(axis=2), truth.mean(axis=1)[:, None])
    assert np.allclose(shape.mean(axis=2), samples.mean(axis=2))
    assert np.allclose(D.centered(shape), D.centered(truth)[:, None])
    both = D.oracle_replace(samples, truth, "both")
    assert np.allclose(both, truth[:, None])
    assert np.allclose(D.ensemble_metrics(both, truth)["crps"], 0, atol=1e-12)
    assert (level < 0).any(), "Diagnostic oracles must not silently clip and break component identities"


def test_residual_decomposition_and_utc_phase_are_exact():
    hour = np.arange(24)
    truth = np.tile(300 + 100 * np.cos(2 * np.pi * (hour - 5) / 24), (3, 1))
    point = truth + 80
    result = D.residual_metrics(point, truth)
    assert np.allclose(result["signed_level_error_truth_minus_point"], -80)
    assert np.allclose(result["shape_mae"], 0, atol=1e-12)
    summary = D.shape_summary(truth)
    assert summary["profile_peak_utc"] == 5 and summary["profile_trough_utc"] == 17
    assert summary["first_harmonic_peak_utc"] == pytest.approx(5)
    assert summary["shape_rms"] == pytest.approx(100 / np.sqrt(2))


def test_monthly_weighted_gap_uses_exactly_the_same_retained_hours():
    frame = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC"), "valid": True})
    for fuel in D.FUELS:
        frame[f"generation_{fuel}"] = 0.
    frame.loc[:11, "generation_coal"] = 100.
    frame.loc[12:, "generation_solar"] = 1000.
    coarse = pd.DataFrame({"region": ["TEST"], "period_start": pd.to_datetime(["2026-01-01"], utc=True),
        "observed_hours": [24], **{f"fuelshare_{fuel}": [1 / 11 if fuel == "coal" else 10 / 11 if fuel == "solar" else 0.]
                                  for fuel in D.FUELS}})
    summary, monthly, _, days, truth = D.distribution_summary(frame, "TEST", 2026, "diagnostic", coarse)
    assert len(days) == 1 and truth.shape == (1, 24)
    assert summary["uniform_hourly_mean_cif"] == 500.
    assert summary["energy_weighted_cif"] == pytest.approx(1000 / 11)
    assert monthly[0]["weighted_minus_uniform"] == pytest.approx(1000 / 11 - 500)
    assert monthly[0]["released_minus_retrospective_weighted"] == pytest.approx(0)


def test_latest_asof_respects_extra_lag_and_never_uses_unreleased_month():
    coarse = pd.DataFrame({"region": ["TEST", "TEST"],
        "period_start": pd.to_datetime(["2025-11-01", "2025-12-01"], utc=True),
        "period_end": pd.to_datetime(["2025-12-01", "2026-01-01"], utc=True),
        "available_at": pd.to_datetime(["2025-12-31", "2026-01-31"], utc=True),
        **{f"fuelshare_{fuel}": [float(fuel == "coal"), float(fuel == "gas")] for fuel in D.FUELS}})
    days = pd.DatetimeIndex(["2026-01-31T00:00Z"])
    assert D.latest_asof(coarse, "TEST", days, 0).latest_coarse_cif.iloc[0] == 400.
    assert D.latest_asof(coarse, "TEST", days, 30).latest_coarse_cif.iloc[0] == 1000.


def test_standardized_bank_support_detects_outside_shapes():
    bank = SimpleNamespace(level_paths_=np.array([-1., 0., 1.]),
                           shape_paths_=np.array([np.tile([-1., 1.], 12), np.zeros(24), np.tile([1., -1.], 12)]))
    truth = np.tile(np.tile([-10., 10.], 12), (2, 1)) + 50
    archive = {"corrected_point": np.zeros((2, 24)), "level_scale": np.ones(2), "shape_scale": np.ones(2)}
    result = D.bank_diagnostics(bank, archive, truth, 1.)
    assert result["level_error_outside_source_bank_range_fraction"] == 1.
    assert result["shape_rms_above_source_bank_max_fraction"] == 1.
    assert result["shape_hour_error_outside_source_bank_range_fraction"] == 1.


def test_existing_diagnostic_output_is_never_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        D.diagnose(tmp_path / "source", tmp_path / "external", tmp_path / "base", tmp_path)


@pytest.mark.parametrize("mutation", (None, "duplicate", "missing"))
def test_native_quality_stitches_prior_local_year_and_requires_exact_timestamp_set(mutation):
    from transcif.data.eia930 import MODERN_FUEL_COLUMNS
    starts = pd.date_range("2026-01-01", periods=24, freq="h", tz="UTC")
    frames = []
    for region in D.TARGETS:
        raw = pd.DataFrame({"Balancing Authority": region, "UTC Time at End of Hour": starts + pd.Timedelta(hours=1),
            "Demand (MW)": 100., "Demand Forecast (MW)": 100., "Net Generation (MW)": 100., "Total Interchange (MW)": 0.})
        for fuel, aliases in MODERN_FUEL_COLUMNS.items():
            raw["Net Generation (MW) from " + aliases[0]] = {"coal": 60., "gas": 40.}.get(fuel, 0.)
        frames.append(raw)
    raw = pd.concat(frames, ignore_index=True)
    ending = pd.to_datetime(raw["UTC Time at End of Hour"], utc=True)
    prior = raw.loc[ending.le(starts[12])].copy()
    current = raw.loc[ending.gt(starts[12])].copy()
    if mutation == "duplicate":
        current = pd.concat([current, prior.iloc[:1]], ignore_index=True)
    if mutation == "missing":
        current = current.iloc[1:].copy()
    data = {"2025_Jul_Dec.csv": prior, "2026_Jan_Jun.csv": current}
    inputs = SimpleNamespace(csv=lambda path: data[path].copy())
    audit = {"native_sources": [{"source_snapshot": i, "path": name} for i, name in enumerate(data)],
             "regions": {region: {"2026": {"source_snapshots": [0, 1]}} for region in D.TARGETS}}
    hourly = {region: pd.DataFrame({"time": starts, "valid": True}) for region in D.TARGETS}
    if mutation:
        with pytest.raises(ValueError, match="Duplicate native|complete prepared valid-hour set"):
            D.native_quality(audit, hourly, inputs)
    else:
        records = D.native_quality(audit, hourly, inputs)
        assert all(record["retained_native_hours"] == 24 for record in records)
        assert all(record["retained_native_balance_failure_hours"] == 0 for record in records)
