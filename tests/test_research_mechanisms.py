import numpy as np
import pandas as pd
import pytest

from transcif.research.mechanisms import FUELS, GenerationModel, fit, local_hours, make_evaluator


def _model():
    center = np.zeros((5, 8))
    center[0] = np.log([.35, .25, .02, .1, .1, .08, .08, .02])
    center[1, 0] = 1.5
    slow = np.zeros((4, 5, 8))
    slow[0, 0] = 1  # Common scaling is harmless for shares and CIF.
    daily = np.zeros((2, 5, 8))
    daily[0, 1, 1] = .3
    return GenerationModel(center, slow, np.zeros((5, 8)), daily, {})


def _frame(region, start="2023-01-01", end="2023-03-01"):
    time = pd.date_range(start, end, freq="h", inclusive="left", tz="UTC")
    hour = local_hours(time, region)
    scale = 100 + 10 * np.sin(2 * np.pi * hour / 24)
    values = np.outer(scale, [.35, .25, .02, .1, .1, .08, .08, .02])
    values[:, 0] *= np.where(time.month == 1, 1.1, .9)
    frame = pd.DataFrame(values, columns=[f"generation_{fuel}" for fuel in FUELS])
    frame["time"], frame["valid"] = time, True
    return frame


def test_aggregate_uses_energy_weight_and_scaling_is_harmless():
    model = _model()
    z = np.zeros((2, 6))
    z[1, 0] = 2
    generation = model.generation(z, np.arange(24), include_daily=False)
    assert (generation > 0).all()
    expected = generation.sum(axis=1) / generation.sum(axis=(1, 2))[:, None]
    np.testing.assert_allclose(model.aggregate_mix(z), expected)
    hourly_mean = (generation / generation.sum(axis=-1, keepdims=True)).mean(axis=1)
    assert not np.allclose(expected, hourly_mean)
    np.testing.assert_allclose(expected[0], expected[1])
    cif = model.cif(z, np.arange(24), np.arange(8) * 100)
    np.testing.assert_allclose(cif[0], cif[1])
    assert ((cif >= 0) & (cif <= 700)).all()


def test_daily_state_changes_future_only_and_evaluator_validates_inputs():
    model = _model()
    z = np.zeros((2, 6))
    z[1, 4] = 2
    observed = model.aggregate_mix(z[:1])[0]
    evaluate = make_evaluator(model, observed, np.arange(24), np.arange(8) * 100)
    likelihood, cif = evaluate(z)
    np.testing.assert_allclose(likelihood, 0)
    assert not np.allclose(cif[0], cif[1])
    with pytest.raises(ValueError, match="summing to one"):
        make_evaluator(model, np.ones(8), np.arange(24), np.ones(8))
    with pytest.raises(ValueError, match="latent"):
        evaluate(np.zeros((1, 5)))


def test_source_fit_provenance_cutoff_and_determinism():
    source = _frame("ERCO", "2023-01-01", "2024-02-01")
    first = fit({"ERCO": source}, max_year=2023)
    changed = source.copy()
    changed.loc[changed.time >= "2024-01-01", "generation_coal"] *= 100
    second = fit({"ERCO": changed}, max_year=2023)
    np.testing.assert_array_equal(first.center, second.center)
    np.testing.assert_array_equal(first.slow_loadings, second.slow_loadings)
    assert first.dimension == 6
    assert first.metadata["source_ids"] == ["ERCO"]
    assert first.metadata["number_months"] == 12
    assert first.metadata["number_days"] > 300
    assert first.metadata["fit_range"][1].startswith("2023-12-31 23:")
    assert first.metadata["fit_cutoff_exclusive"] == "2024-01-01T00:00:00Z"
    assert 0 <= first.metadata["retained_pca_variance"]["monthly"] <= 1
    assert np.isfinite(first.cif(np.zeros((1, 6)), np.arange(24), np.ones(8))).all()


def test_source_completeness_duplicates_and_timezones():
    frame = _frame("ERCO")
    frame.loc[frame.time < "2023-01-15", "valid"] = False
    with pytest.raises(ValueError, match="two sufficiently"):
        fit({"ERCO": frame})
    duplicated = pd.concat([frame, frame.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        fit({"ERCO": duplicated})
    times = pd.to_datetime(["2023-03-12T06:00Z", "2023-03-12T07:00Z"])
    np.testing.assert_array_equal(local_hours(times, "FPL"), [1, 3])
    with pytest.raises(ValueError, match="timezone"):
        local_hours(times, "UNKNOWN")
