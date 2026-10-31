"""Behavioral invariants for weather gating and legal coarse-factor transfer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import coarse_dynamics_v9 as v9
from scripts import weather_coarse_v14 as m
from transcif.research.fuel_transport_v7 import recompose_paths


def monthly() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", "2025-02-01", freq="MS", tz="UTC")
    values = np.random.default_rng(3).dirichlet(np.ones(8)*3, len(dates))
    result = pd.DataFrame(values, columns=m.COLUMNS)
    result["region"] = "BPAT"
    result["period_start"] = dates
    result["period_end"] = dates+pd.offsets.MonthBegin(1)
    result["available_at"] = result.period_end+pd.Timedelta(days=30)
    return result


def test_legal_law_is_exact_v9_harmonic_distribution() -> None:
    data = monthly()
    days = pd.DatetimeIndex(["2024-01-01", "2024-02-15", "2024-06-30"], tz="UTC")
    law = m.legal_coarse_law(data, "BPAT", days)
    reference = v9.forecast_monthly(data, "BPAT", days)["harmonic_ridge"]
    np.testing.assert_array_equal(law["mix"], reference["mix"])
    np.testing.assert_array_equal(law["draws"], reference["draws"])
    assert law["audit"]["current_month_truth_used"] is False
    assert law["metadata"].shape == (3, 4)


def test_future_month_values_and_unreleased_rows_do_not_change_law() -> None:
    data = monthly()
    days = pd.DatetimeIndex(["2024-01-17"], tz="UTC")
    before = m.legal_coarse_law(data, "BPAT", days)
    changed = data.copy()
    future = changed.available_at.gt(days[0]) | changed.period_end.gt(days[0])
    changed.loc[future, list(m.COLUMNS)] = np.full((future.sum(), 8), 1/8)
    after = m.legal_coarse_law(changed, "BPAT", days)
    np.testing.assert_array_equal(before["mix"], after["mix"])
    np.testing.assert_array_equal(before["draws"], after["draws"])
    changed.loc[future, list(m.COLUMNS)] = np.nan
    ignored = m.legal_coarse_law(changed, "BPAT", days)
    np.testing.assert_array_equal(before["draws"], ignored["draws"])


def test_coarse_features_are_shared_finite_law_summaries() -> None:
    law = m.legal_coarse_law(monthly(), "BPAT", pd.DatetimeIndex(["2024-02-01"], tz="UTC"))
    x = m.coarse_features(law["mix"], law["draws"], law["metadata"])
    assert x.shape == (1, len(m.COARSE_FEATURE_NAMES)) == (1, 25)
    assert np.isfinite(x).all()
    assert x[0, 16] <= x[0, 17] <= x[0, 18]
    np.testing.assert_array_equal(x[0, -4:], law["metadata"][0])


def test_weather_gate_endpoints_and_fuel_daily_amplitudes() -> None:
    rng = np.random.default_rng(4)
    a = rng.random((3, 8, 24, 8)).astype(np.float32)
    b = a[..., ::-1, :].copy()
    a[..., 0] = 0; b[..., 0] = 0
    a[..., 7] = 1; b[..., 7] = 1
    q = m.blend_profiles(a, b, np.array([0, .5, 1]))
    np.testing.assert_array_equal(q[0], a[0])
    np.testing.assert_array_equal(q[2], b[2])
    np.testing.assert_allclose(q.mean(-2), a.mean(-2), atol=1e-7)
    assert (q[..., 0] == 0).all() and (q[..., 7] == 1).all()


def test_profile_diagnostics_measure_real_zero_creation_and_amplitude_error() -> None:
    q = np.ones((1, 2, 24, 8)); q[..., 0, 0] = 0
    updated = q.copy(); updated[..., 0, 0] = 1; updated[..., 1, 0] = 0
    audit = m.profile_audit(q, updated, np.ones((1, 2, 8), dtype=bool))
    assert audit["amplitude_error"] == 0
    assert audit["sampled_zero_creation"]["created_positive_from_zero_count_by_fuel"][0] == 2
    updated[..., 1, 0] = 1
    assert m.profile_audit(q, updated, np.ones((1, 2, 8), dtype=bool))["amplitude_error"] > 0


@pytest.mark.parametrize("gate", [np.array([-1., 0]), np.array([np.nan, 1]), np.array([1.])])
def test_invalid_gate_is_rejected(gate: np.ndarray) -> None:
    q = np.ones((2, 24, 8))
    with pytest.raises(ValueError):
        m.blend_profiles(q, q, gate)


@pytest.mark.parametrize("kind", ["ridge", "forest"])
def test_gate_fit_is_source_only_and_predicts_registered_actions(kind: str) -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(90, 4)); x[0, 1] = np.nan
    losses = np.tile([100., 40., 1.], (90, 1))
    regions = np.repeat(["BPAT", "CISO", "ERCO"], 30)
    gate = m.GainGate(kind).fit(x, losses, regions)
    np.testing.assert_array_equal(gate.gates(x), np.ones(90))
    assert gate.audit_["training_regions"] == ["BPAT", "CISO", "ERCO"]
    with pytest.raises(ValueError):
        m.GainGate(kind).fit(x, losses, np.repeat("PACE", 90))


def test_gate_tie_protects_native() -> None:
    x = np.arange(80).reshape(40, 2)
    gate = m.GainGate("ridge").fit(x, np.ones((40, 3)), np.repeat(["BPAT", "CISO"], 20))
    np.testing.assert_array_equal(gate.gates(x), np.zeros(40))


def test_coupling_preserves_fine_day_multisets_and_positive_rank() -> None:
    rng = np.random.default_rng(9)
    indices = rng.integers(0, 50, size=(2, 128))
    draws = rng.dirichlet(np.ones(8), size=(2, 128))
    z = np.linspace(-100, 200, 50)
    days = pd.date_range("2025-01-01", periods=2, tz="UTC")
    coupled = m.couple_indices(indices, draws, z, days)
    np.testing.assert_array_equal(np.sort(coupled, axis=1), np.sort(indices, axis=1))
    np.testing.assert_array_equal(coupled, m.couple_indices(indices, draws, z, days))
    for i in range(2):
        assert np.all(np.diff(z[coupled[i, np.argsort(draws[i] @ m.EF)]]) >= 0)


def test_coupling_ties_are_reproducible_and_do_not_drop_days() -> None:
    indices = np.arange(128)[None] % 7
    draws = np.full((1, 128, 8), 1/8)
    z = np.zeros(7)
    days = pd.DatetimeIndex(["2026-04-01"], tz="UTC")
    coupled = m.couple_indices(indices, draws, z, days)
    np.testing.assert_array_equal(np.sort(indices), np.sort(coupled))
    np.testing.assert_array_equal(coupled, m.couple_indices(indices, draws, z, days))


def test_conditional_cdf_never_selects_zero_mass_tail() -> None:
    weights = np.array([[.1, .8999999999999998, 0., 0.]])
    uniforms = np.array([[0., .5, np.nextafter(1., 0.), 1.]])
    np.testing.assert_array_equal(m.cdf_indices(weights, uniforms), [[0, 1, 1, 1]])
    with pytest.raises(ValueError):
        m.cdf_indices(weights, np.full((1, 1), 1.01))


def test_factor_decomposition_removes_coarse_error_exactly() -> None:
    rng = np.random.default_rng(10)
    mix, true_mix = rng.dirichlet(np.ones(8), size=2)
    qbar = rng.random((24, 8))+.1
    truth = rng.uniform(200, 600, size=24)
    center = recompose_paths(mix, qbar)
    oracle = recompose_paths(true_mix, qbar)
    errors, fine = (truth-center)[None], (truth-oracle)[None]
    result = m.factor_paths(mix[None], np.tile(true_mix, (1, 128, 1)), qbar[None],
        errors, fine, np.array([(true_mix-mix) @ m.EF]), np.zeros((1, 128), dtype=int), physical=True)
    np.testing.assert_allclose(result, np.broadcast_to(truth, (1, 128, 24)), atol=1e-12)
    np.testing.assert_allclose(errors[0]-fine[0], oracle-center)


def test_physical_curve_is_not_constant_ef_level_shift() -> None:
    rng = np.random.default_rng(11)
    mix = np.full((1, 8), 1/8)
    draws = rng.dirichlet(np.ones(8), size=(1, 128))
    q = rng.random((1, 24, 8))+.1
    ids = np.zeros((1, 128), dtype=int)
    errors, fine, z = np.zeros((1, 24)), np.zeros((1, 24)), np.zeros(1)
    physical = m.factor_paths(mix, draws, q, errors, fine, z, ids, physical=True)
    level = m.factor_paths(mix, draws, q, errors, fine, z, ids, physical=False)
    assert np.max(np.abs(physical-level)) > 1
    assert np.max(np.ptp(physical-level, axis=-1)) > 1


def test_response_cutoff_excludes_unreleased_december_supervision() -> None:
    context = object.__new__(m.Context)
    context.monthly = monthly()
    days = pd.DatetimeIndex(["2024-11-04", "2024-12-04"], tz="UTC")
    keep, rows = context._response_rows("BPAT", days)
    np.testing.assert_array_equal(keep, [True, False])
    assert rows.available_at.iloc[1] > m.CUTOFF


def test_nested_shape_registration_includes_teacher_donor_fold(monkeypatch: pytest.MonkeyPatch) -> None:
    requested = []
    context = object.__new__(m.Context)
    context.shapes = type("FakeCache", (), {"request": lambda self, *args: requested.append(args)})()
    subset = ("BPAT", "CISO", "ERCO")
    context._register_profiles(subset, "FPL", "proper")
    assert (subset, "FPL", "proper") in requested
    assert (("CISO", "ERCO"), "BPAT", "proper") in requested
    assert (("BPAT", "ERCO"), "CISO", "proper") in requested
    with pytest.raises(ValueError):
        context._register_profiles(("BPAT",), "FPL", "proper")


@pytest.mark.parametrize("empty_region", [False, True])
def test_location_scale_held_region_cannot_change_its_scale_training_responses(
        monkeypatch: pytest.MonkeyPatch, empty_region: bool) -> None:
    class MeanModel:
        def fit(self, x, y, regions, **kwargs):
            self.mean = np.asarray(y).mean(0)
            return self

        def predict(self, x):
            return np.broadcast_to(self.mean, (len(x), len(self.mean))).copy()

    class Scale(MeanModel):
        def fit(self, x, y, regions, **kwargs):
            self.mean = np.repeat(np.sqrt(np.mean(np.asarray(y)**2))+1, 2)
            return self

    monkeypatch.setattr(m, "Regression", MeanModel)
    monkeypatch.setattr(m, "ScaleModel", Scale)
    monkeypatch.setattr(m, "JointQuantileForest", MeanModel)
    subset = ("BPAT", "CISO", "ERCO", "FPL") + (("ISNE",) if empty_region else ())
    days = pd.date_range("2024-07-01", periods=2, tz="UTC")

    def run(offset):
        c = object.__new__(m.Context)
        c.risks, c.records = {}, {}
        for i, region in enumerate(subset):
            c.records[region, "calibration"] = dict(days=days, truth=np.full((2, 24), 100+i*30+(offset if i == 0 else 0)),
                features={"B": np.full((2, 8), 1/8)})
        c._profiles = lambda *args: (np.ones((2, 24, 8)), np.ones((2, 24, 8)))
        law = dict(draws=np.full((2, 128, 8), 1/8), metadata=np.ones((2, 4)))
        c._law = lambda *args: law
        rows = pd.DataFrame(np.full((2, 8), 1/8), columns=m.COLUMNS)
        rows["available_at"] = pd.Timestamp("2024-08-31", tz="UTC")
        rows["period_end"] = pd.Timestamp("2024-08-01", tz="UTC")
        c._response_rows = lambda region, dates: (np.full(2, region != "ISNE", dtype=bool), rows)
        cal = c._calibration(subset, "B", "fixed_donor")
        assert set(cal["regions"]) == {"BPAT", "CISO", "ERCO", "FPL"}
        assert len(cal["errors"]) == 8
        return cal["location_scales"][cal["regions"] == "BPAT"]

    np.testing.assert_array_equal(run(0), run(500))
