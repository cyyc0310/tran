import itertools
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from transcif.physics.envelope import dispatch_cif_envelope


def test_same_public_capacity_and_supply_allow_opposite_cif_extremes():
    result = dispatch_cif_envelope([100, 100], 100, [0, 1000])
    assert result.lower == 0 and result.upper == 1000
    assert result.midpoint == 500 and result.radius == 500
    np.testing.assert_equal(result.lower_generation, [100, 0])
    np.testing.assert_equal(result.upper_generation, [0, 100])
    # A learned point can improve average error with extra source assumptions;
    # it cannot eliminate this capacity-only worst-case ambiguity.
    for point in [-20, 0, 400, 500, 1000, 1200]:
        assert max(abs(point - result.lower), abs(point - result.upper)) >= result.radius


def test_extremizers_match_enumerated_dispatches_with_minimums_and_ef_intervals():
    capacity, minimum, total = np.array([2, 4, 3]), np.array([1, 0, 1]), 6
    low, high = np.array([0, 400, 800]), np.array([20, 500, 1200])
    result = dispatch_cif_envelope(capacity, total, low, high, minimum)
    feasible = np.array([g for g in itertools.product(range(3), range(5), range(4))
                         if sum(g) == total and np.all(g >= minimum)])
    assert result.lower == pytest.approx((feasible @ low / total).min())
    assert result.upper == pytest.approx((feasible @ high / total).max())
    for witness, ef, bound in [(result.lower_generation, low, result.lower),
                                (result.upper_generation, high, result.upper)]:
        assert witness.sum() == total and (witness >= minimum).all() and (witness <= capacity).all()
        assert witness @ ef / total == pytest.approx(bound)


def test_arbitrary_fuel_count_batches_units_and_zero_capacity():
    capacity = np.array([[0, 2, 0, 3, 2, 1, 2, 1], [1, 2, 3, 0, 2, 4, 1, 2]])
    total = np.array([5, 8])
    ef = np.arange(8) * 100
    result = dispatch_cif_envelope(capacity, total, ef)
    scaled = dispatch_cif_envelope(capacity * 1e6, total * 1e6, ef)
    np.testing.assert_allclose(result.lower, scaled.lower)
    np.testing.assert_allclose(result.upper, scaled.upper)
    for witness in [result.lower_generation, result.upper_generation]:
        np.testing.assert_allclose(witness.sum(-1), total)
        assert np.all(witness <= capacity) and np.all(witness >= 0)
    singleton = dispatch_cif_envelope([0, 3], 2, [1000, 50])
    assert singleton.lower == singleton.upper == 50


@pytest.mark.parametrize("capacity,total,low,high,minimum", [
    ([1, 1], 3, [0, 1000], None, None),
    ([1, 1], 1, [0, 1000], None, [1, 1]),
    ([1, -1], 1, [0, 1000], None, None),
    ([1, 1], 0, [0, 1000], None, None),
    ([1, 1], 1, [0, np.nan], None, None),
    ([1, 1], 1, [0, 1000], [0, 900], None),
    ([1, 1], 1, [-10, 1000], None, None),
    ([1, 1], [1], [0, 1000], None, None),
    ([1, 1], 1, [0, 500, 1000], None, None),
    ([], 1, [], None, None),
])
def test_ambiguous_or_infeasible_accounting_is_rejected(capacity, total, low, high, minimum):
    with pytest.raises(ValueError):
        dispatch_cif_envelope(capacity, total, low, high, minimum)


def test_public_diagnostic_does_not_read_or_hash_labels_and_marks_shortage(bundle):
    manifest, doc, grids = bundle
    spec = importlib.util.spec_from_file_location("diagnose_public_envelope",
        Path(__file__).resolve().parents[1] / "scripts" / "diagnose_public_envelope.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    target = grids[doc["target"]]
    target.labels_path.unlink()
    result = module.diagnose(manifest, doc["origins"][0])
    assert len(result["input_sha256"]) == 3
    assert str(target.labels_path.resolve()) not in result["input_sha256"]
    assert result["mean_conditional_width"] >= 0
    forecasts_path = manifest.parent / next(r for r in doc["regions"] if r["name"] == target.name)["forecasts"]
    import pandas as pd
    forecasts = pd.read_csv(forecasts_path)
    forecasts["demand_forecast_mw"] = 1e12
    forecasts.to_csv(forecasts_path, index=False)
    result = module.diagnose(manifest, doc["origins"][0])
    assert result["infeasible_member_hours"] == 4 * doc["horizon"]
    assert result["mean_conditional_width"] is None
    assert all(x is None for row in result["lower"] for x in row)
