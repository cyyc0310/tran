import numpy as np
import pytest

from transcif.research.stress_cases import StressCase
from transcif.research.toy import ToyMechanism


@pytest.mark.parametrize("case", ["harmful", "harmless", "resolved"])
def test_stress_reduces_to_original_model(case):
    model = StressCase(case, harmless=case == "harmless", resolved=case == "resolved")
    z = np.random.default_rng(189).normal(size=(200, 2))
    expected = ToyMechanism(case).evaluate(z)
    result = model.evaluate(z)
    np.testing.assert_allclose(result[0], expected[0])
    np.testing.assert_allclose(result[1], expected[1], atol=1e-10)


def test_rotated_nuisance_coordinates_are_invariant():
    model = StressCase("rotation", dimension=6, rotation_seed=18, curve=.5)
    active = np.random.default_rng(1).normal(size=(10, 2))
    q = np.linalg.qr(np.column_stack([model.axes, np.eye(6)]))[0]
    z = active @ model.axes.T
    displaced = z + np.random.default_rng(2).normal(size=(10, 4)) @ q[:, 2:].T
    np.testing.assert_allclose(model.evaluate(z)[0], model.evaluate(displaced)[0], atol=1e-10)
    np.testing.assert_allclose(model.evaluate(z)[1], model.evaluate(displaced)[1], atol=1e-10)


def test_reference_recovers_known_symmetric_mode_mass():
    reference = StressCase("symmetric", curve=0).reference(points=1201, secondary_points=81)
    assert reference["weights"][reference["v"] > 0].sum() == pytest.approx(.5, abs=1e-10)
    asymmetric = StressCase("shift", shift=.65).reference(points=1201, secondary_points=81)
    assert asymmetric["weights"][asymmetric["v"] > 0].sum() > .8


def test_quadrature_change_of_coordinates_matches_prior_in_flat_limit():
    reference = StressCase("curved_flat", curve=.55, shift=.3, noise=1e10).reference(1601, 321)
    v, w, weights = reference["v"], reference["w"], reference["weights"]
    u = v - .3 - .55 * (w*w - 1)
    # The transformed box truncates a small part of the curved prior tail.
    assert weights @ u == pytest.approx(0, abs=.002)
    assert weights @ (u*u) == pytest.approx(1, abs=.004)
    assert weights @ (w*w) == pytest.approx(1, abs=.004)
