"""Tests for transcif.physics decomposition and bounds."""

import numpy as np

from transcif.physics.decompose import cif_from_shares
from transcif.physics.bounds import (
    cif_identity, validate_identity, config_distance, compute_weighted_config_distance,
)


def test_cif_from_shares_identity():
    # At rs=1 only renewable contribution; at rs=0 only non-renewable.
    assert cif_from_shares(1.0, 0.0, 800.0) == 0.0
    assert cif_from_shares(0.0, 0.0, 800.0) == 800.0
    assert cif_from_shares(0.5, 0.0, 800.0) == 400.0


def test_validate_identity_zero_error():
    rs = np.linspace(0.1, 0.9, 100)
    ef_r, ef_nr = 0.0, 800.0
    cif = cif_identity(rs, ef_r, ef_nr)
    max_err, mean_err, frac = validate_identity(rs, cif, ef_r, ef_nr, verbose=False)
    assert max_err < 1e-4
    assert mean_err < 1e-4
    assert frac == 1.0


def test_config_distance_symmetric():
    a = [0.3, 0.8]
    b = [0.5, 0.5]
    assert config_distance(a, b) == config_distance(b, a)


def test_weighted_config_distance():
    sources = [[0.3, 0.8], [0.4, 0.7]]
    target = [0.35, 0.75]
    d = compute_weighted_config_distance(sources, target)
    assert d > 0
    assert d < config_distance([0.3, 0.8], target) + 1
