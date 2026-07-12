import numpy as np
import pytest
from transcif.physics.cif import cif_from_shares, get_emission_factors, EMISSION_FACTOR_TABLES


def test_cif_from_shares_full_renewable():
    result = cif_from_shares(np.array([1.0]), renew_factor=24.0, nonrenew_factor=850.0)
    np.testing.assert_allclose(result, [24.0])


def test_cif_from_shares_full_nonrenewable():
    result = cif_from_shares(np.array([0.0]), renew_factor=24.0, nonrenew_factor=850.0)
    np.testing.assert_allclose(result, [850.0])


def test_cif_from_shares_half_and_half():
    result = cif_from_shares(np.array([0.5]), renew_factor=20.0, nonrenew_factor=800.0)
    np.testing.assert_allclose(result, [410.0])


def test_get_emission_factors_known_region():
    renewable, nonrenewable = get_emission_factors("AU_NSW")
    assert renewable == EMISSION_FACTOR_TABLES["AU_NSW"]["renewable"]
    assert nonrenewable == EMISSION_FACTOR_TABLES["AU_NSW"]["nonrenewable"]


def test_get_emission_factors_unknown_region_raises():
    with pytest.raises(KeyError):
        get_emission_factors("UNKNOWN_REGION")
