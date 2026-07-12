"""Stage 2: CIF physics formula (Zhang et al., AAAI-26, Eq. CIF_avg,t) reduced to a
two-category (renewable / non-renewable) aggregation, matching the RenewShare
prediction target from Stage 1."""

import numpy as np

# NOTE: placeholder figures (gCO2/kWh) for pipeline development and testing only.
# MUST be replaced with sourced values (IPCC / EIA / ElectricityMaps) before running
# real experiments — see the data-acquisition follow-up plan referenced in the design doc.
EMISSION_FACTOR_TABLES = {
    "AU_NSW": {"renewable": 24.0, "nonrenewable": 850.0},
    "AU_SA": {"renewable": 30.0, "nonrenewable": 700.0},
    "EU_ES": {"renewable": 35.0, "nonrenewable": 550.0},
    "EU_DE": {"renewable": 30.0, "nonrenewable": 750.0},
    "US_EPE": {"renewable": 40.0, "nonrenewable": 490.0},
}


def cif_from_shares(renew_share: np.ndarray, renew_factor: float, nonrenew_factor: float) -> np.ndarray:
    """CIF_avg,t = RenewShare_t * C_renew + (1 - RenewShare_t) * C_nonrenew."""
    nonrenew_share = 1 - renew_share
    return renew_share * renew_factor + nonrenew_share * nonrenew_factor


def get_emission_factors(region_code: str) -> tuple:
    """Look up (renewable_factor, nonrenewable_factor) for a region code. Raises KeyError
    for unknown region codes rather than silently defaulting."""
    table = EMISSION_FACTOR_TABLES[region_code]
    return table["renewable"], table["nonrenewable"]
