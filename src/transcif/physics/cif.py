"""Stage 2: CIF physics formula (Zhang et al., AAAI-26, Eq. CIF_avg,t) reduced to a
two-category (renewable / non-renewable) aggregation, matching the RenewShare
prediction target from Stage 1."""

import numpy as np

# AU_* figures (gCO2/kWh) are sourced from real 2023 AEMO/NEM dispatch data via NEMED
# (generator-level Plant_Emissions_Intensity, generation-weighted per region, threshold
# 0.02 tCO2/MWh for the renewable/non-renewable split). EU_*/US_* remain placeholder
# figures pending the same real-data treatment — do not use them for reported results.
EMISSION_FACTOR_TABLES = {
    "AU_NSW": {"renewable": 0.09, "nonrenewable": 875.23},
    "AU_SA": {"renewable": 0.00, "nonrenewable": 490.43},
    "AU_QLD": {"renewable": 0.00, "nonrenewable": 841.59},
    "AU_VIC": {"renewable": 0.00, "nonrenewable": 1160.12},
    "EU_ES": {"renewable": 35.0, "nonrenewable": 550.0},
    "EU_DE": {"renewable": 30.0, "nonrenewable": 750.0},
    "US_EPE": {"renewable": 40.0, "nonrenewable": 490.0},
}


def cif_from_shares(renew_share: np.ndarray, renew_factor: float, nonrenew_factor: float) -> np.ndarray:
    """CIF_avg,t = RenewShare_t * C_renew + (1 - RenewShare_t) * C_nonrenew."""
    nonrenew_share = 1 - renew_share
    return renew_share * renew_factor + nonrenew_share * nonrenew_factor


def get_emission_factors(region_code: str) -> tuple[float, float]:
    """Look up (renewable_factor, nonrenewable_factor) for a region code. Raises KeyError
    for unknown region codes rather than silently defaulting."""
    table = EMISSION_FACTOR_TABLES[region_code]
    return table["renewable"], table["nonrenewable"]
