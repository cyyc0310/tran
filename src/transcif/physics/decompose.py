"""Physics decomposition layer (Theorem 1 / Theorem 2).

The core identity: CIF = rs * ef_r + (1 - rs) * ef_nr when expressed in
gCO2/kWh, where rs is renewable share, ef_r / ef_nr are the renewable and
non-renewable emission factors.  This closed-form reconstruction is what makes
zero-shot transfer possible using only two scalar region configs.
"""

import numpy as np


def cif_from_shares(rs, ef_r, ef_nr):
    """Closed-form CIF reconstruction from renewable share.

    Args:
        rs    : renewable share in [0, 1] (scalar or array)
        ef_r  : renewable emission factor (tCO2/MWh)
        ef_nr : non-renewable emission factor (tCO2/MWh)

    Returns CIF in gCO2/kWh.  In gCO2/kWh the identity is
    ``rs*ef_r + (1-rs)*ef_nr`` (1 tCO2/MWh == 1000 gCO2/kWh, factors cancel).
    """
    return np.asarray(rs) * ef_r + (1.0 - np.asarray(rs)) * ef_nr
