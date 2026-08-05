"""Physics decomposition layer (Theorem 1 / Theorem 2).

The core identity: CIF = rs * ef_r + (1 - rs) * ef_nr when expressed in
gCO2/kWh, where rs is renewable share, ef_r / ef_nr are the renewable and
non-renewable emission factors.  This closed-form reconstruction is what makes
zero-shot transfer possible using only two scalar region configs.
"""

import numpy as np

import torch


def cif_from_shares(rs, ef_r, ef_nr):
    """Closed-form CIF reconstruction from renewable share.

    Args:
        rs    : renewable share in [0, 1] (scalar or array).  Accepts a numpy
                array or a torch tensor (including one that requires grad —
                it is detached internally so this is safe inside training loops).
        ef_r  : renewable emission factor (tCO2/MWh)
        ef_nr : non-renewable emission factor (tCO2/MWh)

    Returns CIF in gCO2/kWh as a numpy array.  In gCO2/kWh the identity is
    ``rs*ef_r + (1-rs)*ef_nr`` (1 tCO2/MWh == 1000 gCO2/kWh, factors cancel).
    """
    if isinstance(rs, torch.Tensor):
        rs = rs.detach().cpu().numpy()
    return np.asarray(rs) * ef_r + (1.0 - np.asarray(rs)) * ef_nr
