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
        ef_r  : renewable emission factor (tCO2/MWh).  Accepts a scalar, numpy
                array, or torch tensor; converted to numpy internally.
        ef_nr : non-renewable emission factor (tCO2/MWh).  Same as ef_r.

    Returns CIF in gCO2/kWh as a numpy array.  In gCO2/kWh the identity is
    ``rs*ef_r + (1-rs)*ef_nr`` (1 tCO2/MWh == 1000 gCO2/kWh, factors cancel).
    """
    if isinstance(rs, torch.Tensor):
        rs = rs.detach().cpu().numpy()
    if isinstance(ef_r, torch.Tensor):
        ef_r = ef_r.detach().cpu().numpy()
    if isinstance(ef_nr, torch.Tensor):
        ef_nr = ef_nr.detach().cpu().numpy()
    rs = np.asarray(rs, dtype=np.float64)
    return rs * ef_r + (1.0 - rs) * ef_nr


def cif_from_fuel_shares(fuel_shares, fuel_efs):
    """Multi-fuel CIF reconstruction: CIF = Σ_f share_f × ef_f.

    This is the N-fuel generalisation of ``cif_from_shares``.  When the fuel
    breakdown is collapsed to two buckets (renewable vs non-renewable) it
    reduces exactly to the 2-class identity.

    Args:
        fuel_shares : dict {fuel_key: share in [0,1]} or array of shares.
                      Shares should sum to ~1.0; they are NOT renormalised here
                      so callers control the convention.
        fuel_efs    : dict {fuel_key: emission factor in gCO2/kWh} or array.
                      gCO2/kWh is used (not tCO2/MWh) so the result is directly
                      comparable to ``cif_from_shares`` output.  Since
                      1 tCO2/MWh == 1000 gCO2/kWh the unit factors cancel.

    Returns CIF in gCO2/kWh as a float.
    """
    if isinstance(fuel_shares, dict):
        keys = list(fuel_shares.keys())
        s = np.array([fuel_shares[k] for k in keys], dtype=np.float64)
        e = np.array([fuel_efs[k] for k in keys], dtype=np.float64)
    else:
        s = np.asarray(fuel_shares, dtype=np.float64)
        e = np.asarray(fuel_efs, dtype=np.float64)
    return float(np.dot(s, e))
