"""Theorem 1 / Theorem 2 bound formulas, as reusable functions.

These are the mathematical cores extracted from the verification scripts
(``scripts/verify/theorem1_physics_bound.py`` and ``theorem2_transfer_bound.py``).
The CLI scripts remain as thin wrappers that call into this module.
"""

import numpy as np


def cif_identity(rs, ef_r, ef_nr):
    """Theorem 1 closed-form: CIF = rs*ef_r + (1-rs)*ef_nr (gCO2/kWh)."""
    return np.asarray(rs) * ef_r + (1.0 - np.asarray(rs)) * ef_nr


def validate_identity(rs_seq, cif_seq, ef_r, ef_nr, verbose=True):
    """Theorem 1 validation: confirm the identity matches observed CIF.

    Returns ``(max_abs_error, mean_abs_error, valid_fraction)`` over all hours
    where rs is in (0.05, 0.95).  Max error is expected < 0.05 gCO2/kWh.
    """
    rs = np.asarray(rs_seq)
    cif = np.asarray(cif_seq)
    mask = (rs > 0.05) & (rs < 0.95) & np.isfinite(cif)
    cif_pred = cif_identity(rs[mask], ef_r, ef_nr)
    err = np.abs(cif_pred - cif[mask])
    if verbose:
        print(f"[Theorem1] n_valid={mask.sum():,}  "
              f"max_err={err.max():.4f}  mean_err={err.mean():.4f}  "
              f"valid_frac={mask.mean():.3f}")
    return float(err.max()), float(err.mean()), float(mask.mean())


def config_distance(config_a, config_b, weights=(1.0, 1.0)):
    """Weighted L1 distance between two region config vectors.

    Config vector is ``[mean_rs, ef_nr/1000]``.  Used by Theorem 2 to quantify
    source/target dissimilarity that drives zero-shot transfer difficulty.
    """
    wa, wb = weights
    return wa * abs(config_a[0] - config_b[0]) + wb * abs(config_a[1] - config_b[1])


def compute_weighted_config_distance(source_configs, target_config):
    """Mean weighted config distance from a target to a set of sources."""
    return float(np.mean([config_distance(s, target_config) for s in source_configs]))
