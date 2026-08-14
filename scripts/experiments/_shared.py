"""Shared helpers for the experiments orchestrator scripts.

Previously the origin-splitting and direction-predictor builders were
copy-pasted between ``run_joint_train_full`` / ``run_fused_five_full`` /
``run_fused_five_variants`` / ``probe_calibration_curve``; they now live here
as the single source of truth. Import as::

    from scripts.experiments._shared import split_origins, zs_plus_origins
"""

import numpy as np

from transcif.config import HORIZON, TEST_STRIDE, TRAIN_FRACTION


def zs_plus_origins(rs, cif=None):
    """Compute ZS+ origins aligned with ``build_windows`` TEST output.

    The split point is derived from ``rs`` while the range bound uses ``cif``
    (identical when both come from the same region dict, as they always do).
    """
    split = int(len(rs) * TRAIN_FRACTION)
    n = len(cif) if cif is not None else len(rs)
    return [split + st for st in range(0, n - split - HORIZON + 1, TEST_STRIDE)]


def split_origins(rs: np.ndarray, n_train: int = 12, n_eval: int = 12):
    """Get disjoint train + eval origin lists from the test split."""
    all_origins = zs_plus_origins(rs)
    if len(all_origins) < n_train + n_eval:
        # Fall back to fewer if series is short
        n_train = max(2, len(all_origins) // 2)
        n_eval = len(all_origins) - n_train
    return all_origins[:n_train], all_origins[n_train:n_train + n_eval]


def build_direction_predictors(regions, target, seed, device):
    """Train the 5 direction models for one target; return predictor dict.

    Args:
        regions: Small dict (target + a few donor regions). Train functions
            iterate ``regions.items()`` for the auxiliary/donor pool, so
            passing the full 29-region dict here makes each train call ~24x
            slower (~80s vs ~3.3s with 4 regions). See commit history for the
            profiling that isolated this bottleneck.
        target: Target region name.
        seed: Random seed.
        device: Torch device (or None for CPU).
    """
    predictors = {}
    from transcif.models.zeroshot.rag import train_rag_zero_shot, predict_rag_zs
    from transcif.models.zeroshot.phys_irm import train_phys_irm, predict_phys_irm
    from transcif.models.zeroshot.causal import train_causal_zero_shot, predict_causal_zs
    from transcif.models.zeroshot.icl import train_icl, predict_icl_zs
    from transcif.models.zeroshot.hier import train_hier, predict_hier_zs

    m, bank = train_rag_zero_shot(regions, target, seed=seed, device=device)
    predictors["rag"] = lambda x, cfg, ef_r, ef_nr, m=m, b=bank: predict_rag_zs(
        m, b, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m, _ = train_phys_irm(regions, target, seed=seed, gamma_irm=0.1,
                          lambda_cif=0.5, device=device)
    predictors["phys"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_phys_irm(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m, _ = train_causal_zero_shot(regions, target, seed=seed, device=device)
    predictors["causal"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_causal_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m = train_icl(regions, target, seed=seed, device=device)
    predictors["icl"] = lambda x, cfg, ef_r, ef_nr, m=m, r=regions, t=target: (
        predict_icl_zs(m, r, t, x.astype(np.float32), ef_r, ef_nr)
    )
    m = train_hier(regions, target, seed=seed, device=device)
    predictors["hier"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_hier_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)

    return predictors
