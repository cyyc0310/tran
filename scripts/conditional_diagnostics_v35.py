"""Zero-fit V35 empirical forecast scores, point summaries and Ridge replay.

Pure arrays only. Empirical mid-CDF is not a randomized uniform-rank claim;
the local particle median is not the complete conditional CIF median.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from scripts import coarse_context_v30 as v30


RIDGE_WIDTHS = {'Size-Ridge': 31, 'SizeCap-Ridge': 36,
                'IntensiveCap-Ridge': 35, 'DirectEF-Ridge': 36}


def _array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        if np.iscomplexobj(value):
            raise ValueError('Real finite arrays required')
        array = np.asarray(value, dtype=float)
    except (TypeError, OverflowError) as exc:
        raise ValueError('Real finite arrays required') from exc
    if (shape is not None and array.shape != shape) or not np.isfinite(array).all():
        raise ValueError('Finite arrays with the required shape needed')
    return array


def _simplex(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = _array(value, shape)
    if np.any(array < 0) or not np.allclose(array.sum(-1), 1., atol=1e-12, rtol=0):
        raise ValueError('Finite nonnegative G5 simplex required; no renormalization')
    return array


def forecast_case(draws: np.ndarray, truth_g5: np.ndarray, past_g5: np.ndarray) -> dict[str, Any]:
    """Score 128 G5 particles using empirical V-statistic CRPS/energy scores.

    Intervals include their endpoints; std uses ddof=0 and ties use exact equality.
    The last legally available past G5 is supplied and time-checked by the runner.
    """
    ensemble = _simplex(draws, (128, 5))
    truth = _simplex(truth_g5, (5,))
    past = _simplex(past_g5, (5,))
    pos, y = ensemble[:, 0], truth[0]
    mean, median = float(pos.mean()), float(np.median(pos))
    q05, q10, q50, q90, q95 = np.quantile(pos, [.05, .1, .5, .9, .95], method='linear')
    low, high = float(pos.min()), float(pos.max())
    less, equal = int(np.count_nonzero(pos < y)), int(np.count_nonzero(pos == y))
    crps = np.abs(pos-y).mean() - .5*np.abs(pos[:, None]-pos[None, :]).mean()
    energy = (np.linalg.norm(ensemble-truth, axis=1).mean()
              - .5*np.linalg.norm(ensemble[:, None, :]-ensemble[None, :, :], axis=2).mean())
    return dict(pos_mean=mean, pos_median=median, pos_std=float(pos.std(ddof=0)),
        pos_mean_bias=float(mean-y), pos_mean_abs_error=float(abs(mean-y)),
        pos_median_bias=float(median-y), pos_median_abs_error=float(abs(median-y)),
        pos_q05=float(q05), pos_q10=float(q10), pos_q50=float(q50), pos_q90=float(q90), pos_q95=float(q95),
        pos_coverage80=bool(q10 <= y <= q90), pos_coverage90=bool(q05 <= y <= q95),
        pos_min=low, pos_max=high, pos_minmax_contains=bool(low <= y <= high),
        pos_less_count=less, pos_equal_count=equal, pos_mid_cdf=(less+.5*equal)/128,
        pos_crps=float(crps), g5_energy=float(energy), past_pos_abs_error=float(abs(past[0]-y)),
        past_g5_euclidean_error=float(np.linalg.norm(past-truth)))


def point_case(cif_draws: np.ndarray, ef_draws: np.ndarray,
               truth_cif: float, truth_ef: float) -> dict[str, dict[str, float]]:
    """Compare local ensemble mean/median points under absolute and squared loss."""
    cif, ef = _array(cif_draws, (128,)), _array(ef_draws, (128,))
    truth = np.array([float(_array(truth_cif, ())), float(_array(truth_ef, ()))])
    result = {}
    try:
        with np.errstate(over='raise', invalid='raise'):
            for mode, summarize in (('mean', np.mean), ('median', np.median)):
                points = np.array([summarize(cif), summarize(ef)])
                bias = points-truth
                squared = bias*bias
                if not np.isfinite(points).all() or not np.isfinite(squared).all():
                    raise ValueError('Finite point summaries and errors required')
                result[mode] = dict(cif_point=float(points[0]), ef_point=float(points[1]),
                    cif_bias=float(bias[0]), cif_abs_error=float(abs(bias[0])), cif_sq_error=float(squared[0]),
                    ef_bias=float(bias[1]), ef_abs_error=float(abs(bias[1])), ef_sq_error=float(squared[1]))
    except FloatingPointError as exc:
        raise ValueError('Finite point summaries and errors required') from exc
    return result


def ridge_ef(fit: Mapping[str, Any], features: np.ndarray) -> np.ndarray:
    """Replay fixed coefficients/scaler, then the original ratio/scalar projection.

    The runner checks feature-name order and sealed provenance. This function
    checks the four allowed head identities, dimensions and finite values only;
    it never estimates coefficients, recomputes a scaler or fits a model.
    """
    try:
        arm = fit['arm']
        if (arm not in RIDGE_WIDTHS or fit['model'] != 'Ridge'
                or fit['family'] != arm.removesuffix('-Ridge')):
            raise ValueError('One of the four registered frozen Ridge heads required')
        width = RIDGE_WIDTHS[arm]
        x = _array(features)
        if x.ndim < 1 or x.shape[-1] != width or not x.size:
            raise ValueError('Nonempty correctly sized feature arrays required')
        mean, scale = _array(fit['scaler_mean'], (width,)), _array(fit['scaler_scale'], (width,))
        if np.any(scale <= 0):
            raise ValueError('Strictly positive frozen scaler required')
        scalar = arm == 'DirectEF-Ridge'
        coef = _array(fit['coefficients'], (width,) if scalar else (4, width))
        intercept = _array(fit['intercept'], () if scalar else (4,))
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            raw = ((x-mean)/scale) @ coef.T + intercept
        if not np.isfinite(raw).all():
            raise ValueError('Finite frozen Ridge predictions required')
        ef = np.clip(raw, .4, 1.)*1000 if scalar else v30.project(raw) @ v30.EF[v30.POS]
        return np.asarray(ef, dtype=float)
    except KeyError as exc:
        raise ValueError('Missing frozen Ridge field: '+str(exc)) from exc
    except FloatingPointError as exc:
        raise ValueError('Finite frozen Ridge predictions required') from exc
