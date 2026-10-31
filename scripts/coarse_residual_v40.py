"""Pure coarse-history feature means and ordinary six-source residual calibrators."""
from __future__ import annotations

from typing import Any

import numpy as np


ARMS = ('EnsembleNone', 'ResidualMean', 'ResidualD', 'ResidualS', 'ResidualDS')
FEATURE_NAMES = ('D', 'S')
SELECTED_COLUMNS = ((), (0,), (1,), (0, 1))
EF = np.array([1000., 400., 800., 400.])
ALPHA = 1.


def _real(value: Any, name: str, shape: tuple[int, ...] | None = None,
          nonnegative: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f'Real {name} required')
    try:
        array = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'Real numeric {name} required') from exc
    if ((shape is not None and array.shape != shape) or not np.isfinite(array).all()
            or (nonnegative and np.any(array < 0))):
        raise ValueError(f'Finite correctly shaped {name} required')
    return array


def features(C: np.ndarray, POS_MW: np.ndarray, U: np.ndarray) -> dict[str, Any]:
    """Mean coarse residual D and predicted carbon structure S; no fine labels."""
    C = _real(C, 'capacities', nonnegative=True)
    if C.ndim != 2 or C.shape[1] != 4 or not len(C):
        raise ValueError('Nonempty (N,4) capacities required')
    n = len(C)
    POS_MW = _real(POS_MW, 'coarse POS MW', (n,), nonnegative=True)
    U = _real(U, 'source profile rates', (n, 4), nonnegative=True)
    total = _real(C.sum(1), 'total POS capacity')
    if np.any(total <= 0):
        raise ValueError('Strictly positive total POS capacity required')
    q = C/total[:, None]
    xhat = _real(q*U, 'predicted contributions')
    pred_mass = _real(xhat.sum(1), 'predicted POS contribution mass')
    if np.any(pred_mass <= 0):
        raise ValueError('Strictly positive predicted contribution mass required')
    z = _real(POS_MW/total, 'coarse POS contribution')
    d = _real(z-pred_mass, 'coarse contribution residual')
    ef_pred = _real((xhat/pred_mass[:, None])@EF, 'profile EF')
    x = _real(np.array([d.mean(), ((ef_pred-400.)/600.).mean()]), 'mean coarse features', (2,))
    return dict(X=x, audit=dict(q=q, xhat=xhat, z=z, d=d, EFpred=ef_pred))


def fit_predict(X: np.ndarray, bias_g: np.ndarray, queryX: np.ndarray,
                baseEF: np.ndarray) -> dict[str, Any]:
    """SUM-ridge with alpha=1 slopes and unpenalized intercept; clip then ensemble."""
    X = _real(X, 'six-source features', (6, 2))
    bias_g = _real(bias_g, 'true-minus-predicted EF biases', (6,))
    queryX = _real(queryX, 'query component features', (6, 6, 2))
    baseEF = _real(baseEF, 'query component baseline EF', (6, 6))
    y = bias_g/1000.
    raw = [baseEF.copy()]
    fits = []
    for arm, ids in zip(ARMS[1:], SELECTED_COLUMNS):
        selected = X[:, ids]
        constant = np.all(selected == selected[:1], axis=0)
        variable = ~constant
        mean, scale = selected[0].copy(), np.ones(len(ids))
        mean[variable] = selected[:, variable].mean(0)
        scale[variable] = selected[:, variable].std(0, ddof=0)
        _real(mean, 'source feature means'); _real(scale, 'source population standard deviations')
        if np.any(scale <= 0):
            raise ValueError('Nonconstant source columns require positive finite standard deviation')
        standardized = np.zeros_like(selected)
        query = np.zeros((6, 6, len(ids)))
        standardized[:, variable] = (selected[:, variable]-mean[variable])/scale[variable]
        query[..., variable] = (queryX[..., ids][..., variable]-mean[variable])/scale[variable]
        _real(standardized, 'standardized source features'); _real(query, 'standardized query features')
        Z = np.c_[np.ones(6), standardized]
        query_Z = np.concatenate([np.ones((6, 6, 1)), query], axis=2)
        penalty = np.diag(np.r_[0., np.full(len(ids), ALPHA)])
        normal_matrix = _real(Z.T@Z+penalty, 'normal matrix')
        rhs = _real(Z.T@y, 'normal response')
        theta = _real(np.linalg.solve(normal_matrix, rhs), 'calibration parameters')
        normal_residual = _real(normal_matrix@theta-rhs, 'normal equation residual')
        fitted_residual = _real(Z@theta-y, 'source response residual')
        correction = _real(query_Z@theta, 'query normalized EF correction')
        raw.append(_real(baseEF+1000.*correction, 'raw corrected component EF'))
        squared_error = float(_real(fitted_residual@fitted_residual, 'source squared error'))
        penalty_value = float(_real(ALPHA*(theta[1:]@theta[1:]), 'slope penalty'))
        fits.append(dict(arm=arm, selected_columns=ids,
            selected_feature_names=tuple(FEATURE_NAMES[i] for i in ids),
            scaler_mean=mean.copy(), scaler_scale=scale.copy(), constant_mask=constant.copy(),
            Z=Z, y=y.copy(), theta=theta, normal_equation_residual=normal_residual,
            normal_equation_residual_l2=float(np.linalg.norm(normal_residual)),
            objective_sum_squared_error=squared_error, objective_penalty=penalty_value,
            objective_penalized=float(_real(squared_error+penalty_value, 'penalized objective')),
            alpha=ALPHA, loss_reduction='sum', penalize_intercept=False))
    raw_component = np.stack(raw)
    component = np.clip(raw_component, 400., 1000.)
    return dict(raw_component_EF=raw_component, component_EF=component,
                clipmask=raw_component != component, ensemble_EF=component.mean(2),
                audit=dict(arms=ARMS, fits=fits, feature_names=FEATURE_NAMES,
                           response='source late EF_true_minus_pred / 1000'))
