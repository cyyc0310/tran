"""Exact Cartesian coarse/whole-shape controls, retaining empirical within weights."""
from __future__ import annotations

import json

import numpy as np

from rebased_coarse_lift import RebasedCoarseLift
from v151_hourly_law import EF, Law
from v179_increment_support_diagnostic import project_simplex

DIAGONAL = 'rebased_increment_history_scalar8'
PRODUCT = ('coarse_shape_independent', 'coarse_shape_hellinger')
REFERENCES = ('world_gaussian', DIAGONAL)
METHODS = REFERENCES + PRODUCT


def product_laws(values, coarse_weights, shape_weights, cells):
    """Merge only identical coarse vectors; enumerate every remaining Cartesian atom."""
    values = np.asarray(values, float)
    p = np.asarray(coarse_weights, float)
    if (values.shape != (len(p), 8) or not np.isfinite(values).all() or (values < 0).any()
            or max(abs(values.sum(axis=1) - 1)) > 1e-12 or (p <= 0).any()
            or not np.isfinite(p).all() or abs(p.sum() - 1) > 1e-12):
        raise ValueError('invalid product coarse law')
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    pg = np.bincount(inverse, weights=p, minlength=len(unique))
    output = {name: [] for name in shape_weights}
    diagnostic = {name: {'rejected_mass': [], 'effective_coarse_weights': [], 'effective_shape_weights': [],
                          'fuel_mean': [], 'precondition_marginal_error': 0.} for name in shape_weights}
    for name, s in shape_weights.items():
        if np.shape(s) != np.shape(p) or not np.isfinite(s).all() or (s <= 0).any() or abs(np.sum(s) - 1) > 1e-12:
            raise ValueError('invalid product shape probabilities')
        error = max(float(max(abs(p * np.sum(s) - p))), float(max(abs(s * np.sum(p) - s))))
        diagnostic[name]['precondition_marginal_error'] = error
        if error > 1e-12:
            raise ValueError('product precondition marginals differ')
    for shape, component, within, _ in cells:
        n, a = len(unique), len(shape)
        numerator, total = np.empty((n, a)), np.empty((n, a))
        # Keep per-atom eight-fuel arithmetic, rather than projecting an EF anchor.
        for start in range(0, n, 8):
            fuel = (unique[start:start + 8, None] * shape[None]).reshape(-1, 8)
            numerator[start:start + 8] = (fuel @ EF).reshape(-1, a)
            total[start:start + 8] = fuel.sum(axis=1).reshape(-1, a)
        good = total > 0
        if not good.any():
            raise ValueError('no positive Cartesian generation')
        x = numerator[good] / total[good]
        if not np.isfinite(x).all() or min(x) < -1e-10 or max(x) > 1000 + 1e-10:
            raise ValueError('invalid Cartesian CIF')
        support, positions = np.unique(x, return_inverse=True)
        for name, s in shape_weights.items():
            atom_shape_mass = s[component] * within
            joint = pg[:, None] * atom_shape_mass[None]
            # No tiny-probability cutoffs. Reject underflow, rather than silently
            # changing support or claiming an approximate product is exact.
            if (joint <= 0).any() or not np.isfinite(joint).all():
                raise ValueError('Cartesian mass underflow/nonfinite')
            kept = float(joint[good].sum())
            if kept <= 0:
                raise ValueError('no positive Cartesian probability')
            mass = np.bincount(positions, weights=joint[good], minlength=len(support))
            output[name].append(Law(support, mass))
            group_survival = good @ atom_shape_mass
            posterior_coarse = p * group_survival[inverse] / kept
            shape_survival = pg @ good
            posterior_shape = np.bincount(component, weights=atom_shape_mass * shape_survival, minlength=len(p)) / kept
            if (max(abs(np.asarray([posterior_coarse.sum(), posterior_shape.sum()]) - 1)) > 1e-12
                    or min(posterior_coarse) < 0 or min(posterior_shape) < 0):
                raise ValueError('effective Cartesian marginal normalization')
            d = diagnostic[name]
            d['rejected_mass'].append(float(joint[~good].sum() / joint.sum()))
            d['effective_coarse_weights'].append(posterior_coarse.tolist())
            d['effective_shape_weights'].append(posterior_shape.tolist())
            d['fuel_mean'].append(((p @ values) * (atom_shape_mass @ shape) / kept).tolist())
    return output, diagnostic


def diagonal_diagnostic(values, p, cells):
    result = {'rejected_mass': [], 'effective_coarse_weights': [], 'effective_shape_weights': [], 'fuel_mean': []}
    for shape, component, within, _ in cells:
        fuel = values[component] * shape
        mass = p[component] * within
        good = fuel.sum(axis=1) > 0
        kept = float(mass[good].sum())
        posterior = np.bincount(component[good], weights=mass[good], minlength=len(p)) / kept
        result['rejected_mass'].append(float(mass[~good].sum() / mass.sum()))
        result['effective_coarse_weights'].append(posterior.tolist())
        result['effective_shape_weights'].append(posterior.tolist())
        result['fuel_mean'].append((mass @ fuel / kept).tolist())
    return result


class CoarseShapeCoupling:
    def __init__(self, frozen, monthly, archive):
        self.frozen, self.monthly = frozen, monthly
        self.rebased = RebasedCoarseLift(frozen, monthly, archive)

    def build(self, q):
        base, forecast, _ = self.rebased.build(q)
        p = self.rebased.weights(q, 'forest_history_scalar8')
        pH = self.rebased.weights(q, 'allmonth_hellinger')
        values = project_simplex(self.monthly.shares[forecast['local']] + self.rebased.delta)
        laws, diagnostics = product_laws(values, p, dict(zip(PRODUCT, (p, pH))), self.rebased.cells)
        laws.update({name: base[name] for name in REFERENCES})
        diagnostics[DIAGONAL] = diagonal_diagnostic(values, p, self.rebased.cells)
        missing = self.monthly.shares[self.rebased.profiles] == 0
        details = []
        for name, shape_p in ((DIAGONAL, p), (PRODUCT[0], p), (PRODUCT[1], pH)):
            d = diagnostics[name]
            unknown = float(p @ (values * missing).sum(axis=1)) if name == DIAGONAL else float((p @ values) @ (shape_p @ missing))
            d |= {'method': name, 'source_profiles': self.rebased.profiles.tolist(), 'coarse_support': values.tolist(),
                  'coarse_weights': p.tolist(), 'shape_weights': shape_p.tolist(), 'unknown_shape_energy_mass': unknown,
                  'precondition_marginal_error': float(d.get('precondition_marginal_error', 0.)),
                  'unique_coarse_values': len(np.unique(values, axis=0))}
            details.append({key: json.dumps(value) if isinstance(value, list) else value for key, value in d.items()})
        return laws, details


def check_inputs(predictor, q):
    monthly, data = predictor.monthly, predictor.frozen.data
    before, details = predictor.build(q)
    ids = monthly.groups[monthly.index.region.iloc[q]]
    ids = ids[(monthly.release.iloc[ids] > monthly.dates.iloc[q]).to_numpy()]
    saved = monthly.M[ids].copy(), monthly.shares[ids].copy()
    raw = data.raw
    changed = np.random.default_rng(q + 181).uniform(1, 1000, size=saved[0].shape)
    try:
        monthly.M[ids] = changed
        monthly.shares[ids] = changed / changed.sum(axis=1, keepdims=True)
        data.raw = None
        after, fresh = predictor.build(q)
    finally:
        data.raw = raw
        monthly.M[ids], monthly.shares[ids] = saved
    if details != fresh:
        raise ValueError('future query data changed Cartesian diagnostics')
    for method in METHODS:
        for a, b in zip(before[method], after[method]):
            if not np.array_equal(a.x, b.x) or not np.array_equal(a.w, b.w):
                raise ValueError('future query data changed Cartesian law')
    return before, details
