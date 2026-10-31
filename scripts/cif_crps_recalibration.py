"""Standard direct-CRPS map controls on the exact frozen PIT forest kernels."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cif_crps_quadratic import PACKED, solve_mean, solve_group_max
from cif_pit_recalibration import (PITRecalibration, REFERENCES, GRID, pit_targets,
                                   recalibrate, check_maps)
from monthly_history_features import monthly_context
from v154_hourly_strong_baselines import balanced

SETTINGS = {f'{prefix}_crps_{kind}_a{int(alpha * 100)}': (reference, kind, alpha)
            for prefix, reference in zip(('conditional', 'gaussian'), REFERENCES)
            for kind in ('global', 'conditional20', 'conditional43', 'group_max') for alpha in (.5, 1.)}
METHODS = REFERENCES + tuple(SETTINGS)


class CRPSRecalibration(PITRecalibration):
    """Change only the calibration objective, retaining source data and kernels."""

    def __init__(self, frozen, monthly, pit_archive: Path, statistics_archive: Path):
        super().__init__(frozen, monthly, pit_archive)
        with np.load(statistics_archive / 'quadratics.npz', allow_pickle=False) as saved:
            for key, array in [('raw_id', self.raw_ids), ('profile_id', self.profiles),
                               ('region', self.regions), ('inner_fold', self.inner)]:
                if not np.array_equal(saved[key], array):
                    raise ValueError('quadratic/PIT source identities differ')
            stats = saved['statistics'].copy()
        if stats.shape != (len(self.raw_ids), 2, PACKED) or not np.isfinite(stats).all():
            raise ValueError('quadratic dimensions/finiteness')
        with np.load(pit_archive / 'oof.npz', allow_pickle=False) as old:
            lower, upper = old['lower'].copy(), old['upper'].copy()
        contexts = {int(q): monthly_context(monthly, int(q)) for q in np.unique(self.profiles)}
        X20 = np.concatenate([frozen.activity.sources[r]['X'] for r in frozen.fit])
        X43 = np.column_stack([X20, np.asarray([contexts[int(q)]['history'][13:] for q in self.profiles])])
        if X20.shape != (len(self.raw_ids), 20) or X43.shape != (len(self.raw_ids), 43):
            raise ValueError('kernel training feature dimensions')
        weight = balanced(self.regions)
        self.tables, self.fixed_maps, self.fixed_statistics = {}, {}, {}
        for j, reference in enumerate(REFERENCES):
            target = pit_targets(lower[:, j], upper[:, j])
            response = np.column_stack([stats[:, j], target])
            weighted = response * weight[:, None]
            for kind, X in [('conditional20', X20), ('conditional43', X43)]:
                forest = self.models[(reference, kind)]
                if forest.bootstrap or forest.n_estimators != 32:
                    raise ValueError('unregistered kernel forest')
                leaves = forest.apply(X)
                tables = []
                for tree, estimator in enumerate(forest.estimators_):
                    count = estimator.tree_.node_count
                    mass = np.bincount(leaves[:, tree], weights=weight, minlength=count)
                    sums = np.column_stack([np.bincount(leaves[:, tree], weights=weighted[:, k], minlength=count)
                                            for k in range(response.shape[1])])
                    tables.append(np.divide(sums, mass[:, None], out=np.zeros_like(sums), where=mass[:, None] > 0))
                self.tables[(reference, kind)] = tables
            global_stats = np.average(stats[:, j], axis=0, weights=weight)
            regional_stats = np.stack([stats[self.regions == r, j].mean(axis=0) for r in frozen.fit])
            for name, (base, kind, alpha) in SETTINGS.items():
                if base == reference and kind in ('global', 'group_max'):
                    self.fixed_statistics[name] = global_stats if kind == 'global' else regional_stats
                    mapping, certificate = (solve_mean(global_stats, alpha) if kind == 'global'
                                            else solve_group_max(regional_stats, alpha))
                    self.fixed_maps[name] = (mapping, certificate)

    def conditional_statistics(self, q):
        if q not in self.frozen.allowed_queries or self.monthly.index.region.iloc[q] in self.frozen.fit:
            raise ValueError('query outside frozen nontraining roster')
        context = monthly_context(self.monthly, q)
        if context is None or context['local'] != self.frozen.data.local(q, 'asof_30d'):
            raise ValueError('query prefix mismatch')
        X20 = self.frozen.data.features(q, context['local'], np.arange(48))
        X43 = np.column_stack([X20, np.tile(context['history'][13:], (48, 1))])
        averaged, kernel_error = {}, 0.
        for reference in REFERENCES:
            for kind, X in [('conditional20', X20), ('conditional43', X43)]:
                forest = self.models[(reference, kind)]
                leaves = forest.apply(X)
                rows = sum(table[leaves[:, t]] for t, table in enumerate(self.tables[(reference, kind)])) / 32
                error = float(np.max(abs(rows[:, PACKED:] - forest.predict(X))))
                kernel_error = max(kernel_error, error)
                if error > 1e-12:
                    raise ValueError('PIT forest kernel weights not reproduced')
                averaged[(reference, kind)] = rows[:, :PACKED]
        return averaged, kernel_error

    def direct_maps(self, q):
        averaged, kernel_error = self.conditional_statistics(q)
        maps, certificates = {}, []
        for name, (reference, kind, alpha) in SETTINGS.items():
            if kind in ('global', 'group_max'):
                mapping, certificate = self.fixed_maps[name]
                maps[name] = np.tile(mapping, (48, 1))
                certificates.extend({'method': name, 'state': h, **certificate} for h in range(48))
            else:
                results = [solve_mean(row, alpha) for row in averaged[(reference, kind)]]
                maps[name] = check_maps(np.asarray([r[0] for r in results]))
                certificates.extend({'method': name, 'state': h, **r[1]} for h, r in enumerate(results))
        return maps, certificates, kernel_error

    def build(self, q):
        laws, forecast = self.physical.build(q)
        laws = {name: cells + cells for name, cells in laws.items()}
        maps, certificates, kernel_error = self.direct_maps(q)
        details = []
        for name, (reference, _, alpha) in SETTINGS.items():
            laws[name] = [recalibrate(cell, mapping, alpha) for cell, mapping in zip(laws[reference], maps[name])]
            details.append({'method': name, 'maps': json.dumps(maps[name].tolist()), 'kernel_pit_error': kernel_error,
                            'state_mean_before': json.dumps([float(c.mean) for c in laws[reference]]),
                            'state_mean_after': json.dumps([float(c.mean) for c in laws[name]])})
        return laws, forecast, details, certificates
