"""Ordinary same-task native-offset residual controls; no new uncertainty law."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from scripts.local_sketch_v17 import SOURCES, calendar, balanced
from scripts.native_center_v18 import project, shifted_paths, summarize_paths, scaled_paths

PACKETS = ('D2', 'D8', 'T8', 'G5')
INPUTS = ('native_only', *('packet_only_'+p for p in PACKETS), *('joint_'+p for p in PACKETS))
PROJECTIONS, ALPHAS, FACTORS = ('day', 'block', 'full'), (.5, 1.), (.75, 1., 1.25, 1.5, 2.)
SEED, TREES, LEAF = 17023, 150, 10
PACKET_WIDTHS = {'D2': 19, 'D8': 51, 'T8': 51, 'G5': 35}


def arms() -> tuple[tuple[str, str, float], ...]:
    return (('native', 'none', 0.),) + tuple((name, p, a) for name in INPUTS for p in PROJECTIONS for a in ALPHAS)


def native_features(raw: np.ndarray, origins: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if (raw.shape != (len(origins), 128, 24) or not np.isfinite(raw).all()
            or np.any(raw < 0) or np.any(raw > 1000)):
        raise ValueError('Finite bounded native128 paths and matching origins required')
    mean, sd = raw.mean(1), raw.std(1)
    return np.column_stack((mean, sd, np.stack([calendar(d) for d in origins]))), mean, sd


def input_features(raw: np.ndarray, origins: pd.DatetimeIndex, packets: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    x, _, _ = native_features(raw, origins)
    if set(packets) != set(PACKETS):
        raise ValueError('Exactly four packet feature matrices required')
    for p, values in packets.items():
        if values.shape != (len(origins), PACKET_WIDTHS[p]) or not np.isfinite(values).all():
            raise ValueError('Common finite complete packet-feature cohort required')
        # All packet X already contain the common calendar; do not duplicate it in joint.
        if not np.array_equal(values[:, -10:-4], x[:, -6:]):
            raise ValueError('Packet feature calendar differs from native query calendar')
    return {'native_only': x, **{'packet_only_'+p: packets[p].copy() for p in PACKETS},
            **{'joint_'+p: np.column_stack((x[:, :48], packets[p])) for p in PACKETS}}


class ResidualModel:
    def fit(self, x: np.ndarray, response: np.ndarray, regions: np.ndarray,
            origins: pd.DatetimeIndex, *, excluded: str) -> ResidualModel:
        if (excluded not in SOURCES or excluded in set(regions) or len(set(regions)) < 2
                or not set(regions) <= set(SOURCES) or x.ndim != 2 or len(x) != len(regions)
                or regions.shape != (len(x),) or response.shape != (len(x), 24) or len(origins) != len(x)
                or not np.isfinite(x).all() or not np.isfinite(response).all()
                or origins.hasnans or not origins.equals(origins.normalize())
                or np.any(origins < pd.Timestamp('2024-01-25', tz='UTC'))
                or np.any(origins >= pd.Timestamp('2024-07-01', tz='UTC'))
                or pd.MultiIndex.from_arrays([regions, origins]).has_duplicates):
            raise ValueError('Finite H1 proper region-days excluding held source required')
        weights = balanced(regions)*len(x)
        self.scaler = StandardScaler().fit(x, sample_weight=weights)
        self.model = ExtraTreesRegressor(n_estimators=TREES, min_samples_leaf=LEAF,
                                        random_state=SEED, n_jobs=1)
        self.model.fit(self.scaler.transform(x), response, sample_weight=weights)
        self.regions = tuple(sorted(set(regions)))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if x.ndim != 2 or x.shape[1] != self.scaler.n_features_in_ or not np.isfinite(x).all():
            raise ValueError('Finite matching query features required')
        result = self.model.predict(self.scaler.transform(x))
        if result.shape != (len(x), 24) or not np.isfinite(result).all():
            raise ValueError('Nonfinite residual center prediction')
        return result  # No target/center clipping; raw translation is clipped only at scoring.


def reconstruct(native: np.ndarray, deltas: np.ndarray, arm: tuple[str, str, float]) -> tuple[np.ndarray, np.ndarray]:
    if (deltas.shape != (9, len(native), 24) or not np.isfinite(deltas).all() or arm not in arms()):
        raise ValueError('Exactly nine fitted deltas and a registered arm required')
    name, resolution, alpha = arm
    if name == 'native':
        shift = np.zeros((len(native), 24))
    else:
        shift = project(deltas[INPUTS.index(name)], resolution)
    raw = shifted_paths(native, shift, alpha)
    requested = native.mean(1) if alpha == 0 else native.mean(1)+alpha*shift
    return raw, requested


def diagnostics(native: np.ndarray, deltas: np.ndarray) -> dict[str, np.ndarray]:
    rows = []
    for arm in arms():
        raw, requested = reconstruct(native, deltas, arm)
        summary = summarize_paths(raw, native, requested)
        summary['unclipped_width90'] = (np.sort(raw, axis=1)[:, 121]-np.sort(raw, axis=1)[:, 6]).mean(1)
        clipped = np.sort(np.clip(raw, 0, 1000), axis=1)
        summary['initial_clipped_width90'] = (clipped[:, 121]-clipped[:, 6]).mean(1)
        rows.append(summary)
    return {k: np.stack([row[k] for row in rows]) for k in rows[0]}
