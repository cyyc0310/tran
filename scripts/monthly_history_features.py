"""Monthly trajectories and coarse backtests built exclusively from released prefixes."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from v151_hourly_law import EF


class MonthlyInformation:
    """Load the coarse panel without accessing packed generation or timestamps."""

    def __init__(self, root):
        directory = root / 'results/v144-canonical-profile-cache-r1'
        self.index = pd.read_csv(directory / 'profile_index.csv')
        with np.load(directory / 'profiles.npz', allow_pickle=False) as packed:
            self.M = np.asarray([[float(Fraction(str(v))) for v in row]
                                 for row in packed['margin_fractions']])
        self.shares = self.M / self.M.sum(axis=1, keepdims=True)
        self.dates = pd.to_datetime(self.index.month)
        self.release = self.dates + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)
        self.groups = {r: np.asarray(ids, int) for r, ids in self.index.groupby('region').indices.items()}

    def local(self, q: int, budget: str) -> int | None:
        if budget != 'asof_30d':
            raise ValueError('monthly history probe only supports asof')
        ids = self.groups[self.index.region.iloc[q]]
        known = ids[(self.release.iloc[ids] <= self.dates.iloc[q]).to_numpy()]
        return int(known[np.argmax(self.dates.iloc[known].to_numpy())]) if len(known) else None


def month_number(date) -> int:
    return 12 * date.year + date.month


def monthly_context(data, q: int) -> dict | None:
    origin = data.dates.iloc[q]
    region = data.index.region.iloc[q]
    ids = data.groups[region]
    prefix = ids[(data.release.iloc[ids] <= origin).to_numpy()]
    prefix = prefix[np.argsort(data.dates.iloc[prefix].to_numpy())]
    if not len(prefix):
        return None
    local = int(prefix[-1])
    if data.local(q, 'asof_30d') != local:
        raise ValueError('last released statistic differs')
    S = data.shares[local]
    anchor = float(S @ EF)
    month = origin.month
    base = np.r_[S, np.log1p(data.M[local].sum()),
                 (origin - data.dates.iloc[local]).days / 365,
                 np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12), anchor / 1000]
    slope, log_slope, gap = np.zeros(8), 0., 0.
    previous = None
    if len(prefix) >= 2:
        previous = int(prefix[-2])
        gap = month_number(data.dates.iloc[local]) - month_number(data.dates.iloc[previous])
        if gap <= 0:
            raise ValueError('nonpositive history interval')
        slope = (S - data.shares[previous]) / gap
        log_slope = (np.log1p(data.M[local].sum()) - np.log1p(data.M[previous].sum())) / gap
    coarse_errors = []
    for observed in prefix:
        observed = int(observed)
        historical_origin = data.dates.iloc[observed]
        known = ids[(data.release.iloc[ids] <= historical_origin).to_numpy()]
        if not len(known):
            continue
        prediction_id = int(known[np.argmax(data.dates.iloc[known].to_numpy())])
        prediction = float(data.shares[prediction_id] @ EF)
        actual = float(data.shares[observed] @ EF)
        coarse_errors.append({'observed_profile_id': observed, 'forecast_origin': str(historical_origin),
                              'prediction_input_profile_id': prediction_id,
                              'prediction': prediction, 'actual': actual, 'error': actual - prediction})
    coarse_errors = coarse_errors[-6:]
    values = np.asarray([e['error'] for e in coarse_errors])
    rms = float(np.sqrt(np.mean(values**2))) if len(values) else 0.
    bias = float(np.mean(values)) if len(values) else 0.
    total_mix = data.M[prefix].sum(axis=0)
    total_mix /= total_mix.sum()
    historic_anchor = data.shares[prefix] @ EF
    history = np.r_[base, slope, log_slope, gap / 12, total_mix,
                    np.std(historic_anchor) / 1000, np.log1p(len(prefix)),
                    rms / 1000, bias / 1000, np.log1p(len(coarse_errors))]
    if len(base) != 13 or len(history) != 36 or not np.isfinite(history).all():
        raise ValueError('monthly history feature contract')
    horizon = month_number(origin) - month_number(data.dates.iloc[local])
    trend = anchor + float(slope @ EF) * horizon
    harmonic = anchor
    if len(prefix) >= 4:
        start = month_number(data.dates.iloc[prefix[0]])
        t = np.asarray([month_number(data.dates.iloc[p]) - start for p in prefix]) / 12
        m = np.asarray([data.dates.iloc[p].month for p in prefix])
        design = np.column_stack([t, np.sin(2 * np.pi * m / 12), np.cos(2 * np.pi * m / 12)])
        model = Ridge(alpha=1).fit(design, historic_anchor)
        query = [[(month_number(origin) - start) / 12,
                  np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]]
        harmonic = float(model.predict(query)[0])
    return {'base': base, 'history': history, 'anchor': anchor, 'local': local,
            'prefix': prefix.tolist(), 'previous': previous, 'coarse_errors': coarse_errors,
            'persistence': anchor, 'trend': float(np.clip(trend, 0, 1000)),
            'harmonic': float(np.clip(harmonic, 0, 1000))}
