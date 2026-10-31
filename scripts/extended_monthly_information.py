"""Monthly-only extended history, preserving all original canonical profile ids."""
from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from monthly_history_features import MonthlyInformation


class ExtendedMonthlyInformation(MonthlyInformation):
    def __init__(self, root, history_path):
        super().__init__(root)
        original_M, original_shares = self.M.copy(), self.shares.copy()
        self.canonical_count = len(self.index)
        columns = ['profile_id', 'region', 'month', 'hours']
        original_index = self.index[columns].copy()
        history = pd.read_csv(history_path, dtype={f'margin_{f}': str for f in ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')})
        roster = set(pd.read_csv(root / 'results/v159-fuel-activity-source-pilot-r1/source_sampling.csv').region)
        if (len(history) != 661 or history.duplicated(['region', 'month']).any()
                or not set(history.region).issubset(roster) or not history.year.isin([2022, 2023]).all()):
            raise ValueError('extended monthly history roster/year/inventory')
        history = history.sort_values(['region', 'month']).reset_index(drop=True)
        fuels = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
        M = np.asarray([[float(Fraction(row[f'margin_{f}'])) for f in fuels] for _, row in history.iterrows()])
        if not np.isfinite(M).all() or (M < 0).any() or (M.sum(axis=1) <= 0).any():
            raise ValueError('invalid monthly margins')
        dates = pd.to_datetime(history.month)
        release = pd.to_datetime(history.available_at, utc=True).dt.tz_localize(None)
        if not (release == dates + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)).all():
            raise ValueError('extended synthetic release rule differs')
        self.historical_ids = np.arange(self.canonical_count, self.canonical_count + len(history))
        extra_index = pd.DataFrame({'profile_id': self.historical_ids, 'region': history.region,
                                    'month': history.month, 'hours': history.observed_hours})
        # This container exposes no raw offsets: added history has no hour bank.
        self.index = pd.concat([original_index, extra_index], ignore_index=True)
        self.M = np.concatenate([self.M, M])
        self.shares = np.concatenate([self.shares, M / M.sum(axis=1, keepdims=True)])
        self.dates = pd.concat([self.dates, dates], ignore_index=True)
        self.release = pd.concat([self.release, release], ignore_index=True)
        self.groups = {r: np.asarray(ids, int) for r, ids in self.index.groupby('region').indices.items()}
        if (not np.array_equal(self.M[:self.canonical_count], original_M)
                or not np.array_equal(self.shares[:self.canonical_count], original_shares)
                or not self.index.profile_id.eq(np.arange(len(self.index))).all()):
            raise ValueError('original canonical identities/margins changed')
        pd.testing.assert_frame_equal(self.index.iloc[:self.canonical_count], original_index)
