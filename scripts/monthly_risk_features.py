"""Public monthly-history features for a later hourly-risk value experiment."""
from __future__ import annotations

import numpy as np

from hourly_fuel_data import HourlyFuelData
from monthly_history_features import monthly_context

NAMES = ('direct20', 'short43', 'long43', 'long84')


def volatility_features(data, prefix):
    """Use at most twelve released monthly share vectors, without interpolation."""
    ids = prefix[-12:]
    if not ids:
        raise ValueError('monthly risk features require a released prefix')
    shares = data.shares[ids]
    return np.r_[shares.mean(axis=0), shares.std(axis=0), shares.min(axis=0),
                 shares.max(axis=0), (shares == 0).mean(axis=0), np.log1p(len(ids))]


def feature_matrices(short, long, q):
    """Build four ordinary feature controls; neither container exposes raw hours."""
    contexts = [monthly_context(data, q) for data in (short, long)]
    if any(c is None for c in contexts):
        raise ValueError('missing monthly risk prefix')
    cells = np.arange(48)
    # This existing function only reads monthly fields and calendar-state IDs.
    base = [HourlyFuelData.features(data, q, c['local'], cells) for data, c in zip((short, long), contexts)]
    if not np.array_equal(base[0], base[1]):
        raise ValueError('long history changed the original twenty base features')
    short43, long43 = [np.column_stack([base[0], np.tile(c['history'][13:], (48, 1))]) for c in contexts]
    extra = volatility_features(long, contexts[1]['prefix'])
    matrices = dict(zip(NAMES, (base[0], short43, long43, np.column_stack([long43, np.tile(extra, (48, 1))]))))
    if any(X.shape != (48, d) or not np.isfinite(X).all() for X, d in zip(matrices.values(), (20, 43, 43, 84))):
        raise ValueError('monthly risk feature dimensions or finite values')
    return matrices, contexts, extra

