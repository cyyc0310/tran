import numpy as np
import pandas as pd
import pytest

from scripts.analyze_information_value_v10 import METRICS, paired, take, holdout_decision


def test_month_pair_preserves_equal_region_weights_and_exact_support():
    records = []
    for region, copies in [('A', 2), ('B', 1)]:
        for month in range(1, 7):
            for day in range(1, copies + 1):
                records.append(dict(region=region, origin=f'2026-{month:02d}-{day:02d}',
                    **{m: (2. if region == 'A' else 8.) for m in METRICS}))
    left = pd.DataFrame(records)
    right = left.copy()
    right[METRICS] = 0.
    result = paired(left, right, n_bootstrap=50)
    overall = result[result.label.eq('overall')]
    np.testing.assert_allclose(overall.primary_mean, 5.)
    np.testing.assert_allclose(overall.ci95_low, 5.)
    np.testing.assert_allclose(overall.ci95_high, 5.)
    with pytest.raises(ValueError, match='identical paired'):
        paired(left, right.iloc[:-1], n_bootstrap=50)
    with pytest.raises(ValueError, match='unique'):
        paired(pd.concat([left, left.iloc[:1]]), right, n_bootstrap=50)


def test_fixed_config_and_lag_cannot_silently_change():
    data = pd.DataFrame([dict(lag=7,budget='B',model='ridge',factor=1.)])
    assert len(take(data, dict(budget='B',model='ridge',factor=1.))) == 1
    with pytest.raises(ValueError, match='Missing fixed'):
        take(data, dict(budget='B',model='ridge',factor=1.),30)


def test_weather_proxy_track_cannot_alone_open_unseen_regions():
    assert holdout_decision({'a': True}, {'weather_csv': None})['new_holdout_triggered']
    assert not holdout_decision({'a': False}, {'weather_csv': None})['new_holdout_triggered']
    assert not holdout_decision({'a': True}, {'weather_csv': 'proxy.csv'})['new_holdout_triggered']
