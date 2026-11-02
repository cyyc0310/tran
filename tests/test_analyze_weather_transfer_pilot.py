from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze_weather_transfer_pilot import aggregate, paired_days  # noqa: E402


def fixture():
    return pd.DataFrame({'configuration': 'US_to_GB', 'country': 'GB', 'region': 'GB',
        'budget': 'monthly', 'origin': '2023-07-15T00:00Z',
        'valid_at': ['2023-07-15T00:00Z', '2023-07-15T01:00Z'],
        'y': [100., 200.], 'point': [120., 180.], 'lower': [80., 160.], 'upper': [140., 220.]})


def test_pairing_is_by_forecast_key_and_not_row_order():
    right = fixture(); left = right.copy()
    left['point'], left['lower'], left['upper'] = [100., 200.], [90., 190.], [110., 210.]
    result = paired_days(left.iloc[::-1], right).iloc[0]
    assert result.delta_mae == -20. and result.delta_shape_mae == -20.
    assert result.delta_coverage90 == 0. and result.delta_width90 == -40.
    assert result.delta_interval_score90 == -40.


def test_pairing_rejects_missing_duplicate_or_inconsistent_truth():
    f = fixture()
    with pytest.raises(ValueError, match='different cohorts'):
        paired_days(f, f.iloc[:1])
    with pytest.raises(ValueError, match='unique'):
        paired_days(pd.concat([f, f]), f)
    g = f.copy(); g.loc[0, 'y'] = 999.
    with pytest.raises(ValueError, match='truth differs'):
        paired_days(g, f)


def test_macro_does_not_overweight_regions_or_dates():
    a = paired_days(fixture(), fixture())
    a['comparison'], a['attribution'] = 'pair', 'matched_weather_increment'
    b = a.copy(); b['region'], b['origin'], b['left_mae'] = 'other', 'other_date', 100.
    # Many observations for one region must not outweigh the other region.
    days = pd.concat([a]*4+[b], ignore_index=True)
    _, _, macro = aggregate(days)
    np.testing.assert_allclose(macro.left_mae, 60.)
