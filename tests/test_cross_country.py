"""Information-contract regression tests for the country-held-out pilot."""
import numpy as np
import pandas as pd
import pytest

from transcif.research.cross_country import (
    COARSE, FUELS, FEATURES, METHODS, fit_models, metrics,
    monthly_inputs, predict_models,
)


def fixture_frame():
    clock = pd.date_range('2022-01-01', '2022-05-01', freq='h', inclusive='left')
    frame = pd.DataFrame(np.ones((len(clock), len(FUELS))), columns=FUELS)
    frame['clock'] = clock
    frame['time'] = clock
    frame['country'], frame['region'] = 'GB', 'GB'
    frame['expected_hours'] = clock.days_in_month * 24
    return frame


def test_future_and_contemporaneous_generation_cannot_change_march_inputs():
    frame = fixture_frame()
    before = monthly_inputs(frame)
    frame.loc[frame.clock.ge('2022-02-01'), 'coal'] *= 9
    after = monthly_inputs(frame)
    mask = before.clock.dt.month.eq(3)
    pd.testing.assert_frame_equal(before.loc[mask, list(FEATURES)], after.loc[mask, list(FEATURES)])
    assert before.loc[mask, 'observed_month'].eq('2022-01').all()


def test_incomplete_observed_month_is_not_replaced_with_newer_month():
    frame = fixture_frame()
    frame = frame.loc[~frame.clock.between('2022-01-01', '2022-01-02')].reset_index(drop=True)
    inputs = monthly_inputs(frame)
    assert inputs.loc[inputs.clock.dt.month.eq(3), 'thermal'].isna().all()
    assert inputs.loc[inputs.clock.dt.month.eq(4), 'thermal'].notna().all()


def test_proxy_subfuel_labels_do_not_leak_into_coarse_inputs():
    frame = fixture_frame()
    before = monthly_inputs(frame)
    frame['coal'], frame['gas'] = 2., 0.
    after = monthly_inputs(frame)
    pd.testing.assert_frame_equal(before[list(FEATURES)], after[list(FEATURES)])
    with pytest.raises(ValueError):
        monthly_inputs(frame, lag=0)


def test_frozen_prediction_accepts_feature_only_frame():
    raw = fixture_frame()
    features = monthly_inputs(raw).dropna(subset=list(COARSE))
    models = fit_models(features, raw.loc[features.index, list(FUELS)].to_numpy())
    feature_only = features[['clock', *FEATURES]].copy()
    predictions = predict_models(models, feature_only)
    assert set(predictions) == set(METHODS)
    for value in predictions.values():
        assert value.shape == (len(features),)
        assert np.isfinite(value).all()


def test_gaussian_crps_known_value_and_no_false_coverage():
    inputs = pd.DataFrame({'clock': pd.date_range('2024-01-01', periods=2, freq='h')})
    y = np.array([0., 10.])
    scores = metrics(inputs, y, y, bias=0., sigma=2.)
    assert scores['crps_gaussian'] == pytest.approx(2*(np.sqrt(2)-1)/np.sqrt(np.pi))
    assert scores['coverage90'] == 1.
    assert metrics(inputs, y, y+100, 0., 2.)['coverage90'] == 0.
