import numpy as np
import pandas as pd
import pytest

from transcif.research.coarse_weather_shapes import delta_transport, physical_shapes, fit_heads, predict_heads, fit_crossfit, predict_methods
from transcif.research import coarse_trajectories as ct
from transcif.research.multicountry import GROUPS, public_features


def test_zero_increment_is_exact_and_amplitudes_support_are_preserved():
    rng = np.random.default_rng(416)
    q = rng.uniform(.1, 3., (4, 24, 6)).astype('float32')
    q[:, :, 5] = 1.; q[0, :, 1] = 0.
    support = np.ones((4, 6), dtype=bool); support[:, 5] = False
    a = rng.uniform(0., 2., (24, 6)); b = rng.uniform(0., 2., (4, 24, 6))
    original = delta_transport(q, a, a, b, b, support, strength=1.)
    assert original.dtype == q.dtype and original.tobytes() == q.tobytes()
    zero = delta_transport(q, a, a*.5, b, b*.5, support, strength=0.)
    assert zero.dtype == q.dtype and zero.tobytes() == q.tobytes()
    result = delta_transport(q, a, a*.5, b, b*.5, support, strength=.5)
    np.testing.assert_allclose(result.mean(axis=1), q.astype(float).mean(axis=1), atol=1e-12)
    np.testing.assert_array_equal(result[:, :, 5], q[:, :, 5])
    np.testing.assert_array_equal(result[0, :, 1], q[0, :, 1])
    assert (result >= 0).all()


def frame():
    clock = pd.Series(pd.date_range('2022-05-15', periods=24, freq='h'))
    months = pd.DataFrame(np.ones((5, 6)), columns=GROUPS, index=pd.period_range('2022-01',periods=5,freq='M'))
    f = public_features(months, clock); f['country'], f['region'], f['split'] = 'S', 'R', 'train'
    f['temperature_c'], f['wind_speed_ms'], f['shortwave_w_per_m2'] = 20., 0., 0.
    return f


def test_physical_zero_weather_is_flat_shape_not_physical_generation():
    f = frame(); result = physical_shapes(f)
    np.testing.assert_array_equal(result, np.ones((24, 6)))
    f['shortwave_w_per_m2'] = np.maximum(0., np.sin(np.arange(24)*np.pi/12))
    result = physical_shapes(f)
    np.testing.assert_allclose(result.mean(axis=0), 1.)
    assert result[18, 4] == 0. and result[6, 4] > 1.


def test_shape_heads_reject_non_source_or_future_fit_and_ignore_target_labels():
    f = frame(); coarse = np.ones((24, 6))
    fitted = fit_heads(f, coarse, allowed_countries=('S',))
    before = predict_heads(fitted, f)
    f['target_cif'] = -999.; f['label_proxy_cif'] = 10000.
    after = predict_heads(fitted, f)
    for a, b in zip(before, after):np.testing.assert_array_equal(a, b)
    f['country'] = 'TARGET'
    with pytest.raises(ValueError, match='source'):
        fit_heads(f, coarse, allowed_countries=('S',))
    f['country'] = 'S'; f['clock'] += pd.DateOffset(years=1)
    with pytest.raises(ValueError, match='2022'):
        fit_heads(f, coarse, allowed_countries=('S',))


def test_source_head_cannot_use_hidden_hourly_mix_as_public_monthly_features():
    f = frame(); f.loc[0, 'thermal'] += .05; f.loc[0, 'hydro'] -= .05
    with pytest.raises(ValueError, match='fixed valid public monthly'):
        fit_heads(f, np.ones((24, 6)), allowed_countries=('S',))


def test_transport_rejects_corrupt_support_even_for_zero_strength():
    q = np.ones((2, 24, 6)); q[0, 0, 0] = 2.
    with pytest.raises(ValueError, match='unsupported'):
        delta_transport(q, q, q, q, q, np.zeros((2, 6), bool), strength=0.)


def test_six_fuel_transport_matches_frozen_v12_math_with_dummy_fuels():
    from scripts.weather_increment_v12 import delta_transport as legacy
    rng = np.random.default_rng(51)
    q = rng.uniform(.1, 3., (3, 24, 6))
    query_w, query_b = rng.uniform(0., 2., (2, 24, 6))
    donor_w, donor_b = rng.uniform(0., 2., (2, 3, 24, 6))
    def pad(a):
        return np.concatenate([a, np.ones((*a.shape[:-1], 2))], axis=-1)
    expected = legacy(pad(q), pad(query_w), pad(query_b), pad(donor_w), pad(donor_b),
                      np.tile([True]*6+[False]*2, (3, 1)), strength=.5)[..., :6]
    actual = delta_transport(q, query_w, query_b, donor_w, donor_b, np.ones((3, 6), bool), strength=.5)
    np.testing.assert_array_equal(actual, expected)


def training_days():
    frames = []
    for month in (5, 6, 8):
        f = frame(); f['clock'] += pd.DateOffset(months=month-5)
        f['origin'] = f.clock.dt.strftime('%Y-%m-%d')
        f['wind_speed_ms'] = np.linspace(1., 10., 24)+month/10.
        f['shortwave_w_per_m2'] = np.maximum(0., np.sin(np.arange(24)*np.pi/12))*month
        frames.append(f)
    f = pd.concat(frames, ignore_index=True)
    fuel = np.ones((len(f), 9)); fuel[:, 6] = f.wind_speed_ms.to_numpy()
    coarse = np.column_stack([fuel[:, :4].sum(axis=1), fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7], fuel[:, 8]])
    return f, fuel, coarse


def test_held_donor_weather_prediction_cannot_use_own_date_labels():
    f, _, coarse = training_days()
    before = fit_crossfit(f, coarse, allowed_countries=('S',))
    changed = coarse.copy(); changed[:24] *= np.linspace(.1, 9., 24)[:, None]
    after = fit_crossfit(f, changed, allowed_countries=('S',))
    np.testing.assert_array_equal(before['donor_b'][0], after['donor_b'][0])
    np.testing.assert_array_equal(before['donor_w'][0], after['donor_w'][0])
    for fold in before['audit']['folds']:
        assert fold['held_origin'] not in fold['proper_origins']


def test_identity_members_match_original_bank_and_query_order():
    f, fuel, coarse = training_days(); bank = ct.fit(f, fuel)
    fitted = fit_crossfit(f, coarse, allowed_countries=('S',))
    query = f.copy(); query['clock'] += pd.DateOffset(years=1)
    original, _ = ct.predict(bank, query)
    before = predict_methods(bank, fitted, query)
    np.testing.assert_array_equal(original['joint'], before['learned_zero'])
    missing_weather = query.drop(columns=['temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2'])
    fallback = predict_methods(bank, fitted, missing_weather, methods=('learned_zero',))
    np.testing.assert_array_equal(original['joint'], fallback['learned_zero'])
    reversed_query = query.iloc[::-1].copy(); reversed_query['target_cif'] = 999.
    after = predict_methods(bank, fitted, reversed_query)
    for method in before:
        np.testing.assert_array_equal(before[method], after[method][::-1])
