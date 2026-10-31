"""Meaningful time-gate, composition and innovation-conditioning regressions."""
import numpy as np
import pandas as pd
import pytest

from scripts.daily_context_v16 import (
    ARMS, EF, FUELS, DailyHistory, analog_weights, carbon_coordinates,
    conditional_ids, daily_law, normalized_shapes, particle_analog_weights, ratio, sample_indices, simplex,
)


def table(start='2024-01-01', periods=400):
    days = pd.date_range(start, periods=periods, freq='D', tz='UTC')
    t = np.arange(periods)
    energy = np.column_stack([20+10*np.sin(t/20), 15+5*np.cos(t/13), np.zeros(periods),
                              10+0*t, 2+0*t, 10+5*np.sin(t/7), 4+2*np.cos(t/30), 1+0*t])
    frame = pd.DataFrame({f'energy_{f}': energy[:, i] for i, f in enumerate(FUELS)})
    return frame.assign(region='BPAT', day=days)


@pytest.mark.parametrize('lag', [7, 30])
@pytest.mark.parametrize('method', ['latest', 'weekday', 'ewma', 'seasonal_kalman'])
def test_future_values_cannot_change_forecasts_or_past_errors(lag, method):
    frame = table()
    at = pd.Timestamp('2025-01-01', tz='UTC')
    base = DailyHistory(frame, lag)
    released = frame.day+pd.Timedelta(days=1+lag) <= at
    damaged = frame.copy()
    damaged.loc[~released, [f'energy_{f}' for f in FUELS]] = np.arange(1,9)*1e8
    other = DailyHistory(damaged, lag)
    np.testing.assert_array_equal(base.forecast(at, method)[0], other.forecast(at, method)[0])
    np.testing.assert_array_equal(base.errors(at, method)[0], other.errors(at, method)[0])
    prefix = DailyHistory(frame.loc[released], lag)
    np.testing.assert_array_equal(base.forecast(at, method)[0], prefix.forecast(at, method)[0])


def test_daily_release_boundary_and_zero_history_are_explicit():
    frame = table('2024-01-01', 20)
    h = DailyHistory(frame, 7)
    origin = pd.Timestamp('2024-01-10', tz='UTC')
    expected = frame.iloc[1][[f'energy_{f}' for f in FUELS]].to_numpy(float)
    np.testing.assert_allclose(h.forecast(origin, 'latest')[0], expected/expected.sum(), atol=1e-15)
    assert h.forecast(origin, 'latest')[1] == 2
    assert h.forecast(pd.Timestamp('2024-01-01', tz='UTC'), 'latest')[1] == 0
    with pytest.raises(ValueError, match='prequential'):
        daily_law(h, origin, 'latest', [.2, .7])


def test_law_uses_only_released_prequential_labels_and_applied_shocks():
    h = DailyHistory(table(), 7)
    at = pd.Timestamp('2025-01-01', tz='UTC')
    law = daily_law(h, at, 'ewma', [.1, .2, .7, .9])
    assert (law['error_release'] <= at.value).all()
    assert (law['error_days'] < law['error_release']).all()
    np.testing.assert_allclose(law['draws'].sum(1), 1)
    assert (law['draws'] >= 0).all()
    np.testing.assert_array_equal(law['innovations'], law['draws']-law['point'])
    np.testing.assert_array_equal(law['raw_innovations'], law['errors'][law['error_indices']])


def test_daily_shape_normalization_preserves_predicted_daily_fuel_budget():
    rng = np.random.default_rng(8)
    q = rng.uniform(0, 4, (3,24,8)); q[1,:,5] = 0
    s = normalized_shapes(q)
    np.testing.assert_allclose(s.mean(1), 1)
    np.testing.assert_array_equal(s[1,:,5], np.ones(24))
    m = rng.dirichlet(np.ones(8), 3)
    a = m[:,None]*s
    np.testing.assert_allclose(a.mean(1), m)
    cif = ratio(m,s)
    weighted = (cif*a.sum(-1)).sum(1)/a.sum((1,2))
    np.testing.assert_allclose(weighted,m@EF)
    # Monthly-normalized q would generally change the intended day mix.
    assert not np.allclose((m[:,None]*q).mean(1),m)


def test_carbon_coordinates_preserve_coarse_ef_shock_not_hourly_sufficiency():
    rng = np.random.default_rng(4)
    d = rng.normal(size=(50,8)); d -= d.mean(1,keepdims=True)
    np.testing.assert_allclose(carbon_coordinates(d).sum(1)*1000,d@EF)
    # Substituting wind for solar is invisible to these coarse carbon axes.
    swap = np.zeros(8); swap[5:7] = [1,-1]
    np.testing.assert_array_equal(carbon_coordinates(swap),np.zeros(3))


def test_innovation_conditioning_and_permutation_change_relevant_donor():
    e = np.zeros((2,8)); e[:,0] = [1,-1]; e[:,3] = [-1,1]
    ids, ess = conditional_ids([.5,.5], e, e, e, [.5,.5])
    np.testing.assert_array_equal(ids,[0,1])
    flipped, _ = conditional_ids([.5,.5],e[::-1],e,e,[.5,.5])
    np.testing.assert_array_equal(flipped,[1,0])
    assert np.all(ess >= 1) and np.all(ess <= 2)


def test_no_innovation_signal_reduces_to_state_analog_weights():
    weights = np.array([.1,.3,.6]); u = np.array([.15,.75])
    ids, _ = conditional_ids(weights,np.zeros((3,8)),np.zeros((2,8)),np.zeros((4,8)),u)
    np.testing.assert_array_equal(ids,sample_indices(weights,u))
    assert set(ARMS) >= {'daily_analog','innovation_carbon','innovation_full','innovation_permuted'}


def test_analog_prior_balances_regions_without_composition_or_calendar_difference():
    mix = np.ones(8)/8
    weights = analog_weights(mix,np.tile(mix,(3,1)),[1,1,1],['A','A','B'],pd.Timestamp('2025-01-01',tz='UTC'))
    np.testing.assert_allclose(weights,[.25,.25,.5])


def test_particle_analog_is_same_metric_without_hidden_innovation_inputs():
    mix = np.array([[.7,.3,0,0,0,0,0,0],[.1,.9,0,0,0,0,0,0]])
    at = pd.Timestamp('2025-01-01',tz='UTC')
    weights = particle_analog_weights(mix,mix,[1,1],['A','B'],at)
    assert weights[0,0] > weights[0,1] and weights[1,1] > weights[1,0]
    for i in range(2):
        np.testing.assert_array_equal(weights[i],analog_weights(mix[i],mix,[1,1],['A','B'],at))


def test_bad_input_rejected():
    with pytest.raises(ValueError): simplex([np.nan]*8)
    with pytest.raises(ValueError): sample_indices([0,0],[.5])
    with pytest.raises(ValueError): sample_indices([1],[1.])
    with pytest.raises(ValueError): DailyHistory(table(),0)
    with pytest.raises(ValueError): normalized_shapes(-np.ones((1,24,8)))


def test_proper_hour_reader_discards_nonproper_values_before_numeric_parse(tmp_path):
    from scripts.run_daily_context_v16 import read_proper
    path = tmp_path/'BPAT_hourly.csv'
    path.write_text('time,region,generation_coal,valid\n'
        '2024-01-01 00:00:00+00:00,BPAT,POISON_BEFORE,True\n'
        '2024-01-25 00:00:00+00:00,BPAT,2,True\n'
        '2024-07-01 00:00:00+00:00,BPAT,POISON_AFTER,True\n')
    frame = read_proper(path)
    assert len(frame) == 1 and frame.generation_coal.iloc[0] == 2
    assert frame.time.iloc[0] == pd.Timestamp('2024-01-25',tz='UTC')


def test_duplicate_or_missing_prediction_grid_rejected_before_truth(tmp_path,monkeypatch):
    from scripts import run_daily_context_v16 as runner
    monkeypatch.setattr(runner,'OUT',tmp_path)
    seal = dict(entries=[],complete_grid=True,source_hourly_2025_scoring_read_by_prediction=False,
                isolated_aggregation_hourly_read_disclosed=True,target_values_opened=False)
    with pytest.raises(ValueError,match='Exact complete'):
        runner.check_predictions(seal)


def test_source_selection_is_serializable_and_does_not_select_stress_lag():
    import json
    from scripts.run_daily_context_v16 import METRICS, SOURCES, select
    records = []
    for arm,crps,lag in [('native',20,0),('daily_analog',15,7),('innovation_full',14,7),('innovation_carbon',1,30)]:
        for region in SOURCES:
            record = dict(region=region,lag=lag,forecaster='latest',arm=arm,factor=1.25,
                          **{m:0. for m in METRICS})
            record.update(crps=crps,empirical_coverage90=.92,empirical_width90=50,
                          empirical_interval_score90=10+crps)
            records.append(record)
    selected = select(pd.DataFrame(records))
    assert selected['candidate']['arm'] == 'innovation_full'
    assert selected['source_pilot_gate'] is True
    assert selected['target_test_authorized_by_this_pilot'] is False
    json.dumps(selected,allow_nan=False)
