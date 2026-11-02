"""Expanded comparison keeps old paths and checks source and scoring boundaries."""
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.test_resource_controls import reference_fixture
from transcif.research import coarse_trajectories as ct
from transcif.research import resource_controls_daily as daily


def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    return importlib.import_module('run_expanded_anchor_transfer')


def test_cube_reference_is_label_matched_and_old_three_paths_remain_exact(monkeypatch):
    module = runner(monkeypatch)
    public, coarse, fuel, power, climate = reference_fixture()
    model = daily.fit(public, coarse, power, climate, allowed_countries=('S',))
    bank = ct.fit(public, fuel)
    query = public.copy(); query['clock'] += pd.DateOffset(years=1)
    result = module.recomposition(bank, model, query, climate)
    original = module.paired_recomposition(bank, model, query, climate)
    for method, paths in original.items():
        np.testing.assert_array_equal(result[method], paths)
    expected, _ = daily.predict_methods(bank, model, query, climate)
    np.testing.assert_array_equal(result['paired_cube_weather'], expected['cube_weather_wind_solar'])


def test_source_audit_cannot_load_target_or_other_label_budget(monkeypatch):
    module = runner(monkeypatch)
    module.require_source(dict(source_countries=['GB'], source_hours=5136), ('GB',), 5136)
    for countries, hours in [(['GB', 'US'], 5136), (['GB'], 7344)]:
        with pytest.raises(ValueError, match='proper paired'):
            module.require_source(dict(source_countries=countries, source_hours=hours), ('GB',), 5136)


def test_overlap_checks_invariant_predictions_but_allows_recalibrated_endpoints(monkeypatch):
    module = runner(monkeypatch)
    previous = pd.DataFrame(dict(configuration=['GB_to_US'], country=['US'], region=['CISO'],
        track=['paired_labels'], budget=['capacity'], method=['paired_joint'], risk=['raw'],
        valid_at=['2023-07-15 07:00:00+00:00'], label_row=[10], point=[100.], median=[90.],
        lower=[10.], upper=[200.]))
    current = previous.copy(); current['upper'] = 210.
    assert module.overlap_check(current, previous)['rows'] == 1
    current['point'] = 101.
    with pytest.raises(AssertionError):
        module.overlap_check(current, previous)


def test_shape_scoring_removes_each_origin_level_separately(monkeypatch):
    module = runner(monkeypatch)
    frame = pd.DataFrame(dict(origin=['a', 'a', 'b', 'b'], truth=[10., 20., 100., 110.],
        point=[30., 40., 95., 105.], median=[30., 40., 95., 105.], lower=[0., 0., 0., 0.],
        upper=[200., 200., 200., 200.]))
    result = module.score_group(frame)
    assert result['shape_mae'] == 0.
    assert result['mae'] == pytest.approx(12.5)
    assert result['coverage90'] == 1.


def test_without_issued_branch_matches_frozen_B_paths_and_requires_no_forecast_values(monkeypatch):
    module = runner(monkeypatch)
    public, coarse, fuel, power, climate = reference_fixture()
    model = daily.fit(public, coarse, power, climate, allowed_countries=('S',))
    bank = ct.fit(public, fuel)
    query = public.copy(); query['clock'] += pd.DateOffset(years=1)
    old = module.paired_recomposition(bank, model, query, climate)
    absent = query.drop(columns=['temperature_c', 'wind_speed_ms', 'shortwave_w_per_m2'])
    changed = query.copy()
    changed['temperature_c'] = -1000.; changed['wind_speed_ms'] = 1000.; changed['shortwave_w_per_m2'] = 1e10
    expected = module.without_issued_recomposition(bank, model, absent, climate)
    actual = module.without_issued_recomposition(bank, model, changed, climate)
    for method in ['paired_joint', 'paired_hgb_no_weather']:
        np.testing.assert_array_equal(expected[method], old[method])
        np.testing.assert_array_equal(actual[method], old[method])
