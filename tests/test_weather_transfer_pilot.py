"""Time alignment and legal feature boundaries of the seasonal pilot."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from transcif.research import multicountry as mc

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
from collect_weather_transfer_pilot import cohort, plan  # noqa: E402
from run_weather_transfer_pilot import inputs, shape_anchor, point_fit, save_checked  # noqa: E402


def synthetic(region='CISO', zone='America/Los_Angeles'):
    times = pd.Series(pd.date_range('2022-01-01', '2024-01-01', freq='h', inclusive='left', tz='UTC'))
    clock = times.dt.tz_convert(zone).dt.tz_localize(None)
    f = pd.DataFrame(np.full((len(times), len(mc.FINE)), 10.), columns=mc.FINE)
    f['time'], f['clock'], f['region'], f['country'] = times, clock, region, 'US'
    f['expected_hours'] = clock.map(lambda t: ((t.to_period('M')+1).start_time.tz_localize(zone)
                                              - t.to_period('M').start_time.tz_localize(zone)).total_seconds()/3600)
    return f


def test_dst_elsewhere_does_not_break_selected_day_and_label_times_match():
    f = synthetic()
    assert f.clock.duplicated().any()
    protocol = plan(); protocol['dates'] = {'development': ['2023-07-15']}
    selected, audit = cohort({'CISO': f}, protocol)
    assert audit[0]['eligible'] and len(selected) == 24
    assert selected.clock.dt.hour.tolist() == list(range(24))
    pd.testing.assert_series_equal(f.loc[selected.label_row, 'time'].reset_index(drop=True),
                                   selected.valid_at.reset_index(drop=True), check_names=False)
    assert selected.latest_allowed_period.eq('2023-05').all()


def test_actual_ambiguous_day_is_excluded_not_collapsed():
    f = synthetic()
    protocol = plan(); protocol['dates'] = {'development': ['2023-11-05', '2023-11-15']}
    selected, audit = cohort({'CISO': f}, protocol)
    assert audit[0]['reason'] == 'ambiguous DST day'
    assert len(selected) == 24 and selected.clock.dt.day.eq(15).all()


def test_feature_values_ignore_labels_and_optional_budget_fields():
    f = pd.DataFrame(np.ones((3, len(mc.HISTORY))), columns=mc.HISTORY)
    f['capacity_factor'] = 600.
    f['temperature_c'], f['wind_speed_ms'], f['shortwave_w_per_m2'] = 20., 4., 200.
    f['label_row'], f['target_cif'] = 123, -999.
    base = inputs(f, False, False); weather = inputs(f, True, False)
    f['capacity_factor'], f['target_cif'], f['label_row'] = 10000., 0., 999
    np.testing.assert_array_equal(base, inputs(f, False, False))
    np.testing.assert_array_equal(weather, inputs(f, True, False))
    assert inputs(f, True, True).shape[1] == weather.shape[1]+1


def test_shape_anchor_uses_visible_mix_and_source_profile_only():
    bank = {'prior': np.array([.25, .75]), 'q': np.ones((2, 24, 6))}
    f = pd.DataFrame(np.tile([.8, .1, .1, 0., 0., 0.], (24, 1)), columns=mc.GROUPS)
    f['clock'] = pd.date_range('2023-07-15', periods=24, freq='h')
    np.testing.assert_allclose(shape_anchor(bank, f, np.full(24, 600.)), 480.)


def test_source_model_persistence_checks_reload(tmp_path):
    x = np.arange(60, dtype=float).reshape(20, 3)/10.
    model = point_fit(x, x[:, 0]*2.+1., np.full(20, .05), 'ridge')
    path = tmp_path/'model.joblib'
    assert len(save_checked(model, path, x)) == 64 and path.exists()
