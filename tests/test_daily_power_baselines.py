import numpy as np
import pandas as pd
import pytest

from tests.test_resource_controls import reference_fixture
from transcif.research import daily_power_baselines as power


def test_source_contract_rejects_target_labels_and_mismatched_month_power():
    f, c, _, past, climate = reference_fixture()
    for changed in [f.assign(country='T'), f.assign(split='development'),
                    f.assign(clock=f.clock+pd.DateOffset(years=1))]:
        with pytest.raises(ValueError, match='source-2022'):
            power.source_inputs(changed, c, past, climate, allowed_countries=('S',))
    wrong = past.copy(); wrong[:, 0] *= 2.
    with pytest.raises(ValueError, match='visible monthly'):
        power.source_inputs(f, c, wrong, climate, allowed_countries=('S',))
    future = f.assign(latest_allowed_period='2022-06')
    with pytest.raises(ValueError, match='reference month'):
        power.source_inputs(future, c, past, climate, allowed_countries=('S',))


def test_no_weather_head_ignores_weather_and_unrelated_label_columns():
    f, c, _, past, climate = reference_fixture()
    daily, y = power.source_inputs(f, c, past, climate, allowed_countries=('S',))
    for learner in ('ridge', 'hgb'):
        fitted = power.fit_head(daily, y['total'], weather=False, learner=learner,
                                allowed_countries=('S',))
        changed = daily.copy()
        changed.loc[:, [x for x in changed if x.startswith(('log_weather', 'forecast_'))]] = 1e8
        changed['target_total'] = -1e30
        np.testing.assert_array_equal(power.predict_head(fitted, daily), power.predict_head(fitted, changed))
        with pytest.raises(ValueError, match='proper-source'):
            power.fit_head(daily.assign(country='T'), y['total'], weather=False,
                           learner=learner, allowed_countries=('S',))


def test_amplitude_reference_is_mean_power_not_hourly_cif_average():
    f, c, _, past, climate = reference_fixture()
    daily, y = power.source_inputs(f, c, past, climate, allowed_countries=('S',))
    groups = power.daily_controls.day_groups(f)
    for j, ix in enumerate(groups):
        assert y['total'][j] == pytest.approx(c[ix].sum(axis=1).mean()/past[j].sum())
        assert y['thermal'][j] == pytest.approx(c[ix, 0].mean()/past[j, 0])
    assert len(y['total']) == len(daily)
    # A zero lagged thermal denominator is unidentifiable, never imputed from a target day.
    p = past.copy(); p[:, 0] = 0.
    changed = f.copy()
    for j, ix in enumerate(groups):
        changed.loc[changed.index[ix], list(power.GROUPS)] = p[j]/p[j].sum()
    _, z = power.source_inputs(changed, c, p, climate, allowed_countries=('S',))
    assert np.isnan(z['thermal']).all() and np.isfinite(z['total']).all()
