import numpy as np
import pytest

from transcif.research import resource_controls as control
from transcif.research import coarse_trajectories as ct
from tests.test_resource_amplitudes import fixture


def reference_fixture():
    f,c,fuel,past,climate=fixture()
    climate['wind_curve_resource']=climate.wind_resource/2000.
    return f,c,fuel,past,climate


def test_fixed_curve_cutin_rated_cutout_and_invalid_speeds():
    v=np.array([0.,2.,3.,6.,12.,20.,24.99,25.,30.])
    y=control.wind_curve(v)
    np.testing.assert_array_equal(y[[0,1,2,7,8]],np.zeros(5))
    np.testing.assert_array_equal(y[[4,5,6]],np.ones(3))
    assert 0<y[3]<1
    with pytest.raises(ValueError):control.wind_curve(np.array([-1.]))


def test_learners_share_source_contract_and_bounds_ignore_target_labels():
    f,c,_,past,climate=reference_fixture()
    model=control.fit(f,c,past,climate,allowed_countries=('S',))
    daily,_,_=control.day_inputs(f,climate)
    for learner in ['ridge','rf','hgb']:
        for kind in ['B','W']:
            values=control.predict_amplitudes(model,daily,learner,kind,bounded=True)
            assert np.isfinite(values).all() and (values>=0).all()
            assert (values<=model['upper']+1e-12).all()
    original=control.predict_amplitudes(model,daily,'rf','W')
    f['label_proxy_cif']=1e15;f['target_cif']=-100.
    changed,_,_=control.day_inputs(f,climate)
    np.testing.assert_array_equal(original,control.predict_amplitudes(model,changed,'rf','W'))
    with pytest.raises(ValueError,match='source-2022'):
        control.fit(f,c,past,climate,allowed_countries=('T',))


def test_control_baseline_and_old_cube_paths_exact_and_curve_missing_fallback():
    from transcif.research import resource_amplitudes as base
    f,c,fuel,past,climate=reference_fixture();bank=ct.fit(f,fuel)
    fitted=control.fit(f,c,past,climate,allowed_countries=('S',))
    old=base.fit(f,c,past,climate,allowed_countries=('S',))
    q=f.copy();q['clock']+=__import__('pandas').DateOffset(years=1)
    values,_=control.predict_methods(bank,fitted,q,climate)
    expected,_=base.predict_methods(bank,old,q,climate)
    for new,prior in [('joint_original','joint_original'),('weather_solar','weather_solar'),
                      ('cube_weather_wind_solar','weather_wind_solar')]:
        np.testing.assert_array_equal(values[new],expected[prior])
    empty=climate.copy();empty[['wind_resource','solar_resource','wind_curve_resource']]=0.
    fallback,_=control.predict_methods(bank,fitted,q,empty)
    for method in fallback:np.testing.assert_array_equal(fallback[method],expected['joint_original'])
    reversed_values,_=control.predict_methods(bank,fitted,q.iloc[::-1],climate)
    for method in values:np.testing.assert_array_equal(values[method],reversed_values[method][::-1])
