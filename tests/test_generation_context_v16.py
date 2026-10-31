"""Synthetic tests for legal netG shape context and particle margin controls."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest
from scripts import generation_context_v16 as m


def telemetry(start='2024-10-01', periods=24*90):
    ends = pd.date_range(start, periods=periods, freq='h', tz='UTC')+pd.Timedelta(hours=1)
    return pd.Series(np.tile(np.arange(1.,25.),periods//24),index=ends)


def test_shape_is_same_weekday_mean_one_and_all_ends_asof():
    origin=pd.Timestamp('2025-01-01',tz='UTC')
    pool,audit=m.history_shapes(telemetry(),origin,7,28)
    assert pool.shape==(4,24)
    np.testing.assert_allclose(pool.mean(1),1)
    assert audit['same_weekday_used'] and not audit['missing_fallback']
    assert pd.Timestamp(audit['max_selected_hour_end'])<=origin-pd.Timedelta(days=7)
    assert all(pd.Timestamp(s).dayofweek==origin.dayofweek for s in audit['selected_day_starts'])


def test_history_never_parses_future_or_other_window_values():
    origin=pd.Timestamp('2025-01-01',tz='UTC')
    series=telemetry().astype(object)
    first,a=m.history_shapes(series,origin,7,28)
    cutoff=origin-pd.Timedelta(days=7)
    series.loc[series.index>cutoff]='DO_NOT_PARSE'
    series.loc[series.index<=cutoff.normalize()-pd.Timedelta(days=28)]='DO_NOT_PARSE'
    second,b=m.history_shapes(series,origin,7,28)
    np.testing.assert_array_equal(first,second)
    assert a==b


def test_complete_positive_days_only_and_sameweekday_fallback():
    # One Tuesday complete day, but querying Wednesday: use all days fallback.
    ends=pd.date_range('2024-12-17 01:00',periods=24,freq='h',tz='UTC')
    series=pd.Series(np.ones(24),index=ends)
    pool,audit=m.history_shapes(series,pd.Timestamp('2025-01-01',tz='UTC'),7,28)
    assert pool.shape==(1,24) and not audit['same_weekday_used']
    series.iloc[0]=-1
    pool,audit=m.history_shapes(series,pd.Timestamp('2025-01-01',tz='UTC'),7,28)
    assert not len(pool) and audit['missing_fallback']
    series.iloc[0]=1
    pool,_=m.history_shapes(series.iloc[:-1],pd.Timestamp('2025-01-01',tz='UTC'),7,28)
    assert not len(pool)


def test_30day_lag_never_uses_7day_window():
    origin=pd.Timestamp('2025-01-01',tz='UTC')
    _,a=m.history_shapes(telemetry(),origin,30,56)
    assert a['cutoff']=='2024-12-02T00:00:00+00:00'
    assert pd.Timestamp(a['max_selected_hour_end'])<=origin-pd.Timedelta(days=30)


def test_telemetry_parser_discards_non_source_and_unavailable_before_g_parse(tmp_path):
    path=tmp_path/'total.csv'
    pd.DataFrame([dict(region='PACE',hour_end='bad',g_mw='DO_NOT_PARSE'),
        dict(region='BPAT',hour_end='2025-07-01T00:00:00Z',g_mw='DO_NOT_PARSE'),
        dict(region='BPAT',hour_end='2024-12-01T01:00:00Z',g_mw='10')]).to_csv(path,index=False)
    result=m.read_telemetry(path,pd.Timestamp('2025-06-23',tz='UTC'))
    assert set(result)==set(m.SOURCES) and result['BPAT'].tolist()==['10']


def test_raking_preserves_fuel_columns_and_attains_soft_rows():
    rng=np.random.default_rng(4)
    a=rng.uniform(.1,3,(5,24,8)); a[:,:,2]=0
    shape=np.linspace(.3,1.7,24)
    result,audit=m.rake(a,shape)
    assert audit['converged'].all()
    np.testing.assert_allclose(result.sum(1),a.sum(1),atol=1e-10)
    np.testing.assert_allclose(result.sum(2),shape[None]*a.sum((1,2))[:,None]/24,atol=1e-6)
    assert (result[:,:,2]==0).all()
    assert np.max(audit['final_fuel_column_error'])<1e-12


def test_zero_prior_infeasibility_falls_back_tiny_prior_is_explicit_relaxation():
    a=np.zeros((1,24,8)); a[0,:12,0]=1; a[0,12:,1]=1
    shape=np.r_[np.full(12,1.5),np.full(12,.5)]
    zero,z=m.rake(a,shape,0.,max_steps=30)
    assert not z['converged'][0]
    np.testing.assert_array_equal(zero,a)
    relaxed,r=m.rake(a,shape,.001)
    assert r['converged'][0] and r['created_positive_cells'][0]>0
    assert r['created_positive_mass'][0]>0
    np.testing.assert_allclose(relaxed.sum(1),a.sum(1),atol=1e-12)


def test_finite_nonconvergence_is_not_used_as_success():
    rng=np.random.default_rng(6)
    a=rng.random((2,24,8)); target=np.linspace(.1,1.9,24)
    result,audit=m.rake(a,target,max_steps=1,tolerance=1e-16)
    assert not audit['converged'].all()
    np.testing.assert_array_equal(result[~audit['converged']],a[~audit['converged']])


@pytest.mark.parametrize('shape', [np.zeros(24),np.ones(23),np.full(24,np.nan),np.full(24,2.)])
def test_bad_shapes_rejected(shape):
    with pytest.raises(ValueError):
        m.rake(np.ones((2,24,8)),shape)


def test_row_scaling_is_cif_invariant_but_does_not_preserve_columns():
    rng=np.random.default_rng(7)
    a=rng.random((3,24,8)); a[:,0]=0
    fallback=np.full((3,24),123.)
    result=m.row_only(a,np.linspace(.5,1.5,24))
    np.testing.assert_allclose(m.generation_cif(a,fallback),m.generation_cif(result,fallback),atol=1e-12)
    assert not np.allclose(result.sum(1),a.sum(1))


def test_ratio_roundoff_at_support_endpoint_does_not_abort_valid_particles():
    a=np.zeros((1,24,8)); a[:,:,0]=1.; a[:,:,2]=1e-16
    result=m.generation_cif(a,np.zeros((1,24)))
    assert np.isfinite(result).all()
    assert result.min()>=0 and result.max()<=1000
    np.testing.assert_allclose(result,1000,atol=1e-12)


class FakeBank:
    def __init__(self):
        rng=np.random.default_rng(11)
        self.q_=rng.uniform(.2,2,(4,24,8))
        self.days_=pd.date_range('2024-01-25',periods=4,tz='UTC')
    def _weights(self,*args):
        return np.array([.1,.2,.3,.4]),None,None,None
    _rng=staticmethod(m.old.v11.FuelTrajectoryBank._rng)


def test_analog_cdf_cannot_select_trailing_zero_mass_donor():
    class ExtremeBank:
        q_=np.ones((11,24,8))
        q_[-1,1:]=0
        days_=pd.date_range('2024-01-25',periods=11,tz='UTC')
        def _weights(self,*args):
            return np.full(11,1/11),None,None,None
        def _rng(self,*args):
            class Stream:
                def random(self,shape):
                    return np.full(shape,np.nextafter(1.,0.))
            return Stream()
    bank=ExtremeBank(); mix=np.full(8,1/8); day=pd.Timestamp('2025-01-01',tz='UTC')
    assert (m.sample_donors(bank,mix,day,np.ones(24))<10).all()
    # Native branch is left bit-identical to the frozen sampler.
    assert (m.sample_donors(bank,mix,day)==10).all()


def test_native_rng_same_stream_and_missing_shape_all_native():
    bank=FakeBank(); day=pd.Timestamp('2025-01-01',tz='UTC'); mix=np.full(8,1/8)
    expected=np.searchsorted(np.cumsum([.1,.2,.3,.4]),bank._rng(m.SEED,day).random((128,8))[:,0],side='right')
    np.testing.assert_array_equal(m.sample_donors(bank,mix,day),expected)
    draws=np.tile(mix,(128,1))
    raw,audit=m.particles(bank,mix,draws,day,np.empty((0,24)))
    assert raw.shape==(7,128,24)
    assert all(np.array_equal(raw[0],x) for x in raw)
    np.testing.assert_array_equal(audit['donor_indices'],expected)


def test_mixture_is_exactly64_native64_rake_no_path_average():
    bank=FakeBank(); day=pd.Timestamp('2025-01-01',tz='UTC'); mix=np.full(8,1/8)
    raw,audit=m.particles(bank,mix,np.tile(mix,(128,1)),day,np.array([np.linspace(.2,1.8,24)]))
    np.testing.assert_array_equal(raw[6,::2],raw[0,::2])
    np.testing.assert_array_equal(raw[6,1::2],raw[4,1::2])
    np.testing.assert_allclose(raw[1],raw[0],atol=1e-9)
    assert not np.allclose(raw[6],.5*(raw[0]+raw[4]))
    assert np.isfinite(raw).all()


def test_extreme_draw_with_zero_donor_fuel_energy_returns_native_fallback():
    bank=FakeBank(); bank.q_[:,:,2]=0
    day=pd.Timestamp('2025-01-01',tz='UTC'); mix=np.eye(8)[2]
    raw,audit=m.particles(bank,mix,np.tile(mix,(128,1)),day,np.array([np.linspace(.2,1.8,24)]))
    np.testing.assert_array_equal(raw,np.full((7,128,24),800.))
    for evidence in audit['rake'].values():
        assert evidence['zero_energy_fallback'].all()
        assert not evidence['converged'].any()
        assert (evidence['iterations']==0).all()


def test_score_fails_before_labels_without_complete_prediction_seal(tmp_path,monkeypatch):
    touched=[]
    monkeypatch.setattr(m.old.v10,'_truth_lookup',lambda *args: touched.append(True))
    with pytest.raises(FileNotFoundError):
        m.score(tmp_path)
    assert not touched


def test_exclusive_outputs_and_frozen_mutation_guard(tmp_path):
    path=tmp_path/'record.json'
    m.write(path,{'a':1})
    expected=m.sha(path)
    with pytest.raises(FileExistsError):
        m.write(path,{'a':2})
    path.write_text(json.dumps({'a':3}))
    with pytest.raises(ValueError,match='Changed frozen bytes'):
        m.verify({str(path):expected})
