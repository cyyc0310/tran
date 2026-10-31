"""Checks of V12 selection, native preservation and source-first boundaries."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import run_weather_increment_v12 as r


def selection_table(round_number):
    rows=[]
    for region in r.SOURCES:
        for config in r.configurations(round_number):
            for factor in r.FACTORS:
                metrics={m:1. for m in r.METRICS}
                metrics.update(empirical_coverage90=.925,crps=20+abs(factor-1))
                rows.append(dict(stage='source',region=region,semester='H1',lag=7,
                                 factor=factor,**{k:config[k] for k in ('budget','model')},**metrics))
    return pd.DataFrame(rows)


def test_fixed_round_grids_and_primary_contract():
    assert len(r.configurations(1))==11
    assert len(r.configurations(2))==13
    assert {c['model'] for c in r.configurations(1) if c['primary_eligible']}==set(r.ALPHAS)
    assert {c['model'] for c in r.configurations(2) if c['primary_eligible']}=={
        'donor','global_total','hetero_total','qrf_0','qrf_25','qrf_50'}
    with pytest.raises(ValueError): r.configurations(3)


def test_round1_locks_only_donor_strength_from_canonical_source(tmp_path):
    table=selection_table(1)
    table.loc[table.model.eq('qrf_0'),'crps']=0
    table.loc[table.model.eq('donor_a050'),'crps']=1+abs(table.factor-1)
    path=tmp_path/'source.csv'; table.to_csv(path,index=False)
    choice=r.selection(pd.read_csv(path),1)
    assert choice['primary']['model']=='donor_a050'
    assert choice['primary']['factor']==1
    assert choice['locked_alpha']==.5
    assert choice['diagnostic_crps_best']['model']=='qrf_0'
    assert choice==r.selection(pd.read_csv(path),1)


def test_round2_coverage_gate_and_worst_region_precede_crps():
    table=selection_table(2)
    table.loc[table.model.eq('qrf_50'),'crps']=0
    table.loc[table.model.eq('qrf_50') & table.region.eq(r.SOURCES[0]),'empirical_coverage90']=.84
    choice=r.selection(table,2)
    assert choice['primary']['model']!='qrf_50'
    assert choice['diagnostic_crps_best']['model']=='qrf_50'
    with pytest.raises(ValueError,match='Incomplete'): r.selection(table.iloc[:-1],2)


def test_labels_cannot_be_opened_before_complete_phase_seal(tmp_path,monkeypatch):
    def forbidden(*args): raise AssertionError('labels opened')
    monkeypatch.setattr(r.v10,'_truth_lookup',forbidden)
    with pytest.raises(ValueError,match='seal required'):
        r.score_phase(tmp_path,{},'target',{})


def test_known_region_boundary():
    assert set(r.allowed_sources())==set(r.SOURCES)
    assert 'CISO' not in r.allowed_sources('CISO')
    with pytest.raises(ValueError): r.allowed_sources('TVA')
    with pytest.raises(ValueError): r._bank({},('CISO','AZPS'))


def test_cache_registers_double_excluded_donors_and_calibration():
    subset=tuple(sorted(r.SOURCES[:-1])); outer=r.SOURCES[-1]
    cache=r.ShapeCache({},[subset],{subset:[outer]})
    for cal in subset:
        inner=tuple(x for x in subset if x!=cal)
        assert (cal,'calibration') in cache.requests[inner]
        for donor in inner:
            nested=tuple(x for x in inner if x!=donor)
            assert (donor,'proper') in cache.requests[nested]
            assert outer not in nested and cal not in nested and donor not in nested


def test_center_is_ratio_of_weighted_q_and_zero_strength_identity():
    class Bank:
        q_=np.ones((2,24,8)); fuel_support_=np.ones((2,8),bool)
        days_=pd.date_range('2024-01-25',periods=2,tz='UTC')
        def _weights(self,*args): return np.array([.25,.75]),None
    bank=Bank(); bank.q_[0,:,0]=np.linspace(.1,2,24); bank.q_[1,:,1]=np.linspace(3,.2,24)
    mix=np.full((1,8),1/8); days=pd.date_range('2025-01-01',periods=1,tz='UTC')
    query=np.ones((1,24,8)); donors=np.ones((2,24,8)); audit={}
    result=r.center_paths(bank,mix,days,query,query,donors,donors,alpha=0.,diagnostics=audit)
    expected=r.recompose_paths(mix[0],.25*bank.q_[0]+.75*bank.q_[1])
    np.testing.assert_array_equal(result[0],expected)
    assert audit['fullbank_q_mean_max_error']==0
    assert not np.allclose(expected,.25*r.recompose_paths(mix[0],bank.q_[0])+.75*r.recompose_paths(mix[0],bank.q_[1]))


def test_round2_source_requires_prior_before_creating_directory(tmp_path,monkeypatch):
    monkeypatch.setattr(r.v11,'verify',lambda _:dict(data=str(tmp_path/'data'),weather_csv='x',points_json='y'))
    with pytest.raises(ValueError,match='requires --prior-round'):
        r.freeze(tmp_path/'new',tmp_path/'data',round_number=2)
    assert not (tmp_path/'new').exists()


def test_target_gate_requires_other_source_round(tmp_path,monkeypatch):
    calls=[]
    monkeypatch.setattr(r,'verify_phase',lambda p,s:calls.append((Path(p),s)))
    monkeypatch.setattr(r,'verify',lambda _:dict(round=2,prior_round=str(tmp_path/'r1')))
    peer=r._target_source_gate(tmp_path/'r1',dict(round=1),tmp_path/'r2')
    assert peer==tmp_path/'r2'
    assert calls==[(tmp_path/'r1','source'),(tmp_path/'r2','source')]
    monkeypatch.setattr(r,'verify',lambda _:dict(round=1))
    with pytest.raises(ValueError,match='Both V12 source rounds'):
        r._target_source_gate(tmp_path/'r1',dict(round=1),tmp_path/'r2')


def test_qrf_day_ids_share_uniforms_with_global_and_preserve_pairs():
    class Bank:
        _rng=staticmethod(lambda seed,day:np.random.default_rng(seed+day.day))
    days=pd.date_range('2025-01-01',periods=2,tz='UTC')
    uniforms=r._uniforms(Bank(),days)
    assert uniforms.shape==(2,128)
    np.testing.assert_array_equal(uniforms[0],Bank._rng(r.SEED,days[0]).random((128,8))[:,1])


def test_shape_scores_keep_energy_and_level_diagnostics():
    paths=np.zeros((2,128,24)); truth=np.full((2,24),10.)
    metrics=r._extra_metrics(paths.mean(1),paths,truth,paths.mean(1))
    np.testing.assert_array_equal(metrics['level_crps'],[10.,10.])
    np.testing.assert_array_equal(metrics['centered_shape_energy_score'],[0.,0.])


def test_native_exact_archives_and_total_risk_never_double_coarse_noise(tmp_path,monkeypatch):
    days=pd.date_range('2025-01-01',periods=2,tz='UTC')
    class Bank:
        source_regions_=tuple(r.SOURCES[:3])
        days_=pd.date_range('2024-01-25',periods=3,tz='UTC')
        q_=np.ones((3,24,8)); fuel_support_=np.ones((3,8),bool)
        raw_cif_=np.full((3,24),100.); donor_mix_=np.full((3,8),1/8)
        _rng=staticmethod(lambda seed,day:np.random.default_rng(seed+day.day))
        def _weights(self,*args): return np.ones(3)/3,None
    class Shapes:
        def get(self,*args): return np.ones((2,24,8))
    class Scale:
        audit_={}
        def predict(self,x): return np.ones((len(x),2))
    class QRF:
        audit_={}
        def predict_weights(self,x,global_fraction=0.):
            return np.tile([.25,.75,0.],(len(x),1))
        def predict_quantiles(self,x,levels,global_fraction=0.):
            return np.broadcast_to(np.array(levels)[None,:,None],(len(x),len(levels),24))
    bank=Bank(); region='AZPS'
    records={(region,'evaluation'):dict(days=days,features={'B':np.ones((2,25)),'W':np.ones((2,99))})}
    cal=dict(regions=np.array(r.SOURCES[:3]),
             days=pd.date_range('2024-07-01',periods=3,tz='UTC'),errors=np.arange(72.).reshape(3,24),
             model=Scale(),qrf=QRF(),scales=np.ones((3,2)),audit=[])
    monkeypatch.setattr(r,'_bank',lambda *a:bank)
    monkeypatch.setattr(r,'_indices',lambda *a:(np.zeros((2,128),int),np.zeros((2,128),int)))
    monkeypatch.setattr(r,'_donor_predictions',lambda *a:np.ones((3,24,8)))
    mix=np.full((2,8),1/8); draws=np.broadcast_to(mix[:,None],(2,128,8)).copy()
    frozen=dict(round=2,alpha=.5,native_paths={})
    def write_baselines(draws):
        native=np.stack([r.recompose_paths(draws[i],bank.q_[np.zeros(128,int)]) for i in range(2)])
        simple=bank.raw_cif_[np.zeros((2,128),int)]+((draws-bank.donor_mix_[np.zeros((2,128),int)]) @ r.EF)[...,None]
        for mode,raw in [('native_recompose',native),('simple_anchor',simple)]:
            path=tmp_path/f'{mode}.npz'; ensemble=np.clip(raw,0,1000)
            np.savez(path,origins=days.as_unit('ns').asi8,raw_ensemble=raw,ensemble=ensemble,
                     point=ensemble.mean(1),center=raw.mean(1),donor_indices=np.zeros((2,128),int))
            frozen['native_paths'][f'target/{region}/{mode}']=str(path)
    write_baselines(draws)
    first=r._predict_region({},records,Shapes(),bank.source_regions_,region,mix,draws,{'B':cal,'W':cal},frozen,'target')[0]
    draws[:]=0; draws[:,:,0]=1
    # Different coarse particles change both strong baselines and structural
    # donor paths, while every end-to-end error reference keeps its center.
    with pytest.raises(ValueError,match='exact identity'):
        r._predict_region({},records,Shapes(),bank.source_regions_,region,mix,draws,{'B':cal,'W':cal},frozen,'target')
    write_baselines(draws)
    second=r._predict_region({},records,Shapes(),bank.source_regions_,region,mix,draws,{'B':cal,'W':cal},frozen,'target')[0]
    assert len(first)==13
    for budget in ('B','W'):
        for mode in ('global_total','hetero_total','qrf_0','qrf_25','qrf_50'):
            np.testing.assert_array_equal(first[budget,mode]['raw'],second[budget,mode]['raw'])
        assert np.max(first[budget,'qrf_0']['conditional_residual_indices'])==1
    assert not np.array_equal(first['W','donor']['raw'],second['W','donor']['raw'])
    assert first['W','donor']['amplitude_error']<1e-12
    assert first['W','qrf_0']['raw_conditional_quantiles'].shape==(2,3,24)


def test_target_requires_explicit_predict_or_score_mode_and_source_rejects_flags(tmp_path):
    for args in ({},{'predict_only':True,'score_only':True}):
        with pytest.raises(ValueError,match='exactly one'):
            r.run_phase(tmp_path/'new',tmp_path/'data','target',round_number=1,**args)
    with pytest.raises(ValueError,match='target-only'):
        r.run_phase(tmp_path/'new',tmp_path/'data','source',round_number=1,predict_only=True)
    assert not (tmp_path/'new').exists()


def test_score_phase_cannot_read_any_target_truth_without_other_prediction_seal(tmp_path,monkeypatch):
    peer=tmp_path/'peer'; peer.mkdir()
    seal=dict(other_round_source=str(peer))
    r.write(tmp_path/'target_prediction_seal.json',seal)
    monkeypatch.setattr(r,'check_entries',lambda *a:None)
    monkeypatch.setattr(r,'_target_source_gate',lambda *a:peer)
    monkeypatch.setattr(r.v10,'_truth_lookup',lambda *a:pytest.fail('labels opened'))
    with pytest.raises(ValueError,match='Both target prediction seals required'):
        r.score_phase(tmp_path,{},'target',seal)
    assert not (tmp_path/'target_scores').exists()


def test_target_predict_only_cannot_overwrite_sealed_predictions(tmp_path,monkeypatch):
    data=tmp_path/'data'; seal=tmp_path/'target_prediction_seal.json'; seal.write_text('{}')
    monkeypatch.setattr(r,'verify',lambda *a:dict(round=1,data=str(data)))
    monkeypatch.setattr(r,'_target_source_gate',lambda *a:tmp_path/'peer')
    monkeypatch.setattr(r,'_load_training',lambda *a:pytest.fail('attempted new prediction'))
    with pytest.raises(FileExistsError,match='immutable'):
        r.run_phase(tmp_path,data,'target',round_number=1,predict_only=True)
    assert seal.read_text()=='{}'


def test_archived_qrf_weights_and_source_bank_are_independently_auditable():
    n=2; regions=np.array(r.SOURCES[:2],dtype='U')
    errors=np.arange(48.).reshape(2,24); ids=np.zeros((n,128),dtype=int)
    center=np.full((n,24),100.)
    arrays=dict(calibration_errors=errors,calibration_regions=regions,
        calibration_origins=pd.date_range('2024-07-01',periods=2,tz='UTC').as_unit('ns').asi8,
        conditional_weights=np.tile([.25,.75],(n,1)),residual_uniforms=np.full((n,128),.1),
        conditional_residual_indices=ids,center=center,raw_ensemble=center[:,None]+errors[ids])
    r._check_qrf_archive(arrays,n,tuple(regions))
    bad={**arrays,'conditional_weights':np.tile([.25,.25],(n,1))}
    with pytest.raises(ValueError,match='conditional weights/source calibration'):
        r._check_qrf_archive(bad,n,tuple(regions))
    bad={**arrays,'calibration_regions':np.array(['AZPS',regions[1]],dtype='U')}
    with pytest.raises(ValueError,match='conditional weights/source calibration'):
        r._check_qrf_archive(bad,n,tuple(regions))
    bad={**arrays,'conditional_residual_indices':np.full((n,128),2,dtype=int)}
    with pytest.raises(ValueError,match='conditional weights/source calibration'):
        r._check_qrf_archive(bad,n,tuple(regions))
    bad={**arrays,'raw_ensemble':arrays['raw_ensemble']+1}
    with pytest.raises(ValueError,match='detached'):
        r._check_qrf_archive(bad,n,tuple(regions))
