"""Synthetic stage-order, source-role and aggregation checks; never actual inputs."""
from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts import run_disjoint_calibration_v50 as run


def test_existing_attempt_rejected_before_contract(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'contract', lambda: pytest.fail('existing attempt read dependencies'))
    with pytest.raises(FileExistsError): run.prepare()


def test_prediction_restart_preserves_marker(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    (tmp_path/'prediction_started.json').write_text('{}')
    monkeypatch.setattr(run, 'monthly_rows', lambda *_: pytest.fail('restart consumed rows'))
    with pytest.raises(FileExistsError): run.predict()


def test_calibration_requires_complete_prediction_before_any_response(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    def incomplete(): raise ValueError('incomplete77')
    monkeypatch.setattr(run, 'predictions', incomplete)
    monkeypatch.setattr(run.helper, 'external_truth', lambda *_: pytest.fail('calibration response opened'))
    with pytest.raises(ValueError, match='incomplete77'): run.calibrate()
    assert not (tmp_path/'calibration_started.json').exists()


def test_score_requires_complete_calibration_before_target_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    def incomplete(): raise ValueError('unsealed_calibration')
    monkeypatch.setattr(run, 'calibration', incomplete)
    monkeypatch.setattr(run.common, 'source_days', lambda *_: pytest.fail('2025 response opened'))
    with pytest.raises(ValueError, match='unsealed_calibration'): run.score()
    assert not (tmp_path/'score_started.json').exists()


def test_external_support_lag_scenarios_require_all_ten(monkeypatch):
    paths = {s: '/fake/'+s+'_hourly.csv' for s in run.EXTERNAL}
    monkeypatch.setattr(run.helper, 'metadata', lambda p, s: [s+'/2024-07-01'])
    support, counts = run.external_support(paths)
    assert set(support) == set(run.EXTERNAL)
    assert all(counts[str(lag)][source] == 1 for lag in run.LAGS for source in run.EXTERNAL)
    monkeypatch.setattr(run.helper, 'metadata', lambda p, s: [] if s == 'SCL' else [s+'/2024-07-01'])
    with pytest.raises(ValueError): run.external_support(paths)


def test_external_report_must_be_parent_bound_before_read_and_cannot_add_paths(monkeypatch):
    report_path='/fake/report.json'
    records=[];closure={report_path:'report-sha'}
    for source in run.EXTERNAL:
        folder='external-v4-data' if source in run.EXTERNAL[:6] else 'external-v5-data'
        path=str(run.ROOT/'results'/folder/(source+'_hourly.csv'))
        closure[path]='file-sha'
        records.append(dict(region=source,path=path,input_sha256='file-sha',status='present_metadata_verified'))
    report=dict(fixed_donor_sources=list(run.EXTERNAL),metadata_only=True,records=records)
    monkeypatch.setattr(run,'read',lambda _:pytest.fail('unbound/tampered report was parsed'))
    with pytest.raises(ValueError):run.external_report({},report_path)
    def changed(_):raise ValueError('changed bound report')
    monkeypatch.setattr(run,'verify',changed)
    with pytest.raises(ValueError,match='changed bound'):run.external_report(closure,report_path)
    monkeypatch.setattr(run,'verify',lambda _:None)
    monkeypatch.setattr(run,'read',lambda _:report)
    assert len(run.external_report(closure,report_path)[0])==10
    records[0]['path']='/not-in-parent/SWPP_hourly.csv'
    with pytest.raises(ValueError,match='parent input'):run.external_report(closure,report_path)


def task_entries():
    return [dict(outer=o, held=h, pair=sorted((o, h)), label_region=h,
                 path='/fake/'+o+'__'+h+'.npz', risk_path='/risk/'+o+'__'+h+'.npz',
                 source_label_path='/labels/'+h+'.npz', input_path='/inputs/'+h+'.npz')
            for o in run.SOURCES for h in run.SOURCES if o != h]


def test_proper_tasks_exactly_six_and_no_reverse_or_held_alias():
    entries = task_entries()
    assert {e['held'] for e in run.proper_tasks(entries, 'BPAT')} == set(run.SOURCES)-{'BPAT'}
    with pytest.raises(ValueError): run.proper_tasks(entries[:-1], 'BPAT')
    entries[0]['label_region'] = 'BPAT'
    with pytest.raises(ValueError): run.proper_tasks(entries, 'BPAT')


def test_capacity_filters_metadata_before_values(tmp_path):
    path = tmp_path/'cap.csv'
    columns = ['year', 'region', *['capacity_'+s+'_mw' for s in run.helper.FUELS], 'capacity_pos_mw', 'capacity_total_mw']
    pd.DataFrame([['2022', 'SWPP', *([1.]*8), 4., 8.],
                  ['2023', 'SWPP', *(['poison']*8), 'poison', 'poison'],
                  ['2022', 'PACE', *(['poison']*8), 'poison', 'poison']], columns=columns).to_csv(path, index=False)
    q, c = run.capacity(str(path), 'SWPP', run.CAL_DAYS[0])
    np.testing.assert_array_equal(q, np.full(4, .25)); assert c == 4
    with pytest.raises(ValueError): run.capacity(str(path), 'PACE', run.CAL_DAYS[0])


def test_query_rows_keep_only_whitelisted_strings(tmp_path):
    legacy, v4, v5 = [tmp_path/name for name in ('legacy.csv', 'v4.csv', 'v5.csv')]
    for p, regions in ((legacy, ['BPAT', 'PACE']), (v4, ['SWPP', 'AVA', 'LGEE']), (v5, ['SCL', 'WACM', 'PACE'])):
        pd.DataFrame(dict(region=regions, fuelshare_coal=['poison']*len(regions))).to_csv(p, index=False)
    train, query = run.monthly_rows(dict(legacy_monthly=str(legacy), external_monthly=[str(v4), str(v5)]))
    assert [r['region'] for r in train] == ['BPAT']
    assert {r['region'] for r in query} == {'BPAT', 'SWPP', 'AVA', 'SCL', 'WACM'}
    assert all(r['fuelshare_coal'] == 'poison' for r in query)


def test_query_loader_bank_provenance_is_a_metadata_whitelist(tmp_path):
    p = tmp_path/'query.npz'
    np.savez(p, bank_regions=['CISO'], bank_days=np.array([1], np.int64), bank_row_ids=[0], bank_cutoff=np.array(2),
             forbidden_truth=np.array([object()], object))
    result = run.bank_query(str(p))
    assert set(result) == {'bank_regions', 'bank_days', 'bank_row_ids', 'bank_cutoff'}


def test_bank_actual_regions_checked_before_poison_q_decode(tmp_path):
    path=tmp_path/'bad_bank.npz'
    expected=[s for s in run.SOURCES if s not in ('BPAT','CISO')]
    np.savez(path,excluded=['BPAT','CISO'],source_regions=expected,regions=['BPAT',*expected[1:]],
             q=np.array([object()],object))
    frozen=dict(banks_by_outer={'BPAT':[str(path),'/never-second-bank']},query_paths={'BPAT':'/never-query'})
    with pytest.raises(ValueError,match='exclude outer before numeric'):
        run.bank6(frozen,'BPAT')


def test_infinite_quantile_is_json_null_and_flag():
    row = run.quantile_record(dict(q=np.inf, K=6, finite_level=1.05, rule='HCP', source_counts={'SWPP':1}),
                              'BPAT', 0, 0, 90)
    assert row['q'] is None and row['is_infinite'] is True


def test_audit_json_real_nested_numpy_structures_roundtrip_and_no_nonfinite_hiding(tmp_path):
    audit=dict(risk_diagnostics=[dict(cqr_crossing_mask=np.array([[True,False]]),count=np.int64(1))],
        risk_audits=[dict(training_regions=np.array(['CISO','ERCO']),sample_weight_days=np.array([1.,1.]),
                         oof_bias_clipped=np.array([[2.,3.]]),location_scale_log_target=np.array([[.1,.2]]))],
        point_fit=dict(center_fits=[dict(training_X=np.arange(6.).reshape(2,3),training_y=np.eye(2),
                                        coefficients=np.ones((2,3)),intercept=np.array([0.,1.]))]))
    path=tmp_path/'audit.json';run.write(path,audit)
    actual=json.loads(path.read_text())
    assert actual['risk_diagnostics'][0]['cqr_crossing_mask']==[[True,False]]
    assert actual['risk_audits'][0]['training_regions']==['CISO','ERCO']
    assert actual['point_fit']['center_fits'][0]['training_X']==[[0.,1.,2.],[3.,4.,5.]]
    for i,value in enumerate((np.array([np.inf]),np.float64(np.nan),{'a':float('-inf')})):
        with pytest.raises(ValueError): run.write(tmp_path/f'bad{i}.json',{'v':value})
    with pytest.raises(FileExistsError): run.write(path,audit)


def test_prediction_archive_and_real_audit_write_integration(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    (tmp_path/'predictions').mkdir()
    n=2;source='SWPP'
    size=np.zeros((n,128,31));size[...,:5]=.2
    cap=np.concatenate((size,np.broadcast_to([.25,.25,.25,.25,1.],(n,128,5))),axis=-1)
    value=dict(region=source,origins=run.CAL_DAYS[:n].asi8,
               keys=np.array([f'{source}/{at:%Y-%m-%d}' for at in run.CAL_DAYS[:n]]),
               features=dict(Size=size,SizeCap=cap),MG_draws=size[...,:5],audit=dict(legal=True))
    prediction=dict(raw=np.full((5,n,128,24),500.),audit=dict(new_fits=np.int64(0)))
    class Risk:
        def predict(self,X,p,h,l,s,regions,keys):
            return dict(lo0=np.broadcast_to(p-1,(6,n,24)).copy(),hi0=np.broadcast_to(p+1,(6,n,24)).copy(),
                        scales=np.ones((6,n,24)),centers=np.broadcast_to(p,(6,n,24)).copy(),
                        diagnostics=dict(cqr_crossing_mask=np.zeros((n,24),bool)),
                        audit=dict(new_fits=np.int64(0),names=('a','b')))
    entry=run._save_prediction('BPAT',source,value,prediction,[Risk() for _ in range(5)],0.)
    with np.load(entry['path'],allow_pickle=False) as z:
        assert z['lo0'].shape==(5,6,n,24) and np.all(z['point']==500.)
    audit=json.loads(Path(entry['audit_path']).read_text())
    assert audit['risk_diagnostics'][0]['cqr_crossing_mask']==[[False]*24]*n
    assert audit['risk_audits'][0]['new_fits']==0


def test_region_equal_macro_not_day_pooled_and_loso():
    rows = []
    for i, source in enumerate(run.SOURCES):
        for day in range(i+1):
            rows.append(dict(region=source, base='a', budget='B0', method='GlobalABS', lag=7, rule='HCP',
                             **{name: float(i) for name in run.METRICS}))
    region, macro, loso = run.tables(pd.DataFrame(rows), ['base', 'budget', 'method', 'lag', 'rule'])
    assert len(region) == 7 and len(macro) == 1 and len(loso) == 7
    assert macro.iloc[0].width90 == 3
    assert loso.loc[loso.excluded_region == 'BPAT', 'width90'].iloc[0] == 3.5


def test_partial_prediction_registry_fails_before_arrays(tmp_path, monkeypatch):
    monkeypatch.setattr(run, 'OUT', tmp_path)
    monkeypatch.setattr(run, 'check', lambda: {})
    run.write(tmp_path/'prediction_started.json', {})
    run.write(tmp_path/'prediction_seal.json', dict(complete_grid=True, entries=[], proper=[]))
    monkeypatch.setattr(np, 'load', lambda *_a, **_k: pytest.fail('partial grid opened numeric archive'))
    with pytest.raises(ValueError): run.predictions()


def test_calibration_phase_exact_source_weights_intervals_and_second_seal(tmp_path,monkeypatch):
    monkeypatch.setattr(run,'OUT',tmp_path)
    predictions=tmp_path/'predictions';predictions.mkdir()
    entries=[]
    for outer in run.SOURCES:
        for source in (outer,*run.EXTERNAL):
            n=len(run.TEST_DAYS) if source==outer else len(run.CAL_DAYS)
            p=predictions/f'{outer}__{source}.npz'
            run._save(p,lo0=np.full((5,6,n,24),500.),hi0=np.full((5,6,n,24),500.),scales=np.ones((5,6,n,24)))
            entries.append(dict(outer=outer,region=source,path=str(p)))
    frozen=dict(external_paths={s:'/fake/'+s+'_hourly.csv' for s in run.EXTERNAL},
        cal_keys={s:[s+'/2024-07-01'] for s in run.EXTERNAL},
        lag_counts={str(lag):{s:1 for s in run.EXTERNAL} for lag in run.LAGS})
    run.write(tmp_path/'freeze.json',frozen);run.write(tmp_path/'prediction_seal.json',{})
    monkeypatch.setattr(run,'check',lambda:frozen)
    events=[]
    def sealed():events.append('seal');return entries
    monkeypatch.setattr(run,'predictions',sealed)
    def truth(path,source,keys):
        assert events[0]=='seal' and (tmp_path/'calibration_started.json').is_file()
        assert keys==[source+'/2024-07-01']
        events.append(source)
        return np.array([0]),np.full((1,24),501.+run.EXTERNAL.index(source))
    monkeypatch.setattr(run.helper,'external_truth',truth)
    monkeypatch.setattr(run.risk.HistGradientBoostingRegressor,'fit',
        lambda *_a,**_k:pytest.fail('Calibration refit a risk head'))
    run.calibrate()
    assert events==['seal',*run.EXTERNAL]
    records=json.loads((tmp_path/'calibration.json').read_text())['records']
    assert len(records)==1260
    assert all(row['q']==(9. if row['rule']=='equal_region' else 10.) for row in records)
    final=run.calibration()
    assert len(final)==7
    with np.load(final[0]['path'],allow_pickle=False) as z:
        np.testing.assert_array_equal(z['lo'][:,:,:,0],491.)
        np.testing.assert_array_equal(z['hi'][:,:,:,0],509.)
        np.testing.assert_array_equal(z['lo'][:,:,:,1],490.)
        np.testing.assert_array_equal(z['hi'][:,:,:,1],510.)
    with pytest.raises(FileExistsError):run.calibrate()


def test_full_parity_reads_large_archive_fields_once(monkeypatch):
    days=run.TEST_DAYS[:2];monkeypatch.setattr(run,'TEST_DAYS',days)
    t=len(days);keys=np.array([f'BPAT/{at:%Y-%m-%d}' for at in days])
    mg=np.zeros((t,128,5));mg[...,1]=1
    fine=np.zeros((t,128,8));fine[...,3]=1
    mean=np.full((t,128,4),.25)
    result=dict(MG_draws=mg,A=np.zeros((t,128,24)),R=np.zeros((t,128,24)),fine=fine,
        U=np.zeros((t,128)),P=np.zeros((t,128)),raw=np.zeros((5,t,128,24)),query_mean=mean,
        query_scale=np.ones((t,128)),source_scale=np.ones((t,128)),scale_ratio=np.ones((t,128)),
        donor_ids=np.zeros((t,128),int),fallback_ids=np.full((t,128),-1,int),
        effective_pos_ids=np.zeros((t,128),int),effective_month_ids=np.zeros((t,128),int),
        center_raw_ratios=np.full((2,t,128,4),.25),center_ratios=np.full((2,t,128,4),.25),
        center_EF=np.full((2,t,128),650.))
    scale=SimpleNamespace(oof_mean=np.full((1,4),.25),oof_scale=np.ones(1),oof_residual=np.zeros((1,4)),
        oof_covariance3=np.zeros((1,3,3)),oof_weights=np.ones((1,1)),oof_leaf_ids=np.zeros((1,128),int),
        final=SimpleNamespace(metadata=dict(training_keys=['CISO/2023-01']),X=np.full((1,5),.2),
                              y=np.full((1,4),.25),weights=np.ones(1)))
    query=dict(region='BPAT',MG_mix=mg[:,0],MG_metadata=np.zeros((t,4)),
               features=dict(Size=np.zeros((t,128,31)),SizeCap=np.zeros((t,128,36))))
    class Archive(dict):
        def __init__(self,**kwargs):super().__init__(**kwargs);self.reads={}
        def __getitem__(self,name):self.reads[name]=self.reads.get(name,0)+1;return super().__getitem__(name)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    v29=Archive(origins=days.asi8,arms=np.array(run.common.V29_ARMS),raw=np.zeros((10,t,128,24)),
        fine_draws=np.broadcast_to(fine,(10,*fine.shape)),MG_draws=mg,query_mean=mean,
        query_scale=result['query_scale'],source_scale=result['source_scale'],scale_ratio=result['scale_ratio'],
        donor_ids=result['donor_ids'],fallback_ids=result['fallback_ids'],effective_pos_ids=result['effective_pos_ids'],
        effective_month_ids=result['effective_month_ids'],bank_oof_mean=scale.oof_mean,bank_oof_scale=scale.oof_scale,
        bank_oof_residuals=scale.oof_residual,bank_oof_covariance3=scale.oof_covariance3,
        bank_oof_weights=scale.oof_weights,bank_oof_leaf_ids=scale.oof_leaf_ids,training_keys=np.array(['CISO/2023-01']),
        training_X=scale.final.X,training_ratios=scale.final.y,training_weights=scale.final.weights)
    v47=Archive(origins=days.asi8,keys=keys,arms=np.array(['Scale-anchor']),raw=np.zeros((1,t,128,24)),
        query_feature_Size=query['features']['Size'],query_feature_SizeCap=query['features']['SizeCap'],
        head_EF_raw=np.full((12,t,128),650.),EF_draws=np.full((15,t,128),650.),head_projection_l1=np.zeros((12,t,128)))
    v48=Archive(origins=days.asi8,keys=keys,arms=np.array([*run.point.CACHE_NAMES[1:],*[f'extra{i}' for i in range(71)]]),
                raw=np.zeros((75,t,128,24)))
    files=dict(v25=Archive(origins=days.asi8,MG_mix=query['MG_mix'],MG_metadata=query['MG_metadata']),
               v29=v29,v47=v47,v48=v48)
    monkeypatch.setattr(np,'load',lambda path,**kwargs:files[path])
    errors=run.parity({name:name for name in files},query,result,dict(scale=scale))
    assert max(errors.values())==0
    assert v48.reads['raw']==1 and v47.reads['raw']==1
    assert all(v29.reads[name]==1 for name in ('raw','fine_draws','MG_draws','query_mean'))


def test_query_inputs_cached_once_per_source_and_readonly(monkeypatch):
    calls=[]
    monkeypatch.setattr(run,'capacity',lambda *_:(np.full(4,.25),1000.))
    def query(rows,source,dates,q=None,c=None):
        calls.append((source,q is not None))
        return dict(region=source,features=dict(Size=np.ones((2,128,31))),MG_draws=np.full((2,128,5),.2))
    monkeypatch.setattr(run.point,'query',query)
    cached=run.query_cache({'capacity_path':'/fake'},[])
    assert len(calls)==34 and len(set(calls))==34
    assert all(not value['MG_draws'].flags.writeable and not value['features']['Size'].flags.writeable
               for pair in cached.values() for value in pair)
    def prediction(ctx,value):
        return dict(raw=np.full((5 if value is cached['SWPP'][0] else 3,2,128,24),500.))
    monkeypatch.setattr(run.point,'predict',prediction)
    run._query({},'SWPP',cached);run._query({},'SWPP',cached)
    assert len(calls)==34
    with pytest.raises(ValueError): cached['SWPP'][0]['MG_draws'][0,0,0]=.1


@pytest.mark.parametrize('corrupt_point',[False,True])
def test_proper_reads_only_six_held_h1_and_never_npz_truth_errors(tmp_path,monkeypatch,corrupt_point):
    days=pd.DatetimeIndex(pd.to_datetime(['2024-01-25','2024-01-26','2024-07-01'],utc=True)).as_unit('ns')
    monkeypatch.setattr(run,'HISTORY_DAYS',days)
    all_entries=task_entries()
    frozen=dict(proper_tasks={s:[e for e in all_entries if e['outer']==s] for s in run.SOURCES},
                H1_keys={},source_paths={s:str(tmp_path/(s+'_hourly.csv')) for s in run.SOURCES})
    names=('Scale-anchor','Scale-recompose','B0-SizeET-KeepA',*run.BASES[1:3],
           'B1-SizeCapRidge-KeepA',*run.BASES[3:])
    for e in frozen['proper_tasks']['BPAT']:
        held=e['held'];keys=np.array([f'{held}/{at:%Y-%m-%d}' for at in days])
        frozen['H1_keys'][held]=keys[:2].tolist()
        for field in ('path','risk_path','input_path','source_label_path'):
            e[field]=str(tmp_path/(held+'_'+field+'.npz'))
        np.savez(e['source_label_path'],keys=keys,origins=days.asi8,positions=np.arange(3),
                 truth=np.array([object()],object))
        np.savez_compressed(e['path'],keys=keys,origins=days.asi8,arms=np.array(names),raw=np.full((8,3,128,24),500.))
        size=np.zeros((3,128,31));size[...,:5]=.2
        cap=np.concatenate((size,np.broadcast_to([.25,.25,.25,.25,1.],(3,128,5))),axis=-1)
        np.savez(e['input_path'],keys=keys,origins=days.asi8,feature_Size=size,feature_SizeCap=cap,
                 MG_draws=size[...,:5])
        np.savez(e['risk_path'],keys=keys,arms=np.array(names),point=np.full((8,3,24),499. if corrupt_point else 500.),
                 hourly_error=np.array([object()],object))
    calls=[]
    def source_days(path,start,stop):
        assert path.name!='BPAT_hourly.csv' and (start,stop)==('2024-01-25','2024-07-01')
        calls.append(path.name)
        g=np.zeros((24,8));g[:,1]=1
        return [(at,g) for at in days[:2]]
    monkeypatch.setattr(run.common,'source_days',source_days)
    if corrupt_point:
        with pytest.raises(ValueError,match='cached risk point'): run.proper(frozen,'BPAT')
        assert calls==[]
    else:
        data=run.proper(frozen,'BPAT')
        assert len(calls)==6 and set(data['regions'])==set(run.SOURCES)-{'BPAT'}
        assert data['truth'].shape==(12,24) and np.all(data['truth']==400.)
        assert data['X'][0].shape==(12,24,40) and data['X'][3].shape==(12,24,45)
        assert all(key.endswith(('2024-01-25','2024-01-26')) for key in data['keys'])
