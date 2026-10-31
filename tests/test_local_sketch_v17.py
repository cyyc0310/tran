"""Synthetic information closure, time gates, disjoint learning and source seals."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts import local_sketch_v17 as m


def packet_table(packet,arrays,dates=None):
    dates=pd.date_range('2024-01-01',periods=len(arrays),tz='UTC') if dates is None else dates
    return pd.DataFrame([dict(region='BPAT',day=d,**{f'v{i}':v for i,v in enumerate(a)}) for d,a in zip(dates,arrays)],
        columns=['region','day',*[f'v{i}' for i in range(m.PACKETS[packet])]])


def test_full_packet_channels_are_invertible_not_hidden_information():
    a=np.random.default_rng(1).uniform(.2,3,(24,8));packets=m.packets_from_day(a)
    for packet,values in packets.items():
        c=m.channels(values,packet);g=np.expm1(c[-2]);anchor=c[-1]/1000
        if packet=='D8':recovered=c[:8]*g
        elif packet=='T8':
            bg=c[4:8]*g;recovered=np.column_stack((c[:4]/1000*bg,bg)).ravel()
        elif packet=='G5':recovered=np.r_[anchor*g,c[:4]*g]
        else:recovered=np.array([anchor*g,g])
        np.testing.assert_allclose(recovered,values,rtol=1e-12,atol=1e-12)


def test_denominator_control_and_daily_vs_block_information_negative_control():
    a=np.full((24,8),2.);b=a.copy()
    b[:6,0]+=1;b[:6,1]-=1;b[6:12,0]-=1;b[6:12,1]+=1
    p,q=m.packets_from_day(a),m.packets_from_day(b)
    for packet in ('D2','D8','G5'):np.testing.assert_allclose(p[packet],q[packet],atol=1e-12)
    assert not np.allclose(p['T8'],q['T8'])
    # Even T8 cannot distinguish within-block phase without more information.
    phase=b.copy();phase[:6]=phase[:6][::-1]
    for packet,v in m.packets_from_day(phase).items():np.testing.assert_allclose(v,q[packet])


def test_G5_exposes_denominator_timing_without_changing_daily_fuels():
    a=np.full((24,8),2.);b=a.copy();b[:6]*=.5;b[6:12]*=1.5
    p,q=m.packets_from_day(a),m.packets_from_day(b)
    for packet in ('D2','D8'):np.testing.assert_array_equal(p[packet],q[packet])
    assert not np.array_equal(p['G5'],q['G5'])
    np.testing.assert_allclose(m.channels(p['T8'],'T8')[:4],m.channels(q['T8'],'T8')[:4])


@pytest.mark.parametrize('lag',[7,30])
def test_all_packets_share_raw_energy_ewma_anchor_and_day_mask(lag):
    rng=np.random.default_rng(2);days=pd.date_range('2024-01-01',periods=45,tz='UTC')
    all_packets=[m.packets_from_day(rng.uniform(.1,10,(24,8))*(1+i)) for i in range(len(days))]
    at=pd.Timestamp('2024-02-20',tz='UTC');anchors=[]
    for packet in m.PACKETS:
        h=m.PacketHistory(packet_table(packet,[p[packet] for p in all_packets],days),packet,lag)
        _,anchor,audit=h.features(at);anchors.append(anchor)
        assert pd.Timestamp(audit['max_release'])<=at and audit['history_days']<=28
    np.testing.assert_allclose(anchors,anchors[0],atol=1e-10,rtol=0)
    eligible=np.flatnonzero(days+pd.Timedelta(days=1+lag)<=at);last=days[eligible[-1]]
    keep=eligible[days[eligible]>=last-pd.Timedelta(days=27)]
    w=np.exp(-np.asarray((last-days[keep]).days)/14);w/=w.sum()
    raw=np.stack([all_packets[i]['D2'] for i in keep]);expected=1000*(w@raw[:,0])/(w@raw[:,1])
    assert anchors[0]==pytest.approx(expected)


def test_runtime_common_anchor_guard_rejects_legal_but_inconsistent_packet():
    from scripts import run_local_sketch_v17 as r
    packets=m.packets_from_day(np.ones((24,8)));at=pd.Timestamp('2024-01-10',tz='UTC')
    histories={(p,'BPAT',7):m.PacketHistory(packet_table(p,[v]),p,7) for p,v in packets.items()}
    for packet in m.PACKETS:
        *_,difference=r.checked_features(histories,packet,'BPAT',7,at)
        assert difference<1e-9
    altered=packets['G5'].copy();altered[0]*=.9  # Valid H/G, timing and shape; wrong shared daily H.
    histories['G5','BPAT',7]=m.PacketHistory(packet_table('G5',[altered]),'G5',7)
    with pytest.raises(ValueError,match='Common D2 anchor mismatch'):
        r.checked_features(histories,'G5','BPAT',7,at)


@pytest.mark.parametrize('packet',list(m.PACKETS))
def test_future_packets_not_converted_or_used(packet):
    rng=np.random.default_rng(3);days=pd.date_range('2024-01-01',periods=60,tz='UTC')
    arrays=[m.packets_from_day(rng.uniform(.1,10,(24,8)))[packet] for _ in days]
    t=packet_table(packet,arrays,days);at=pd.Timestamp('2024-02-01',tz='UTC')
    x,a,d=m.PacketHistory(t,packet,7).features(at)
    damaged=t.copy();columns=[c for c in t if c.startswith('v')];damaged[columns]=damaged[columns].astype(object)
    damaged.loc[days+pd.Timedelta(days=8)>at,columns]='POISON_FUTURE'
    xx,aa,dd=m.PacketHistory(damaged,packet,7).features(at)
    np.testing.assert_array_equal(x,xx);assert a==aa and d==dd


def test_no_history_fallback_does_not_drop_query_and_release_boundary_exact():
    at=pd.Timestamp('2024-01-09',tz='UTC')
    for packet in m.PACKETS:
        values=m.packets_from_day(np.ones((24,8)))[packet]
        t=packet_table(packet,[values]);h=m.PacketHistory(t,packet,7)
        x,a,d=h.features(at-pd.Timedelta(days=1));assert a==500 and d['missing'] and x[-1]==1
        x,a,d=h.features(at);assert not d['missing'] and d['history_days']==1
        empty=packet_table(packet,[]);xx,aa,dd=m.PacketHistory(empty,packet,7).features(at)
        assert np.isfinite(xx).all() and aa==500 and dd['missing']


def data(period,n=40):
    rng=np.random.default_rng(12);x=rng.normal(size=(n,4));y=rng.uniform(100,500,(n,24))
    labels=np.repeat(['BPAT','CISO'],n//2)
    days=pd.DatetimeIndex(list(pd.date_range(m.PERIODS[period][0],periods=n//2,tz='UTC'))*2)
    return x,y,labels,days


@pytest.mark.parametrize('period',['proper','riskfit','riskpool'])
def test_held_region_and_wrong_time_rejected(period):
    x,y,r,d=data(period)
    with pytest.raises(ValueError,match='excluding held'):
        m.validate_training(x,y,r,d,period,'BPAT')
    with pytest.raises(ValueError,match='fixed training period'):
        m.validate_training(x,y,r,d+pd.Timedelta(days=400),period,'ERCO')


def test_ridge_point_is_source_proper_only_and_finite():
    x,y,r,d=data('proper');anchor=np.full(len(x),300.)
    fit=m.PointModel('ridge').fit(x,y,anchor,r,d,exclude='ERCO')
    assert fit.model.alpha==30 and fit.regions==('BPAT','CISO')
    assert fit.predict(x,anchor).shape==(len(x),24)
    with pytest.raises(ValueError):fit.predict(x,np.full(len(x),np.nan))


def test_qrf_splits_and_scaler_are_out_of_sample_from_riskpool_labels():
    x,y,r,d=data('riskfit',80);px,py,pr,pd_=data('riskpool',40)
    a=m.RiskModel('QRF').fit(x,y-300,r,d,px,py-300,pr,pd_,exclude='ERCO')
    b=m.RiskModel('QRF').fit(x,y-300,r,d,px,py-293,pr,pd_,exclude='ERCO')
    assert a.model.n_estimators==128 and a.model.min_samples_leaf==20 and a.model.bootstrap
    np.testing.assert_array_equal(a.scaler.mean_,b.scaler.mean_)
    for ta,tb in zip(a.model.estimators_,b.model.estimators_):
        np.testing.assert_array_equal(ta.tree_.threshold,tb.tree_.threshold)
        np.testing.assert_array_equal(ta.tree_.value,tb.tree_.value)
    np.testing.assert_array_equal(a.pool_leaves,b.pool_leaves)
    w,_,_=a.weights(px[:3]);ww,_,_=b.weights(px[:3]);np.testing.assert_array_equal(w,ww)
    assert w.shape==(3,len(px)) and np.allclose(w.sum(1),1)


def test_leaf_mapping_uses_all_pool_rows_and_empty_leaf_global_fallback():
    class Identity:
        n_features_in_=1
        def transform(self,x):return x
    class Forest:
        n_estimators=2
        def apply(self,x):return np.array([[1,9]])
    risk=m.RiskModel('QRF');risk.scaler=Identity();risk.model=Forest()
    risk.pool_errors=np.zeros((3,24));risk.pool_regions=np.array(['BPAT','BPAT','CISO'])
    risk.global_weights=m.balanced(risk.pool_regions);risk.pool_leaves=np.array([[1,0],[1,0],[2,0]])
    w,_,empty=risk.weights(np.zeros((1,1)))
    np.testing.assert_allclose(w,[[.375,.375,.25]])
    assert empty==1


def test_global_whole_day_sampling_and_constant_hetero_inverse_scale():
    x,y,r,d=data('riskfit');px,py,pr,pd_=data('riskpool')
    pool_errors=py-300;fit_errors=np.full_like(y,10.)
    u=np.tile(np.linspace(0,np.nextafter(1.,0.),128),(2,1));q=px[:2];center=np.full((2,24),350.)
    global_model=m.RiskModel('global').fit(x,fit_errors,r,d,px,pool_errors,pr,pd_,exclude='ERCO')
    hetero=m.RiskModel('hetero').fit(x,fit_errors,r,d,px,pool_errors,pr,pd_,exclude='ERCO')
    g,h=global_model.draw(center,q,u),hetero.draw(center,q,u)
    np.testing.assert_array_equal(g['raw'],center[:,None]+pool_errors[g['residual_indices']])
    np.testing.assert_allclose(h['raw'],g['raw'],atol=1e-10)
    np.testing.assert_allclose(h['query_scales'],np.sqrt(125),atol=1e-10)
    assert hetero.model.n_estimators==150 and hetero.model.min_samples_leaf==10


def test_positive_support_cdf_endpoint_excludes_underflow_tail():
    w=np.array([[0.,.1,.9,0.]])
    ids=m.cdf_ids(w,np.array([[0.,.1,np.nextafter(1.,0.)]]))
    np.testing.assert_array_equal(ids,[[1,2,2]])


def test_proper_and_risk_reader_never_parses_2025_fine(tmp_path):
    from scripts import run_local_sketch_v17 as r
    columns=['generation_'+f for f in m.FUELS]
    t=pd.DataFrame(dict(time=pd.date_range('2024-01-25',periods=24,freq='h',tz='UTC'),region='BPAT',valid=True,
                        **{c:np.ones(24) for c in columns}))
    poisoned=t.iloc[[0]].copy();poisoned['time']='2025-01-01T00:00:00Z';poisoned[columns]='DO_NOT_PARSE'
    p=tmp_path/'BPAT_hourly.csv';pd.concat([t,poisoned]).to_csv(p,index=False)
    days=r.source_days(p,'2024-01-25','2025-01-01');assert len(days)==1
    with pytest.raises(ValueError,match='registered source'):r.source_days(tmp_path/'PACE_hourly.csv','2024-','2025-')


def test_incomplete_seal_rejected_before_2025_fine_reader(tmp_path,monkeypatch):
    from scripts import run_local_sketch_v17 as r
    monkeypatch.setattr(r,'OUT',tmp_path);(tmp_path/'freeze.json').write_text('{}')
    monkeypatch.setattr(r,'check',lambda:({},{}));touched=[]
    monkeypatch.setattr(r,'source_days',lambda *args:touched.append(True))
    r.write(tmp_path/'prediction_seal.json',dict(freeze_sha256=r.sha(tmp_path/'freeze.json'),entries=[],
        complete_grid=True,source2025fine_labels_read=False,target_values_opened=False))
    with pytest.raises(ValueError,match='Exact63prediction'):r.score()
    assert not touched


def test_registered_grid_with_malformed_particle_count_is_rejected(tmp_path,monkeypatch):
    from scripts import run_local_sketch_v17 as r
    monkeypatch.setattr(r,'OUT',tmp_path);folder=tmp_path/'predictions';folder.mkdir()
    entries=[]
    for source in r.SOURCES:
        entries.append(dict(region=source,lag=0,packet='native',models=['native'],
            path=str(folder/f'source-{source}-native.npz'),sha256='pending'))
        for lag in r.LAGS:
            for packet in r.PACKETS:
                entries.append(dict(region=source,lag=lag,packet=packet,models=list(r.MODELS),
                    path=str(folder/f'source-{source}-{lag}-{packet}.npz'),sha256='pending'))
    path=Path(entries[0]['path'])
    with path.open('xb') as stream:
        np.savez_compressed(stream,origins=pd.date_range('2025-01-01','2025-06-30',tz='UTC').asi8,
            raw=np.zeros((181,127,24)),models=np.array(['native']))
    entries[0]['sha256']=r.sha(path)
    seal=dict(entries=entries,complete_grid=True,source2025fine_labels_read=False,target_values_opened=False)
    with pytest.raises(ValueError,match='Invalid prediction array'):r.check_predictions(seal)


def test_selection_keeps_lag30_stress_out_and_matches_G5_same_model_factor():
    from scripts import run_local_sketch_v17 as r
    records=[]
    for packet,method,lag,crps in [('native','native',0,30),('D2','ET_global',7,29),('D8','ET_global',7,28),
                                 ('G5','ET_global',7,27),('T8','ET_global',7,25),('T8','ET_QRF',30,1)]:
        for region in m.SOURCES:
            row=dict(region=region,packet=packet,model=method,lag=lag,factor=1.25,
                **{k:0. for k in r.METRICS});row.update(crps=crps,empirical_coverage90=.92,
                    empirical_width90=100,empirical_interval_score90=crps+100);records.append(row)
    choice=r.selection(pd.DataFrame(records));assert choice['selections']['T8']['lag']==7
    assert choice['source_information_gate'] and not choice['target_test_authorized']
    assert choice['H_timing_gain_detected']
    assert choice['T8_same_model_same_factor_G5']['model']=='ET_global'
    json.dumps(choice,allow_nan=False)
    # A stronger daily H/G baseline blocks information promotion even if D8/native lose.
    altered=pd.DataFrame(records);altered.loc[altered.packet.eq('D2'),'crps']=24.
    assert not r.selection(altered)['source_information_gate']
    # Shared G5 can already explain all gains: no numerator-timing attribution.
    altered=pd.DataFrame(records);altered.loc[altered.packet.eq('G5'),'crps']=24.
    choice=r.selection(altered)
    assert choice['source_information_gate'] and not choice['H_timing_gain_detected']


def test_exclusive_receipt_and_changed_bytes_fail_closed(tmp_path):
    from scripts import run_local_sketch_v17 as r
    p=tmp_path/'seal.json';r.write(p,{'v':1});old=r.sha(p)
    with pytest.raises(FileExistsError):r.write(p,{'v':2})
    p.write_text('{}')
    with pytest.raises(ValueError,match='Changed frozen bytes'):r.verify({str(p):old})
