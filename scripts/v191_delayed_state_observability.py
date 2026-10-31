"""Bounded source-supervision/observability check; no calibration/test scoring."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v188-forward-zero-risk-r1'
OUT=ROOT/'results/v191-delayed-state-observability-r1'
DOC=ROOT/'docs/V191_DELAYED_STATE_OBSERVABILITY.md'


def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[Path(__file__),DOC,ROOT/'scripts/hourly_fuel_data.py',ROOT/'scripts/v151_hourly_law.py',
           ROOT/'results/v144-canonical-profile-cache-r1/profiles.npz',ROOT/'results/v144-canonical-profile-cache-r1/profile_index.csv',
           ROOT/'results/v153-calendar-times-r1/timestamps.npz']
    paths += [BASE/f'fold_{f}'/n for f in range(6) for n in ('training_matrices.npz','training_provenance.json')]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    data=HourlyFuelData(ROOT);states=[];transitions=[];ranks=[];roster=set()
    with np.load(ROOT/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as p:times=p['unix_seconds']
    for fold in range(6):
        proper=json.loads((BASE/f'fold_{fold}/training_provenance.json').read_text())['proper_regions'];roster.update(proper)
        with np.load(BASE/f'fold_{fold}/training_matrices.npz',allow_pickle=False) as p:
            raw=p['raw_id'].copy();profiles=p['profile_id'].copy();regions=p['region'].copy()
        if not set(regions).issubset(proper) or not np.all(times[raw]<1719792000):raise ValueError('proper/time violation')
        if not (data.release.iloc[profiles]<=pd.Timestamp('2024-07-01')).all():raise ValueError('source month not released')
        rows=[];counts=[]
        for region in proper:
            selected=[]
            for q in sorted(set(profiles[regions==region])):
                month=str(data.index.month.iloc[q]);date=pd.to_datetime(month,format='%Y-%m-%d',errors='raise')
                if date.day!=1 or date.strftime('%Y-%m-01')!=month:raise ValueError('canonical month format')
                ids=raw[profiles==q];g=data.raw[ids];a=float(data.shares[q]@EF);mu=float(np.mean(g@EF/g.sum(axis=1)))
                row={'fold':fold,'region':region,'profile_id':int(q),'month':data.index.month.iloc[q],'sampled_hours':len(ids),
                     'energy_anchor':a,'sample_time_mean':mu,'sample_kappa_proxy':a-mu}
                states.append(row);selected.append(row)
            counts.append(len(selected))
            for a,b in zip(selected,selected[1:]):
                gap=(pd.Timestamp(b['month']).year-pd.Timestamp(a['month']).year)*12+pd.Timestamp(b['month']).month-pd.Timestamp(a['month']).month
                if gap==1:
                    row={'fold':fold,'region':region,'from_profile':a['profile_id'],'to_profile':b['profile_id'],
                         'x_a':a['energy_anchor'],'x_k':a['sample_kappa_proxy'],'y_a':b['energy_anchor'],'y_k':b['sample_kappa_proxy']}
                    transitions.append(row);rows.append(row)
        if any(c>2 for c in counts):raise ValueError('unexpected time support')
        n=len(rows);X=np.asarray([[r['x_a'],r['x_k']] for r in rows]);B=np.eye(n)
        rank=int(np.linalg.matrix_rank(np.column_stack([X,B])));brank=int(np.linalg.matrix_rank(B))
        if len({r['region'] for r in rows})!=n or rank!=brank or rank!=n:raise ValueError('intercept alias rank')
        ranks.append({'fold':fold,'proper_regions':len(proper),'zero_states':counts.count(0),'one_state':counts.count(1),
                      'two_states':counts.count(2),'transitions':n,'design_rank':rank,'intercept_rank':brank,'additional_A_rank':rank-brank})
    available=[]
    for r in sorted(roster):
        ix=data.index.loc[data.index.region.eq(r)&data.index.month.between('2024-01-01','2024-05-01')]
        available.append({'region':r,'canonical_months':len(ix),'months':json.dumps(ix.month.tolist()),'scope':'metadata_only_not_additional_hour_labels'})
    A=np.array([[.5,.25],[.125,.5]]);H=np.array([[1.,0.]])
    T=np.block([[A,np.eye(2)],[np.zeros((2,2)),np.eye(2)]]);C=np.array([[1.,0.,0.,0.]])
    v=np.array([0.,1.,-.25,.5]);O=np.vstack([C@np.linalg.matrix_power(T,k) for k in range(8)])
    if not np.array_equal(T@v,v) or np.max(abs(O@v))!=0 or np.linalg.matrix_rank(O)!=3:raise ValueError('gauge observability')
    # Reuse V15's physical two-hour witness, explicitly not a new witness discovery.
    g1=np.array([[0.,1.],[1.,1.]]);g2=np.array([[1.,0.],[0.,2.]]);ef=np.array([1000.,0.])
    a=float(g1.sum(axis=0)@ef/g1.sum());mu1=float(np.mean(g1@ef/g1.sum(axis=1)));mu2=float(np.mean(g2@ef/g2.sum(axis=1)))
    x1=np.array([a,a-mu1]);x2=np.array([a,a-mu2]);b1=(np.eye(2)-A)@x1;b2=(np.eye(2)-A)@x2
    if not np.array_equal(g1.sum(axis=0),g2.sum(axis=0)) or not np.array_equal(g1.sum(axis=1),g2.sum(axis=1)):raise ValueError('physical witness')
    certificate={'A':A.tolist(),'T':T.tolist(),'gauge_vector':v.tolist(),'known_b_observability_rank':int(np.linalg.matrix_rank(np.vstack([H,H@A]))),
                 'unknown_b_augmented_rank':int(np.linalg.matrix_rank(O)),'augmented_dimension':4,'physical_witness_reused_from':'V15',
                 'physical_g1':g1.tolist(),'physical_g2':g2.tolist(),'energy_anchor':a,'time_mean_1':mu1,'time_mean_2':mu2,
                 'x1':x1.tolist(),'x2':x2.tolist(),'b1':b1.tolist(),'b2':b2.tolist()}
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('input changed')
    OUT.mkdir()
    for name,rows in [('source_states',states),('transitions',transitions),('design_ranks',ranks),('early_canonical_metadata',available)]:pd.DataFrame(rows).to_csv(OUT/f'{name}.csv',index=False)
    (OUT/'gauge_certificate.json').write_text(json.dumps(certificate,indent=2)+'\n')
    s={'assertion_failures':0,'transitions_by_fold':[r['transitions'] for r in ranks],'all_additional_A_ranks_zero':True,
       'canonical_metadata_counts':pd.DataFrame(available).canonical_months.value_counts().to_dict(),'models_fitted':False,'cal_test_hour_labels_used':False,
       'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in OUT.iterdir()},'scope':'State-support audit, not a predictor or universal impossibility theorem; kappa proxy uses sampled mean, not exact monthly covariance.'}
    (OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print({k:v for k,v in s.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
