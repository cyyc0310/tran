"""Independent exact-rational gauge proof and reconstruction of V191 source rows."""
from pathlib import Path
from fractions import Fraction as F
import json
import numpy as np
import pandas as pd
from hourly_fuel_data import HourlyFuelData
from v151_hourly_law import EF,sha

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/v191-delayed-state-observability-r1'
OUT=ROOT/'results/v191-delayed-state-observability-audit-r4'
BASE=ROOT/'results/v188-forward-zero-risk-r1'


def mm(a,b):return [[sum((x*y for x,y in zip(row,col)),F(0)) for col in zip(*b)] for row in a]
def add(a,b):return [[x+y for x,y in zip(r,s)] for r,s in zip(a,b)]
def rank(a):
    a=[r.copy() for r in a];r=0
    for c in range(len(a[0])):
        candidates=[i for i in range(r,len(a)) if a[i][c]]
        if not candidates:continue
        i=candidates[0];a[r],a[i]=a[i],a[r];v=a[r][c];a[r]=[x/v for x in a[r]]
        for i in range(len(a)):
            if i!=r:
                v=a[i][c];a[i]=[x-v*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r


def main():
    if OUT.exists():raise FileExistsError(OUT)
    summary=json.loads((SOURCE/'summary.json').read_text());hashes=summary['input_hashes'].copy()
    for name in ('summary.json',*summary['output_hashes']):hashes[str((SOURCE/name).relative_to(ROOT))]=sha(SOURCE/name)
    hashes[str(Path(__file__).relative_to(ROOT))]=sha(Path(__file__))
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('input hash')
    data=HourlyFuelData(ROOT);saved=pd.read_csv(SOURCE/'source_states.csv');transitions=pd.read_csv(SOURCE/'transitions.csv');ranks=pd.read_csv(SOURCE/'design_ranks.csv')
    with np.load(ROOT/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as p:times=p['unix_seconds']
    rebuilt=[];expected_counts=[];max_error=0.
    for fold in range(6):
        proper=json.loads((BASE/f'fold_{fold}/training_provenance.json').read_text())['proper_regions']
        with np.load(BASE/f'fold_{fold}/training_matrices.npz',allow_pickle=False) as p:ids=p['raw_id'];profiles=p['profile_id'];regions=p['region']
        if not np.all(times[ids]<1719792000):raise ValueError('independent raw timestamp cutoff')
        mapped=np.searchsorted(data.index.stop.to_numpy(),ids,side='right')
        if not np.array_equal(mapped,profiles) or not np.array_equal(data.index.region.to_numpy()[mapped],regions) or not set(regions).issubset(proper):raise ValueError('source mapping/role')
        if not (data.release.iloc[profiles]<=pd.Timestamp('2024-07-01')).all():raise ValueError('source label cutoff')
        pairs=0
        for region in proper:
            qids=sorted(set(profiles[regions==region]));pairs+=sum((pd.Timestamp(data.index.month.iloc[b]).to_period('M')-pd.Timestamp(data.index.month.iloc[a]).to_period('M')).n==1 for a,b in zip(qids,qids[1:]))
            for q in qids:
                raw=ids[profiles==q];g=data.raw[raw];a=float((data.M[q]*EF).sum()/data.M[q].sum());mu=float(np.mean((g*EF).sum(axis=1)/g.sum(axis=1)))
                rec=saved.loc[saved.fold.eq(fold)&saved.profile_id.eq(q)]
                if len(rec)!=1 or int(rec.sampled_hours.iloc[0])!=len(raw) or rec.region.iloc[0]!=region:raise ValueError('state key/count')
                max_error=max(max_error,abs(float(rec.energy_anchor.iloc[0])-a),abs(float(rec.sample_time_mean.iloc[0])-mu),abs(float(rec.sample_kappa_proxy.iloc[0])-(a-mu)))
        expected_counts.append(pairs)
        t=transitions.loc[transitions.fold.eq(fold)];r=ranks.loc[ranks.fold.eq(fold)].iloc[0]
        if len(t)!=pairs or t.region.duplicated().any() or int(r.additional_A_rank)!=0 or int(r.design_rank)!=pairs:raise ValueError('transition/intercept rank')
        D=[[F(str(row.x_a)),F(str(row.x_k))]+[F(i==j) for j in range(pairs)] for i,row in enumerate(t.itertuples(index=False))]
        B=[[F(i==j) for j in range(pairs)] for i in range(pairs)]
        null_a=[[F(1)],[F(0)]]+[[-row[0]] for row in D]
        null_k=[[F(0)],[F(1)]]+[[-row[1]] for row in D]
        if rank(D)!=rank(B) or rank(D)!=pairs or any(x[0] for x in mm(D,null_a)) or any(x[0] for x in mm(D,null_k)):raise ValueError('exact full design/nullspace')
        for row in t.itertuples(index=False):
            for profile,a,k in [(row.from_profile,row.x_a,row.x_k),(row.to_profile,row.y_a,row.y_k)]:
                state=saved.loc[saved.fold.eq(fold)&saved.profile_id.eq(profile)]
                if len(state)!=1 or state.region.iloc[0]!=row.region or abs(float(state.energy_anchor.iloc[0])-a)>1e-9 or abs(float(state.sample_kappa_proxy.iloc[0])-k)>1e-9:raise ValueError('transition/state binding')
    if expected_counts!=[19,17,17,17,18,20] or max_error>1e-9:raise ValueError('source support reconstruction')
    A=[[F(1,2),F(1,4)],[F(1,8),F(1,2)]];I=[[F(1),F(0)],[F(0),F(1)]]
    T=[A[0]+I[0],A[1]+I[1],[F(0),F(0),F(1),F(0)],[F(0),F(0),F(0),F(1)]]
    H=[[F(1),F(0),F(0),F(0)]];v=[[F(0)],[F(1)],[F(-1,4)],[F(1,2)]]
    O=[];row=H
    for _ in range(12):O+=row;row=mm(row,T)
    if rank([r[:2] for r in O[:2]])!=2 or rank(O)!=3 or any(r[0] for r in mm(O,v)) or mm(T,v)!=v:raise ValueError('exact observability/gauge')
    mu=[F(1),F(-1),F(0),F(0)];kappa=[F(0),F(1),F(0),F(0)]
    if rank(O+[mu])!=4 or rank(O+[kappa])!=4 or rank(O+H)!=3:raise ValueError('minimal added statistic gauge test')
    # Exact physical example already present in V15: same fuel totals and full hourly G.
    g1=[[F(0),F(1)],[F(1),F(1)]];g2=[[F(1),F(0)],[F(0),F(2)]]
    totals1=[sum(col,F(0)) for col in zip(*g1)];totals2=[sum(col,F(0)) for col in zip(*g2)]
    if totals1!=totals2 or totals1!=[F(1),F(2)]:raise ValueError('exact physical fuel monthly totals')
    means=[sum((F(1000)*r[0]/sum(r) for r in g),F(0))/2 for g in (g1,g2)];a=F(1000,3)
    if means!=[F(250),F(500)] or [sum(r) for r in g1]!=[sum(r) for r in g2]:raise ValueError('physical witness')
    x1=[[a],[a-means[0]]];x2=[[a],[a-means[1]]];delta=F(-250)
    B=[[I[i][j]-A[i][j] for j in range(2)] for i in range(2)];b1=mm(B,x1);b2=mm(B,x2)
    for _ in range(12):
        if x1[0]!=x2[0] or x2[1][0]-x1[1][0]!=delta:raise ValueError('monthly likelihood alias')
        x1=add(mm(A,x1),b1);x2=add(mm(A,x2),b2)
    certificate=json.loads((SOURCE/'gauge_certificate.json').read_text())
    if certificate['time_mean_1']!=250 or certificate['time_mean_2']!=500 or certificate['unknown_b_augmented_rank']!=3:raise ValueError('saved certificate')
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('final hash')
    OUT.mkdir();s={'assertion_failures':0,'source_state_rows_rebuilt':len(saved),'transitions_by_fold':expected_counts,'max_state_proxy_error':max_error,
        'fraction_known_b_rank':2,'fraction_unknown_b_rank':3,'fraction_mu_added_rank':4,'fraction_kappa_added_rank':4,'fraction_extra_month_same_H_rank':3,
        'original_raw_timestamps_cutoff_checked':True,'physical_fuel_totals_checked_exact':True,'physical_months_checked_exact':12,'models_fitted':False,'cal_test_hour_labels_used':False,'input_hashes':hashes,'output_hashes':{},
        'scope':'Bounded gauge/intercept audit; no universal impossibility, no new theorem or predictive experiment.'}
    (OUT/'summary.json').write_text(json.dumps(s,indent=2)+'\n');print({k:v for k,v in s.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
