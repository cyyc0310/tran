"""Proper-source conditional distributions and interval-only quantile controls."""
from __future__ import annotations
import json,hashlib,argparse
from strict_histogram_quantile import StrictHistogramPinball
from pathlib import Path
from fractions import Fraction
import numpy as np
import pandas as pd
from scipy.special import ndtri
from sklearn.ensemble import ExtraTreesRegressor,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from v151_hourly_law import Law,EF,sha
from calendar_cif_law import CalendarLaw

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/v144-canonical-profile-cache-r1'
V153=ROOT/'results/v153-calendar-law-r1'
SOURCE_MODE=argparse.ArgumentParser()
SOURCE_MODE.add_argument('--source-mode',choices=('sampled','full'),required=True)
SOURCE_MODE=SOURCE_MODE.parse_args().source_mode
OUT=ROOT/f'results/v156-anchored-hourly-{SOURCE_MODE}-r1'
PROTOCOL=ROOT/'docs/V156_ANCHORED_HOURLY_RESIDUAL_PROTOCOL.md'
METHODS=('distribution_forest','heteroscedastic_normal','conditional_quantile')
FEATURES=[f'share_{i}' for i in range(8)]+['log_total','age_years','energy_anchor','month_sin','month_cos',
    'hour_sin','hour_cos','weekend','coal_hour_sin','coal_hour_cos','solar_hour_sin','solar_hour_cos']

def balanced(groups):
    c=pd.Series(groups).value_counts();w=np.asarray([1/c[g] for g in groups]);return w/w.mean()

def hgb(loss='squared_error',quantile=None):
    return HistGradientBoostingRegressor(loss=loss,quantile=quantile,max_iter=100,learning_rate=.1,
        max_leaf_nodes=15,min_samples_leaf=64,l2_regularization=1,early_stopping=False,random_state=20261028)

class Interval:
    def __init__(self,pred,states):self.pred=pred;self.states=states;self.mean=pred[:,1][states]
    def quantile(self,p):return self.pred[:,0 if p==.05 else 2][self.states]

class CompressedLaw(Law):
    def __init__(self,values,weights):
        values=np.asarray(values,float);weights=np.asarray(weights,float)
        unique,inverse=np.unique(values,return_inverse=True)
        merged=np.bincount(inverse,weights=weights,minlength=len(unique))
        super().__init__(unique,merged)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/v151_hourly_law.py',ROOT/'scripts/calendar_cif_law.py',P/'profiles.npz',P/'profile_index.csv',
        ROOT/'results/v153-calendar-times-r1/timestamps.npz',V153/'summary.json',V153/'query_inputs.csv',V153/'region_roles.csv',
        V153/'target_month_metrics.csv']
    paths += [ROOT/'scripts/strict_histogram_quantile.py',ROOT/'scripts/v154_hourly_strong_baselines.py']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    old=json.loads((V153/'summary.json').read_text())
    if old['assertion_failures']:raise ValueError('V153 audit failure')
    for name,digest in old['output_hashes'].items():
        if sha(V153/name)!=digest:raise ValueError('upstream hash mismatch')
    index=pd.read_csv(P/'profile_index.csv');packed=np.load(P/'profiles.npz',allow_pickle=False);raw=packed['generation']
    M=np.asarray([[float(Fraction(str(v))) for v in row] for row in packed['margin_fractions']])
    shares=M/M.sum(axis=1,keepdims=True);dates=pd.to_datetime(index.month)
    release=dates+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30)
    times=pd.DatetimeIndex(pd.to_datetime(np.load(paths[6])['unix_seconds'],unit='s',utc=True))
    hour=times.hour.to_numpy();weekend=(times.dayofweek.to_numpy()>=5).astype(int);states=hour+24*weekend
    groupids={r:np.asarray(ids,int) for r,ids in index.groupby('region').indices.items()}
    roles=pd.read_csv(V153/'region_roles.csv');queries=pd.read_csv(V153/'query_inputs.csv')
    def local_id(q,budget):
        if budget=='same_month':return q
        ids=groupids[index.region.iloc[q]];eligible=ids[(release.iloc[ids]<=dates.iloc[q]).to_numpy()]
        return int(eligible[np.argmax(dates.iloc[eligible].to_numpy())]) if len(eligible) else None
    def features(q,local,cell):
        hour=cell%24;sin=np.sin(2*np.pi*hour/24);cos=np.cos(2*np.pi*hour/24);mix=shares[local]
        mn=int(index.month.iloc[q][5:7]);age=(dates.iloc[q]-dates.iloc[local]).days/365
        return np.column_stack([np.tile(mix,(len(cell),1)),np.full(len(cell),np.log1p(M[local].sum())),np.full(len(cell),age),
            np.full(len(cell),mix@EF/1000),np.full(len(cell),np.sin(2*np.pi*mn/12)),np.full(len(cell),np.cos(2*np.pi*mn/12)),
            sin,cos,cell//24,mix[0]*sin,mix[0]*cos,mix[6]*sin,mix[6]*cos])
    datasets={}
    for budget in ('same_month','asof_30d'):
        for region,ids in groupids.items():
            candidates=[q for q in ids if index.month.iloc[q].startswith('2024') and local_id(q,budget) is not None]
            all_ids=np.concatenate([np.arange(int(index.offset.iloc[q]),int(index.stop.iloc[q])) for q in candidates]) if candidates else np.array([],int)
            if not len(all_ids):continue
            seed=int(hashlib.sha256(f'20261028/{region}/{budget}'.encode()).hexdigest()[:16],16)
            chosen=all_ids if SOURCE_MODE=='full' else np.sort(np.random.default_rng(seed).choice(all_ids,min(4096,len(all_ids)),replace=False))
            xs=[];ys=[];ordered=[]
            for q in candidates:
                pick=chosen[(chosen>=index.offset.iloc[q])&(chosen<index.stop.iloc[q])]
                if not len(pick):continue
                local=local_id(q,budget);g=raw[pick];y=g@EF/g.sum(axis=1)
                xs.append(features(q,local,states[pick]));ys.extend(y-shares[local]@EF);ordered.extend(pick)
            datasets[(region,budget)]=(np.concatenate(xs),np.asarray(ys),np.asarray(ordered,int))
    outputs=[];radii=[];sampling=[];oof_roles=[];cal_records=[]
    with threadpool_limits(limits=2):
        for target in ('PACE','LGEE','SC'):
            for seed in (41,83):
                roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)]
                fit=sorted(roster.loc[roster.role.eq('fit'),'region']);cal=set(roster.loc[roster.role.eq('calibration'),'region'])
                if len(fit)!=30 or len(cal)!=12 or set(fit)&cal or target in set(fit)|cal:raise ValueError('roles')
                inner={r:j%5 for j,r in enumerate(sorted(fit,key=lambda r:hashlib.sha256(f'20261028/{r}'.encode()).hexdigest()))}
                for region,fold in inner.items():oof_roles.append({'target_region':target,'seed':seed,'region':region,'inner_fold':fold})
                for budget in ('same_month','asof_30d'):
                    X=np.concatenate([datasets[(r,budget)][0] for r in fit]);y=np.concatenate([datasets[(r,budget)][1] for r in fit])
                    groups=np.concatenate([np.full(len(datasets[(r,budget)][1]),r) for r in fit]);w=balanced(groups)
                    if len(FEATURES)!=X.shape[1] or not np.isfinite(X).all():raise ValueError('features')
                    for r in fit:
                        sampling.append({'target_region':target,'seed':seed,'budget':budget,'region':r,
                            'sampled_raw_hour_indices':json.dumps(datasets[(r,budget)][2].tolist()),'sampled_hours':len(datasets[(r,budget)][1])})
                    forest=ExtraTreesRegressor(n_estimators=32,min_samples_leaf=64,max_features=1.,random_state=20261028,n_jobs=2)
                    forest.fit(X,y,sample_weight=w);leaf_ids=forest.apply(X);leaf_maps=[]
                    for t in range(32):
                        order=np.argsort(leaf_ids[:,t],kind='stable');ids=leaf_ids[order,t]
                        cuts=np.r_[0,np.flatnonzero(np.diff(ids))+1,len(ids)];mapping={}
                        for a,b in zip(cuts[:-1],cuts[1:]):
                            selected=order[a:b];mapping[int(ids[a])]=(y[selected],w[selected]/w[selected].sum()/32)
                        leaf_maps.append(mapping)
                    oof=np.full(len(y),np.nan)
                    for fold in range(5):
                        mask=np.array([inner[r]==fold for r in groups]);model=hgb()
                        model.fit(X[~mask],y[~mask],sample_weight=balanced(groups[~mask]));oof[mask]=model.predict(X[mask])
                    if not np.isfinite(oof).all():raise ValueError('source region OOF missing')
                    mean_model=hgb().fit(X,y,sample_weight=w)
                    variance_model=hgb(loss='poisson').fit(X,(y-oof)**2+25,sample_weight=w)
                    quantiles=[hgb(loss=StrictHistogramPinball(q)).fit(X,y,sample_weight=w) for q in (.05,.5,.95)]
                    errors={m:{} for m in METHODS};target_laws={}
                    fold_queries=queries.loc[queries.target_region.eq(target)&queries.seed.eq(seed)&queries.budget.eq(budget)]
                    for query in fold_queries.itertuples(index=False):
                        q=int(query.profile_id);local=local_id(q,budget)
                        if local is None or index.month.iloc[local]!=query.input_month:raise ValueError('query input release mismatch')
                        cells=np.arange(48);qx=features(q,local,cells);leaves=forest.apply(qx)
                        anchor=shares[local]@EF
                        laws_f=[];laws_g=[];mu=np.clip(anchor+mean_model.predict(qx),0,1000)
                        sigma=np.clip(np.sqrt(np.maximum(variance_model.predict(qx)-25,1)),1,500)
                        anchor=shares[local]@EF;z=ndtri((np.arange(1025)+.5)/1025)
                        for s in range(48):
                            yy=[];ww=[]
                            for t in range(32):
                                values,prob=leaf_maps[t][int(leaves[s,t])];yy.extend(values);ww.extend(prob)
                            laws_f.append(CompressedLaw(np.clip(np.asarray(yy)+anchor,0,1000),ww));laws_g.append(Law(np.clip(mu[s]+sigma[s]*z,0,1000),np.ones(1025)))
                        row=index.iloc[q];requested=states[int(row.offset):int(row.stop)]
                        qpred=np.clip(anchor+np.sort(np.column_stack([model.predict(qx) for model in quantiles]),axis=1),0,1000)
                        laws={'distribution_forest':CalendarLaw(laws_f,requested),'heteroscedastic_normal':CalendarLaw(laws_g,requested),
                            'conditional_quantile':Interval(qpred,requested)}
                        if query.role=='target':target_laws[q]=laws
                        else:
                            actual=raw[int(row.offset):int(row.stop)];actual=actual@EF/actual.sum(axis=1)
                            for method,law in laws.items():
                                err=np.maximum.reduce([law.quantile(.05)-actual,actual-law.quantile(.95),np.zeros(len(actual))])
                                errors[method].setdefault(query.query_region,[]).append(err)
                    for method in METHODS:
                        err={r:np.concatenate(v) for r,v in errors[method].items()}
                        if set(err)!=cal:raise ValueError('calibration regions')
                        pooled=Law(np.concatenate(list(err.values())),np.concatenate([np.full(len(v),1/(12*len(v))) for v in err.values()])).quantile(.9)
                        region_q={r:Law(v,np.ones(len(v))).quantile(.9) for r,v in err.items()}
                        for r,value in region_q.items():cal_records.append({'target_region':target,'seed':seed,'budget':budget,'method':method,'region':r,'q90_error':value})
                        for calibration,radius in (('pooled',pooled),('max_region',max(region_q.values()))):
                            radii.append({'target_region':target,'seed':seed,'budget':budget,'method':method,'calibration':calibration,'radius':radius})
                            for q,laws in target_laws.items():
                                row=index.iloc[q];g=raw[int(row.offset):int(row.stop)];actual=g@EF/g.sum(axis=1);law=laws[method]
                                lo=np.maximum(0,law.quantile(.05)-radius);hi=np.minimum(1000,law.quantile(.95)+radius)
                                if np.any(lo>hi):raise ValueError('interval inversion')
                                outputs.append({'region':target,'month':row.month,'seed':seed,'budget':budget,'method':method,'calibration':calibration,'hours':len(actual),
                                    'crps':float(law.crps(actual).mean()) if method!='conditional_quantile' else np.nan,
                                    'mae':float(np.abs(actual-law.mean).mean()),'coverage':float(((actual>=lo)&(actual<=hi)).mean()),
                                    'width':float(np.mean(hi-lo)),'winkler':float(np.mean(hi-lo+20*np.maximum(lo-actual,0)+20*np.maximum(actual-hi,0)))})
                    print(f'conditional models completed {target}/{seed}/{budget}; {len(y)} source hours',flush=True)
    rows=pd.DataFrame(outputs);keys=['region','month','seed','budget','method','calibration']
    if len(rows)!=864 or rows.duplicated(keys).any():raise ValueError('target denominator')
    regional=[]
    for key,g in rows.groupby(['region','seed','budget','method','calibration']):
        if len(g)!=12:raise ValueError('target months')
        regional.append(dict(zip(['region','seed','budget','method','calibration'],key))|
            {name:float(np.average(g[name],weights=g.hours)) for name in ['crps','mae','coverage','width','winkler']})
    region=pd.DataFrame(regional)
    macro=region.groupby(['seed','budget','method','calibration']).agg(crps=('crps','mean'),mae=('mae','mean'),coverage=('coverage','mean'),
        worst_coverage=('coverage','min'),width=('width','mean'),winkler=('winkler','mean'),regions=('region','size')).reset_index()
    if not macro.regions.eq(3).all():raise ValueError('macro denominator')
    if any(sha(ROOT/name)!=digest for name,digest in hashes.items()):raise ValueError('input changed')
    OUT.mkdir(parents=True)
    for name,frame in [('target_month_metrics',rows),('region_metrics',region),('macro_metrics',macro),('calibration_radii',pd.DataFrame(radii)),
        ('calibration_regions',pd.DataFrame(cal_records)),('source_sampling',pd.DataFrame(sampling)),('inner_source_folds',pd.DataFrame(oof_roles)),('region_roles',roles)]:frame.to_csv(OUT/f'{name}.csv',index=False)
    result={'status':'ANCHORED_HOURLY_RESIDUAL_DEVELOPMENT','source_mode':SOURCE_MODE,'scored_rows':len(rows),'assertion_failures':0,'feature_columns':FEATURES,
        'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Known development targets and simulated releases; no original-method claim.',
            'CQR has interval/point metrics only; normal law uses 1025-point discretization.',
            'Whole-source-region five-fold residuals; physical input anchor added to residual predictions; sampled/full modes are predefined.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
