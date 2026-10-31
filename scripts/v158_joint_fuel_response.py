"""Joint fuel innovation controls and a proper-source response-aware split probe."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from threadpoolctl import threadpool_limits
from v154_hourly_strong_baselines import balanced,CompressedLaw,FEATURES
from v151_hourly_law import Law,EF,sha
from calendar_cif_law import CalendarLaw
from hourly_fuel_data import HourlyFuelData,repair_response

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v156-anchored-hourly-sampled-r1'
V153=ROOT/'results/v153-calendar-law-r1'
OUT=ROOT/'results/v158-joint-fuel-response-r1'
PROTOCOL=ROOT/'docs/V158_JOINT_FUEL_RESPONSE_PROTOCOL.md'
METHODS=('scalar','joint_linear','joint_nonnegative','joint_budget','response_budget')

class LeafMixture:
    def __init__(self,X,split_labels,scalar,joint,w):
        self.forest=ExtraTreesRegressor(n_estimators=32,min_samples_leaf=64,max_features=1.,random_state=20261028,n_jobs=2)
        self.forest.fit(X,split_labels,sample_weight=w);ids=self.forest.apply(X)
        self.maps=[];self.scalar=scalar;self.joint=joint;self.w=w;self.cache={}
        for t in range(32):
            order=np.argsort(ids[:,t],kind='stable');sorted_ids=ids[order,t]
            cuts=np.r_[0,np.flatnonzero(np.diff(sorted_ids))+1,len(order)]
            self.maps.append({int(sorted_ids[a]):order[a:b] for a,b in zip(cuts[:-1],cuts[1:])})
    def pieces(self,signature):
        ids=[];probs=[]
        for t,leaf in enumerate(signature):
            selected=self.maps[t][int(leaf)];ids.extend(selected)
            probs.extend(self.w[selected]/self.w[selected].sum()/32)
        return np.asarray(ids,int),np.asarray(probs)
    def scalar_law(self,signature,anchor):
        ids,prob=self.pieces(signature)
        return CompressedLaw(np.clip(self.scalar[ids]+anchor,0,1000),prob)
    def vectors(self,signature):
        key=tuple(signature)
        if key not in self.cache:
            ids,prob=self.pieces(signature)
            values=np.ascontiguousarray(self.joint[ids])
            # Exact bitwise grouping avoids NumPy's costly structured-float row sort.
            # Signed zeros may remain separate; their masses are merged after CIF mapping.
            keys=values.view(np.dtype((np.void,values.dtype.itemsize*values.shape[1]))).reshape(-1)
            _,first,inverse=np.unique(keys,return_index=True,return_inverse=True)
            unique=values[first]
            if len(self.cache)>=256:self.cache.pop(next(iter(self.cache)))
            self.cache[key]=(unique,np.bincount(inverse,weights=prob,minlength=len(unique)))
        return self.cache[key]

def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/hourly_fuel_data.py',ROOT/'scripts/v154_hourly_strong_baselines.py',
        ROOT/'scripts/v151_hourly_law.py',ROOT/'scripts/calendar_cif_law.py',
        ROOT/'results/v144-canonical-profile-cache-r1/profiles.npz',ROOT/'results/v144-canonical-profile-cache-r1/profile_index.csv',
        ROOT/'results/v153-calendar-times-r1/timestamps.npz',BASE/'summary.json',BASE/'source_sampling.csv',BASE/'region_roles.csv',
        BASE/'target_month_metrics.csv',V153/'query_inputs.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    previous=json.loads((BASE/'summary.json').read_text())
    if previous['assertion_failures']:raise ValueError('upstream failure')
    for name,h in previous['output_hashes'].items():
        if sha(BASE/name)!=h:raise ValueError('upstream changed')
    data=HourlyFuelData(ROOT);roles=pd.read_csv(BASE/'region_roles.csv');sampling=pd.read_csv(BASE/'source_sampling.csv')
    queries=pd.read_csv(V153/'query_inputs.csv');outputs=[];radii=[];prototypes=[];counts=[]
    with threadpool_limits(limits=2):
        for target in ('PACE','LGEE','SC'):
            for seed in (41,83):
                roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)]
                fit=sorted(roster.loc[roster.role.eq('fit'),'region']);cal=set(roster.loc[roster.role.eq('calibration'),'region'])
                if len(fit)!=30 or len(cal)!=12 or set(fit)&cal or target in set(fit)|cal:raise ValueError('role leakage')
                for budget in ('same_month','asof_30d'):
                    draw=sampling.loc[sampling.target_region.eq(target)&sampling.seed.eq(seed)&sampling.budget.eq(budget)].set_index('region')
                    if set(draw.index)!=set(fit):raise ValueError('source draw identity')
                    source=[data.source(r,budget,json.loads(draw.loc[r,'sampled_raw_hour_indices'])) for r in fit]
                    X=np.concatenate([v[0] for v in source]);y=np.concatenate([v[1] for v in source]);d=np.concatenate([v[2] for v in source])
                    groups=np.concatenate([np.full(len(v[1]),r) for r,v in zip(fit,source)]);w=balanced(groups)
                    if X.shape[1]!=len(FEATURES):raise ValueError('feature dimension')
                    input_ids=np.unique(np.concatenate([v[3] for v in source]))
                    if not set(data.index.region.iloc[input_ids]).issubset(fit):raise ValueError('response prototype leakage')
                    order=np.lexsort((input_ids,data.shares[input_ids,0]));ordered=input_ids[order]
                    chosen=ordered[np.rint(np.linspace(0,len(ordered)-1,5)).astype(int)]
                    responses=[];training_fallback=0
                    for q in chosen:
                        response,number=repair_response(data.shares[q],d,budget=='same_month')
                        responses.append(response);training_fallback+=number
                    response=np.column_stack(responses);mu=np.average(response,axis=0,weights=w)
                    sd=np.sqrt(np.average((response-mu)**2,axis=0,weights=w));active=sd>1e-8
                    embedding=(response[:,active]-mu[active])/sd[active] if active.any() else np.zeros((len(y),1))
                    prototypes.append({'target_region':target,'seed':seed,'budget':budget,'prototype_profile_ids':json.dumps(chosen.tolist()),
                        'active_response_dimensions':int(active.sum()),'training_rows':len(y),'training_repair_fallbacks':training_fallback})
                    models={'scalar':LeafMixture(X,y,y,d,w),'joint':LeafMixture(X,d,y,d,w),'response':LeafMixture(X,embedding,y,d,w)}
                    errors={m:{} for m in METHODS};target_laws={};query_fallbacks=0
                    selected=queries.loc[queries.target_region.eq(target)&queries.seed.eq(seed)&queries.budget.eq(budget)]
                    for query in selected.itertuples(index=False):
                        q=int(query.profile_id);local=data.local(q,budget)
                        if local is None or data.index.month.iloc[local]!=query.input_month:raise ValueError('input timing mismatch')
                        S=data.shares[local];anchor=float(S@EF);qx=data.features(q,local,np.arange(48))
                        leaves={name:model.forest.apply(qx) for name,model in models.items()};cell_laws={m:[] for m in METHODS}
                        for state in range(48):
                            cell_laws['scalar'].append(models['scalar'].scalar_law(leaves['scalar'][state],anchor))
                            delta,prob=models['joint'].vectors(leaves['joint'][state])
                            cell_laws['joint_linear'].append(CompressedLaw(np.clip(anchor+delta@EF,0,1000),prob))
                            for method,mask in (('joint_nonnegative',False),('joint_budget',budget=='same_month')):
                                values,number=repair_response(S,delta,mask);query_fallbacks+=number
                                cell_laws[method].append(CompressedLaw(values,prob))
                            delta,prob=models['response'].vectors(leaves['response'][state])
                            values,number=repair_response(S,delta,budget=='same_month');query_fallbacks+=number
                            cell_laws['response_budget'].append(CompressedLaw(values,prob))
                        row=data.index.iloc[q];requested=data.states[int(row.offset):int(row.stop)]
                        laws={name:CalendarLaw(values,requested) for name,values in cell_laws.items()}
                        if query.role=='target':target_laws[q]=laws
                        else:
                            actual=data.truth(q)
                            for name,law in laws.items():
                                score=np.maximum.reduce([law.quantile(.05)-actual,actual-law.quantile(.95),np.zeros(len(actual))])
                                errors[name].setdefault(query.query_region,[]).append(score)
                    counts.append({'target_region':target,'seed':seed,'budget':budget,'query_repair_fallbacks':query_fallbacks})
                    for name in METHODS:
                        e={r:np.concatenate(v) for r,v in errors[name].items()}
                        if set(e)!=cal:raise ValueError('calibration denominator')
                        pool=Law(np.concatenate(list(e.values())),np.concatenate([np.full(len(v),1/(12*len(v))) for v in e.values()])).quantile(.9)
                        maximum=max(Law(v,np.ones(len(v))).quantile(.9) for v in e.values())
                        for calibration,radius in (('pooled',pool),('max_region',maximum)):
                            radii.append({'target_region':target,'seed':seed,'budget':budget,'method':name,'calibration':calibration,'radius':radius})
                            for q,laws in target_laws.items():
                                law=laws[name];actual=data.truth(q);row=data.index.iloc[q]
                                lo=np.maximum(0,law.quantile(.05)-radius);hi=np.minimum(1000,law.quantile(.95)+radius)
                                if np.any(lo>hi):raise ValueError('interval inversion')
                                outputs.append({'region':target,'month':row.month,'seed':seed,'budget':budget,'method':name,'calibration':calibration,
                                    'hours':len(actual),'crps':float(law.crps(actual).mean()),'mae':float(np.abs(actual-law.mean).mean()),
                                    'coverage':float(((actual>=lo)&(actual<=hi)).mean()),'width':float(np.mean(hi-lo)),
                                    'winkler':float(np.mean(hi-lo+20*np.maximum(lo-actual,0)+20*np.maximum(actual-hi,0))),
                                    'mean':float(np.mean(law.mean)),'lower':float(np.mean(lo)),'upper':float(np.mean(hi))})
                    print(f'joint response completed {target}/{seed}/{budget}',flush=True)
    rows=pd.DataFrame(outputs);keys=['region','month','seed','budget','method','calibration']
    if len(rows)!=1440 or rows.duplicated(keys).any():raise ValueError('target denominator')
    old=pd.read_csv(BASE/'target_month_metrics.csv');old=old.loc[old.method.eq('distribution_forest')].copy();old['method']='scalar'
    match=rows.loc[rows.method.eq('scalar')].merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if len(match)!=288:raise ValueError('scalar reproduction denominator')
    for metric in ('crps','mae','coverage','width','winkler'):
        if not np.allclose(match[metric+'_new'],match[metric+'_old'],atol=1e-7,rtol=0):raise ValueError('V156 scalar mismatch '+metric)
    regional=[]
    for key,g in rows.groupby(['region','seed','budget','method','calibration']):
        if len(g)!=12:raise ValueError('target month count')
        regional.append(dict(zip(['region','seed','budget','method','calibration'],key))|
            {m:float(np.average(g[m],weights=g.hours)) for m in ['crps','mae','coverage','width','winkler']})
    region=pd.DataFrame(regional)
    macro=region.groupby(['seed','budget','method','calibration']).agg(crps=('crps','mean'),mae=('mae','mean'),coverage=('coverage','mean'),
        worst_coverage=('coverage','min'),width=('width','mean'),winkler=('winkler','mean'),regions=('region','size')).reset_index()
    if not macro.regions.eq(3).all():raise ValueError('macro region count')
    if any(sha(ROOT/name)!=h for name,h in hashes.items()):raise ValueError('frozen input/code changed')
    OUT.mkdir(parents=True)
    for name,frame in [('target_month_metrics',rows),('region_metrics',region),('macro_metrics',macro),('calibration_radii',pd.DataFrame(radii)),
        ('response_prototypes',pd.DataFrame(prototypes)),('repair_counts',pd.DataFrame(counts)),('source_sampling',sampling),('region_roles',roles)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    result={'status':'JOINT_FUEL_RESPONSE_SPLIT_DEVELOPMENT','scored_rows':len(rows),'scalar_reference_reproduced_records':len(match),'assertion_failures':0,
        'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Response feature embedding is a mechanism probe, not an established novelty claim.',
            'Share repair respects nonnegativity/support but not monthly energy totals; asof zeros never hard-mask future fuels.',
            'Opened targets, retrospective quality and simulated release budget; no independent confirmation.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
