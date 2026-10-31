"""Exact empirical hourly CIF laws under retrospective/as-of monthly budgets."""
from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from v139_physical_reserve_calibration import sha

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/v144-canonical-profile-cache-r1'
OUT=ROOT/'results/v157-matched-source-hours-r1'
PROTOCOL=ROOT/'docs/V157_MATCHED_SOURCE_HOURS_PROTOCOL.md'
EF=np.array([1000,400,800,0,0,0,0,400.])
METHODS=('matched_full','sampled')

class Law:
    def __init__(self,values,weights):
        x,w=np.asarray(values,float),np.asarray(weights,float)
        if x.shape!=w.shape or not np.isfinite(x).all() or not np.isfinite(w).all() or (w<0).any() or w.sum()<=0:
            raise ValueError('invalid probability law')
        order=np.argsort(x,kind='stable');self.x=x[order];self.w=w[order]/w.sum()
        self.c=np.cumsum(self.w);self.s=np.cumsum(self.w*self.x)
        self.mean=float(self.w@self.x)
        self.half_pair=float(np.sum(self.w*self.x*(2*self.c-self.w-1)))
    def quantile(self,p):return float(self.x[min(int(np.searchsorted(self.c,p,side='left')),len(self.x)-1)])
    def crps(self,y):
        ids=np.searchsorted(self.x,y,side='left');c=np.r_[0,self.c][ids];s=np.r_[0,self.s][ids]
        values=y*(2*c-1)+self.mean-2*s-self.half_pair
        if values.min() < -1e-8:raise ValueError('negative CRPS')
        return np.maximum(values,0)

def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[PROTOCOL,Path(__file__),P/'summary.json',P/'profile_index.csv',P/'canonical_monthly_panel.csv',P/'profiles.npz',
           ROOT/'results/v145-recomposed-risk-baselines-r1/region_roles.csv']
    paths += [ROOT/'scripts/calendar_cif_law.py', ROOT/'scripts/v151_hourly_law.py',
        ROOT/'results/v153-calendar-times-r1/timestamps.npz',ROOT/'results/v153-calendar-times-r1/summary.json',
        ROOT/'results/v151-hourly-law-r1/target_month_metrics.csv']
    paths += [ROOT/'scripts/sampled_calendar_source.py',ROOT/'results/v154-hourly-strong-baselines-r1/source_sampling.csv',ROOT/'results/v154-hourly-strong-baselines-r1/summary.json']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    summary=json.loads((P/'summary.json').read_text())
    if sha(P/'profiles.npz')!=summary['profiles_sha256']:raise ValueError('profile cache hash')
    index=pd.read_csv(P/'profile_index.csv');panel=pd.read_csv(P/'canonical_monthly_panel.csv')
    roles=pd.read_csv(ROOT/'results/v145-recomposed-risk-baselines-r1/region_roles.csv');packed=np.load(P/'profiles.npz',allow_pickle=False);raw=packed['generation']
    margins=np.asarray([[float(Fraction(str(v))) for v in row] for row in packed['margin_fractions']])
    shares=margins/margins.sum(axis=1,keepdims=True)
    from calendar_cif_law import CalendarSource
    time_dir=ROOT/'results/v153-calendar-times-r1'
    time_summary=json.loads((time_dir/'summary.json').read_text())
    if sha(time_dir/'timestamps.npz')!=time_summary['timestamp_sha256']:raise ValueError('timestamp hash')
    calendar=CalendarSource(index,raw,margins,np.load(time_dir/'timestamps.npz')['unix_seconds'])
    if not index.profile_id.eq(np.arange(len(index))).all():raise ValueError('profile ordering')
    dates=pd.to_datetime(index.month);years=dates.dt.year.to_numpy();months=dates.dt.month.to_numpy()
    released=dates+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30)
    groupids={r:np.asarray(ids,int) for r,ids in index.groupby('region').indices.items()}
    def generation(q):
        row=index.iloc[q];return raw[int(row.offset):int(row.stop)]
    def truth(q):
        g=generation(q);total=g.sum(axis=1)
        if (total<=0).any():raise ValueError('nonpositive total in eligible hours')
        return g@EF/total
    def input_id(q,budget):
        if budget=='same_month':return q
        candidates=groupids[index.region.iloc[q]]
        eligible=candidates[(released.iloc[candidates]<=dates.iloc[q]).to_numpy()]
        return int(eligible[np.argmax(dates.iloc[eligible].to_numpy())]) if len(eligible) else None
    from sampled_calendar_source import SampledCalendarSource
    sampling=pd.read_csv(ROOT/'results/v154-hourly-strong-baselines-r1/source_sampling.csv')
    provenance=[];availability=[];outputs=[];radii=[];cal_counts=[];source_fallbacks=[]
    for target in ('PACE','LGEE','SC'):
        for seed in (41,83):
            roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)]
            fit=sorted(roster.loc[roster.role.eq('fit'),'region']);cal=sorted(roster.loc[roster.role.eq('calibration'),'region'])
            if len(set(fit))!=30 or len(set(cal))!=12 or set(fit)&set(cal) or target in fit+cal:raise ValueError('region roles')
            donor_bank={r:groupids[r][years[groupids[r]]==2024] for r in fit}
            if any(len(v)<2 for v in donor_bank.values()):raise ValueError('source year support')
            for budget in ('same_month','asof_30d'):
                selected_sampling=sampling.loc[sampling.target_region.eq(target)&sampling.seed.eq(seed)&sampling.budget.eq(budget)]
                if set(selected_sampling.region)!=set(fit):raise ValueError('matched source identity')
                allowed=np.concatenate([np.asarray(json.loads(v),int) for v in selected_sampling.sampled_raw_hour_indices])
                profile_ids=np.unique(np.searchsorted(index.stop.to_numpy(),allowed,side='right'))
                if not set(index.region.iloc[profile_ids]).issubset(fit):raise ValueError('source sample leakage')
                matched_bank={r:np.asarray([q for q in profile_ids if index.region.iloc[q]==r],int) for r in fit}
                if any(len(ids)<2 for ids in matched_bank.values()):raise ValueError('two source months unavailable')
                sampled_source=SampledCalendarSource(index,raw,margins,np.load(time_dir/'timestamps.npz')['unix_seconds'],allowed)
                full_source=CalendarSource(index,raw,margins,np.load(time_dir/'timestamps.npz')['unix_seconds'])
                cal_queries=[q for r in cal for q in groupids[r] if years[q]==2024 and input_id(q,budget) is not None]
                target_queries=[q for q in groupids[target] if years[q]==2025 and input_id(q,budget) is not None]
                if not target_queries or set(index.region.iloc[cal_queries])!=set(cal):raise ValueError('evaluation denominator')
                query_laws={};errors={m:{} for m in METHODS}
                for q in cal_queries+target_queries:
                    local=input_id(q,budget);query=index.iloc[q];M=margins[local]
                    if budget=='asof_30d' and released.iloc[local]>dates.iloc[q]:raise ValueError('release leakage')
                    availability.append({'target_region':target,'seed':seed,'budget':budget,'profile_id':q,'query_region':query.region,
                        'query_month':query.month,'input_month':index.month.iloc[local],
                        'statistic_age_days':(dates.iloc[q]-dates.iloc[local]).days,'hours':int(query.hours),
                        'role':'target' if query.region==target else 'calibration'})
                    donor_ids=[];prior=[];labels=[]
                    for r,ids in matched_bank.items():
                        distance=np.abs(months[ids]-months[q]);distance=np.minimum(distance,12-distance)
                        chosen=ids[np.lexsort((ids,distance))[:2]]
                        donor_ids.extend(chosen);prior.extend([1/60]*2);labels.extend([r]*2)
                    donor_ids=np.asarray(donor_ids,int);prior=np.asarray(prior)
                    if query.region in labels or any(years[d]!=2024 for d in donor_ids):raise ValueError('hourly donor leakage')
                    hellinger=.5*((np.sqrt(shares[donor_ids])-np.sqrt(shares[local]))**2).sum(axis=1)
                    conditional=prior*np.exp(-hellinger/.3**2);conditional/=conditional.sum()
                    missing=(shares[local]*(margins[donor_ids]==0)).sum(axis=1)
                    provenance.append({'target_region':target,'seed':seed,'budget':budget,'profile_id':q,
                        'donor_ids':json.dumps(donor_ids.tolist()),'donor_regions':json.dumps(labels),
                        'conditional_weights':json.dumps(conditional.tolist()),'unsupported_mass':float(conditional@missing)})
                    laws={'matched_full':full_source.build(q,donor_ids,conditional,local,False,False),
                          'sampled':sampled_source.build(q,donor_ids,conditional,local,False,False)}
                    if query.region==target:query_laws[q]=laws
                    else:
                        y=truth(q)
                        for method,law in laws.items():
                            a,b=law.quantile(.05),law.quantile(.95)
                            err=np.maximum.reduce([a-y,y-b,np.zeros(len(y))])
                            errors[method].setdefault(query.region,[]).append(err)
                source_fallbacks.append({'target_region':target,'seed':seed,'budget':budget,
                    'allowed_source_hours':len(allowed),'sampled_empty_cell_fallbacks':sampled_source.empty_cell_fallbacks,
                    'full_empty_cell_fallbacks':full_source.empty_cell_fallbacks})
                for method in METHODS:
                    err={r:np.concatenate(v) for r,v in errors[method].items()}
                    if set(err)!=set(cal):raise ValueError('calibration identity missing')
                    values=np.concatenate(list(err.values()));w=np.concatenate([np.full(len(v),1/(12*len(v))) for v in err.values()])
                    pooled=Law(values,w).quantile(.9)
                    region_q={r:Law(v,np.ones(len(v))).quantile(.9) for r,v in err.items()}
                    for r,v in err.items():cal_counts.append({'target_region':target,'seed':seed,'budget':budget,'method':method,'region':r,'hours':len(v),'q90_error':region_q[r]})
                    for calibration,radius in (('pooled',pooled),('max_region',max(region_q.values()))):
                        radii.append({'target_region':target,'seed':seed,'budget':budget,'method':method,'calibration':calibration,'radius':radius})
                        for q in target_queries:
                            law=query_laws[q][method];y=truth(q);a,b=law.quantile(.05),law.quantile(.95)
                            lo=np.maximum(0,a-radius);hi=np.minimum(1000,b+radius);coverage=(y>=lo)&(y<=hi)
                            winkler=hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
                            if np.any(lo<0) or np.any(lo>hi) or np.any(hi>1000):raise ValueError('hourly interval bounds')
                            outputs.append({'region':target,'month':index.month.iloc[q],'seed':seed,'budget':budget,'method':method,
                                'calibration':calibration,'hours':len(y),'crps':float(law.crps(y).mean()),
                                'mae':float(np.abs(y-law.mean).mean()),'coverage':float(coverage.mean()),
                                'width':float(np.mean(hi-lo)),'winkler':float(winkler.mean()),'lower':float(np.mean(lo)),'upper':float(np.mean(hi)),'mean':float(np.mean(law.mean))})
            print(f'hourly laws completed {target}/{seed}',flush=True)
    rows=pd.DataFrame(outputs)
    keys=['region','month','seed','budget','method','calibration']
    if rows.duplicated(keys).any():raise ValueError('duplicate target records')
    expected=index.loc[index.region.isin(['PACE','LGEE','SC']) & dates.dt.year.eq(2025)]
    if len(rows)!=len(expected)*2*2*2*2:raise ValueError('common target denominator')
    grouped=[]
    for key,g in rows.groupby(['region','seed','budget','method','calibration']):
        grouped.append(dict(zip(['region','seed','budget','method','calibration'],key))|{'months':len(g),'hours':int(g.hours.sum())}|
            {name:float(np.average(g[name],weights=g.hours)) for name in ['crps','mae','coverage','width','winkler']})
    region=pd.DataFrame(grouped)
    macro=region.groupby(['seed','budget','method','calibration']).agg(crps=('crps','mean'),mae=('mae','mean'),coverage=('coverage','mean'),
        worst_coverage=('coverage','min'),width=('width','mean'),winkler=('winkler','mean'),regions=('region','size')).reset_index()
    if not macro.regions.eq(3).all():raise ValueError('macro denominator')
    # Direct finite-atom pair formula audit, independent of prefix-sum calculation.
    toy=Law([0,2,5],[.2,.3,.5]);y=np.array([-1.,0.,1.,5.,7.])
    brute=np.abs(toy.x[:,None]-y).T@toy.w-.5*np.sum(toy.w[:,None]*toy.w[None,:]*np.abs(toy.x[:,None]-toy.x[None,:]))
    if not np.allclose(toy.crps(y),brute,atol=1e-12):raise ValueError('CRPS identity')
    if any(sha(ROOT/name)!=h for name,h in hashes.items()):raise ValueError('frozen inputs changed')
    OUT.mkdir(parents=True)
    for name,frame in [('target_month_metrics',rows),('region_metrics',region),('macro_metrics',macro),('query_inputs',pd.DataFrame(availability)),
        ('donor_provenance',pd.DataFrame(provenance)),('calibration_radii',pd.DataFrame(radii)),('calibration_regions',pd.DataFrame(cal_counts)),('region_roles',roles),('source_fallbacks',pd.DataFrame(source_fallbacks))]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    result={'status':'MATCHED_SOURCE_HOURLY_RECOMPOSITION_DEVELOPMENT','target_region_months':len(expected),'scored_rows':len(rows),'assertion_failures':0,
        'crps_finite_atom_identity_checks':len(y),'source_sample_rows':len(sampling),'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Known development identities, nominal production CIF proxy, retrospective quality screen, simulated release lag.',
            'UTC calendar conditioning and expectation raking are standard controls, not original mechanisms or blind confirmation.',
            'Interval calibration does not change the predictive law used for CRPS.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
