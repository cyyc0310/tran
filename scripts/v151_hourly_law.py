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
OUT=ROOT/'results/v151-hourly-law-r1'
PROTOCOL=ROOT/'docs/V151_HOURLY_LAW_PROTOCOL.md'
EF=np.array([1000,400,800,0,0,0,0,400.])
METHODS=('unconditional','level_residual','recompose_pooled','recompose_conditional')

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
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    summary=json.loads((P/'summary.json').read_text())
    if sha(P/'profiles.npz')!=summary['profiles_sha256']:raise ValueError('profile cache hash')
    index=pd.read_csv(P/'profile_index.csv');panel=pd.read_csv(P/'canonical_monthly_panel.csv')
    roles=pd.read_csv(paths[-1]);packed=np.load(P/'profiles.npz',allow_pickle=False);raw=packed['generation']
    margins=np.asarray([[float(Fraction(str(v))) for v in row] for row in packed['margin_fractions']])
    shares=margins/margins.sum(axis=1,keepdims=True)
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
    provenance=[];availability=[];outputs=[];radii=[];cal_counts=[]
    for target in ('PACE','LGEE','SC'):
        for seed in (41,83):
            roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)]
            fit=sorted(roster.loc[roster.role.eq('fit'),'region']);cal=sorted(roster.loc[roster.role.eq('calibration'),'region'])
            if len(set(fit))!=30 or len(set(cal))!=12 or set(fit)&set(cal) or target in fit+cal:raise ValueError('region roles')
            donor_bank={r:groupids[r][years[groupids[r]]==2024] for r in fit}
            if any(len(v)<2 for v in donor_bank.values()):raise ValueError('source year support')
            for budget in ('same_month','asof_30d'):
                residuals=[];resweights=[];unconditional=[];unweights=[]
                for r,ids in donor_bank.items():
                    eligible=[q for q in ids if input_id(q,budget) is not None]
                    if not eligible:raise ValueError('no training history')
                    for q in eligible:
                        y=truth(q);anchor=float(shares[input_id(q,budget)]@EF)
                        w=np.full(len(y),1/(30*len(eligible)*len(y)))
                        residuals.extend(y-anchor);resweights.extend(w)
                    for q in ids:
                        y=truth(q);unconditional.extend(y);unweights.extend(np.full(len(y),1/(30*len(ids)*len(y))))
                unconditional_law=Law(unconditional,unweights)
                residuals=np.asarray(residuals);resweights=np.asarray(resweights)
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
                    for r,ids in donor_bank.items():
                        distance=np.abs(months[ids]-months[q]);distance=np.minimum(distance,12-distance)
                        chosen=ids[np.lexsort((ids,distance))[:2]]
                        donor_ids.extend(chosen);prior.extend([1/60]*2);labels.extend([r]*2)
                    donor_ids=np.asarray(donor_ids,int);prior=np.asarray(prior)
                    if query.region in labels or any(years[d]!=2024 for d in donor_ids):raise ValueError('hourly donor leakage')
                    hellinger=.5*((np.sqrt(shares[donor_ids])-np.sqrt(shares[local]))**2).sum(axis=1)
                    conditional=prior*np.exp(-hellinger/.3**2);conditional/=conditional.sum()
                    atoms=[];pw=[];cw=[]
                    for j,d in enumerate(donor_ids):
                        g=generation(d);means=margins[d]/len(g)
                        shape=np.divide(g,means,out=np.ones_like(g),where=means>0)
                        hypothetical=shape*M;total=hypothetical.sum(axis=1)
                        cif=np.divide(hypothetical@EF,total,out=np.zeros(len(g)),where=total>0)
                        atoms.extend(cif);pw.extend(np.full(len(g),prior[j]/len(g)));cw.extend(np.full(len(g),conditional[j]/len(g)))
                    missing=(shares[local]*(margins[donor_ids]==0)).sum(axis=1)
                    provenance.append({'target_region':target,'seed':seed,'budget':budget,'profile_id':q,
                        'donor_ids':json.dumps(donor_ids.tolist()),'donor_regions':json.dumps(labels),
                        'conditional_weights':json.dumps(conditional.tolist()),'unsupported_mass':float(conditional@missing)})
                    laws={'unconditional':unconditional_law,
                        'level_residual':Law(np.clip(residuals+shares[local]@EF,0,1000),resweights),
                        'recompose_pooled':Law(atoms,pw),'recompose_conditional':Law(atoms,cw)}
                    if query.region==target:query_laws[q]=laws
                    else:
                        y=truth(q)
                        for method,law in laws.items():
                            a,b=law.quantile(.05),law.quantile(.95)
                            err=np.maximum.reduce([a-y,y-b,np.zeros(len(y))])
                            errors[method].setdefault(query.region,[]).append(err)
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
                            lo=max(0,a-radius);hi=min(1000,b+radius);coverage=(y>=lo)&(y<=hi)
                            winkler=hi-lo+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
                            if not 0<=lo<=hi<=1000:raise ValueError('hourly interval bounds')
                            outputs.append({'region':target,'month':index.month.iloc[q],'seed':seed,'budget':budget,'method':method,
                                'calibration':calibration,'hours':len(y),'crps':float(law.crps(y).mean()),
                                'mae':float(np.abs(y-law.mean).mean()),'coverage':float(coverage.mean()),
                                'width':hi-lo,'winkler':float(winkler.mean()),'lower':lo,'upper':hi,'mean':law.mean})
            print(f'hourly laws completed {target}/{seed}',flush=True)
    rows=pd.DataFrame(outputs)
    keys=['region','month','seed','budget','method','calibration']
    if rows.duplicated(keys).any():raise ValueError('duplicate target records')
    expected=index.loc[index.region.isin(['PACE','LGEE','SC']) & dates.dt.year.eq(2025)]
    if len(rows)!=len(expected)*2*2*4*2:raise ValueError('common target denominator')
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
        ('donor_provenance',pd.DataFrame(provenance)),('calibration_radii',pd.DataFrame(radii)),('calibration_regions',pd.DataFrame(cal_counts)),('region_roles',roles)]:
        frame.to_csv(OUT/f'{name}.csv',index=False)
    result={'status':'HOURLY_DISTRIBUTION_INFORMATION_AGE_DEVELOPMENT','target_region_months':len(expected),'scored_rows':len(rows),'assertion_failures':0,
        'crps_finite_atom_identity_checks':len(y),'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Known development identities, nominal production CIF proxy, retrospective quality screen, simulated release lag.',
            'Monthly marginal hourly law has no hour-specific phase; no original method claim or blind confirmation.',
            'Interval calibration does not change the predictive law used for CRPS.']}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(macro.to_string(index=False))

if __name__=='__main__':main()
