"""Evaluation-only moment attribution of frozen hourly marginal CIF laws."""
from __future__ import annotations
import json
from fractions import Fraction
from pathlib import Path
import numpy as np
import pandas as pd
from v151_hourly_law import Law, EF, sha

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/v144-canonical-profile-cache-r1'
V151=ROOT/'results/v151-hourly-law-r1'
OUT=ROOT/'results/v152-hourly-moment-diagnostic-r2'
PROTOCOL=ROOT/'docs/V152_HOURLY_MOMENT_DIAGNOSTIC_PROTOCOL.md'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/v151_hourly_law.py',P/'profiles.npz',P/'profile_index.csv',
        V151/'summary.json',V151/'query_inputs.csv',V151/'donor_provenance.csv',V151/'target_month_metrics.csv',V151/'calibration_radii.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    previous=json.loads((V151/'summary.json').read_text())
    for name,h in previous['output_hashes'].items():
        if sha(V151/name)!=h:raise ValueError('V151 output mismatch')
    idx=pd.read_csv(P/'profile_index.csv');packed=np.load(P/'profiles.npz',allow_pickle=False)
    raw=packed['generation'];M=np.asarray([[float(Fraction(str(v))) for v in row] for row in packed['margin_fractions']])
    lookup={(r.region,r.month):r.profile_id for r in idx.itertuples()}
    inputs=pd.read_csv(V151/'query_inputs.csv');inputs=inputs.loc[inputs.role.eq('target')]
    provenance=pd.read_csv(V151/'donor_provenance.csv')
    queries=inputs.merge(provenance,on=['target_region','seed','budget','profile_id'],validate='one_to_one')
    if len(queries)!=144:raise ValueError('query denominator')
    radii=pd.read_csv(V151/'calibration_radii.csv').set_index(['target_region','seed','budget','method','calibration']).radius
    archived=pd.read_csv(V151/'target_month_metrics.csv')
    reference=archived.loc[archived.method.str.startswith('recompose')&archived.calibration.eq('pooled')].set_index(['region','month','seed','budget','method'])
    def generation(q):
        row=idx.iloc[q];return raw[int(row.offset):int(row.stop)]
    outputs=[];moments=[];reproduced=0;max_identity=0.
    for query in queries.itertuples(index=False):
        q=int(query.profile_id);local=lookup[(query.query_region,query.input_month)]
        input_anchor=float(M[local]@EF/M[local].sum())
        true_g=generation(q);G=true_g.sum(axis=1);y=true_g@EF/G
        true_mean=float(y.mean());true_sd=float(y.std());true_anchor=float(M[q]@EF/M[q].sum())
        kappa=float(np.mean((G-G.mean())*(y-y.mean()))/G.mean())
        identity=abs(true_mean-(true_anchor-kappa));max_identity=max(max_identity,identity)
        if identity>1e-8:raise ValueError('energy/time mean identity')
        truth_law=Law(y,np.ones(len(y)));self_score=float(truth_law.crps(y).mean())
        atoms=[];priors=[];conditional=[]
        donors=json.loads(query.donor_ids);weights=json.loads(query.conditional_weights)
        for j,d in enumerate(donors):
            if idx.region.iloc[d]==query.query_region or not idx.month.iloc[d].startswith('2024'):raise ValueError('donor isolation')
            g=generation(d);shape=np.divide(g,M[d]/len(g),out=np.ones_like(g),where=M[d]>0)
            hypo=shape*M[local];total=hypo.sum(axis=1)
            a=np.divide(hypo@EF,total,out=np.zeros(len(g)),where=total>0)
            atoms.extend(a);priors.extend(np.full(len(g),1/(len(donors)*len(g))));conditional.extend(np.full(len(g),weights[j]/len(g)))
        for method,w in (('recompose_pooled',priors),('recompose_conditional',conditional)):
            law=Law(atoms,w);sd=float(np.sqrt(np.sum(law.w*(law.x-law.mean)**2)))
            ratio=true_sd/sd if sd>1e-12 else 1.
            pred_kappa=input_anchor-law.mean
            decomposition=(input_anchor-true_anchor)-(pred_kappa-kappa)
            if abs(decomposition-(law.mean-true_mean))>1e-8:raise ValueError('bias decomposition')
            moments.append({'region':query.query_region,'month':query.query_month,'seed':query.seed,'budget':query.budget,'method':method,
                'hours':len(y),'input_anchor':input_anchor,'actual_energy_anchor':true_anchor,'true_time_mean':true_mean,
                'predicted_mean':law.mean,'mean_error':law.mean-true_mean,'stale_anchor_error':input_anchor-true_anchor,
                'true_covariance_ratio':kappa,'predicted_covariance_ratio':pred_kappa,'covariance_ratio_error':pred_kappa-kappa,
                'actual_sd':true_sd,'predicted_sd':sd,'sd_ratio':ratio,'marginal_self_crps':self_score,'identity_error':identity})
            variants={'raw':law,'oracle_location':Law(law.x+true_mean-law.mean,law.w),
                'oracle_scale':Law(law.mean+(law.x-law.mean)*ratio,law.w),
                'oracle_affine':Law(true_mean+(law.x-law.mean)*ratio,law.w),'oracle_marginal':truth_law}
            radius=float(radii.loc[(query.query_region,query.seed,query.budget,method,'pooled')])
            for variant,distribution in variants.items():
                lo=max(0,distribution.quantile(.05)-radius);hi=min(1000,distribution.quantile(.95)+radius)
                if lo>hi:raise ValueError('oracle interval inverted')
                crps=float(distribution.crps(y).mean())
                if crps<self_score-1e-8:raise ValueError('empirical marginal proper-score identity')
                coverage=float(((y>=lo)&(y<=hi)).mean())
                item={'region':query.query_region,'month':query.query_month,'seed':query.seed,'budget':query.budget,
                    'method':method,'variant':variant,'hours':len(y),'crps':crps,'excess_crps':crps-self_score,
                    'mean':distribution.mean,'coverage':coverage,'width':hi-lo,'lower':lo,'upper':hi,
                    'outside_physical_mass':float(distribution.w[(distribution.x < -1e-8)|(distribution.x > 1000+1e-8)].sum())}
                if variant=='raw':
                    old=reference.loc[(query.query_region,query.query_month,query.seed,query.budget,method)]
                    if any(abs(item[k]-old[k])>1e-7 for k in ('crps','mean','lower','upper','coverage')):raise ValueError('V151 reproduction')
                    reproduced+=1
                outputs.append(item)
    rows=pd.DataFrame(outputs)
    if len(rows)!=1440 or reproduced!=288:raise ValueError('complete denominator')
    keys=['region','month','seed','budget','method','variant']
    if rows.duplicated(keys).any():raise ValueError('duplicate score')
    regional=[]
    for key,g in rows.groupby(['region','seed','budget','method','variant']):
        if len(g)!=12:raise ValueError('regional denominator')
        regional.append(dict(zip(['region','seed','budget','method','variant'],key))|
            {name:float(np.average(g[name],weights=g.hours)) for name in ['crps','excess_crps','coverage','width','outside_physical_mass']})
    if any(sha(ROOT/name)!=h for name,h in hashes.items()):raise ValueError('input/code changed')
    OUT.mkdir(parents=True)
    rows.to_csv(OUT/'target_month_diagnostics.csv',index=False)
    pd.DataFrame(regional).to_csv(OUT/'region_diagnostics.csv',index=False)
    pd.DataFrame(moments).to_csv(OUT/'moment_decomposition.csv',index=False)
    summary={'status':'EVALUATION_ONLY_HOURLY_MOMENT_DIAGNOSTIC','scored_rows':len(rows),'v151_reproduced_rows':reproduced,
        'maximum_energy_time_identity_error':max_identity,'assertion_failures':0,'input_hashes':hashes,
        'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Target-label oracle diagnostics only; not deployable methods or optimal location/scale bounds.',
            'Marginal self-score is not irreducible hourly conditional uncertainty.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(pd.DataFrame(regional).loc[lambda x:x.method.eq('recompose_conditional')&x.budget.eq('same_month')].to_string(index=False))

if __name__=='__main__':main()
