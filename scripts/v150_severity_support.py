"""Dimension and source-influence controls for positive risk-response errors."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from v149_hurdle_risk import weights, record, summarize, sha, DELTAS

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/v147-conditional-curve-scale-r1'
C=ROOT/'results/v146-critical-composition-calibration-r1'
V149=ROOT/'results/v149-hurdle-risk-r1'
OUT=ROOT/'results/v150-severity-support-r1'
PROTOCOL=ROOT/'docs/V150_SEVERITY_SUPPORT_PROTOCOL.md'
FAMILIES=('horizontal_composition_joint','vertical_hull_joint')
FUELS=('coal','gas','oil','nuclear','hydro','wind','solar','other')
COMPOSITION=[f'fuelshare_{f}' for f in FUELS]+['log_total_mwh','month_sin','month_cos']
GEOMETRY=['fuelshare_coal','rsp_conditional_low_q05_tau600','rsp_conditional_high_q95_tau600',
    'rsp_conditional_high_q95_tau900','rsp_conditional_low_sd_tau600','rsp_conditional_high_sd_tau600',
    'rsp_conditional_high_sd_tau900','rsp_conditional_unsupported_mass_tau600','rsp_conditional_effective_regions_tau600']

def main():
    if OUT.exists():raise FileExistsError(OUT)
    paths=[PROTOCOL,Path(__file__),ROOT/'scripts/v149_hurdle_risk.py',ROOT/'scripts/v146_critical_composition_calibration.py',
        ROOT/'scripts/v142_pace_lgee_monthly_holdout.py',P/'fit_inverse_labels.csv',P/'scale_features.csv',P/'region_roles.csv',
        P/'summary.json',C/'curves.npz',C/'curve_index.csv',C/'calibration_inverse_scores.csv',C/'summary.json',
        V149/'target_rows.csv',V149/'summary.json',ROOT/'results/v144-canonical-profile-cache-r1/canonical_monthly_panel.csv']
    hashes={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for directory in (P,C,V149):
        summary=json.loads((directory/'summary.json').read_text())
        if summary['assertion_failures']:raise ValueError('upstream audit failure')
        for name,h in summary['output_hashes'].items():
            if sha(directory/name)!=h:raise ValueError(f'upstream changed: {name}')
    labels=pd.read_csv(P/'fit_inverse_labels.csv')
    features=pd.read_csv(P/'scale_features.csv',float_precision='round_trip')
    roles=pd.read_csv(P/'region_roles.csv')
    inverse=pd.read_csv(C/'calibration_inverse_scores.csv')
    ci=pd.read_csv(C/'curve_index.csv')
    data=np.load(C/'curves.npz',allow_pickle=False)
    if list(data['methods'])!=list(FAMILIES) or not np.array_equal(data['deltas'],DELTAS):raise ValueError('curve conventions')
    curves=data['curves'];panel=pd.read_csv(paths[-1]);monthly=panel.loc[panel.tau.eq(600)].set_index('profile_id')
    full=json.loads((P/'summary.json').read_text())['feature_columns']
    if any('factor_box' in c or 'floor' in c or 'nominal_K' in c for c in full):raise ValueError('label in features')
    outputs=[];predictions=[];diagnostics=[];influence=[];radii=[];scores=[]
    for target in ('PACE','LGEE','SC'):
        for seed in (41,83):
            roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)]
            fit=set(roster.loc[roster.role.eq('fit'),'region']);cal=set(roster.loc[roster.role.eq('calibration'),'region'])
            if len(fit)!=30 or len(cal)!=12 or fit&cal or target in fit|cal:raise ValueError('role separation')
            x=features.loc[features.target_region.eq(target)&features.seed.eq(seed)].set_index('profile_id')
            if not x.index.is_unique:raise ValueError('duplicate query feature')
            fold=ci.loc[ci.target_region.eq(target)&ci.seed.eq(seed)].set_index('profile_id');qids=fold.index.to_numpy(int)
            for fi,family in enumerate(FAMILIES):
                train=labels.loc[labels.target_region.eq(target)&labels.seed.eq(seed)&labels.family.eq(family)].set_index('profile_id')
                ca=inverse.loc[inverse.target_region.eq(target)&inverse.seed.eq(seed)&inverse.method.eq(family)].set_index('profile_id')
                if set(train.index)!=set(monthly.loc[monthly.region.isin(fit)].index) or not train.supported.all():raise ValueError('training denominator/support')
                if set(ca.index)!=set(monthly.loc[monthly.region.isin(cal)].index):raise ValueError('calibration denominator')
                positive=train.training_delta.gt(0).to_numpy();regions=train.region.to_numpy();positive_regions=sorted(set(regions[positive]))
                y=np.log1p(256*train.training_delta.to_numpy(float));w=weights(regions)
                constant=float(np.average(y[positive],weights=weights(regions[positive]))) if positive.any() else 0.
                log_predictions={'constant':np.full(len(qids),constant)}
                for name,columns in (('full',full),('composition',COMPOSITION),('geometry',GEOMETRY)):
                    scaler=StandardScaler().fit(x.loc[train.index,columns],sample_weight=w)
                    tx=scaler.transform(x.loc[train.index,columns]);qx=scaler.transform(x.loc[qids,columns])
                    model=Ridge(alpha=10).fit(tx[positive],y[positive],sample_weight=weights(regions[positive]))
                    log_predictions[name]=model.predict(qx)
                    if name in ('full','geometry'):
                        omitted=[]
                        for omitted_region in positive_regions:
                            mask=positive&(regions!=omitted_region)
                            refit=Ridge(alpha=10).fit(tx[mask],y[mask],sample_weight=weights(regions[mask]))
                            pred=refit.predict(qx);omitted.append(pred)
                            for j,qid in enumerate(qids):influence.append({'target_region':target,'seed':seed,'family':family,
                                'model':name,'omitted_region':omitted_region,'profile_id':int(qid),'log_prediction':float(pred[j]),
                                'diagnostic_scope':'fixed_teacher_fit_influence_only'})
                        if name=='geometry':log_predictions['geometry_jackknife']=np.mean(omitted,axis=0)
                truth=np.log1p(256*ca.delta.to_numpy(float));active=ca.supported.to_numpy()&ca.grid_index.gt(0).to_numpy()
                for model,pred in log_predictions.items():
                    scale=pd.Series(np.clip(np.maximum(0,np.expm1(np.clip(pred,-30,20))/256),1/256,1),index=qids)
                    predicted_log=pd.Series(pred,index=qids).loc[ca.index].to_numpy()
                    error=float(np.average(np.abs(predicted_log[active]-truth[active]),weights=weights(ca.region.to_numpy()[active]))) if active.any() else None
                    diagnostics.append({'target_region':target,'seed':seed,'family':family,'model':model,
                        'positive_fit_months':int(positive.sum()),'positive_fit_regions':len(positive_regions),
                        'positive_cal_months':int(active.sum()),'positive_cal_log_mae':error})
                    method=f'{"horizontal" if fi==0 else "vertical"}_positive_{model}'
                    for qid,value in scale.items():predictions.append({'target_region':target,'seed':seed,'method':method,'profile_id':int(qid),'scale':float(value)})
                    score=np.where(ca.supported,ca.delta.to_numpy()/scale.loc[ca.index].to_numpy(),np.inf)
                    pool=pd.DataFrame({'region':ca.region.to_numpy(),'score':score}).groupby('region').score.max()
                    if set(pool.index)!=cal:raise ValueError('calibration region count')
                    for region,value in pool.items():scores.append({'target_region':target,'seed':seed,'method':method,'region':region,'score':float(value) if np.isfinite(value) else None})
                    for alpha in (.1,.2):
                        rank=math.ceil(13*(1-alpha));radius=float(pool.sort_values().iloc[rank-1])
                        radii.append({'target_region':target,'seed':seed,'method':method,'alpha':alpha,'rank':rank,
                            'radius':radius if np.isfinite(radius) else None,'infinite_radius':not np.isfinite(radius)})
                        for row in panel.loc[panel.region.eq(target)].itertuples(index=False):
                            product=radius*scale.loc[row.profile_id];fallback=not np.isfinite(product) or product>1
                            if fallback:lo,hi=int(row.physical_lower),int(row.physical_upper)
                            else:
                                grid=min(int(np.searchsorted(DELTAS,product,side='left')),256)
                                a,b=curves[int(fold.loc[row.profile_id,'curve_id']),fi,(600,900).index(row.tau),grid]
                                lo=max(int(row.physical_lower),math.floor(a*row.hours+1e-9));hi=min(int(row.physical_upper),math.ceil(b*row.hours-1e-9))
                            if not row.physical_lower<=lo<=hi<=row.physical_upper:raise ValueError('physical containment')
                            outputs.append({**record(row,seed,method,alpha,lo,hi),'physical_fallback':fallback,
                                'factor_floor':row.factor_box_K_upper-row.factor_box_K_lower,'calibration_scope':'joint_two_thresholds'})
            print(f'completed severity support {target}/{seed}',flush=True)
    rows=pd.DataFrame(outputs);keys=['region','month','tau','seed','method','alpha']
    if len(rows)!=5280 or rows.duplicated(keys).any():raise ValueError('target denominator')
    for key,g in rows.groupby(['region','tau','seed','method','alpha']):
        if len(g)!=(18 if key[0]=='PACE' else 24):raise ValueError('per-cell denominator')
    for _,g in rows.groupby(keys[:-1]):
        wide,narrow=g.sort_values('alpha').itertuples(index=False)
        if wide.lower>narrow.lower or wide.upper<narrow.upper:raise ValueError('alpha nesting')
    old=pd.read_csv(V149/'target_rows.csv');reference=old.loc[old.method.str.endswith('_positive_severity')].copy()
    reference['method']=reference.method.str.replace('_positive_severity','_positive_full',regex=False)
    check=rows.loc[rows.method.str.endswith('_positive_full')].merge(reference,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    if len(check)!=1056 or not check.lower_new.eq(check.lower_old).all() or not check.upper_new.eq(check.upper_old).all():raise ValueError('V149 full model reproduction')
    frontier,joint=summarize(pd.concat([rows,old],ignore_index=True))
    if any(sha(ROOT/p)!=h for p,h in hashes.items()):raise ValueError('frozen input changed')
    OUT.mkdir(parents=True)
    for name,frame in [('target_rows',rows),('comparison_frontier',frontier),('joint_coverage',joint),('predicted_scales',pd.DataFrame(predictions)),
        ('source_severity_diagnostics',pd.DataFrame(diagnostics)),('fixed_teacher_influence',pd.DataFrame(influence)),
        ('calibration_radii',pd.DataFrame(radii)),('calibration_region_scores',pd.DataFrame(scores)),('region_roles',roles)]:frame.to_csv(OUT/f'{name}.csv',index=False)
    summary={'status':'SEVERITY_SUPPORT_DIMENSION_DEVELOPMENT','scored_rows':len(rows),'assertion_failures':0,'v149_reproduced_rows':len(check),
        'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in sorted(OUT.iterdir())},
        'limitations':['Fixed-teacher omissions are fit influence diagnostics, not source-region-independent validation.',
            'All methods are standard controls; monthly endpoint proxy coverage does not establish hourly CIF prediction.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('5280 rows; V149 1056 rows reproduced; assertion failures: 0')

if __name__=='__main__':main()
