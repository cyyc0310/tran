"""Full refit and source/leaf/expanded-Brier audit of the oil deletion stress test."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from monthly_risk_features import feature_matrices
from source_block_oil_risk import SourceBlockOil,OIL,HOURLY,COMPONENTS,guard
from v151_hourly_law import sha
from v154_hourly_strong_baselines import balanced
from v170_fuel_marginal_diagnosis import verify
from v171_two_scale_zero import METRICS
from v179_audit_increment_support import compare_frames
from v189_source_block_oil_risk import ROOT,OUT as SOURCE,BASE,BASELINE,roster,load_fold,aggregates,score_frame

OUT=ROOT/'results/v189-source-block-oil-audit-r2'


def check_source(data,arrays,proper,times):
    original=pd.read_csv(ROOT/'results/v159-fuel-activity-source-pilot-r1/source_sampling.csv')
    original=original.loc[original.budget.eq('asof_30d')].drop_duplicates('region').set_index('region')
    raw=[];profiles=[]
    for region in proper:
        sample=np.asarray(json.loads(original.loc[region,'sampled_raw_hour_indices']),int)
        for q in data.groups[region]:
            row=data.index.iloc[q]
            if not row.month.startswith('2024') or data.release.iloc[q]>pd.Timestamp('2024-07-01'):continue
            selected=sample[(sample>=row.offset)&(sample<row.stop)]
            raw.extend(selected.tolist());profiles.extend([int(q)]*len(selected))
    if (not np.array_equal(raw,arrays['raw_id']) or not np.array_equal(profiles,arrays['profile_id'])
        or not np.all(times[arrays['raw_id']]<1719792000)
        or not np.array_equal(arrays['region'],data.index.region.to_numpy()[arrays['profile_id']])
        or not np.array_equal(arrays['zero'],data.raw[arrays['raw_id']]==0)
        or not np.array_equal(arrays['positive'],data.M[arrays['profile_id']]>0)):
        raise ValueError('original source role/time/target contract')


def tables(model,states):
    result={}
    for name,forest in model.models.items():
        key='direct'+name.removeprefix('direct').removeprefix('cond')
        take=np.ones(len(model.raw_ids),bool) if name.startswith('direct') else model.positive
        X=model.X[key][take];y=model.zero[take].astype(float);weights=balanced(model.regions[take]);leaves=forest.apply(X)
        result[name]=[]
        for t,tree in enumerate(forest.estimators_):
            mass=np.bincount(leaves[:,t],weights=weights,minlength=tree.tree_.node_count)
            event=np.bincount(leaves[:,t],weights=weights*y,minlength=tree.tree_.node_count)
            result[name].append(np.divide(event,mass,out=np.zeros_like(event),where=mass>0))
    w=balanced(model.regions[model.positive]);h=states[model.raw_ids[model.positive]];y=model.zero[model.positive]
    mass=np.bincount(h,weights=w,minlength=48);event=np.bincount(h,weights=w*y,minlength=48)
    clock=np.divide(event,mass,out=np.full(48,float(w@y/w.sum())),where=mass>0)
    if not np.allclose(model.clock,clock,rtol=0,atol=1e-12):raise ValueError('independent positive clock')
    return result


def leaf_check(model,tables,short,long,q,hourly,components):
    features,_,_=feature_matrices(short,long,q);error=0.
    for name,forest in model.models.items():
        X=features['long'+name.removeprefix('direct').removeprefix('cond')];leaves=forest.apply(X)
        expected=sum(table[leaves[:,t]] for t,table in enumerate(tables[name]))/32
        actual=(hourly if name.startswith('direct') else components)[name]
        error=max(error,float(np.max(abs(expected-actual))))
    if error>1e-12:raise ValueError('independent leaf means')
    return error


def score(data,rec,scenario,hourly,components,affected):
    q=int(rec.profile_id);row=data.index.iloc[q];h=data.states[int(row.offset):int(row.stop)]
    y=(data.raw[int(row.offset):int(row.stop),OIL]==0).astype(float);local=data.local(q,'asof_30d')
    identity={'fold':int(rec.fold),'period':rec.period,'profile_id':q,'region':rec.region,'month':rec.month,'hours':len(h),'fuel':'oil',
              'scenario':scenario,'input_fuel_positive':bool(data.M[local,OIL]>0),'current_fuel_positive':bool(data.M[q,OIL]>0),'affected':bool(affected)}
    def metrics(p):return {'brier':float(np.mean(p*p-2*p*y+y)),'mean_forecast':float(p.mean()),'observed_fraction':float(y.mean()),'calibration_bias':float(p.mean()-y.mean())}
    return {'hourly':[identity|{'method':m,**metrics(p[h])} for m,p in hourly.items()],
            'components':[identity|{'method':m,**metrics(p[h])} for m,p in components.items()] if data.M[q,OIL]>0 else []}


def audit_fold(data,short,long,fold,times):
    folder=SOURCE/f'fold_{fold}';meta=verify(folder)
    if meta['manifest_sha256']!=sha(SOURCE/'run_manifest.json'):raise ValueError('fold manifest')
    arrays,queries,old,proper=load_fold(fold);check_source(data,arrays,proper,times)
    pd.testing.assert_frame_equal(queries,pd.read_csv(folder/'queries.csv'),check_exact=True)
    fingerprints=json.loads((folder/'model_fingerprints.json').read_text());provenance=json.loads((folder/'training_provenance.json').read_text())
    if set(fingerprints)!={BASELINE,*proper} or set(provenance)!=set(fingerprints):raise ValueError('fit block inventory')
    with np.load(folder/'probabilities.npz',allow_pickle=False) as packed:
        if set(packed.files)!={'profile_id','scenario','hourly','components'} or not np.array_equal(packed['profile_id'],queries.profile_id.to_numpy(int)) or list(packed['scenario'])!=roster():raise ValueError('probability identity')
        saved={k:packed[k].copy() for k in ('hourly','components')}
    for k,a in saved.items():
        shape=(30,len(queries),5 if k=='hourly' else 3,48)
        if a.shape!=shape or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('probability schema/domain')
    predictions={};maxleaf=0.;source_rows=[]
    for deleted in (BASELINE,*proper):
        model=SourceBlockOil(arrays,data.states,deleted)
        expected=arrays['raw_id'][arrays['region']!=deleted]
        if not np.array_equal(model.raw_ids,expected) or model.fingerprint()!=fingerprints[deleted] or model.provenance()!=provenance[deleted]:raise ValueError('deletion source/fingerprint/provenance')
        for q in queries.profile_id.iloc[[0,-1]]:guard(model,data,short,long,int(q))
        tt=tables(model,data.states);outputs=[]
        for q in queries.profile_id:
            hourly,components,c=model.predict(short,long,int(q));outputs.append((hourly,components))
            maxleaf=max(maxleaf,leaf_check(model,tt,short,long,int(q),hourly,components))
            m=float(long.M[c['local'],OIL]==0)
            for name,p in components.items():
                if not np.allclose(hourly['persistence_'+name],1-(1-m)*(1-p),rtol=0,atol=1e-12):raise ValueError('hurdle identity')
        predictions[deleted]={k:np.asarray([np.stack([v[i][m] for m in names]) for v in outputs]) for i,(k,names) in enumerate([('hourly',HOURLY),('components',COMPONENTS)])}
        if deleted==BASELINE:
            from v188_forward_zero_risk import KINDS
            for k,names in [('hourly',HOURLY),('components',COMPONENTS)]:
                target=np.stack([old[k][:,list(KINDS[k]).index(m),:,OIL] for m in names],axis=1)
                if not np.array_equal(predictions[deleted][k],target):raise ValueError('V188 baseline reproduction')
        elif not np.any(arrays['region']==deleted):
            if any(not np.array_equal(a,predictions[BASELINE][k]) for k,a in predictions[deleted].items()):raise ValueError('zero-amount placebo')
        if deleted!=BASELINE and not np.any((arrays['region']==deleted)&arrays['positive'][:,OIL]):
            if not np.array_equal(predictions[deleted]['components'],predictions[BASELINE]['components']):raise ValueError('zero-positive-oil support placebo changed conditional probabilities')
        source_rows.append({'fold':fold,**model.provenance()})
    rows={k:[] for k in saved}
    for si,scenario in enumerate(roster()):
        expected=predictions[scenario if scenario in proper else BASELINE]
        if any(not np.array_equal(a,saved[k][si]) for k,a in expected.items()):raise ValueError('scenario/nonproper reuse identity')
        for j,rec in enumerate(queries.itertuples(index=False)):
            h={m:saved['hourly'][si,j,k] for k,m in enumerate(HOURLY)};c={m:saved['components'][si,j,k] for k,m in enumerate(COMPONENTS)}
            scored=score(data,rec,scenario,h,c,scenario in proper and bool(np.any(arrays['region']==scenario)))
            for k,v in scored.items():rows[k].extend(v)
    errors=[compare_frames(score_frame(v),pd.read_csv(folder/f'{k}_scores.csv'),['scenario','profile_id','method']) for k,v in rows.items()]
    print(f'V189 audit fold {fold} complete',flush=True)
    return {'fold':fold,'fit_blocks':len(fingerprints),'forests_refitted':4*len(fingerprints),'scenario_query_forecasts':30*len(queries),
            'max_score_error':max(errors),'max_leaf_error':maxleaf},source_rows


def pairs(kind):
    if kind=='components':return [('cond43','clock'),('cond84','clock'),('cond84','cond43')]
    return [('direct84','direct43'),('persistence_cond43','persistence_clock'),('persistence_cond84','persistence_clock'),
            ('persistence_cond84','persistence_cond43'),*[(a,b) for a in ('persistence_cond43','persistence_cond84') for b in ('direct43','direct84')]]


def comparisons(tables):
    rng=np.random.default_rng(189);rows=[];changes=[]
    def interval(delta):
        boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
        return {'paired_difference':float(delta.mean()),'region_bootstrap_lo':float(np.quantile(boot,.025)),'region_bootstrap_hi':float(np.quantile(boot,.975))}
    for task,t in tables.items():
        for (scenario,period,cohort),group in t.groupby(['scenario','period','cohort']):
            for method,ref in pairs(task):
                a,b=[group.loc[group.method.eq(m)].sort_values('region') for m in (method,ref)]
                if not len(a) or not np.array_equal(a.region,b.region) or a.region.duplicated().any():raise ValueError('paired model cohort')
                delta=a.brier.to_numpy()-b.brier.to_numpy();base=float(b.brier.mean())
                rows.append({'task':task,'scenario':scenario,'period':period,'cohort':cohort,'method':method,'reference':ref,'regions':len(a),
                             'brier':float(a.brier.mean()),'reference_brier':base,'reduction_percent':float(-100*delta.mean()/base) if base>0 else None,**interval(delta)})
                if task=='components' and scenario!=BASELINE and ref=='clock':
                    baseline=t.loc[t.scenario.eq(BASELINE)&t.period.eq(period)&t.cohort.eq(cohort)]
                    aa,bb=[baseline.loc[baseline.method.eq(m)].sort_values('region') for m in (method,ref)]
                    if not np.array_equal(a.region,aa.region) or not np.array_equal(a.region,bb.region):raise ValueError('baseline deletion pairing')
                    original=aa.brier.to_numpy()-bb.brier.to_numpy()
                    changes.append({'scenario':scenario,'period':period,'cohort':cohort,'method':method,'regions':len(a),
                                    'baseline_model_minus_clock':float(original.mean()),'deleted_model_minus_clock':float(delta.mean()),**interval(delta-original)})
    if len(rows)!=1980 or len(changes)!=348:raise ValueError('comparison denominators')
    return pd.DataFrame(rows),pd.DataFrame(changes)



def affected_comparisons(rows,mode='fine'):
    """Same-query baseline pairing on training-availability-defined affected folds."""
    rng=np.random.default_rng(18901);records=[];coverage=[]
    for source in roster()[1:]:
        eligible=[]
        for fold in range(6):
            arrays,_,_,proper=load_fold(fold)
            if source in proper and np.any((arrays['region']==source)&(arrays['positive'][:,OIL] if mode=='positive_oil' else np.ones(len(arrays['region']),bool))):eligible.append(fold)
        source_rows=rows.loc[rows.scenario.eq(source)]
        if mode=='fine' and not np.array_equal(source_rows.affected.to_numpy(),source_rows.fold.isin(eligible).to_numpy()):raise ValueError('independent affected-fold definition')
        affected=source_rows.loc[source_rows.fold.isin(eligible)]
        for period in ('H2','Q3','Q4'):
            part=affected if period=='H2' else affected.loc[affected.period.eq(period)]
            for cohort,part in [('all_available',part),('input_fuel_positive',part.loc[part.input_fuel_positive])]:
                baseline=rows.loc[rows.scenario.eq(BASELINE)&rows.profile_id.isin(part.profile_id)]
                if len(baseline)!=len(part):raise ValueError('affected exact query pairing')
                coverage.append({'source':source,'period':period,'cohort':cohort,'eligible_folds':json.dumps(eligible),
                                 'query_count':part.profile_id.nunique(),'region_count':part.region.nunique(),
                                 'regions':json.dumps(sorted(set(part.region))),'hours':int(part.drop_duplicates('profile_id').hours.sum())})
                if len(part):
                    def regional(t):
                        return pd.DataFrame([{'region':key[0],'method':key[1],'brier':float(np.average(g.brier,weights=g.hours))}
                                             for key,g in t.groupby(['region','method'])]).pivot(index='region',columns='method',values='brier').sort_index()
                    a,b=regional(part),regional(baseline)
                    if not a.index.equals(b.index) or not np.isfinite(a.to_numpy()).all() or not np.isfinite(b.to_numpy()).all():raise ValueError('affected regional pairing')
                for method in ('cond43','cond84'):
                    row={'source':source,'period':period,'cohort':cohort,'method':method,'regions':part.region.nunique()}
                    if not len(part):
                        row.update(status='no_eligible_fold' if not eligible else 'no_task_rows',baseline_model_minus_clock=None,deleted_model_minus_clock=None,paired_difference=None,region_bootstrap_lo=None,region_bootstrap_hi=None)
                    else:
                        original=(b[method]-b['clock']).to_numpy();deleted=(a[method]-a['clock']).to_numpy();delta=deleted-original
                        boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(axis=1)
                        row.update(status='ok',baseline_model_minus_clock=float(original.mean()),deleted_model_minus_clock=float(deleted.mean()),
                                   paired_difference=float(delta.mean()),region_bootstrap_lo=float(np.quantile(boot,.025)),region_bootstrap_hi=float(np.quantile(boot,.975)))
                    records.append(row)
    if len(records)!=348 or len(coverage)!=174:raise ValueError('affected identity denominator')
    return pd.DataFrame(records),pd.DataFrame(coverage)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    summary=verify(SOURCE);hashes=summary['input_hashes'].copy()
    for p in [SOURCE/'summary.json',*[SOURCE/k for k in summary['output_hashes']],Path(__file__),ROOT/'scripts/v179_audit_increment_support.py',ROOT/'docs/V189_SUPPORT_EXPOSED_ADDENDUM.md']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit input changed')
    data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT);long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    with np.load(ROOT/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as packed:times=packed['unix_seconds'].copy()
    with threadpool_limits(limits=2):checked=[audit_fold(data,short,long,f,times) for f in range(6)]
    tables={};component_rows=None
    for k,count in [('hourly',23700),('components',5490)]:
        rows=pd.concat([pd.read_csv(SOURCE/f'fold_{f}/{k}_scores.csv') for f in range(6)],ignore_index=True)
        if len(rows)!=count or rows.duplicated(['scenario','profile_id','method']).any() or (SOURCE/f'{k}_scores.csv').read_text()!=rows.to_csv(index=False):raise ValueError('root binding/count')
        r,m=aggregates(rows)
        if k=='components':component_rows=rows
        compare_frames(r,pd.read_csv(SOURCE/f'{k}_region_metrics.csv'),['scenario','period','cohort','region','method','fuel'])
        compare_frames(m,pd.read_csv(SOURCE/f'{k}_macro_metrics.csv'),['scenario','period','cohort','method','fuel']);tables[k]=r
    comparison,change=comparisons(tables)
    affected,coverage=affected_comparisons(component_rows)
    exposed,exposed_coverage=affected_comparisons(component_rows,'positive_oil')
    if sum(c[0]['fit_blocks'] for c in checked)!=122 or sum(c[0]['scenario_query_forecasts'] for c in checked)!=4740:raise ValueError('full refit/forecast scope')
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit final input changed')
    OUT.mkdir();pd.DataFrame([c[0] for c in checked]).to_csv(OUT/'fold_checks.csv',index=False)
    (OUT/'source_support.json').write_text(json.dumps([r for c in checked for r in c[1]],indent=2,allow_nan=False)+'\n')
    comparison.to_csv(OUT/'comparisons.csv',index=False);change.to_csv(OUT/'deletion_advantage_changes.csv',index=False)
    affected.to_csv(OUT/'affected_advantage_changes.csv',index=False);coverage.to_csv(OUT/'affected_coverage.csv',index=False)
    exposed.to_csv(OUT/'support_exposed_advantage_changes.csv',index=False);exposed_coverage.to_csv(OUT/'support_exposed_coverage.csv',index=False)
    result={'assertion_failures':0,'fit_blocks_refitted':122,'forests_refitted':488,'scenario_query_forecasts':4740,
            'hourly_scores_rebuilt':23700,'conditional_scores_rebuilt':5490,'model_comparisons':1980,'deletion_advantage_changes':348,'affected_comparison_identities':348,'affected_comparisons_estimable':int(affected.status.eq('ok').sum()),'support_exposed_comparison_identities':len(exposed),'support_exposed_comparisons_estimable':int(exposed.status.eq('ok').sum()),
            'max_score_error':max(c[0]['max_score_error'] for c in checked),'max_leaf_error':max(c[0]['max_leaf_error'] for c in checked),
            'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in OUT.iterdir()},
            'scope':'Full refit/leaf/expanded loss audit sharing feature construction and tree growth; all 30 scenarios reuse previously viewed 158 queries.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
