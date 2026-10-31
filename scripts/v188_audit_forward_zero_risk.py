"""Refit V188 and independently audit cutoff eligibility, leaf means and Brier."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation,monthly_context
from monthly_risk_features import feature_matrices
from forward_zero_risk import MONTH_HEADS,HURDLES,check_inputs
from v151_hourly_law import sha
from v154_hourly_strong_baselines import balanced
from v170_fuel_marginal_diagnosis import verify,FUELS
from v171_two_scale_zero import METRICS
from v179_audit_increment_support import compare_frames
from v188_forward_zero_risk import ROOT,OUT as SOURCE,INV,KINDS,make_predictor,aggregate_periods,provenance

OUT=ROOT/'results/v188-forward-zero-risk-audit-r1'
BASE=ROOT/'results/v159-fuel-activity-source-pilot-r1'


def source_contract(model,timestamps):
    """Re-enumerate allowed original samples and labels, without inventory selections."""
    samples=pd.read_csv(BASE/'source_sampling.csv')
    samples=samples.loc[samples.budget.eq('asof_30d')].drop_duplicates('region').set_index('region')
    metadata=json.loads((BASE/f'fold_{model.fold}_asof_30d/summary.json').read_text())
    if sorted(metadata['source_regions'])!=model.fit:raise ValueError('proper source roster')
    expected_raw=[];expected_profiles=[];expected_months=[]
    for region in model.fit:
        original=np.asarray(json.loads(samples.loc[region,'sampled_raw_hour_indices']),int)
        for q in model.data.groups[region]:
            row=model.data.index.iloc[q]
            if not row.month.startswith('2024') or model.data.release.iloc[q]>model.cutoff:continue
            picked=original[(original>=row.offset)&(original<row.stop)]
            expected_raw.extend(picked.tolist());expected_profiles.extend([int(q)]*len(picked))
        for q in model.long.groups[region]:
            if model.long.release.iloc[q]>model.cutoff:continue
            eligible=model.long.groups[region][(model.long.release.iloc[model.long.groups[region]]<=model.long.dates.iloc[q]).to_numpy()]
            if len(eligible):expected_months.append(int(q))
    if (not np.array_equal(expected_raw,model.raw_ids) or not np.array_equal(expected_profiles,model.profiles)
        or not np.array_equal(expected_months,model.month_ids)
        or not np.all(timestamps[model.raw_ids]<int(model.cutoff.tz_localize('UTC').timestamp()))):raise ValueError('independent raw/current-month cutoff qualification')
    if len(np.unique(model.raw_ids))!=len(model.raw_ids) or len(np.unique(model.month_ids))!=len(model.month_ids):raise ValueError('source duplicate indices')
    if not np.array_equal(model.zero,model.data.raw[model.raw_ids]==0) or not np.array_equal(model.positive,model.data.M[model.profiles]>0):raise ValueError('source targets or positive filter')
    for f in range(8):
        take=model.positive[:,f];states=model.data.states[model.raw_ids[take]]
        weights=balanced(model.regions[take]);target=model.zero[take,f].astype(float)
        mass=np.bincount(states,weights=weights,minlength=48)
        zeros=np.bincount(states,weights=weights*target,minlength=48)
        expected=np.divide(zeros,mass,out=np.full(48,float(weights@target/weights.sum())),where=mass>0)
        if not np.allclose(expected,model.clock[:,f],rtol=0,atol=1e-12):raise ValueError('independent weighted positive-month clock')
    expected_queries=pd.read_csv(BASE/f'fold_{model.fold}_asof_30d/query_inputs.csv')
    expected_queries=expected_queries.loc[expected_queries.role.eq('test') & expected_queries.month.ge('2024-07-01')]
    if not np.array_equal(expected_queries.profile_id,model.queries.profile_id):raise ValueError('frozen original H2 test roster')
    for rec in model.queries.itertuples(index=False):
        if rec.region in model.fit or model.long.index.region.iloc[rec.profile_id]!=rec.region or rec.month!=model.long.index.month.iloc[rec.profile_id]:raise ValueError('query source identity')
        if rec.period!=('Q3' if pd.Timestamp(rec.month).month<10 else 'Q4'):raise ValueError('query period')
    for row in model.prefixes():
        data=model.short if row['kind']=='fine_short' else model.long
        q=int(row['profile_id']);prefix=json.loads(row['prefix'])
        expected=data.groups[row['region']][(data.release.iloc[data.groups[row['region']]]<=data.dates.iloc[q]).to_numpy()]
        expected=expected[np.argsort(data.dates.iloc[expected].to_numpy())]
        if not np.array_equal(prefix,expected) or row['local']!=prefix[-1] or q in prefix:raise ValueError('training legal prefix identity')
        if row['region'] not in model.fit:raise ValueError('prefix proper role')


def leaf_tables(model):
    result={}
    for name,forests in model.models.items():
        for fuel,forest in enumerate(forests):
            if name in MONTH_HEADS:X=model.month_X[MONTH_HEADS[name]];y=model.month_zero[:,fuel].astype(float);regions=model.month_regions
            else:
                key=name if name.startswith('direct') else 'direct'+name.removeprefix('cond')
                take=np.ones(len(model.raw_ids),bool) if name.startswith('direct') else model.positive[:,fuel]
                X=model.X[key][take];y=model.zero[take,fuel].astype(float);regions=model.regions[take]
            weights=balanced(regions);leaves=forest.apply(X);tables=[]
            for t,tree in enumerate(forest.estimators_):
                mass=np.bincount(leaves[:,t],weights=weights,minlength=tree.tree_.node_count)
                value=np.bincount(leaves[:,t],weights=weights*y,minlength=tree.tree_.node_count)
                tables.append(np.divide(value,mass,out=np.zeros_like(value),where=mass>0))
            result[(name,fuel)]=tables
    return result


def check_kernels(model,q,hourly,monthly,components,tables):
    features,contexts,extra=feature_matrices(model.short,model.long,q);c=contexts[1]
    mx={'base13':c['base'][None],'history36':c['history'][None],'history77':np.r_[c['history'],extra][None]}
    error=0.
    for name,forests in model.models.items():
        X=mx[MONTH_HEADS[name]] if name in MONTH_HEADS else features['long'+name.removeprefix('direct').removeprefix('cond')]
        for f,forest in enumerate(forests):
            leaves=forest.apply(X);expected=sum(table[leaves[:,t]] for t,table in enumerate(tables[(name,f)]))/32
            actual=monthly[name][f] if name in MONTH_HEADS else (hourly if name.startswith('direct') else components)[name][:,f]
            error=max(error,float(np.max(abs(expected-actual))))
    if error>1e-12:raise ValueError('independent empirical leaves')
    return error


def independent_scores(data,rec,outputs,fold):
    hourly,monthly,components=outputs;q=int(rec.profile_id);row=data.index.iloc[q]
    states=data.states[int(row.offset):int(row.stop)];actual=(data.raw[int(row.offset):int(row.stop)]==0).astype(float)
    local=data.local(q,'asof_30d');rows={k:[] for k in KINDS}
    def score(p,y):
        p,y=np.broadcast_arrays(np.asarray(p,float),np.asarray(y,float))
        return {'brier':float(np.mean(p*p-2*p*y+y)),'mean_forecast':float(p.mean()),'observed_fraction':float(y.mean()),'calibration_bias':float(p.mean()-y.mean())}
    for f,fuel in enumerate(FUELS):
        identity={'fold':fold,'period':rec.period,'profile_id':q,'region':rec.region,'month':rec.month,'hours':len(states),'fuel':fuel,
                  'input_fuel_positive':bool(data.M[local,f]>0),'current_fuel_positive':bool(data.M[q,f]>0)}
        for name,p in hourly.items():rows['hourly'].append(identity|{'method':name,**score(p[states,f],actual[:,f])})
        for name,p in monthly.items():rows['monthly'].append(identity|{'method':name,'is_world_proxy':False,**score(p[f],data.M[q,f]==0)})
        if data.M[q,f]>0:
            for name,p in components.items():rows['components'].append(identity|{'method':name,**score(p[states,f],actual[:,f])})
    return rows


def audit_fold(data,short,long,fold,timestamps):
    folder=SOURCE/f'fold_{fold}';meta=verify(folder);model=make_predictor(data,short,long,fold)
    if meta['manifest_sha256']!=sha(SOURCE/'run_manifest.json'):raise ValueError('fold manifest')
    source_contract(model,timestamps)
    if provenance(model)!=json.loads((folder/'training_provenance.json').read_text()) or model.fingerprint()!=json.loads((folder/'model_fingerprints.json').read_text()):raise ValueError('training provenance/fingerprint')
    with np.load(folder/'training_matrices.npz',allow_pickle=False) as saved:
        arrays=model.arrays()
        if set(saved.files)!=set(arrays) or any(not np.array_equal(v,saved[k]) for k,v in arrays.items()):raise ValueError('source matrices')
    if pd.DataFrame(model.prefixes()).to_csv(index=False)!=pd.read_csv(folder/'training_prefix.csv').to_csv(index=False):raise ValueError('prefix CSV binding')
    pd.testing.assert_frame_equal(model.queries.reset_index(drop=True),pd.read_csv(folder/'queries.csv'),check_exact=True)
    for q in model.queries.profile_id.iloc[[0,-1]]:check_inputs(model,int(q))
    with np.load(folder/'probabilities.npz',allow_pickle=False) as packed:
        if set(packed.files)!={'profile_id',*KINDS} or not np.array_equal(packed['profile_id'],model.queries.profile_id.to_numpy(int)):raise ValueError('query NPZ identity')
        saved={k:packed[k].copy() for k in KINDS}
    for k,a in saved.items():
        shape=(len(model.queries),len(KINDS[k]),8) if k=='monthly' else (len(model.queries),len(KINDS[k]),48,8)
        if a.shape!=shape or not np.isfinite(a).all() or a.min()<0 or a.max()>1:raise ValueError('probability axes/domain')
    tables=leaf_tables(model);rows={k:[] for k in KINDS};contexts=[];leaf_error=hurdle_error=0.
    predictions=[]
    for j,rec in enumerate(model.queries.itertuples(index=False)):
        q=int(rec.profile_id);outputs=model.predict(q);predictions.append((rec,outputs[:3]));hourly,monthly,components,c=outputs
        for k,values in zip(KINDS,outputs[:3]):
            if not np.array_equal(np.stack([values[m] for m in KINDS[k]]),saved[k][j]):raise ValueError('refit probability')
        leaf_error=max(leaf_error,check_kernels(model,q,hourly,monthly,components,tables))
        for method,(m,r) in HURDLES.items():hurdle_error=max(hurdle_error,float(np.max(abs(hourly[method]-(1-(1-monthly[m][None])*(1-components[r]))))))
        contexts.append({'profile_id':q,'local':int(c['local']),'prefix':[int(p) for p in c['prefix']]})
    if contexts!=json.loads((folder/'contexts.json').read_text()) or hurdle_error>1e-12:raise ValueError('context/hurdle identity')
    for rec,outputs in predictions:
        rebuilt=independent_scores(data,rec,outputs,fold)
        for k,v in rebuilt.items():rows[k].extend(v)
    errors=[]
    for k,v in rows.items():errors.append(compare_frames(pd.DataFrame(v),pd.read_csv(folder/f'{k}_scores.csv'),['profile_id','method','fuel']))
    print(f'V188 audit fold {fold} complete',flush=True)
    return {'fold':fold,'query_forecasts':len(model.queries),'forests_refitted':56,'model_query_leaf_checks':len(model.queries)*56,
            'score_rows':{k:len(v) for k,v in rows.items()},'max_score_error':max(errors),'max_leaf_error':leaf_error,'max_hurdle_error':hurdle_error,
            'positive_source_zero_counts':json.dumps({fuel:int(model.zero[model.positive[:,f],f].sum()) for f,fuel in enumerate(FUELS)}),
            'positive_source_hour_counts':json.dumps({fuel:int(model.positive[:,f].sum()) for f,fuel in enumerate(FUELS)}),
            'positive_source_class_counts':json.dumps({fuel:{'zero_event_hours':int((model.positive[:,f]&model.zero[:,f]).sum()),
                'nonzero_hours':int((model.positive[:,f]&~model.zero[:,f]).sum()),
                'zero_event_regions':sorted(set(model.regions[model.positive[:,f]&model.zero[:,f]])),
                'nonzero_regions':sorted(set(model.regions[model.positive[:,f]&~model.zero[:,f]]))} for f,fuel in enumerate(FUELS)})}


def pairs(kind):
    if kind=='components':return [('cond43','clock'),('cond84','clock'),('cond84','cond43')]
    if kind=='monthly':return [('month_history','month_base'),('month_volatility','month_history'),*[(m,'persistence') for m in MONTH_HEADS]]
    result=[('direct84','direct43')]
    for m in MONTH_HEADS:result += [(f'{m}_{a}',f'{m}_{b}') for a,b in [('cond43','clock'),('cond84','clock'),('cond84','cond43')]]
    for r in ('clock','cond43','cond84'):
        result += [(f'month_history_{r}',f'month_base_{r}'),(f'month_volatility_{r}',f'month_history_{r}')]
        result += [(f'{m}_{r}',f'persistence_{r}') for m in MONTH_HEADS]
    result += [(m,r) for m in HURDLES for r in ('direct43','direct84')]
    return result


def comparisons(tables):
    rng=np.random.default_rng(188);rows=[]
    for kind,regional in tables.items():
        pp=pairs(kind)
        if len(pp)!=len(set(pp)) or len(pp)!={'hourly':49,'monthly':5,'components':3}[kind]:raise ValueError('pair inventory')
        for (period,cohort,fuel),table in regional.groupby(['period','cohort','fuel']):
            for method,reference in pp:
                a,b=[table.loc[table.method.eq(x)].sort_values('region') for x in (method,reference)]
                if not len(a) or not np.array_equal(a.region,b.region) or a.region.duplicated().any():raise ValueError('paired cohort')
                delta=a.brier.to_numpy()-b.brier.to_numpy();boot=delta[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
                baseline=float(b.brier.mean())
                rows.append({'task':kind,'period':period,'cohort':cohort,'fuel':fuel,'method':method,'reference':reference,'regions':len(a),
                    'brier':float(a.brier.mean()),'reference_brier':baseline,'paired_brier_difference':float(delta.mean()),
                    'brier_reduction_percent':float(-100*delta.mean()/baseline) if baseline>0 else None,
                    'region_bootstrap_lo':float(np.quantile(boot,.025)),'region_bootstrap_hi':float(np.quantile(boot,.975)),
                    'regions_better_brier':int((delta < -1e-12).sum()),
                    'comparison_phase':'posthoc_direct_reference' if kind=='hourly' and reference in ('direct43','direct84') and method in HURDLES else 'original_protocol'})
    if len(rows)!=2736:raise ValueError('comparison denominator')
    return pd.DataFrame(rows)


def time_pairs(tables):
    """Only common regions enter Q4-minus-Q3, keeping period cohorts explicit."""
    rng=np.random.default_rng(18801);rows=[]
    for task,regional in tables.items():
        for (cohort,fuel,method),table in regional.groupby(['cohort','fuel','method']):
            a=table.loc[table.period.eq('Q3')].set_index('region');b=table.loc[table.period.eq('Q4')].set_index('region')
            common=sorted(set(a.index)&set(b.index))
            if not common:raise ValueError('empty common-region time cohort')
            delta=b.loc[common,'brier'].to_numpy()-a.loc[common,'brier'].to_numpy()
            boot=delta[rng.integers(0,len(common),size=(10000,len(common)))].mean(axis=1)
            rows.append({'task':task,'cohort':cohort,'fuel':fuel,'method':method,'common_regions':len(common),
                         'regions':json.dumps(common),'Q3_brier':float(a.loc[common,'brier'].mean()),'Q4_brier':float(b.loc[common,'brier'].mean()),
                         'Q4_minus_Q3':float(delta.mean()),'region_bootstrap_lo':float(np.quantile(boot,.025)),'region_bootstrap_hi':float(np.quantile(boot,.975))})
    if len(rows)!=336:raise ValueError('time-pair denominator')
    return pd.DataFrame(rows)



def advantage_time_pairs(tables):
    """Common-region change in model-minus-reference loss, not causal drift."""
    rng=np.random.default_rng(18802);rows=[]
    for task,regional in tables.items():
        for (cohort,fuel),table in regional.groupby(['cohort','fuel']):
            for method,reference in pairs(task):
                parts={}
                for period in ('Q3','Q4'):
                    t=table.loc[table.period.eq(period)]
                    a=t.loc[t.method.eq(method)].set_index('region').brier
                    b=t.loc[t.method.eq(reference)].set_index('region').brier
                    if set(a.index)!=set(b.index):raise ValueError('advantage period cohort')
                    parts[period]=a-b
                common=sorted(set(parts['Q3'].index)&set(parts['Q4'].index))
                if not common:raise ValueError('empty temporal advantage cohort')
                a=parts['Q3'].loc[common].to_numpy();b=parts['Q4'].loc[common].to_numpy();delta=b-a
                boot=delta[rng.integers(0,len(common),size=(10000,len(common)))].mean(axis=1)
                rows.append({'task':task,'cohort':cohort,'fuel':fuel,'method':method,'reference':reference,
                             'common_regions':len(common),'regions':json.dumps(common),
                             'comparison_phase':'posthoc_direct_reference' if task=='hourly' and reference in ('direct43','direct84') and method in HURDLES else 'original_protocol',
                             'Q3_model_minus_reference':float(a.mean()),'Q4_model_minus_reference':float(b.mean()),
                             'Q4_minus_Q3_advantage_gap':float(delta.mean()),
                             'region_bootstrap_lo':float(np.quantile(boot,.025)),'region_bootstrap_hi':float(np.quantile(boot,.975))})
    if len(rows)!=912:raise ValueError('temporal advantage denominator')
    return pd.DataFrame(rows)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    summary=verify(SOURCE);hashes=summary['input_hashes'].copy()
    paths=[SOURCE/'summary.json',*[SOURCE/k for k in summary['output_hashes']],Path(__file__),ROOT/'scripts/v179_audit_increment_support.py',ROOT/'docs/V188_TEMPORAL_AUDIT_ADDENDUM.md']
    for p in paths:hashes[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit input changed')
    data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT)
    long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    with np.load(ROOT/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as packed:timestamps=packed['unix_seconds'].copy()
    with threadpool_limits(limits=2):checks=[audit_fold(data,short,long,f,timestamps) for f in range(6)]
    tables={};coverage=[];counts={}
    for k in KINDS:
        rows=pd.concat([pd.read_csv(SOURCE/f'fold_{f}/{k}_scores.csv') for f in range(6)],ignore_index=True)
        if rows.duplicated(['profile_id','method','fuel']).any() or (SOURCE/f'{k}_scores.csv').read_text()!=rows.to_csv(index=False):raise ValueError('root serialization/keys')
        r,m=aggregate_periods(rows)
        compare_frames(r,pd.read_csv(SOURCE/f'{k}_region_metrics.csv'),['period','cohort','region','method','fuel'])
        compare_frames(m,pd.read_csv(SOURCE/f'{k}_macro_metrics.csv'),['period','cohort','method','fuel']);tables[k]=r;counts[k]=len(rows)
        # Coverage derived from canonical outcomes only for the conditional task definition.
        for period in ('H2','Q3','Q4'):
            sub=rows if period=='H2' else rows.loc[rows.period.eq(period)]
            for cohort,t in [('all_available',sub),('input_fuel_positive',sub.loc[sub.input_fuel_positive])]:
                for fuel,g in t.groupby('fuel'):
                    unique=g.drop_duplicates('profile_id')
                    coverage.append({'task':k,'period':period,'cohort':cohort,'fuel':fuel,'query_fuel_pairs':len(unique),
                                     'regions':json.dumps(sorted(set(unique.region))),'region_count':unique.region.nunique(),'hours':int(unique.hours.sum())})
    if counts['hourly']!=17696 or counts['monthly']!=5056:raise ValueError('total records')
    # Independently enumerate real current-positive query/fuel pairs for component denominator.
    queries=pd.read_csv(INV/'queries.csv');positive=sum(int((data.M[int(q)]>0).sum()) for q in queries.profile_id)
    if counts['components']!=3*positive:raise ValueError('conditional denominator')
    comparison=comparisons(tables);time=time_pairs(tables);advantage=advantage_time_pairs(tables)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit final input changed')
    OUT.mkdir();pd.DataFrame(checks).to_csv(OUT/'fold_checks.csv',index=False);pd.DataFrame(coverage).to_csv(OUT/'coverage.csv',index=False)
    comparison.to_csv(OUT/'comparisons.csv',index=False)
    comparison.loc[comparison.comparison_phase.eq('original_protocol')].to_csv(OUT/'original_protocol_comparisons.csv',index=False)
    comparison.loc[comparison.comparison_phase.eq('posthoc_direct_reference')].to_csv(OUT/'posthoc_direct_comparisons.csv',index=False)
    pd.DataFrame([{'fold':c['fold'],'fuel':fuel,**value} for c in checks for fuel,value in json.loads(c['positive_source_class_counts']).items()]).to_csv(OUT/'positive_source_class_counts.csv',index=False)
    time.to_csv(OUT/'time_pairs.csv',index=False)
    advantage.to_csv(OUT/'temporal_advantage_pairs.csv',index=False)
    result={'assertion_failures':0,'forests_refitted':336,'query_forecasts_rebuilt':sum(c['query_forecasts'] for c in checks),
            'model_query_leaf_checks':sum(c['model_query_leaf_checks'] for c in checks),'score_rows_rebuilt':counts,
            'current_positive_query_fuel_pairs':positive,'comparisons':len(comparison),'common_region_time_comparisons':len(time),'common_region_advantage_comparisons':len(advantage),
            'max_score_error':max(c['max_score_error'] for c in checks),'max_leaf_error':max(c['max_leaf_error'] for c in checks),
            'max_hurdle_error':max(c['max_hurdle_error'] for c in checks),'input_hashes':hashes,
            'output_hashes':{p.name:sha(p) for p in OUT.iterdir()},
            'scope':'Full refit, cutoff eligibility and expanded Brier; shared tree growing/features. Descriptive previously viewed 2024 data; no online-vintage or CIF confirmation.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
