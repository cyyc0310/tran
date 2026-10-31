"""Full law/refit audit, independent source-event formulas and CDF-loss integrals."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from calendar_cif_law import CalendarLaw
from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from monthly_risk_features import feature_matrices
from source_event_cif import METHODS,MODIFIED,BASELINES,law_hashes,input_guard
from v151_hourly_law import EF,sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify
from v179_audit_increment_support import compare_frames
from v190_source_event_cif import ROOT,OUT as SOURCE,BASE,ROLES,PRE,METRICS,queries_fold,make_model,aggregate

OUT=ROOT/'results/v190-source-event-cif-audit-r2'
PHYSICAL=('physical84_leaf','physical84_bg','source_uniform','source_delete_mix','event_mass_full','event_risk_half','event_risk_full','oil_nonzero_full')
REFERENCES=tuple(m for m in METHODS if m not in ('event_risk_half','event_risk_full'))


def metadata_compare(new,old,keys):
    a,b=[f.sort_values(keys).reset_index(drop=True) for f in (new,old)]
    if list(a.columns)!=list(b.columns) or a.duplicated(keys).any() or b.duplicated(keys).any():raise ValueError('metadata identity/duplicate')
    pd.testing.assert_frame_equal(a,b,check_dtype=False,check_exact=False,rtol=0,atol=1e-9)


def source_contract(model,fold,times):
    data,long=model.data,model.long;cutoff=pd.Timestamp('2024-07-01')
    proper=sorted(json.loads((ROLES/f'fold_{fold}_asof_30d/summary.json').read_text())['source_regions'])
    if proper!=model.proper:raise ValueError('proper roster')
    samples=pd.read_csv(ROLES/'source_sampling.csv');samples=samples.loc[samples.budget.eq('asof_30d')].drop_duplicates('region').set_index('region')
    raw=[];profiles=[]
    for region in proper:
        original=np.asarray(json.loads(samples.loc[region,'sampled_raw_hour_indices']),int)
        for q in data.groups[region]:
            row=data.index.iloc[q]
            if row.month.startswith('2024') and data.release.iloc[q]<=cutoff:
                chosen=original[(original>=row.offset)&(original<row.stop)];raw.extend(chosen.tolist());profiles.extend([int(q)]*len(chosen))
    if not np.array_equal(raw,model.raw_ids) or not np.array_equal(profiles,model.profiles) or not np.all(times[model.raw_ids]<int(cutoff.tz_localize('UTC').timestamp())):raise ValueError('independent source cutoff/sample mapping')
    for row in model.source_prefixes():
        container=model.short if row['kind']=='short' else long;q=int(row['profile_id']);ids=container.groups[row['region']]
        expected=ids[(container.release.iloc[ids]<=container.dates.iloc[q]).to_numpy()];expected=expected[np.argsort(container.dates.iloc[expected].to_numpy())]
        if not np.array_equal(expected,json.loads(row['prefix'])) or int(expected[-1])!=row['local'] or q in expected:raise ValueError('independent source prefix/local')
    originals=pd.read_csv(ROLES/f'fold_{fold}_asof_30d/query_inputs.csv')
    cal=originals.loc[originals.role.eq('calibration')&originals.month.isin(['2024-04-01','2024-05-01'])]
    tests=pd.read_csv(BASE/f'fold_{fold}/queries.csv');queries=model.queries
    if not np.array_equal(cal.profile_id,queries.loc[queries.role.eq('calibration')].profile_id) or not np.array_equal(tests.profile_id,queries.loc[queries.role.eq('test')].profile_id):raise ValueError('independent cal/test fixed roster')
    if set(cal.region)&set(tests.region) or (set(cal.region)|set(tests.region))&set(proper):raise ValueError('region leakage')
    if not (data.release.iloc[cal.profile_id]<=cutoff).all():raise ValueError('cal month release')
    for q in cal.profile_id:
        row=data.index.iloc[q];ids=np.arange(int(row.offset),int(row.stop))
        if not np.all(times[ids]<int(cutoff.tz_localize('UTC').timestamp())):raise ValueError('cal hourly cutoff')
    for r in queries.itertuples(index=False):
        if r.month!=data.index.month.iloc[r.profile_id] or r.region!=data.index.region.iloc[r.profile_id]:raise ValueError('query metadata')
        if r.role=='test' and r.period!=('Q3' if pd.Timestamp(r.month).month<10 else 'Q4'):raise ValueError('test period')
    raw_g=data.raw[model.raw_ids]
    if not np.isfinite(raw_g).all() or (raw_g<0).any() or (raw_g.sum(axis=1)<=0).any():raise ValueError('source generation')
    positive=long.M[model.profiles]>0
    expected=np.asarray([positive[:,f]&((raw_g[:,f]==0) if cls==0 else raw_g[:,f]>0) for f in range(8) for cls in (0,1)])
    if not np.array_equal(expected,model.layers):raise ValueError('source event bank identity')
    # Rebuild the physical activity from original raw and source past-only monthly inputs.
    local=np.asarray([long.local(int(q),'asof_30d') for q in model.profiles],int)
    prior=np.mean([long.shares[np.unique(local[model.regions==r])].mean(axis=0) for r in model.source_names],axis=0)+1e-6;prior/=prior.sum()
    scale=long.M[local].sum(axis=1)/data.index.hours.to_numpy()[model.profiles]
    activity=raw_g/(scale[:,None]*(.99*long.shares[local]+.01*prior))
    if not np.array_equal(prior,model.prior) or not np.allclose(activity,model.activity,rtol=1e-14,atol=0):raise ValueError('independent activity/prior')
    source_anchor=np.asarray([long.shares[p]@EF for p in local])
    residual=(raw_g*EF[None]).sum(axis=1)/raw_g.sum(axis=1)-source_anchor
    if not np.allclose(residual,model.y,rtol=0,atol=1e-10):raise ValueError('independent CIF residual labels')
    for r,group in model.inner.items():
        if r not in model.source_names or group not in range(5):raise ValueError('proper OOF role')


def independent_weights(values,w,sources,layers,choice):
    w=w/w.sum();order=np.argsort(values,kind='stable');gaps=np.diff(values[order]);all_candidates=[];impacts=[];mass_scores=[];records=[]
    for j,take in enumerate(layers):
        probability=float(w[take].sum());mass=np.asarray([float(w[take&(sources==s)].sum()) for s in range(int(sources.max())+1)])
        support=np.flatnonzero(mass>0);k=len(support);candidate=w.copy();gamma=0.;H=None
        if k:
            r=mass[support]/probability;H=float(sum(x*x for x in r))
        if k>1:
            gamma=float(max(0,min(1,(H-1/k)/(1-1/k))))
            if gamma<=64*np.finfo(float).eps:gamma=0.
            for source in support:
                ids=take&(sources==source);candidate[ids]=(1-gamma)*w[ids]+gamma*probability*w[ids]/(k*mass[source])
        shift=float(gaps@abs(np.cumsum(candidate[order]-w[order])[:-1]))
        if shift<=64*np.finfo(float).eps*max(1.,float(np.max(abs(values)))):shift=0.
        if abs(candidate[take].sum()-probability)>1e-12 or not np.array_equal(candidate[~take],w[~take]):raise ValueError('independent layer preservation')
        records.append({'layer':j,'mass':probability,'sources':k,'concentration':H,'gamma':gamma,'transport_impact':shift,'mass_concentration_score':probability*gamma})
        all_candidates.append(candidate);impacts.append(shift);mass_scores.append(probability*gamma)
    for key,scores in [('risk_layer',impacts),('mass_layer',mass_scores)]:
        selected=choice[key]
        if selected is None:
            if max(scores)>1e-9:raise ValueError('missing selected source event')
        elif scores[int(selected)]<max(scores)-1e-9:raise ValueError('chosen layer is not an objective maximum')
    risk=w if choice['risk_layer'] is None else all_candidates[int(choice['risk_layer'])]
    mass=w if choice['mass_layer'] is None else all_candidates[int(choice['mass_layer'])]
    ps=np.asarray([float(w[sources==s].sum()) for s in range(int(sources.max())+1)]);active=np.flatnonzero(ps>0);k=len(active)
    uniform=w.copy();deleted=np.zeros(len(w))
    for source in active:
        uniform[sources==source]=w[sources==source]/(k*ps[source])
        keep=sources!=source;deleted[keep]+=w[keep]/(k*float(w[keep].sum()))
    return {'event_risk_half':.5*w+.5*risk,'event_risk_full':risk,'event_mass_full':mass,'oil_nonzero_full':all_candidates[5],
            'source_uniform':uniform/uniform.sum(),'source_delete_mix':deleted/deleted.sum()},records


def cdf_crps(law,y):
    """CDF-squared antiderivative, independent of Law.crps's first-moment formula."""
    x,w=law.x,law.w;prob=np.cumsum(w);prob[-1]=1.;dx=np.diff(x)
    left=np.r_[0.,np.cumsum(prob[:-1]**2*dx)];right=np.r_[0.,np.cumsum((1-prob[:-1])**2*dx)]
    y=np.asarray(y,float);index=np.clip(np.searchsorted(x,y,side='right')-1,0,len(x)-1);distance=y-x[index]
    result=left[index]+prob[index]**2*distance+right[-1]-right[index]-(1-prob[index])**2*distance
    result=np.where(y<x[0],x[0]-y+right[-1],result);result=np.where(y>x[-1],left[-1]+y-x[-1],result)
    if result.min() < -1e-8:raise ValueError('negative independent CRPS')
    return np.maximum(result,0)


def brute_cdf(law,y):
    knots=np.unique(np.r_[law.x,float(y)]);mid=(knots[1:]+knots[:-1])/2
    probability=np.r_[0.,np.cumsum(law.w)][np.searchsorted(law.x,mid,side='right')]
    return float(np.diff(knots)@((probability-(mid>=y))**2))


def independent_scores(cells,actual,clock,radius,full):
    low=np.asarray([l.quantile(.05) for l in cells])[clock];high=np.asarray([l.quantile(.95) for l in cells])[clock]
    lo=np.zeros(len(actual)) if full else np.clip(low-radius,0,1000);hi=np.full(len(actual),1000.) if full else np.clip(high+radius,0,1000)
    if (lo>hi).any():raise ValueError('interval inversion')
    mean=np.asarray([l.mean for l in cells])[clock];loss=np.empty(len(actual))
    for state in np.unique(clock):take=clock==state;loss[take]=cdf_crps(cells[int(state)],actual[take])
    tol=64*np.finfo(float).eps*np.maximum.reduce([np.ones(len(actual)),abs(actual),abs(lo),abs(hi)])
    sl=np.asarray([l.x[0] for l in cells])[clock];sh=np.asarray([l.x[-1] for l in cells])[clock]
    return {'crps':float(loss.mean()),'mae':float(np.mean(abs(actual-mean))),'coverage':float(np.mean((actual>=lo)&(actual<=hi))),
            'tolerant_coverage':float(np.mean((actual>=lo-tol)&(actual<=hi+tol))),
            'width':float(np.mean(hi-lo)),'winkler':float(np.mean(hi-lo+20*np.maximum(lo-actual,0)+20*np.maximum(actual-hi,0))),
            'lower_miss':float(np.mean(actual<lo)),
            'upper_miss':float(np.mean(actual>hi)),'support_below':float(np.mean(actual<sl-tol)),'support_above':float(np.mean(actual>sh+tol))}


def query_weight_checks(model,q,laws,diagnostics,contexts):
    from fuel_activity_forest import response_values,carrier
    features,_,_=feature_matrices(model.short,model.long,q);leaves=model.kernels['84'].forest.apply(features['long84'])
    local=model.long.local(q,'asof_30d');values,positive=response_values(carrier(model.long.shares[local],'asof_30d',model.prior),model.activity)
    if not positive.all():raise ValueError('positive source support')
    support,inverse=np.unique(values,return_inverse=True);max_error=0.;diag_lookup={(d['clock'],d['layer']):d for d in diagnostics}
    for h,context in enumerate(contexts):
        ids,prob=model.kernels['84'].atoms(leaves[h]);w=.05*model.weights.copy();w[ids]+=.95*prob;w/=w.sum()
        candidates,independent=independent_weights(values,w,model.source_ids,model.layers,context)
        for row in independent:
            actual=diag_lookup[(h,row['layer'])]
            for name in ('mass','sources','concentration','gamma','transport_impact','mass_concentration_score'):
                if row[name] is None:
                    if actual[name] is not None:raise ValueError('absent concentration status')
                elif abs(row[name]-actual[name])>1e-9:raise ValueError('independent source layer summary')
        for name,wnew in candidates.items():
            expected=np.bincount(inverse,weights=wnew,minlength=len(support));expected/=expected.sum();law=laws[name][h]
            if not np.array_equal(law.x,support):raise ValueError('shared physical support')
            error=float(np.max(abs(law.w-expected)));max_error=max(max_error,error)
            if error>1e-12:raise ValueError('independent reweighting probability')
    return max_error


def audit_fold(data,short,long,fold,times):
    folder=SOURCE/f'fold_{fold}';meta=verify(folder);model=make_model(data,short,long,fold);source_contract(model,fold,times)
    if meta['manifest_sha256']!=sha(SOURCE/'run_manifest.json') or model.fingerprint()!=json.loads((folder/'model_fingerprints.json').read_text()):raise ValueError('semantic model/manifest')
    with np.load(folder/'source_matrices.npz',allow_pickle=False) as packed:
        arrays={'raw_id':model.raw_ids,'profile_id':model.profiles,'regions':np.asarray(model.regions,str),'X20':model.X['20'],'X84':model.X['84'],
                'residual':model.y,'activity':model.activity,'layer':model.layers,'oof':model.oof,'prior':model.prior}
        if set(packed.files)!=set(arrays) or any(not np.array_equal(v,packed[k]) for k,v in arrays.items()):raise ValueError('source matrix/OOF')
    metadata_compare(pd.DataFrame(model.source_prefixes()),pd.read_csv(folder/'source_prefixes.csv'),['kind','profile_id'])
    metadata_compare(pd.DataFrame(model.cal_inventory),pd.read_csv(folder/'calibration_label_inventory.csv'),['profile_id'])
    pd.testing.assert_frame_equal(model.queries,pd.read_csv(folder/'queries.csv'),check_exact=True)
    for q in model.queries.profile_id.iloc[[0,-1]]:input_guard(model,int(q))
    hashes=[];diag_rows=[];state_rows=[];sealed={};bounds={};max_weight=max_synthetic=0.;synthetic_count=0
    selected=set(map(int,model.queries.profile_id.iloc[[0,-1]]))
    # All CIF laws remain outcome-free in this first pass.
    for rec in model.queries.itertuples(index=False):
        q=int(rec.profile_id);laws,diag,states,_=model.build(q);sealed[q]=law_hashes(laws)
        max_weight=max(max_weight,query_weight_checks(model,q,laws,diag,states))
        diag_rows.extend({'fold':fold,'role':rec.role,**d} for d in diag);state_rows.extend({'fold':fold,'role':rec.role,**s} for s in states)
        for name,cells in laws.items():
            hashes.extend({'fold':fold,'role':rec.role,'profile_id':q,'method':name,'clock':h,'law_sha256':digest} for h,digest in enumerate(sealed[q][name]))
            bounds[(q,name)]=(np.asarray([c.quantile(.05) for c in cells]),np.asarray([c.quantile(.95) for c in cells]))
        if q in selected:
            for name in PHYSICAL:
                for h in (0,47):
                    ys=np.linspace(0,1000,11);analytic=cdf_crps(laws[name][h],ys);brute=np.asarray([brute_cdf(laws[name][h],y) for y in ys])
                    max_synthetic=max(max_synthetic,float(np.max(abs(analytic-brute))));synthetic_count+=len(ys)
        del laws
    if max_synthetic>1e-8:raise ValueError('synthetic CDF integral mismatch')
    metadata_compare(pd.DataFrame(hashes),pd.read_csv(folder/'law_hashes.csv'),['profile_id','method','clock'])
    metadata_compare(pd.DataFrame(diag_rows),pd.read_csv(folder/'event_diagnostics.csv'),['profile_id','clock','layer'])
    metadata_compare(pd.DataFrame(state_rows),pd.read_csv(folder/'source_context.csv'),['profile_id','clock'])
    errors={m:{} for m in METHODS}
    for rec in model.queries.loc[model.queries.role.eq('calibration')].itertuples(index=False):
        q=int(rec.profile_id);actual=data.truth(q);row=data.index.iloc[q];clock=data.states[int(row.offset):int(row.stop)]
        for method in METHODS:
            lo,hi=bounds[(q,method)];error=np.zeros(len(actual)) if method=='uniform_range' else np.maximum.reduce([lo[clock]-actual,actual-hi[clock],np.zeros(len(actual))])
            errors[method].setdefault(rec.region,[]).append(error)
    radii=[]
    for method in METHODS:
        # Independently form the region-balanced error CDF and its empirical .9 quantile.
        by_region=[np.concatenate(errors[method][r]) for r in sorted(errors[method])]
        values=np.concatenate(by_region);weights=np.concatenate([np.full(len(a),1/(len(by_region)*len(a))) for a in by_region])
        weights=weights/weights.sum();order=np.argsort(values,kind='stable');index=min(int(np.searchsorted(np.cumsum(weights[order]),.9)),len(values)-1)
        pooled=float(values[order[index]])
        individual=[]
        for a in by_region:
            ordered=np.sort(a,kind='stable');cdf=np.cumsum(np.full(len(a),1/len(a)))
            individual.append(float(ordered[min(int(np.searchsorted(cdf,.9)),len(a)-1)]))
        maximum=max(individual)
        for label,radius in [('pooled',pooled),('max_region',maximum)]:radii.append({'fold':fold,'method':method,'calibration':label,'radius':radius})
    compare_frames(pd.DataFrame(radii),pd.read_csv(folder/'calibration_radii.csv'),['method','calibration'])
    radius_map={(r['method'],r['calibration']):r['radius'] for r in radii};scores=[]
    for rec in model.queries.loc[model.queries.role.eq('test')].itertuples(index=False):
        q=int(rec.profile_id);laws,_,_,_=model.build(q)
        if law_hashes(laws)!=sealed[q]:raise ValueError('sealed test law identity')
        actual=data.truth(q);row=data.index.iloc[q];clock=data.states[int(row.offset):int(row.stop)]
        for method,cells in laws.items():
            for policy in (('recalibrated','fixed') if method in MODIFIED else ('recalibrated',)):
                for calibration in ('pooled','max_region'):
                    radius=radius_map[(method if policy=='recalibrated' else 'physical84_bg',calibration)]
                    scores.append({'fold':fold,'profile_id':q,'region':rec.region,'month':rec.month,'period':rec.period,'hours':len(actual),
                        'method':method,'radius_policy':policy,'calibration':calibration,'radius':radius,
                        'interval_kind':'full_support' if method=='uniform_range' else 'calibrated_equal_tail',
                        **independent_scores(cells,actual,clock,radius,method=='uniform_range')})
        del laws
    score_error=compare_frames(pd.DataFrame(scores),pd.read_csv(folder/'test_month_metrics.csv'),['profile_id','method','radius_policy','calibration'])
    print(f'V190 audited fold {fold}',flush=True)
    return {'fold':fold,'models_refitted':9,'query_laws_rebuilt':len(model.queries),'law_hashes_checked':len(hashes),
            'event_layers_checked':len(diag_rows),'score_rows_rebuilt':len(scores),'independent_weights_error':max_weight,
            'independent_score_error':score_error,'synthetic_crps_integral_cases':synthetic_count,'synthetic_crps_error':max_synthetic}


def comparisons(regional):
    rng=np.random.default_rng(190);records=[]
    for period in ('H2','Q3','Q4'):
        for method in ('event_risk_half','event_risk_full'):
            for policy in ('fixed','recalibrated'):
                for calibration in ('pooled','max_region'):
                    a=regional.loc[regional.period.eq(period)&regional.method.eq(method)&regional.radius_policy.eq(policy)&regional.calibration.eq(calibration)].sort_values('region')
                    for reference in REFERENCES:
                        ref_policy=policy if reference in MODIFIED else 'recalibrated'
                        b=regional.loc[regional.period.eq(period)&regional.method.eq(reference)&regional.radius_policy.eq(ref_policy)&regional.calibration.eq(calibration)].sort_values('region')
                        if not len(a) or not np.array_equal(a.region,b.region):raise ValueError('strong reference matched cohort/policy')
                        delta=a.crps.to_numpy()-b.crps.to_numpy();boot=delta[rng.integers(0,len(a),size=(10000,len(a)))].mean(axis=1)
                        baseline=float(b.crps.mean());gain=float(-100*delta.mean()/baseline) if baseline>0 else None
                        coverage=float(a.coverage.mean());worst=float(a.coverage.min());width=float(a.width.mean());winkler=float(a.winkler.mean());upper=float(np.quantile(boot,.975))
                        passed=coverage>=.9 and worst>=.85 and gain is not None and gain>=5 and width<=float(b.width.mean())+1e-10 and winkler<float(b.winkler.mean()) and upper<0
                        records.append({'period':period,'method':method,'radius_policy':policy,'calibration':calibration,'reference':reference,'reference_policy':ref_policy,
                            'regions':len(a),'crps':float(a.crps.mean()),'reference_crps':baseline,'crps_gain_percent':gain,'paired_crps_difference':float(delta.mean()),
                            'region_bootstrap_lo':float(np.quantile(boot,.025)),'region_bootstrap_hi':upper,'coverage':coverage,'worst_coverage':worst,'width':width,
                            'reference_width':float(b.width.mean()),'winkler':winkler,'reference_winkler':float(b.winkler.mean()),'joint_gate':bool(passed)})
    if len(records)!=336:raise ValueError('strong comparison count')
    return pd.DataFrame(records)


def main():
    if OUT.exists():raise FileExistsError(OUT)
    summary=verify(SOURCE);hashes=summary['input_hashes'].copy()
    for p in [SOURCE/'summary.json',*[SOURCE/k for k in summary['output_hashes']],Path(__file__),ROOT/'scripts/v179_audit_increment_support.py']:
        hashes[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit input changed')
    data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT);long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    with np.load(ROOT/'results/v153-calendar-times-r1/timestamps.npz',allow_pickle=False) as packed:times=packed['unix_seconds'].copy()
    with threadpool_limits(limits=2):checks=[audit_fold(data,short,long,f,times) for f in range(6)]
    rows=pd.concat([pd.read_csv(SOURCE/f'fold_{f}/test_month_metrics.csv') for f in range(6)],ignore_index=True)
    if len(rows)!=7900 or rows.duplicated(['profile_id','method','radius_policy','calibration']).any() or (SOURCE/'test_month_metrics.csv').read_text()!=rows.to_csv(index=False):raise ValueError('root full score binding')
    r,m=aggregate(rows);compare_frames(r,pd.read_csv(SOURCE/'region_metrics.csv'),['period','region','method','radius_policy','calibration']);compare_frames(m,pd.read_csv(SOURCE/'macro_metrics.csv'),['period','method','radius_policy','calibration'])
    c=comparisons(r)
    if sum(x['law_hashes_checked'] for x in checks)!=163584 or sum(x['synthetic_crps_integral_cases'] for x in checks)!=2112:raise ValueError('full audit scope')
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('audit final input changed')
    OUT.mkdir();pd.DataFrame(checks).to_csv(OUT/'fold_checks.csv',index=False);c.to_csv(OUT/'comparisons.csv',index=False)
    gates=c.groupby(['period','method','radius_policy','calibration']).joint_gate.all().reset_index(name='all_strong_references_gate');gates.to_csv(OUT/'strong_gates.csv',index=False)
    result={'assertion_failures':0,'models_refitted':54,'queries_rebuilt':213,'law_hashes_checked':163584,'event_layers_checked':163584,'score_rows_rebuilt':7900,
            'synthetic_crps_integral_cases':2112,'comparisons':336,'all_strong_H2_gates':int(gates.loc[gates.period.eq('H2'),'all_strong_references_gate'].sum()),
            'max_score_error':max(x['independent_score_error'] for x in checks),'max_weight_error':max(x['independent_weights_error'] for x in checks),
            'max_synthetic_crps_error':max(x['synthetic_crps_error'] for x in checks),'input_hashes':hashes,'output_hashes':{p.name:sha(p) for p in OUT.iterdir()},
            'scope':'Full semantic refit/law hashes and real CDF-squared CRPS; independent source/event weights and selected direct CDF integrals. Shared tree growth/feature construction; known-source retrospective development.'}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
