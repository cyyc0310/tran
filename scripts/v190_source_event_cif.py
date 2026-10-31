"""CIF-law event/source concentration probe with pre-July disjoint calibration."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from threadpoolctl import threadpool_limits
from calendar_cif_law import CalendarLaw
from extended_monthly_information import ExtendedMonthlyInformation
from hourly_fuel_data import HourlyFuelData
from monthly_history_features import MonthlyInformation
from source_event_cif import SourceEventCIF,METHODS,BASELINES,MODIFIED,law_hashes,input_guard
from source_event_transport import source_equalization,event_updates,transport_shift
from v151_hourly_law import sha
from v159_fuel_activity_source_pilot import weighted_radius
from v170_fuel_marginal_diagnosis import verify

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/v188-forward-zero-risk-r1'
ROLES=ROOT/'results/v159-fuel-activity-source-pilot-r1'
PRE=ROOT/'results/v190-source-event-cif-preflight-r2'
OUT=ROOT/'results/v190-source-event-cif-r2'
METRICS=('crps','mae','coverage','width','winkler','tolerant_coverage','lower_miss','upper_miss','support_below','support_above')


def dependencies():
    hashes={}
    for folder in (BASE,ROOT/'results/v188-forward-zero-risk-audit-r1',ROOT/'results/v189-source-block-oil-r2',ROOT/'results/v189-source-block-oil-audit-r2'):
        meta=verify(folder)
        for k,v in meta['input_hashes'].items():
            if k in hashes and hashes[k]!=v:raise ValueError('upstream conflict')
            hashes[k]=v
        for k in ('summary.json',*meta['output_hashes']):hashes[str((folder/k).relative_to(ROOT))]=sha(folder/k)
    for p in (Path(__file__),ROOT/'scripts/source_event_transport.py',ROOT/'scripts/source_event_cif.py',
              ROOT/'docs/V190_SOURCE_EVENT_CIF_PROTOCOL.md',ROOT/'docs/V190_SOURCE_EVENT_CIF_LITERATURE.md',ROOT/'docs/V190_THRESHOLD_FINGERPRINT_ADDENDUM.md'):
        hashes[str(p.relative_to(ROOT))]=sha(p)
    if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('dependency changed')
    return hashes


def queries_fold(fold):
    source=pd.read_csv(ROLES/f'fold_{fold}_asof_30d/query_inputs.csv')
    cal=source.loc[source.role.eq('calibration')&source.month.le('2024-05-01')].copy()
    test=pd.read_csv(BASE/f'fold_{fold}/queries.csv');test['role']='test'
    cols=['fold','profile_id','region','month','role']
    frame=pd.concat([cal[cols],test[cols]],ignore_index=True)
    frame['period']=np.where(frame.month.ge('2024-10-01'),'Q4',np.where(frame.month.ge('2024-07-01'),'Q3','preJuly_cal'))
    if frame.duplicated('profile_id').any() or len(cal)!=[10,10,10,10,8,7][fold] or set(cal.region)&set(test.region):raise ValueError('early cal/test roster')
    return frame


def make_model(data,short,long,fold):
    with np.load(BASE/f'fold_{fold}/training_matrices.npz',allow_pickle=False) as packed:arrays={k:packed[k].copy() for k in packed.files}
    proper=json.loads((BASE/f'fold_{fold}/training_provenance.json').read_text())['proper_regions']
    return SourceEventCIF(data,short,long,arrays,proper,queries_fold(fold),ROOT)


def numerical_contracts():
    rng=np.random.default_rng(190);max_w=max_identity=0.
    for i in range(100):
        n=64;values=rng.uniform(0,1000,n);weights=rng.uniform(.01,1,n);weights/=weights.sum();sources=np.repeat(np.arange(8),8)
        take=rng.random(n)<.6
        after,d=source_equalization(weights,sources,take)
        exact=wasserstein_distance(values,values,u_weights=weights,v_weights=after)
        max_w=max(max_w,abs(exact-transport_shift(values,weights,after)))
        if d['sources']>1 and d['gamma']>0:
            mass=np.bincount(sources[take],weights=weights[take],minlength=8);k=int((mass>0).sum());p=d['mass']
            eq=weights[take]/mass[sources[take]]/k
            expected=p*d['gamma']*wasserstein_distance(values[take],values[take],u_weights=weights[take]/p,v_weights=eq)
            max_identity=max(max_identity,abs(exact-expected))
        if abs(after.sum()-1)>1e-12 or abs(after[take].sum()-weights[take].sum())>1e-12:raise ValueError('toy layer mass')
    values=np.full(64,500.);weights=np.r_[np.full(32,.8/32),np.full(32,.2/32)];sources=np.repeat(np.arange(2),32)
    layers=np.zeros((16,64),bool);layers[0]=True;layers[1,:32]=True
    changes,details,_=event_updates(values,weights,sources,layers)
    if not np.allclose(changes['event_risk_full'],weights,rtol=0,atol=1e-15) or details[1]['status']!='single_source' or details[2]['status']!='absent':raise ValueError('constant/single/missing support toy')
    if max_w>1e-10 or max_identity>1e-10:raise ValueError('transport identity')
    return {'random_cases':100,'max_W1_error':max_w,'max_mixture_identity_error':max_identity,'constant_single_missing_cases':True}


def score_laws(cells,actual,clock,radius,full=False):
    distribution=CalendarLaw(cells,clock)
    lo=np.zeros(len(actual)) if full else np.maximum(0,distribution.quantile(.05)-radius)
    hi=np.full(len(actual),1000.) if full else np.minimum(1000,distribution.quantile(.95)+radius)
    if not np.isfinite(lo+hi).all() or (lo>hi).any():raise ValueError('invalid CIF interval')
    tol=64*np.finfo(float).eps*np.maximum.reduce([np.ones(len(actual)),abs(actual),abs(lo),abs(hi)])
    lower=actual<lo;upper=actual>hi
    support_lo=np.asarray([law.x[0] for law in cells])[clock];support_hi=np.asarray([law.x[-1] for law in cells])[clock]
    return {'crps':float(distribution.crps(actual).mean()),'mae':float(np.mean(abs(actual-distribution.mean))),
            'coverage':float(np.mean(~lower&~upper)),'tolerant_coverage':float(np.mean((actual>=lo-tol)&(actual<=hi+tol))),
            'width':float(np.mean(hi-lo)),'winkler':float(np.mean(hi-lo+20*np.maximum(lo-actual,0)+20*np.maximum(actual-hi,0))),
            'lower_miss':float(lower.mean()),'upper_miss':float(upper.mean()),'support_below':float(np.mean(actual<support_lo-tol)),
            'support_above':float(np.mean(actual>support_hi+tol))}


def aggregate(rows):
    keys=['period','region','method','radius_policy','calibration']
    regional=[]
    for period,t in [('H2',rows),('Q3',rows.loc[rows.period.eq('Q3')]),('Q4',rows.loc[rows.period.eq('Q4')])]:
        for key,g in t.groupby(keys[1:]):regional.append({'period':period,**dict(zip(keys[1:],key)),**{m:float(np.average(g[m],weights=g.hours)) for m in METRICS}})
    regional=pd.DataFrame(regional);macro=[]
    for key,g in regional.groupby(['period','method','radius_policy','calibration']):
        macro.append(dict(zip(['period','method','radius_policy','calibration'],key))|{'regions':len(g),'worst_coverage':float(g.coverage.min()),**{m:float(g[m].mean()) for m in METRICS}})
    return regional,pd.DataFrame(macro)



def repeat_fit_contract(model,other):
    if model.fingerprint()!=other.fingerprint():raise ValueError('repeat-fit semantic tree fingerprint differs')
    from monthly_risk_features import feature_matrices
    qfeatures=[feature_matrices(model.short,model.long,int(q))[0] for q in model.queries.profile_id]
    query={'20':np.concatenate([f['direct20'] for f in qfeatures]),'84':np.concatenate([f['long84'] for f in qfeatures])}
    pairs=[(f'kernel{k}',model.kernels[k].forest,other.kernels[k].forest,k) for k in ('20','84')]
    pairs += [(f'oof{i}',a,b,'84') for i,(a,b) in enumerate(zip(model.oof_models,other.oof_models))]
    pairs += [('mean',model.mean_model,other.mean_model,'84'),('variance',model.variance_model,other.variance_model,'84')]
    hashes={}
    for name,a,b,feature in pairs:
        for label,X in [('all_source_rows',model.X[feature]),('all_query_48states',query[feature])]:
            pa,pb=a.predict(X),b.predict(X)
            if pa.shape!=pb.shape or pa.dtype!=pb.dtype or pa.tobytes()!=pb.tobytes():raise ValueError(f'repeat-fit prediction bytes differ: {name}/{label}')
            hashes[name+'/'+label]=hashlib.sha256(pa.tobytes()).hexdigest()
    return {'source_rows':len(model.raw_ids),'query_profiles':len(model.queries),'models_compared':9,'prediction_arrays':hashes}


def run_fold(data,short,long,fold,preflight=False):
    model=make_model(data,short,long,fold)
    for q in model.queries.profile_id.iloc[[0,-1]]:input_guard(model,int(q))
    prefix=pd.DataFrame(model.source_prefixes())
    if prefix.to_csv(index=False)!=pd.read_csv(io.StringIO(prefix.to_csv(index=False))).to_csv(index=False):raise ValueError('source prefix CSV roundtrip')
    fingerprint=model.fingerprint()
    if preflight:
        repeat=make_model(data,short,long,fold)
        check=repeat_fit_contract(model,repeat)
        return fingerprint,check
    if fingerprint!=json.loads((PRE/'model_fingerprints.json').read_text())[str(fold)]:raise ValueError('preflight fit changed')
    sealed={};hash_rows=[];details=[];contexts=[];summaries={}
    # First pass is entirely outcome-free; only hashes and basic functionals persist.
    for rec in model.queries.itertuples(index=False):
        q=int(rec.profile_id);laws,diag,states,c=model.build(q);hashes=law_hashes(laws);sealed[q]=hashes
        details.extend({'fold':fold,'role':rec.role,**r} for r in diag)
        contexts.extend({'fold':fold,'role':rec.role,**r} for r in states)
        for method,cells in laws.items():
            summaries[(q,method)]={'lo':np.asarray([l.quantile(.05) for l in cells]),'hi':np.asarray([l.quantile(.95) for l in cells])}
            hash_rows.extend({'fold':fold,'role':rec.role,'profile_id':q,'method':method,'clock':h,'law_sha256':digest} for h,digest in enumerate(hashes[method]))
        del laws
    errors={m:{} for m in METHODS};radii=[]
    for rec in model.queries.loc[model.queries.role.eq('calibration')].itertuples(index=False):
        q=int(rec.profile_id);actual=data.truth(q);row=data.index.iloc[q];clock=data.states[int(row.offset):int(row.stop)]
        for method in METHODS:
            summary=summaries[(q,method)]
            error=np.zeros(len(actual)) if method=='uniform_range' else np.maximum.reduce([summary['lo'][clock]-actual,actual-summary['hi'][clock],np.zeros(len(actual))])
            errors[method].setdefault(rec.region,[]).append(error)
    for method in METHODS:
        if set(errors[method])!=set(model.queries.loc[model.queries.role.eq('calibration')].region):raise ValueError('calibration source roster')
        for label,radius in zip(('pooled','max_region'),weighted_radius(errors[method])):radii.append({'fold':fold,'method':method,'calibration':label,'radius':radius})
    radius_map={(r['method'],r['calibration']):r['radius'] for r in radii};scores=[]
    for rec in model.queries.loc[model.queries.role.eq('test')].itertuples(index=False):
        q=int(rec.profile_id);laws,_,_,_=model.build(q)
        if law_hashes(laws)!=sealed[q]:raise ValueError('sealed CIF law changed after calibration')
        actual=data.truth(q);row=data.index.iloc[q];clock=data.states[int(row.offset):int(row.stop)]
        for method,cells in laws.items():
            for policy in (('recalibrated','fixed') if method in MODIFIED else ('recalibrated',)):
                for calibration in ('pooled','max_region'):
                    radius=radius_map[(method if policy=='recalibrated' else 'physical84_bg',calibration)]
                    scores.append({'fold':fold,'profile_id':q,'region':rec.region,'month':rec.month,'period':rec.period,'hours':len(actual),
                                   'method':method,'radius_policy':policy,'calibration':calibration,'radius':radius,
                                   'interval_kind':'full_support' if method=='uniform_range' else 'calibrated_equal_tail',
                                   **score_laws(cells,actual,clock,radius,method=='uniform_range')})
        del laws
    frames={'test_month_metrics':pd.DataFrame(scores),'calibration_radii':pd.DataFrame(radii),'law_hashes':pd.DataFrame(hash_rows),
            'event_diagnostics':pd.DataFrame(details),'source_context':pd.DataFrame(contexts),'queries':model.queries,'source_prefixes':prefix,'calibration_label_inventory':pd.DataFrame(model.cal_inventory)}
    if len(scores)!=50*model.queries.role.eq('test').sum() or not np.isfinite(frames['test_month_metrics'][list(METRICS)]).all().all():raise ValueError('fold scores/domain')
    return frames,fingerprint,model


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight',action='store_true');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    dest=PRE if args.preflight else OUT
    if dest.exists() and (args.preflight or not args.resume):raise FileExistsError(dest)
    hashes=dependencies();data,short=HourlyFuelData(ROOT),MonthlyInformation(ROOT);long=ExtendedMonthlyInformation(ROOT,ROOT/'results/v183-coarse-history-export-r1/monthly_history.csv')
    if args.preflight:
        toy=numerical_contracts();fits={};repeat_checks={}
        with threadpool_limits(limits=2):
            for fold in range(6):
                fits[str(fold)],repeat_checks[str(fold)]=run_fold(data,short,long,fold,True)
                print(f'V190 preflight fold {fold} complete',flush=True)
        if dependencies()!=hashes:raise ValueError('preflight input changed')
        dest.mkdir();(dest/'model_fingerprints.json').write_text(json.dumps(fits,indent=2,allow_nan=False)+'\n')
        (dest/'repeat_fit_checks.json').write_text(json.dumps(repeat_checks,indent=2,allow_nan=False)+'\n')
        result={'assertion_failures':0,'models_fitted':108,'selected_models':54,'repeat_fit_models':54,'future_query_raw_guards':12,'held_outcomes_scored':False,'numerical_contracts':toy,
                'input_hashes':hashes,'output_hashes':{k:sha(dest/k) for k in ('model_fingerprints.json','repeat_fit_checks.json')}}
    else:
        if verify(PRE)['input_hashes']!=hashes:raise ValueError('preflight dependency differs')
        for k in ('summary.json','model_fingerprints.json','repeat_fit_checks.json'):hashes[str((PRE/k).relative_to(ROOT))]=sha(PRE/k)
        manifest=json.dumps({'input_hashes':hashes,'methods':METHODS,'cal_counts':[10,10,10,10,8,7],'cutoff':'2024-07-01'},indent=2,sort_keys=True)+'\n'
        if dest.exists():
            if (dest/'run_manifest.json').read_text()!=manifest or (dest/'summary.json').exists():raise ValueError('resume manifest/completed')
        else:dest.mkdir();(dest/'run_manifest.json').write_text(manifest)
        with threadpool_limits(limits=2):
            for fold in range(6):
                block=dest/f'fold_{fold}'
                if block.exists():
                    if verify(block)['manifest_sha256']!=sha(dest/'run_manifest.json'):raise ValueError('saved manifest')
                    continue
                frames,fit,model=run_fold(data,short,long,fold);stage=Path(tempfile.mkdtemp(prefix='.v190-cif-',dir=OUT.parent))
                for k,frame in frames.items():frame.to_csv(stage/f'{k}.csv',index=False)
                np.savez_compressed(stage/'source_matrices.npz',raw_id=model.raw_ids,profile_id=model.profiles,regions=np.asarray(model.regions,str),X20=model.X['20'],X84=model.X['84'],residual=model.y,activity=model.activity,layer=model.layers,oof=model.oof,prior=model.prior)
                (stage/'model_fingerprints.json').write_text(json.dumps(fit,indent=2,allow_nan=False)+'\n')
                meta={'assertion_failures':0,'fold':fold,'models_fitted':9,'proper_regions':model.proper,'fine_regions':model.source_names,
                      'manifest_sha256':sha(dest/'run_manifest.json'),'query_count':len(model.queries),'output_hashes':{p.name:sha(p) for p in stage.iterdir()}}
                (stage/'summary.json').write_text(json.dumps(meta,indent=2,allow_nan=False)+'\n');stage.rename(block);print(f'V190 formal fold {fold} complete',flush=True)
        rows=pd.concat([pd.read_csv(dest/f'fold_{f}/test_month_metrics.csv') for f in range(6)],ignore_index=True)
        if len(rows)!=7900 or rows.duplicated(['profile_id','method','radius_policy','calibration']).any():raise ValueError('root keys/count')
        r,m=aggregate(rows)
        for k,frame in [('test_month_metrics',rows),('region_metrics',r),('macro_metrics',m)]:frame.to_csv(dest/f'{k}.csv',index=False)
        if any(sha(ROOT/k)!=v for k,v in hashes.items()):raise ValueError('final input changed')
        result={'assertion_failures':0,'models_fitted':54,'calibration_queries':55,'test_queries':158,'score_rows':7900,'law_hash_rows':163584,
                'input_hashes':hashes,'output_hashes':{str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()},
                'scope':'Known 2024 source development. July retrospective cross-region calibration, no new-domain coverage guarantee or established novelty.'}
    (dest/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print({k:v for k,v in result.items() if not k.endswith('hashes')},flush=True)
if __name__=='__main__':main()
