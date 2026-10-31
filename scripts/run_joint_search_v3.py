"""Compose every frozen fold, globally seal, then evaluate target outcomes."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transcif.research.joint_v3 import ALPHAS, compose, load_component, select_source_alpha, sha, freeze_inputs, verify_frozen_inputs
from transcif.research.dynamic_v2 import read_year, daily_profiles, EF
from transcif.research.weighted_metrics import PreparedDistribution


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--components', type=Path)
    parser.add_argument('--fresh', action='store_true')
    parser.add_argument('--end')
    parser.add_argument('--train-year', type=int, default=2022)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--year', type=int, default=2025)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable output: choose fresh path')
    root=Path(__file__).resolve().parents[1]
    files=['src/transcif/research/joint_v3.py','scripts/run_joint_search_v3.py','docs/JOINT_SEARCH_V3_PROTOCOL.md',
           'src/transcif/research/dynamic_v2.py','src/transcif/research/references.py','src/transcif/research/eia_proxy.py',
           'src/transcif/research/weighted_metrics.py','src/transcif/research/mechanisms.py','src/transcif/research/structural_v3.py']
    if args.year == 2026:
        files.append('docs/CONFIRMATION_2026_PROTOCOL.md')
    prefit=freeze_inputs(root,args.data,files,args.output)
    if args.fresh:
        from transcif.research.structural_v3 import fit_model, source_policy
        from transcif.research.eia_proxy import REGIONS
        component_dir=args.output.with_name(args.output.name+'-components')
        if component_dir.exists():
            raise FileExistsError('Immutable component output')
        component_dir.mkdir(parents=True)
        deps=['src/transcif/research/structural_v3.py','src/transcif/research/dynamic_v2.py','src/transcif/research/eia_proxy.py','src/transcif/research/references.py','src/transcif/research/mechanisms.py','src/transcif/research/weighted_metrics.py']
        cp={'hashes':{f:sha(root/f) for f in deps},'train_year':args.train_year,'validation_year':args.year,'source_snapshots':str(args.output.resolve())}
        (component_dir/'protocol.json').write_text(json.dumps(cp,indent=2))
        monthly=pd.read_csv(args.data/'monthly.csv')
        frames={r:read_year(args.data/f'{r}_hourly.csv',args.train_year) for r in REGIONS}
        origins=pd.date_range(f'{args.year}-01-01',args.end or f'{args.year}-06-30',tz='UTC')
        for target in REGIONS:
            sources={r:f for r,f in frames.items() if r!=target}
            for delay in (0,30,90):
                winner,residuals,policy=source_policy(sources,monthly,delay,args.train_year)
                model=fit_model(sources,monthly,f'{args.train_year+1}-01-01T00:00:00Z',delay)
                predictions=model.predict(origins,target,monthly)
                predictions['source_policy']=predictions[winner].copy()
                residuals['source_policy']=residuals[winner].copy()
                size=len(residuals[winner])
                indices=np.random.default_rng(13017).choice(size,128,replace=size<128)
                folder=component_dir/f'{target}-{delay}'
                folder.mkdir()
                np.savez_compressed(folder/'predictions.npz',**predictions,**{'residual_'+k:v[indices] for k,v in residuals.items()},origins=origins.asi8)
                (folder/'prediction_seal.json').write_text(json.dumps({'prediction_sha256':sha(folder/'predictions.npz'),'target_labels_read_before_seal':False,'source_regions':list(model.source_regions),'policy':policy,'winner':winner,'calibration_indices':indices.tolist()},indent=2))
        args.components=component_dir
    if args.components is None:
        parser.error('--components or --fresh required')
    component_protocol=json.loads((args.components/'protocol.json').read_text())
    for name, digest in component_protocol['hashes'].items():
        snapshot = Path(component_protocol['source_snapshots'])/name if 'source_snapshots' in component_protocol else args.components/Path(name).name
        if sha(snapshot) != digest:
            raise ValueError('Archived component dependency hash mismatch: '+name)
    components={str(p):sha(p) for p in sorted(args.components.glob('*/predictions.npz'))}
    if not components:
        raise ValueError('No component folds')
    protocol={'year':args.year,'alphas':ALPHAS,'primary_alpha':.5,'primary_delay':30,
              'status':'post-hoc exploratory' if args.year==2025 else 'fixed primary H1 confirmation',
              'hashes':prefit['hashes'],'component_npz_hashes':components,
              'component_protocol_sha256':sha(args.components/'protocol.json'),
              'source_policy':'2022 source residual MAE; same-bank selection/calibration optimism',
              'data_hashes':{Path(p).name:h for p,h in prefit['data_hashes'].items()}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    bundles=[]
    for pathname in components:
        folder=Path(pathname).parent
        region,delay=folder.name.rsplit('-',1)
        predictions,residuals,origins,seal=load_component(folder,component_protocol)
        if region in seal['source_regions']:
            raise ValueError('Target present in source fit')
        if args.year==2026:
            keep=pd.to_datetime(origins,utc=True).month<=6
            origins=origins[keep]
            predictions={k:v[keep] for k,v in predictions.items()}
        alpha, risks=select_source_alpha(residuals)
        points={k:predictions[k] for k in ('latest','analog_shape_half','source_policy')}
        banks={k:residuals[k] for k in points}
        for a in ALPHAS:
            name=f'joint_{a:g}'
            points[name],banks[name]=compose(predictions,residuals,a)
        points['joint_source_alpha'],banks['joint_source_alpha']=compose(predictions,residuals,alpha)
        dest=args.output/folder.name
        dest.mkdir()
        np.savez_compressed(dest/'predictions.npz',**points,**{'residual_'+k:v for k,v in banks.items()},origins=origins)
        (dest/'prediction_seal.json').write_text(json.dumps({'prediction_sha256':sha(dest/'predictions.npz'),
            'target_labels_read_before_seal':False,'source_alpha':alpha,'source_risks':risks,
            'component_sha256':components[pathname],'calibration_indices':seal['calibration_indices']},indent=2))
        bundles.append((region,int(delay),origins,points,banks,alpha))
    (args.output/'all_predictions_sealed.json').write_text(json.dumps({'folds':len(bundles),'target_labels_read':False}))
    verify_frozen_inputs(root,prefit)
    if any(sha(p)!=h for p,h in components.items()):
        raise ValueError('Component changed before target label access')
    if sha(args.components/'protocol.json') != protocol['component_protocol_sha256']:
        raise ValueError('Component protocol changed before labels')
    if args.fresh and any(sha(root/f)!=h for f,h in component_protocol['hashes'].items()):
        raise ValueError('Fresh component dependencies changed before labels')
    rows=[]
    for region,delay,origins,points,banks,alpha in bundles:
        verify_frozen_inputs(root,prefit)
        days,truth=daily_profiles(read_year(args.data/f'{region}_hourly.csv',args.year))
        locations={int(day):i for i,day in enumerate(origins)}
        for day,actual in zip(days,truth):
            if day.value not in locations:
                continue
            i=locations[day.value]
            for name,values in points.items():
                point=values[i]
                metric=PreparedDistribution(np.clip(point+banks[name],0,EF.max()),np.ones(len(banks[name]))).score(actual)
                metric.update(region=region,delay=delay,method=name,origin=str(day),source_alpha=alpha,
                    raw_point_mae=float(np.abs(point-actual).mean()),raw_shape_mae=float(np.abs(point-point.mean()-actual+actual.mean()).mean()),
                    raw_level_mae=float(abs(point.mean()-actual.mean())),raw_point_below_zero=float(np.mean(point<0)),
                    raw_point_above_max=float(np.mean(point>EF.max())))
                rows.append(metric)
        print(f'{region} delay {delay} scored',flush=True)
    scores=pd.DataFrame(rows)
    if scores.empty:
        raise ValueError('No matching target outcomes')
    scores.to_csv(args.output/'day_metrics.csv',index=False)
    metrics=['crps','mae','coverage90','width90','raw_point_mae','raw_shape_mae','raw_level_mae','raw_point_below_zero','raw_point_above_max']
    regional=scores.groupby(['region','delay','method'])[metrics].mean().reset_index()
    regional.to_csv(args.output/'region_metrics.csv',index=False)
    scores.groupby(['region','delay','method']).size().rename('days').to_csv(args.output/'day_counts.csv')
    macro=regional.groupby(['delay','method'])[metrics].mean().reset_index()
    macro.to_csv(args.output/'macro_metrics.csv',index=False)
    verify_frozen_inputs(root,prefit)
    verification={'prefit_inputs_unchanged':True,'hashes_unchanged':all(sha(root/f)==h for f,h in protocol['hashes'].items()),
        'component_hashes_unchanged':all(sha(p)==h for p,h in components.items()),
        'data_hashes_unchanged':all(sha(args.data/f)==h for f,h in protocol['data_hashes'].items()),
        'seals_match':all(sha(p.parent/'predictions.npz')==json.loads(p.read_text())['prediction_sha256'] for p in args.output.glob('*/prediction_seal.json'))}
    (args.output/'postrun_verification.json').write_text(json.dumps(verification,indent=2))
    print(macro.to_string(index=False))

if __name__=='__main__':
    main()
