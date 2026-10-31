"""Immutable source-only structural search; seal forecasts before reading labels."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from transcif.research.structural_v3 import METHODS, POINTS, fit_model, source_policy, read_year, daily_profiles, EF
from transcif.research.eia_proxy import REGIONS
from transcif.research.weighted_metrics import PreparedDistribution


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path('results/research-prefix/data'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-years', type=int, nargs='+', default=[2022])
    parser.add_argument('--validation-year', type=int, default=2023)
    parser.add_argument('--delays', type=int, nargs='+', default=[0,30,90])
    parser.add_argument('--regions', nargs='+', default=list(REGIONS))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Immutable output: choose fresh path')
    if args.validation_year in args.train_years or max(args.train_years) >= args.validation_year:
        raise ValueError('Training must precede validation')
    args.output.mkdir(parents=True)
    root = Path(__file__).resolve().parents[1]
    files = ['src/transcif/research/structural_v3.py','scripts/run_structural_search_v3.py','docs/STRUCTURAL_SEARCH_V3_PROTOCOL.md',
             'src/transcif/research/dynamic_v2.py','src/transcif/research/references.py',
             'src/transcif/research/eia_proxy.py','src/transcif/research/weighted_metrics.py',
             'src/transcif/research/mechanisms.py']
    protocol = {'train_years':args.train_years,'validation_year':args.validation_year,'delays':args.delays,
                'hashes':{f:sha(root/f) for f in files},'data_hashes':{p.name:sha(p) for p in args.data.glob('*.csv')},
                'selection':'leave-source-region January–June fit, July–December MAE; earliest train year',
                'confirmation_rule':'For every target and latency select minimum source-only inner MAE among fixed POINTS; no 2023 metric selection; train2022 only for2025 confirmation',
                'methods':list(METHODS),'calibration':'same 128 source held-region residual indices per method and delay',
                'status':'exploratory; no untouched claim','seed':13017}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for f in files:
        (args.output/Path(f).name).write_bytes((root/f).read_bytes())
    monthly = pd.read_csv(args.data/'monthly.csv')
    monthly = monthly.loc[(pd.to_datetime(monthly.period_start).dt.year <= args.validation_year)].copy()
    frames = {r:pd.concat([read_year(args.data/f'{r}_hourly.csv',y) for y in args.train_years],ignore_index=True) for r in REGIONS}
    origins = pd.date_range(f'{args.validation_year}-01-01',f'{args.validation_year}-12-31',tz='UTC')
    rows = []
    for target in args.regions:
        sources = {r:f for r,f in frames.items() if r != target}
        inner = {r:f.loc[f.time.dt.year == min(args.train_years)] for r,f in sources.items()}
        bundles = {}
        for delay in args.delays:
            winner, residuals, policy = source_policy(inner, monthly, delay, min(args.train_years))
            model = fit_model(sources,monthly,f'{max(args.train_years)+1}-01-01T00:00:00Z',delay)
            predictions = model.predict(origins,target,monthly)
            predictions['source_policy'] = predictions[winner].copy()
            residuals['source_policy'] = residuals[winner].copy()
            size = len(residuals[winner])
            indices = np.random.default_rng(13017).choice(size,128,replace=size<128)
            banks = {k:v[indices] for k,v in residuals.items()}
            folder = args.output/f'{target}-{delay}'
            folder.mkdir()
            np.savez_compressed(folder/'predictions.npz',**predictions,**{f'residual_{k}':v for k,v in banks.items()},origins=origins.asi8)
            (folder/'prediction_seal.json').write_text(json.dumps({'prediction_sha256':sha(folder/'predictions.npz'),
                'target_labels_read_before_seal':False,'source_regions':list(model.source_regions),'policy':policy,'winner':winner,
                'calibration_indices':indices.tolist()},indent=2))
            bundles[delay] = (predictions, banks, winner)
        days, truth = daily_profiles(read_year(args.data/f'{target}_hourly.csv',args.validation_year))
        for delay in args.delays:
            predictions, banks, winner = bundles[delay]
            for day, actual in zip(days,truth):
                i = origins.get_indexer([day])[0]
                if i < 0:
                    continue
                for name in METHODS:
                    point = predictions[name][i]
                    ensemble = np.clip(point+banks[name],0,EF.max())
                    metric = PreparedDistribution(ensemble,np.ones(128)).score(actual)
                    metric.update(region=target,delay=delay,method=name,origin=str(day),winner=winner,
                        raw_point_mae=float(np.abs(point-actual).mean()),
                        raw_shape_mae=float(np.abs(point-point.mean()-actual+actual.mean()).mean()),
                        raw_level_mae=float(abs(point.mean()-actual.mean())))
                    rows.append(metric)
            print(f'{target} delay={delay}: source winner {winner}; sealed and scored',flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(args.output/'day_metrics.csv',index=False)
    metrics = ['crps','mae','coverage90','width90','raw_point_mae','raw_shape_mae','raw_level_mae']
    regional = scores.groupby(['region','delay','method'])[metrics].mean().reset_index()
    regional.to_csv(args.output/'region_metrics.csv',index=False)
    macro = regional.groupby(['delay','method'])[metrics].mean().reset_index()
    macro.to_csv(args.output/'macro_metrics.csv',index=False)
    # Paired calendar-month bootstrap is descriptive; regions/months are dependent.
    scores['month'] = pd.to_datetime(scores.origin).dt.month
    monthly_scores = scores.groupby(['region','delay','month','method']).crps.mean().unstack('method')
    descriptions=[]
    rng=np.random.default_rng(13017)
    for delay in args.delays:
        part=monthly_scores.xs(delay,level='delay')
        for name in METHODS:
            delta=(part[name]-part.latest).to_numpy()
            boot=np.mean(rng.choice(delta,(2000,len(delta)),replace=True),axis=1)
            descriptions.append({'delay':delay,'method':name,'delta_crps_vs_latest':float(delta.mean()),
                'descriptive_month_ci_low':float(np.quantile(boot,.025)),'descriptive_month_ci_high':float(np.quantile(boot,.975))})
    pd.DataFrame(descriptions).to_csv(args.output/'paired_month_descriptive.csv',index=False)
    (args.output/'postrun_verification.json').write_text(json.dumps({'hashes_unchanged':all(sha(root/f)==h for f,h in protocol['hashes'].items()),
        'data_hashes_unchanged':all(sha(args.data/f)==h for f,h in protocol['data_hashes'].items()),
        'seals_match':all(sha(p.parent/'predictions.npz')==json.loads(p.read_text())['prediction_sha256'] for p in args.output.glob('*/prediction_seal.json'))},indent=2))
    print(macro.sort_values(['delay','crps']).to_string(index=False))

if __name__ == '__main__':
    main()
