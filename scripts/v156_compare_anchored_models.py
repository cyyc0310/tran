"""Post-run source-isolation checks and paired comparisons of hourly baselines."""
from pathlib import Path
import hashlib,json,argparse
import numpy as np
import pandas as pd
from v139_physical_reserve_calibration import sha

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--source-mode',choices=('sampled','full'),required=True)
MODE=parser.parse_args().source_mode
NEW=ROOT/f'results/v156-anchored-hourly-{MODE}-r1'
OLD=ROOT/'results/v157-matched-source-hours-r1'
OUT=ROOT/f'results/v156-anchored-{MODE}-audit-r1'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    if not (NEW/'summary.json').exists():raise RuntimeError('V156 not complete; wait for its existing process')
    summary=json.loads((NEW/'summary.json').read_text())
    if summary['assertion_failures']:raise ValueError('upstream audit failed')
    for name,h in summary['input_hashes'].items():
        if sha(ROOT/name)!=h:raise ValueError('input hash mismatch '+name)
    for name,h in summary['output_hashes'].items():
        if sha(NEW/name)!=h:raise ValueError('output hash mismatch '+name)
    index=pd.read_csv(ROOT/'results/v144-canonical-profile-cache-r1/profile_index.csv')
    dates=pd.to_datetime(index.month);released=dates+pd.offsets.MonthBegin(1)+pd.Timedelta(days=30)
    roles=pd.read_csv(NEW/'region_roles.csv');sampling=pd.read_csv(NEW/'source_sampling.csv')
    inner=pd.read_csv(NEW/'inner_source_folds.csv');verified_samples=0
    for row in sampling.itertuples(index=False):
        roster=roles.loc[roles.target_region.eq(row.target_region)&roles.seed.eq(row.seed)]
        if row.region not in set(roster.loc[roster.role.eq('fit'),'region']) or row.region==row.target_region:
            raise ValueError('held-out region used in learner')
        ids=index.index[index.region.eq(row.region)].to_numpy()
        candidates=[]
        for q in ids:
            available=(released.iloc[ids]<=dates.iloc[q]).any()
            if dates.iloc[q].year==2024 and (row.budget=='same_month' or available):candidates.append(q)
        pool=np.concatenate([np.arange(int(index.offset.iloc[q]),int(index.stop.iloc[q])) for q in candidates])
        seed=int(hashlib.sha256(f'20261028/{row.region}/{row.budget}'.encode()).hexdigest()[:16],16)
        expected=pool if MODE=='full' else np.sort(np.random.default_rng(seed).choice(pool,min(4096,len(pool)),replace=False))
        actual=np.asarray(json.loads(row.sampled_raw_hour_indices),int)
        if len(actual)!=row.sampled_hours or not np.array_equal(expected,actual):raise ValueError('source sampling discrepancy')
        verified_samples+=len(actual)
    if len(sampling)!=360:raise ValueError('source sampling table denominator')
    for (target,seed),g in inner.groupby(['target_region','seed']):
        roster=roles.loc[roles.target_region.eq(target)&roles.seed.eq(seed)&roles.role.eq('fit')]
        if len(g)!=30 or set(g.region)!=set(roster.region) or set(g.inner_fold)!=set(range(5)) or not g.groupby('inner_fold').size().eq(6).all():
            raise ValueError('whole-region five-fold role mismatch')
    new_rows=pd.read_csv(NEW/'target_month_metrics.csv');old_rows=pd.read_csv(OLD/'target_month_metrics.csv')
    hour=old_rows.loc[old_rows.method.eq('sampled' if MODE=='sampled' else 'matched_full')].copy()
    if not new_rows.loc[new_rows.method.eq('conditional_quantile'),'crps'].isna().all():raise ValueError('CQR invented CRPS')
    if new_rows.loc[new_rows.method.ne('conditional_quantile'),'crps'].isna().any():raise ValueError('missing probability-model CRPS')
    macro=pd.read_csv(NEW/'macro_metrics.csv');old_macro=pd.read_csv(OLD/'macro_metrics.csv')
    old_macro=old_macro.loc[old_macro.method.eq('sampled' if MODE=='sampled' else 'matched_full')]
    comparison=[];paired=[]
    for row in macro.itertuples(index=False):
        baseline=old_macro.loc[old_macro.seed.eq(row.seed)&old_macro.budget.eq(row.budget)&old_macro.calibration.eq(row.calibration)]
        if len(baseline)!=1:raise ValueError('hour baseline identity')
        h=baseline.iloc[0]
        comparison.append({'seed':row.seed,'budget':row.budget,'calibration':row.calibration,'strong_model':row.method,
            'hour_crps':h.crps,'model_crps':row.crps,
            'hour_relative_crps_percent':100*(h.crps/row.crps-1) if np.isfinite(row.crps) else np.nan,
            'hour_coverage':h.coverage,'hour_worst_coverage':h.worst_coverage,'model_coverage':row.coverage,'model_worst_coverage':row.worst_coverage,
            'hour_width':h.width,'model_width':row.width,'hour_winkler':h.winkler,'model_winkler':row.winkler,
            'hour_passes_coverage':h.coverage>=.9 and h.worst_coverage>=.85,
            'hour_narrower':h.width<=row.width,'hour_better_winkler':h.winkler<row.winkler})
        subset=new_rows.loc[new_rows.seed.eq(row.seed)&new_rows.budget.eq(row.budget)&new_rows.method.eq(row.method)&new_rows.calibration.eq(row.calibration)]
        reference=hour.loc[hour.seed.eq(row.seed)&hour.budget.eq(row.budget)&hour.calibration.eq(row.calibration)]
        match=reference.merge(subset,on=['region','month'],suffixes=('_hour','_model'),validate='one_to_one')
        if len(match)!=36 or not match.hours_hour.eq(match.hours_model).all():raise ValueError('paired comparison denominator')
        if row.calibration=='pooled' and row.method!='conditional_quantile':
            month_names=sorted(match.month.unique());rng=np.random.default_rng(20261028);deltas=[]
            def score(frame):
                return float(np.mean([np.average(g.crps_hour-g.crps_model,weights=g.hours_hour) for _,g in frame.groupby('region')]))
            for _ in range(1000):
                sample=pd.concat([match.loc[match.month.eq(m)] for m in rng.choice(month_names,len(month_names),replace=True)],ignore_index=True)
                deltas.append(score(sample))
            paired.append({'seed':row.seed,'budget':row.budget,'strong_model':row.method,'hour_minus_model_crps':score(match),
                'descriptive_95_lower':float(np.quantile(deltas,.025)),'descriptive_95_upper':float(np.quantile(deltas,.975)),
                'regions':3,'calendar_months':12})
    if len(comparison)!=24 or len(paired)!=8:raise ValueError('comparison denominator')
    OUT.mkdir(parents=True)
    pd.DataFrame(comparison).to_csv(OUT/'hour_vs_strong_models.csv',index=False)
    pd.DataFrame(paired).to_csv(OUT/'paired_month_crps.csv',index=False)
    audit={'assertion_failures':0,'source_sampling_rows_verified':len(sampling),'source_hour_uses_verified':verified_samples,
        'whole_region_oof_role_rows':len(inner),'comparison_cells':len(comparison),'paired_bootstrap_cells':len(paired),
        'source_mode':MODE,'script_sha256':sha(Path(__file__)),'source_summary_sha256':sha(NEW/'summary.json'),
        'scope':'Opened-target descriptive comparisons and data-contract verification; no independent-region or novelty confirmation.'}
    (OUT/'summary.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(pd.DataFrame(comparison).loc[lambda x:x.calibration.eq('pooled')].to_string(index=False))

if __name__=='__main__':main()
