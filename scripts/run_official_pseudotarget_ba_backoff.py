"""Source-only leave-BA pseudo-target expert ensemble on official EIA/NESO labels."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

import run_official_intensity_transfer as official
from run_cross_country import ROOT

PROTOCOL = ROOT/'docs/OFFICIAL_PSEUDOTARGET_BA_BACKOFF_PROTOCOL_20261101.md'
US_REGIONS = official.US_REGIONS
COARSE = official.COARSE
FEATURES = official.FEATURES
TAUS = (np.inf, .1, .25, .5, 1., 2., 5.)
BASE_METHODS = ('source_mean','calendar_ridge','coarse_ridge','coarse_boosting',
                'capacity_boosting','fixed_ef_recomposition')
SCORED_METHODS = (*BASE_METHODS,'pseudo_uniform','pseudo_gated','pseudo_backoff')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def query_for(prepared: dict, region: str, year: int,
              months: tuple[int, ...] | None = None) -> pd.DataFrame:
    q=prepared[region].loc[prepared[region].year.eq(year)].copy()
    if months is not None:
        q=q.loc[q.clock.dt.month.isin(months)].copy()
    return q.reset_index(drop=True)


def aligned(region: str, frame: pd.DataFrame, labels: dict[str,pd.Series]) -> tuple[pd.DataFrame,np.ndarray]:
    y=official.aligned_labels(frame,labels[region])
    good=np.isfinite(y)
    return frame.loc[good].reset_index(drop=True),y[good]


def collect(regions: list[str], prepared: dict,
            labels: dict[str,pd.Series], year: int,
            months: tuple[int,...] | None = None) -> tuple[pd.DataFrame,np.ndarray]:
    frames=[]; responses=[]
    for region in regions:
        q=query_for(prepared,region,year,months)
        q,y=aligned(region,q,labels)
        if len(q):
            frames.append(q);responses.append(y)
    if not frames:raise ValueError(f'empty cohort: {regions=} {year=} {months=}')
    return pd.concat(frames,ignore_index=True),np.concatenate(responses)


def fit_booster(frame: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingRegressor:
    weights=official.region_weights(frame)
    model=HistGradientBoostingRegressor(
        max_iter=100,max_leaf_nodes=15,min_samples_leaf=100,
        l2_regularization=10.,random_state=412,early_stopping=False)
    model.fit(frame[list(FEATURES)],y,sample_weight=weights*len(weights))
    return model


def prototypes(prepared: dict, regions: list[str]) -> dict[str,np.ndarray]:
    result={}
    for region in regions:
        frame=query_for(prepared,region,2022)
        grouped=frame.groupby(frame.clock.dt.month)[list(COARSE)].mean()
        overall=grouped.mean().to_numpy(float)
        values=np.tile(overall,(12,1))
        for month,row in grouped.iterrows():values[int(month)-1]=row.to_numpy(float)
        result[region]=values
    return result


def gate_weights(query: pd.DataFrame, experts: list[str],
                 profile: dict[str,np.ndarray], tau: float) -> np.ndarray:
    if not experts:raise ValueError('no source experts')
    if np.isinf(tau):return np.full((len(query),len(experts)),1./len(experts))
    x=query[list(COARSE)].to_numpy(float)
    months=query.clock.dt.month.to_numpy(int)-1
    proto=np.stack([profile[r][months] for r in experts],axis=1)
    # Scale each coarse category using the cross-expert spread for that month.
    spread=np.stack([np.std(np.stack([profile[r][m] for r in experts]),axis=0)
                     for m in months])
    spread=np.where(spread>1e-5,spread,1.)
    distance=np.sqrt(np.mean(((proto-x[:,None,:])/spread[:,None,:])**2,axis=2))
    logits=-distance/tau
    logits-=logits.max(axis=1,keepdims=True)
    weights=np.exp(logits)
    weights/=weights.sum(axis=1,keepdims=True)
    return weights


def ensemble_prediction(query: pd.DataFrame, experts: list[str],
                        models: dict[str,HistGradientBoostingRegressor],
                        profile: dict[str,np.ndarray], tau: float) -> tuple[np.ndarray,np.ndarray]:
    predictions=np.column_stack([models[r].predict(query[list(FEATURES)]) for r in experts])
    weights=gate_weights(query,experts,profile,tau)
    return np.sum(predictions*weights,axis=1),weights


def shape_mae(clock: pd.Series,y: np.ndarray,p: np.ndarray) -> float:
    temp=pd.DataFrame({'day':pd.to_datetime(clock).dt.strftime('%Y-%m-%d').to_numpy(),
                       'y':y,'p':p})
    centered=temp[['y','p']]-temp.groupby('day')[['y','p']].transform('mean')
    return float(np.abs(centered.y-centered.p).mean())


def calibrate_residuals(parts: dict[str,list[np.ndarray]]) -> dict:
    output={}
    for method,values in parts.items():
        bias=float(np.mean([v.mean() for v in values]))
        sigma=float(np.sqrt(np.mean([np.mean((v-bias)**2) for v in values])))
        output[method]={'bias':bias,'sigma':max(sigma,1.),
                        'regions':len(values),'calibration_hours':int(sum(map(len,values)))}
    return output


def main() -> None:
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output
    out.mkdir(parents=True,exist_ok=False)
    target_dir=official.TARGET_LABEL_ROOT
    prior_target_files=list(target_dir.glob('*')) if target_dir.exists() else []
    if prior_target_files:
        raise RuntimeError('held-out target labels must be absent before pseudo-target forecasts')
    (out/'preseal_target_label_audit.json').write_text(json.dumps({
        'target_label_files_present_before_prediction':0,
        'target_label_directory':str(target_dir.relative_to(ROOT)),
        'pseudo_target_validation':'US 2023H1 only',
        'target':'GB 2024; target label extraction is post-seal',
    },indent=2))

    source_assets=[official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz'
                   for r in US_REGIONS]
    source_assets += [official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                      official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    if not all(p.is_file() for p in source_assets):
        subprocess.run([sys.executable,str(official.PREPARE_LABELS)],check=True)
    table=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    for name,digest in table['input_hashes'].items():
        if sha(ROOT/name)!=digest:raise ValueError(f'annual capacity input changed: {name}')
    code_files=[Path(__file__),Path(official.__file__),official.PREPARE_LABELS,
                ROOT/'scripts/run_cross_country.py',ROOT/'src/transcif/research/cross_country.py',
                ROOT/'src/transcif/research/capacity_transfer.py',PROTOCOL,
                official.PROTOCOL,official.CAPACITY_PROTOCOL,
                ROOT/'scripts/prepare_official_annual_capacity_budget.py']
    data_files=[official.EIA_ROOT/'acquisition_manifest.json',
                official.EIA_ROOT/'coverage_audit_2022_2024.json',
                official.CAPACITY_TABLE_PATH,
                official.SOURCE_LABEL_ROOT/'GB_consumption_2022_2023_halfhour.csv.gz',
                official.EIA_ROOT/'feature_inputs/GB_generation_2022_2024.csv.gz']
    data_files += [official.SOURCE_LABEL_ROOT/f'{r}_consumption_2022_2023.csv.gz'
                   for r in US_REGIONS]
    data_files += [ROOT/f'results/research/data/{r}_hourly.csv' for r in US_REGIONS]
    data_files += [ROOT/name for name in table['input_hashes']]

    prepared_raw,prepared,labels,audit=official.prepare(ROOT,table)
    hashes={'code':{str(p):sha(p) for p in dict.fromkeys(code_files)},
            'data':{str(p):sha(p) for p in dict.fromkeys(data_files)}}
    (out/'startup_hashes.json').write_text(json.dumps(hashes,indent=2))
    (out/'input_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False))
    source=list(US_REGIONS)
    profile=prototypes(prepared,source)

    # Nested leave-two-area-out experts: pseudo target r is absent from each
    # inner model; expert j is also absent, matching the final leave-one-area-out bank.
    pseudo_rows=[];pseudo_records=[]
    alpha_values=(0.,.25,.5,.75,1.)
    tau_values=TAUS
    candidate_grid=[(0.,np.inf)]+[(a,t) for a in alpha_values[1:] for t in tau_values]
    candidate_keys=[f'alpha={a:.2f}|tau={"inf" if np.isinf(t) else str(t)}'
                    for a,t in candidate_grid]
    candidate_values=dict(zip(candidate_keys,candidate_grid))
    for pseudo in source:
        valid=prepared[pseudo].loc[
            prepared[pseudo].year.eq(2023)&prepared[pseudo].clock.dt.month.le(6)].copy()
        valid,truth=aligned(pseudo,valid,labels)
        candidate_experts=[r for r in source if r!=pseudo]
        pseudo_train,pseudo_y=collect(candidate_experts,prepared,labels,2022)
        pooled_pseudo=fit_booster(pseudo_train,pseudo_y)
        pooled_prediction=pooled_pseudo.predict(valid[list(FEATURES)])
        inner_models={}
        for excluded in candidate_experts:
            train_regions=[r for r in candidate_experts if r!=excluded]
            train,y_train=collect(train_regions,prepared,labels,2022)
            inner_models[excluded]=fit_booster(train,y_train)
        ensemble_cache={}
        for tau in tau_values:
            tau_key='inf' if np.isinf(tau) else str(tau)
            ensemble_cache[tau_key]=ensemble_prediction(
                valid,candidate_experts,inner_models,profile,tau)
        for (alpha,tau),candidate in zip(candidate_grid,candidate_keys):
            tau_key='inf' if np.isinf(tau) else str(tau)
            expert_prediction,weights=ensemble_cache[tau_key]
            point=(pooled_prediction if alpha==0. else
                   (1.-alpha)*pooled_prediction+alpha*expert_prediction)
            pseudo_rows.append({'pseudo_target':pseudo,'candidate':candidate,'alpha':alpha,
                'tau':'inf' if np.isinf(tau) else tau,
                'mae':float(np.abs(truth-point).mean()),
                'shape_mae':shape_mae(valid.clock,truth,point),'hours':len(valid),
                'pooled_mae':float(np.abs(truth-pooled_prediction).mean()),
                'expert_mae':float(np.abs(truth-expert_prediction).mean()),
                'mean_max_expert_weight':float(weights.max(axis=1).mean())})
        pseudo_records.append({'pseudo_target':pseudo,'validation_hours':len(valid),
                               'experts':candidate_experts})
        print('pseudo-target',pseudo,'validated',len(valid),flush=True)
    pseudo_validation=pd.DataFrame(pseudo_rows)
    selection=(pseudo_validation.groupby('candidate',sort=False).mae.mean()
               .sort_values(kind='mergesort'))
    selected_key=str(selection.index[0])
    selected_alpha,selected_tau=candidate_values[selected_key]
    (out/'pseudo_target_validation.csv').write_text(pseudo_validation.to_csv(index=False))
    (out/'pseudo_target_selection.json').write_text(json.dumps({
        'selected_candidate':selected_key,'selected_alpha':selected_alpha,
        'selected_tau':'inf' if np.isinf(selected_tau) else selected_tau,
        'equal_ba_macro_mae':{str(k):float(v) for k,v in selection.items()},
        'tie_rule':'stable candidates begin with alpha=0, then lower alpha and uniform tau first',
        'folds':pseudo_records,
    },indent=2))

    # Fit pooled references and the final seven-expert bank on source 2022 only.
    train,y_train=collect(source,prepared,labels,2022)
    pooled=official.train_official(train,y_train)
    fuel_parts=[]
    for region in source:
        q=prepared[region].loc[prepared[region].year.eq(2022)].copy()
        q,y=aligned(region,q,labels)
        fuel_parts.append(prepared_raw[region].iloc[q.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
    proxy=official.fit_models(train.reset_index(drop=True),np.concatenate(fuel_parts))
    experts={}
    for excluded in source:
        train_regions=[r for r in source if r!=excluded]
        q,y=collect(train_regions,prepared,labels,2022)
        experts[excluded]=fit_booster(q,y)

    # Independent source calibration uses the final expert bank and 2023H2 only.
    residual_parts={m:[] for m in SCORED_METHODS}
    for region in source:
        cal=query_for(prepared,region,2023,tuple(range(7,13)))
        cal,truth=aligned(region,cal,labels)
        pred=official.point_predictions(pooled,proxy,cal)
        p_uniform,_=ensemble_prediction(cal,source,experts,profile,np.inf)
        p_gated,_=ensemble_prediction(cal,source,experts,profile,selected_tau)
        pred.update({'pseudo_uniform':p_uniform,'pseudo_gated':p_gated,
                     'pseudo_backoff':(1.-selected_alpha)*pred['coarse_boosting']+
                                      selected_alpha*p_gated})
        for method in SCORED_METHODS:residual_parts[method].append(truth-pred[method])
    calibration=calibrate_residuals(residual_parts)
    (out/'source_calibration.json').write_text(json.dumps(calibration,indent=2))

    # Freeze every target prediction before the target response extraction.
    target='GB';query=query_for(prepared,target,2024)
    predictions=official.point_predictions(pooled,proxy,query)
    p_uniform,uniform_w=ensemble_prediction(query,source,experts,profile,np.inf)
    p_gated,gated_w=ensemble_prediction(query,source,experts,profile,selected_tau)
    predictions.update({'pseudo_uniform':p_uniform,'pseudo_gated':p_gated,
                        'pseudo_backoff':(1.-selected_alpha)*predictions['coarse_boosting']+
                                         selected_alpha*p_gated})
    prediction_path=out/'US_to_GB_pseudotarget_predictions.csv.gz'
    pd.DataFrame({'time_utc':pd.to_datetime(query.time,utc=True).astype(str).to_numpy(),
        'observed_month':query.observed_month.to_numpy(),
        'gated_max_expert_weight':gated_w.max(axis=1),
        **predictions}).to_csv(prediction_path,index=False,compression='gzip')
    seal={'source':'US','target':'GB','path':prediction_path.name,
          'sha256':sha(prediction_path),'hours_predicted':len(query),
          'selected_alpha':selected_alpha,
          'selected_tau':'inf' if np.isinf(selected_tau) else selected_tau}
    (out/'prediction_seal.json').write_text(json.dumps([seal],indent=2))
    if any(sha(Path(p))!=v for p,v in hashes['code'].items()):
        raise RuntimeError('code/protocol changed during source fitting or prediction')
    if any(sha(Path(p))!=v for p,v in hashes['data'].items()):
        raise RuntimeError('source inputs changed during source fitting or prediction')

    subprocess.run([sys.executable,str(official.PREPARE_LABELS),'--target-labels'],check=True)
    target_file=official.TARGET_LABEL_ROOT/'GB_consumption_2024_halfhour.csv.gz'
    (out/'target_scoring_hashes.json').write_text(json.dumps({str(target_file.relative_to(ROOT)):sha(target_file)},indent=2))
    target_labels=official.source_labels('GB',years=(2024,))
    truth=official.aligned_labels(query,target_labels)
    scores=pd.read_csv(prediction_path)
    monthly_rows=[]
    for month in sorted(query.clock.dt.month.unique()):
        mask=query.clock.dt.month.eq(month).to_numpy()
        for method in SCORED_METHODS:
            metric=official.score(truth[mask],scores.loc[mask,method].to_numpy(float),
                calibration[method]['bias'],calibration[method]['sigma'],
                query.loc[mask,'clock'].reset_index(drop=True))
            monthly_rows.append({'source':'US','target':'GB','method':method,
                                 'month':int(month),**metric})
    monthly=pd.DataFrame(monthly_rows);monthly.to_csv(out/'monthly_metrics.csv',index=False)
    perf=['mae','shape_mae','crps_gaussian','coverage90','width90','interval_score90']
    monthly.groupby(['source','target','method'])[perf].mean().reset_index().to_csv(
        out/'country_metrics.csv',index=False)
    (out/'status.json').write_text(json.dumps({
        'status':'DEVELOPMENT_US_TO_GB_OFFICIAL_CONSUMPTION_PSEUDOTARGET_EXPERTS',
        'target_label_used_before_prediction_seal':False,
        'target_gb_2024_previously_viewed_in_prior_experiment':True,
        'source_training_year':2022,'pseudo_target_selection_period':'US BA 2023H1',
        'selected_alpha':selected_alpha,
        'selected_tau':'inf' if np.isinf(selected_tau) else selected_tau,
        'source_calibration_period':'US BA 2023H2','target_score_year':2024,
        'target_inputs':'calendar and exact t-2-or-older coarse monthly shares',
        'capacity_inputs_used':False,'actual_meta_learning':False,
    },indent=2))
    print(pd.read_csv(out/'country_metrics.csv').to_string(index=False))


if __name__=='__main__':main()
