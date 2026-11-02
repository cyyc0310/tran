#!/usr/bin/env python3
"""Oracle diagnostic: does contemporaneous net flow explain consumption-CIF errors?"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_eia923_pseudotarget_transfer as monthly_inputs
import run_official_intensity_transfer as official
from run_cross_country import ROOT
from transcif.research.cross_country import FEATURES

FLOW_ROOT = ROOT / 'data/neso-interconnector-flows-r1'
INTERCONNECTORS = ('IFA_FLOW', 'IFA2_FLOW', 'BRITNED_FLOW', 'MOYLE_FLOW',
                   'EAST_WEST_FLOW', 'NEMO_FLOW', 'NSL_FLOW', 'ELECLINK_FLOW',
                   'VIKING_FLOW', 'GREENLINK_FLOW')
X = 'same_hour_net_import_share_ORACLE'
METHODS = ('coarse_ridge', 'coarse_boosting')


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hourly_flow_features() -> dict[str, pd.Series]:
    features: dict[str, pd.Series] = {}
    coverage = json.loads((official.EIA_ROOT/'coverage_audit_2022_2024.json').read_text())
    for region in official.US_REGIONS:
        path = official.EIA_ROOT/f'{region}.xlsx'
        raw = pd.read_excel(path, sheet_name='Published Hourly Data',
                            usecols=['UTC time','Demand','Total interchange'])
        time = pd.DatetimeIndex(pd.to_datetime(raw['UTC time'], utc=True, errors='raise'))
        demand = pd.to_numeric(raw.Demand, errors='coerce').to_numpy(float)
        interchange = pd.to_numeric(raw['Total interchange'], errors='coerce').to_numpy(float)
        net_import_share = np.divide(-interchange, demand,
            out=np.full(len(demand), np.nan), where=np.isfinite(demand)&(demand != 0)&np.isfinite(interchange))
        values = pd.DataFrame({
            'x': net_import_share,
        }, index=time)
        features[region] = values.x.loc[values.index.year.isin([2022, 2023, 2024])].dropna()

    pieces = []
    for year in range(2022, 2025):
        raw = pd.read_csv(FLOW_ROOT/f'demanddata_{year}.csv')
        date = pd.to_datetime(raw.SETTLEMENT_DATE, format='mixed', dayfirst=True)
        # Settlement periods are ordinal within the local settlement day. Start
        # from unambiguous local midnight, convert once to UTC, then advance in
        # elapsed half-hours; this handles 46/50-period DST days without
        # ambiguous local 01:00 timestamps.
        utc = (date.dt.normalize().dt.tz_localize('Europe/London').dt.tz_convert('UTC')
               + pd.to_timedelta((raw.SETTLEMENT_PERIOD.astype(int)-1)*30, unit='min'))
        cols = [c for c in INTERCONNECTORS if c in raw]
        flow = raw[cols].apply(pd.to_numeric, errors='coerce').sum(axis=1, min_count=len(cols))
        demand = pd.to_numeric(raw.TSD, errors='coerce')
        piece = pd.DataFrame({'utc': utc,
                              'flow': flow, 'demand': demand})
        piece = piece.set_index('utc')
        pieces.append(piece)
    gb = pd.concat(pieces).sort_index()
    gb['flow_mwh'] = gb.flow * .5
    gb['demand_mwh'] = gb.demand * .5
    hourly = gb.groupby(gb.index.floor('h'))[['flow_mwh','demand_mwh']].sum(min_count=2)
    features['GB'] = (hourly.flow_mwh/hourly.demand_mwh.replace(0., np.nan)).replace(
        [np.inf,-np.inf],np.nan).dropna()
    return features


def add_features(prepared: dict[str, pd.DataFrame], hourly: dict[str, pd.Series]):
    result, audit = {}, {}
    for region, frame in prepared.items():
        out = frame.copy()
        times = pd.DatetimeIndex(pd.to_datetime(out.time, utc=True, errors='raise'))
        out[X] = hourly[region].reindex(times).to_numpy(float)
        result[region] = out
        target_rows = out.year.eq(2024)
        audit[region] = {
            'source_rows': {str(y): int(out.loc[out.year.eq(y), X].notna().sum()) for y in (2022,2023)},
            'target_2024_common_hours': int((target_rows & out[X].notna()).sum()),
            'target_2024_input_hours': int(target_rows.sum()),
        }
    return result, audit


def columns(use_flow: bool) -> list[str]:
    return list(FEATURES) + ([X] if use_flow else [])


def fit_model(frame: pd.DataFrame, y: np.ndarray, name: str, use_flow: bool):
    cols = columns(use_flow)
    good = np.isfinite(y) & np.isfinite(frame[cols].to_numpy(float)).all(axis=1)
    train, response = frame.loc[good], y[good]
    weights = official.region_weights(train)
    if name == 'coarse_ridge':
        model = make_pipeline(StandardScaler(), Ridge(alpha=10.))
        model.fit(train[cols], response, standardscaler__sample_weight=weights,
                  ridge__sample_weight=weights*len(weights))
    else:
        model = HistGradientBoostingRegressor(max_iter=100,max_leaf_nodes=15,
            min_samples_leaf=100,l2_regularization=10.,random_state=412,early_stopping=False)
        model.fit(train[cols], response, sample_weight=weights*len(weights))
    return model, int(good.sum())


def metrics(y: np.ndarray, point: np.ndarray, bias: float, sigma: float, clock: pd.Series):
    error = y - (point + bias)
    z = error/sigma
    crps = sigma*(z*(2*ndtr(z)-1)+2*np.exp(-z*z/2)/np.sqrt(2*np.pi)-1/np.sqrt(np.pi))
    radius = 1.6448536269514722*sigma
    dates = pd.to_datetime(clock).reset_index(drop=True)
    d = pd.DataFrame({'day':dates.dt.strftime('%Y-%m-%d'),'y':y,'p':point+bias})
    centered = d[['y','p']]-d.groupby('day')[['y','p']].transform('mean')
    return {'scored_hours':int(len(y)),'mae':float(np.abs(error).mean()),
        'shape_mae':float(np.abs(centered.y-centered.p).mean()),'crps_gaussian':float(crps.mean()),
        'coverage90':float((np.abs(error)<=radius).mean()),'width90':float(2*radius),
        'interval_score90':float(np.mean(2*radius+20*np.maximum(np.abs(error)-radius,0)))}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    capacity=official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw,base_prepared,_,input_audit=official.prepare(ROOT,capacity)
    mix=monthly_inputs.parse_monthly(tuple(range(2020,2025)),official.US_REGIONS)
    prepared_by_arm,_=monthly_inputs.replace_source_features(base_prepared,mix)
    hourly=hourly_flow_features()
    prepared,audit=add_features(prepared_by_arm['eia923_monthly'],hourly)
    source={r:official.source_labels(r,years=(2022,2023)) for r in (*official.US_REGIONS,'GB')}
    targets={r:official.source_labels(r,years=(2024,)) for r in (*official.US_REGIONS,'GB')}

    code=[Path(__file__),Path(official.__file__),Path(monthly_inputs.__file__),
          ROOT/'src/transcif/research/cross_country.py']
    inputs=[official.EIA_ROOT/f'{r}.xlsx' for r in official.US_REGIONS]
    inputs += [FLOW_ROOT/f'demanddata_{y}.csv' for y in range(2022,2025)]
    inputs += [ROOT/'data/neso-gb/df_fuel_ckan.csv']
    hashes={'code':{str(p):sha(p) for p in code},'inputs':{str(p):sha(p) for p in inputs}}
    (out/'hashes.json').write_text(json.dumps(hashes,indent=2))
    (out/'feature_support.json').write_text(json.dumps({'flow_support':audit,'input_audit':input_audit},indent=2))

    rows=[]; seals=[]
    for src_country,tgt_country in (('US','GB'),('GB','US')):
        srcs=[r for r in prepared if raw[r].country.iloc[0]==src_country]
        tgts=[r for r in prepared if raw[r].country.iloc[0]==tgt_country]
        for method in METHODS:
            predictions={}
            for arm,use_flow in (('M',False),('M_plus_X_hour_oracle',True)):
                train_frames=[]; train_y=[]
                for region in srcs:
                    f=prepared[region].loc[prepared[region].year.eq(2022)].copy()
                    y=official.aligned_labels(f,source[region])
                    good=np.isfinite(y)&np.isfinite(f[list(FEATURES)].to_numpy(float)).all(axis=1)
                    # The M comparator uses exactly the source rows observed by the oracle arm.
                    good &= np.isfinite(f[X].to_numpy(float))
                    train_frames.append(f.loc[good]); train_y.append(y[good])
                train=pd.concat(train_frames,ignore_index=True); response=np.concatenate(train_y)
                model,ntrain=fit_model(train,response,method,use_flow)
                cal_resid=[]
                for region in srcs:
                    f=prepared[region].loc[prepared[region].year.eq(2023)].copy()
                    y=official.aligned_labels(f,source[region])
                    good=np.isfinite(y)&np.isfinite(f[list(FEATURES)].to_numpy(float)).all(axis=1)&np.isfinite(f[X].to_numpy(float))
                    if good.any(): cal_resid.append(y[good]-model.predict(f.loc[good,columns(use_flow)]))
                bias=float(np.mean([v.mean() for v in cal_resid]))
                sigma=max(1.,float(np.sqrt(np.mean([np.mean((v-bias)**2) for v in cal_resid]))))
                for region in tgts:
                    f=prepared[region].loc[prepared[region].year.eq(2024)].copy().reset_index(drop=True)
                    good=np.isfinite(f[list(FEATURES)].to_numpy(float)).all(axis=1)&np.isfinite(f[X].to_numpy(float))
                    f=f.loc[good].copy().reset_index(drop=True)
                    pred=np.clip(model.predict(f[columns(use_flow)]),0,1000)
                    path=out/f'{src_country}_to_{region}_{method}_{arm}.csv.gz'
                    pd.DataFrame({'time_utc':pd.to_datetime(f.time,utc=True).astype(str),'prediction':pred,
                                  'bias':bias,'sigma':sigma}).to_csv(path,index=False,compression='gzip')
                    seals.append({'source':src_country,'target':region,'method':method,'arm':arm,
                                  'train_hours':ntrain,'target_hours':len(f),'sha256':sha(path),'file':path.name})
                    truth=official.aligned_labels(f,targets[region]); ok=np.isfinite(truth)
                    score=metrics(truth[ok],pred[ok],bias,sigma,f.loc[ok,'clock'])
                    rows.append({'source':src_country,'target':tgt_country,'region':region,
                                 'method':method,'arm':arm,**score})
                predictions[arm]=model
    result=pd.DataFrame(rows); result.to_csv(out/'region_metrics.csv',index=False)
    metrics_cols=['mae','shape_mae','crps_gaussian','coverage90','width90','interval_score90']
    summary=result.groupby(['source','target','method','arm'],as_index=False)[metrics_cols].mean()
    summary.to_csv(out/'endpoint_metrics.csv',index=False)
    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2))
    (out/'status.json').write_text(json.dumps({
        'status':'ORACLE_DIAGNOSTIC_ONLY_NOT_DEPLOYABLE', 'target_year':2024,
        'source_train_year':2022,'source_calibration_year':2023,
        'oracle_feature':'same-hour observed net interchange divided by observed demand (US) or TSD (GB)',
        'purpose':'Upper-bound information-value diagnostic only; target-hour flow is unavailable at a month-ahead cold-start origin.',
        'target_labels_previously_viewed':True,
        'fairness':'M and oracle arms share identical source training, calibration, and target scoring support.'
    },indent=2))
    print(summary.to_string(index=False))


if __name__=='__main__': main()
