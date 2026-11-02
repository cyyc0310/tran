#!/usr/bin/env python3
"""Strict-CN5 country-pseudotarget shrinkage of the hourly thermal-factor head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_energy_charts_nine_country_cn5_loco as nine
from transcif.research import multicountry_cn5 as model
from transcif.research.monthly_adaptation import weighted_quantile


ROOT = nine.ROOT
COUNTRIES = nine.COUNTRIES
ALPHAS = (0., .25, .5, .75, 1.)
METHODS = ('ratio_recomposition', 'macro_recomposition', 'factor_history',
           'direct_history', 'pseudo_factor_response')
PROTOCOL = ROOT/'docs/CN5_PSEUDOTARGET_FACTOR_RESPONSE_PROTOCOL_20261101.md'


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def source_fit(prepared: dict, countries: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in countries]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([nine.fuel_for(prepared[c][0], part)
                           for c, part in zip(countries, parts)])
    coarse = np.concatenate([
        model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
        for c, part in zip(countries, parts)
    ])
    if train.empty or len(train) != len(fuel) or len(train) != len(coarse):
        raise ValueError(f'misaligned source training rows: {countries}')
    return model.fit(train, fuel, coarse)


def labeled_half(prepared: dict, country: str, half: int) -> tuple[pd.DataFrame, np.ndarray]:
    raw, features = prepared[country]
    query = features.loc[features.year.eq(2023)].copy()
    query = query.loc[query.clock.dt.month.le(6) if half == 1
                      else query.clock.dt.month.gt(6)].copy()
    if query.empty:
        raise ValueError(f'empty pseudo target {country} H{half}')
    fuel = nine.fuel_for(raw, query)
    truth = fuel @ model.FACTORS / fuel.sum(axis=1)
    return query, truth


def pseudo_tasks(prepared: dict, sources: list[str]) -> dict:
    tasks = {}
    for pseudo in sources:
        fit = source_fit(prepared, [c for c in sources if c != pseudo])
        halves = {}
        for half in (1, 2):
            query, truth = labeled_half(prepared, pseudo, half)
            halves[half] = {'query': query, 'truth': truth,
                            'predictions': model.predict(fit, query)}
        tasks[pseudo] = halves
        print('strict-CN5 factor pseudo task', pseudo.upper(), 'complete', flush=True)
    return tasks


def select_alpha(tasks: dict, countries: list[str]) -> tuple[float, pd.DataFrame]:
    rows = []
    for alpha in ALPHAS:
        country_errors = []
        for country in countries:
            task = tasks[country]
            pred = (task['predictions']['ratio_recomposition'] + alpha *
                    (task['predictions']['factor_history'] -
                     task['predictions']['ratio_recomposition']))
            country_errors.append(float(np.abs(task['truth']-pred).mean()))
        rows.append({'alpha': alpha, 'country_equal_mae': float(np.mean(country_errors)),
                     'country_mae': json.dumps(country_errors)})
    scores = pd.DataFrame(rows).sort_values(['country_equal_mae','alpha']).reset_index(drop=True)
    return float(scores.iloc[0].alpha), scores


def calibration_radius(errors: np.ndarray, countries: np.ndarray) -> float:
    weights = model.weights(pd.DataFrame({'country': countries, 'region': countries}))
    return float(weighted_quantile(np.abs(errors), weights, .9))


def shape_mae(query: pd.DataFrame, truth: np.ndarray, prediction: np.ndarray) -> float:
    frame = pd.DataFrame({'truth': truth, 'prediction': prediction,
                          'day': query.clock.dt.strftime('%Y-%m-%d').to_numpy()})
    centered = frame[['truth','prediction']]-frame.groupby('day')[['truth','prediction']].transform('mean')
    return float(np.abs(centered.truth-centered.prediction).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for country in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(country)
        input_paths += [manifest_path, folder/f'{country}_2022_2024.json']
        if country in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder/'audit.json')
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT/'src/transcif/research/cross_country.py',
                  ROOT/'src/transcif/research/monthly_adaptation.py']
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out/'startup_hashes.json').write_text(json.dumps(
        {'code':code_hashes,'inputs':input_hashes},indent=2)+'\n',encoding='utf-8')

    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, features, audits[country] = nine.prepare_country(country)
        prepared[country] = (raw, features)
    (out/'input_audit.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

    seals, predictions = [], {}
    choice_rows, nested_rows = [], []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        tasks = pseudo_tasks(prepared, sources)
        h1 = {c:{'truth':tasks[c][1]['truth'],
                 'predictions':tasks[c][1]['predictions']} for c in sources}
        alpha, table = select_alpha(h1, sources)
        table.to_csv(out/f'{target.upper()}_alpha_scores.csv',index=False)
        for held in sources:
            train_pseudo = [c for c in sources if c != held]
            fold_alpha, _ = select_alpha(h1, train_pseudo)
            task = h1[held]
            prediction = task['predictions']['ratio_recomposition'] + fold_alpha * (
                task['predictions']['factor_history']-task['predictions']['ratio_recomposition'])
            nested_rows.append({'target_country':target.upper(),'heldout_pseudo_country':held.upper(),
                                'selected_alpha':fold_alpha,
                                'selected_mae':float(np.abs(task['truth']-prediction).mean()),
                                'ratio_mae':float(np.abs(task['truth']-
                                    task['predictions']['ratio_recomposition']).mean())})
        choice_rows.append({'target_country':target.upper(),'selected_alpha':alpha,
                            'inner_h1_mae':float(table.iloc[0].country_equal_mae),
                            'nested_cv_mae':float(np.mean([
                                r['selected_mae'] for r in nested_rows
                                if r['target_country']==target.upper()]))})

        # The independent H2 cross-fitted pseudo-country predictions calibrate all methods.
        h2 = [tasks[c][2] for c in sources]
        calibration = {}
        methods = ('ratio_recomposition','macro_recomposition','factor_history','direct_history')
        pooled_truth = np.concatenate([item['truth'] for item in h2])
        pooled_countries = np.concatenate([
            np.repeat(c,len(tasks[c][2]['truth'])) for c in sources])
        for method in methods:
            point = np.concatenate([item['predictions'][method] for item in h2])
            calibration[method] = calibration_radius(pooled_truth-point,pooled_countries)
        shrink_point = np.concatenate([
            item['predictions']['ratio_recomposition']+alpha*(
                item['predictions']['factor_history']-item['predictions']['ratio_recomposition'])
            for item in h2])
        calibration['pseudo_factor_response'] = calibration_radius(
            pooled_truth-shrink_point,pooled_countries)

        final = source_fit(prepared,sources)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        point = model.predict(final,query)
        point['pseudo_factor_response'] = (point['ratio_recomposition']+alpha*(
            point['factor_history']-point['ratio_recomposition']))
        payload = {'time':raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                   **point}
        for method in METHODS:
            payload[f'{method}_lo'] = point[method]-calibration[method]
            payload[f'{method}_hi'] = point[method]+calibration[method]
        path=out/f'{target.upper()}_heldout_predictions.csv.gz'
        pd.DataFrame(payload).to_csv(path,index=False,compression='gzip')
        radius_path=out/f'{target.upper()}_source_radii.json'
        radius_path.write_text(json.dumps(calibration,indent=2)+'\n',encoding='utf-8')
        seals.append({'target':target.upper(),'path':path.name,'sha256':sha(path),
                      'radius_path':radius_path.name,'radius_sha256':sha(radius_path),
                      'hours':len(query),'source_countries':[c.upper() for c in sources]})
        predictions[target]=(query,path,calibration)
        print('sealed strict-CN5 factor target',target.upper(),len(query),'hours alpha',alpha,flush=True)

    (out/'prediction_seal.json').write_text(json.dumps(seals,indent=2)+'\n',encoding='utf-8')
    (out/'alpha_selection.json').write_text(json.dumps(choice_rows,indent=2)+'\n',encoding='utf-8')
    pd.DataFrame(nested_rows).to_csv(out/'nested_pseudotarget_selection.csv',index=False)
    for group, hashes in (('inputs',input_hashes),('code',code_hashes)):
        if any(sha(Path(path))!=digest for path,digest in hashes.items()):
            raise RuntimeError(f'{group} changed during experiment')
    if any(sha(out/row['path'])!=row['sha256'] or
           sha(out/row['radius_path'])!=row['radius_sha256'] for row in seals):
        raise RuntimeError('prediction/radius seal verification failed')

    rows=[]
    for target,(query,path,radii) in predictions.items():
        raw=prepared[target][0]
        fuel=nine.fuel_for(raw,query)
        truth=fuel@model.FACTORS/fuel.sum(axis=1)
        stored=pd.read_csv(path)
        for method in METHODS:
            point=stored[method].to_numpy(float)
            lo=stored[f'{method}_lo'].to_numpy(float)
            hi=stored[f'{method}_hi'].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                take=query.clock.dt.month.eq(month).to_numpy()
                error=truth[take]-point[take]
                rows.append({'target_country':target.upper(),'method':method,'month':int(month),
                    'hours':int(take.sum()),'mae':float(np.abs(error).mean()),
                    'shape_mae':shape_mae(query.loc[take],truth[take],point[take]),
                    'coverage90':float(((truth[take]>=lo[take])&(truth[take]<=hi[take])).mean()),
                    'width90':float(np.mean(hi[take]-lo[take])),
                    'interval_score90':float(np.mean(hi[take]-lo[take]+
                        20*np.maximum(lo[take]-truth[take],0)+
                        20*np.maximum(truth[take]-hi[take],0)))})
    monthly=pd.DataFrame(rows)
    monthly.to_csv(out/'monthly_metrics.csv',index=False)
    fields=['mae','shape_mae','coverage90','width90','interval_score90']
    country=monthly.groupby(['target_country','method'])[fields].mean().reset_index()
    country.to_csv(out/'country_metrics.csv',index=False)
    country.groupby('method')[fields].mean().reset_index().to_csv(out/'macro_metrics.csv',index=False)
    (out/'status.json').write_text(json.dumps({
        'status':'DEVELOPMENT_STRICT_CN5_PSEUDOTARGET_FACTOR_RESPONSE',
        'countries':[c.upper() for c in COUNTRIES],'target_year':2024,
        'source_train_year':2022,'pseudo_selection':'2023H1 leave-country-out',
        'source_calibration':'2023H2 cross-fitted leave-country-out',
        'target_information':'strict five-category t-2 through t-7 monthly shares and calendar',
        'source_supervision':'fine-fuel hourly generation proxy; target receives no fine-fuel inputs',
        'target_labels_previously_viewed':True,'target_labels_scored_after_prediction_seal':True,
        'labels':'fixed-factor generation-side proxy, not measured CIF',
    },ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(country.groupby('method')[fields].mean().to_string())


if __name__=='__main__':
    main()
