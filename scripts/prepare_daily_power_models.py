"""Prepare ordinary daily power references using only sealed source-2022 labels."""
from __future__ import annotations

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd

from collect_weather_transfer_pilot import ROOT, digest
from run_hourly_transfer_baselines import canonical_fuel
from run_resource_amplitudes import past_power
from run_resource_experiment import validate_observed_source_rows
from transcif.research import daily_power_baselines as power
from transcif.research import multicountry as mc

SOURCES = [('DE',), ('GB',), ('US',), ('DE', 'GB'), ('DE', 'US'), ('GB', 'US')]


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = ROOT / 'results/hourly-direct-baselines-r5'
    completed = json.loads((parent/'completion.json').read_text())
    if not completed['complete'] or not completed['strict_paired_source_label_budget']:
        raise ValueError('strict paired parent required')
    inputs = json.loads((parent/'input_code_seal.json').read_text())
    for name, expected in json.loads((parent/'model_seal.json').read_text()).items():
        inputs[str(parent/name)] = expected
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('inherited R5 source/code/model changed')
    runtime = dict(python=platform.python_version(), **{
        name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    if runtime != json.loads((parent/'runtime.json').read_text()):
        raise ValueError('runtime differs from R5')
    training = ROOT/'results/source-observed-weather-2022-r1/cohort.csv'
    climate_path = ROOT/'results/resource-saturated-reference-r1/climatology.csv'
    for path in [Path(__file__), Path(power.__file__), training, climate_path,
                 ROOT/'tests/test_daily_power_baselines.py', ROOT/'docs/DAILY_POWER_SOURCE_PROTOCOL.md',
                 parent/'input_code_seal.json', parent/'model_seal.json', parent/'completion.json',
                 parent/'runtime.json']:
        name, current = str(path.resolve()), digest(path)
        if name in inputs and inputs[name] != current:
            raise ValueError('inherited input changed')
        inputs[name] = current
    raw, _ = mc.load_countries(ROOT)
    frame = pd.read_csv(training, parse_dates=['clock', 'valid_at'])
    validate_observed_source_rows(frame, raw)
    if len(frame) != 35448:
        raise ValueError('fixed paired 35448-hour source cohort required')
    fuel = canonical_fuel(frame, raw)
    climate = pd.read_csv(climate_path)
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'models').mkdir()
    (out/'input_code_seal.json').write_text(json.dumps(inputs, indent=2)+'\n')
    (out/'runtime.json').write_text(json.dumps(runtime, indent=2)+'\n')
    (out/'execution_plan.json').write_text(json.dumps(dict(source_sets=SOURCES,
        source_hours=len(frame), hgb_options=power.OPTIONS, ridge_alpha=10.,
        response=['total', 'thermal'], weather=[False, True], learners=['ridge', 'hgb'],
        source_observed_ERA5=True, target_predictions=False, source_calibration=False,
        calendar_blocks=['through April', 'May-June', 'July-September'],
        novelty_established=False), indent=2)+'\n')
    seals, scores = {}, []
    for source in SOURCES:
        key = '+'.join(source); sub = frame.loc[frame.country.isin(source)].copy()
        selected_fuel = fuel[sub.index]
        with np.load(parent/'models'/f'{key}_monthly_thermal_no_weather.oof.npz', allow_pickle=False) as old:
            np.testing.assert_array_equal(old['row'], sub.index.to_numpy())
            np.testing.assert_array_equal(old['response'],
                1000.*selected_fuel[:, :4].sum(axis=1)/selected_fuel.sum(axis=1))
        coarse = mc.coarse_values(pd.DataFrame(selected_fuel, columns=mc.FINE))
        past, past_audit = past_power(sub, raw)
        daily, responses = power.source_inputs(sub, coarse, past, climate, allowed_countries=source)
        (out/f'{key}_past_power.json').write_text(json.dumps(past_audit, indent=2)+'\n')
        daily.to_csv(out/f'{key}_daily.csv', index=False)
        month = pd.to_datetime(daily.day).dt.month.to_numpy()
        blocks = np.where(month <= 4, 0, np.where(month <= 6, 1, 2))
        for response_name, all_response in responses.items():
            active = np.isfinite(all_response)
            design = daily.loc[active].reset_index(drop=True); y = all_response[active]
            block = blocks[active]
            if set(block) != {0, 1, 2}:
                raise ValueError('all three fixed source calendar blocks required')
            for weather in (False, True):
                for learner in ('ridge', 'hgb'):
                    name = f'{key}_{response_name}_{learner}_'+('weather' if weather else 'no_weather')
                    oof = np.full(len(y), np.nan)
                    for held in range(3):
                        train, test = block != held, block == held
                        fitted = power.fit_head(design.loc[train], y[train], weather=weather,
                            learner=learner, allowed_countries=source)
                        oof[test] = power.predict_head(fitted, design.loc[test])
                    if not np.isfinite(oof).all():
                        raise ValueError('incomplete source OOF predictions')
                    fitted = power.fit_head(design, y, weather=weather, learner=learner,
                                             allowed_countries=source)
                    fitted['audit'].update(response=response_name, source_hours=len(sub),
                        active_days=len(y), excluded_zero_reference_days=int((~active).sum()))
                    model_path = out/'models'/(name+'.joblib')
                    joblib.dump(fitted, model_path, compress=3)
                    np.testing.assert_array_equal(power.predict_head(fitted, design),
                        power.predict_head(joblib.load(model_path), design))
                    oof_path = model_path.with_suffix('.oof.npz')
                    np.savez_compressed(oof_path, day_row=np.flatnonzero(active), response=y,
                        oof=oof, block=block, weights=mc.weights(design))
                    for path in (model_path, oof_path):
                        seals[str(path.relative_to(out))] = digest(path)
                    for (country, region), ix in design.groupby(['country', 'region']).indices.items():
                        scores.append(dict(source=key, response=response_name, learner=learner,
                            weather=weather, country=country, region=region, days=len(ix),
                            amplitude_mae=float(np.abs(oof[ix]-y[ix]).mean()),
                            persistence_mae=float(np.abs(1.-y[ix]).mean()),
                            log1p_mse=float(np.square(np.log1p(oof[ix])-np.log1p(y[ix])).mean())))
                    print(name, 'source OOF and full fit complete', flush=True)
        for name in [f'{key}_daily.csv', f'{key}_past_power.json']:
            seals[name] = digest(out/name)
    region_scores = pd.DataFrame(scores)
    region_scores.to_csv(out/'region_oof_metrics.csv', index=False)
    columns = ['source', 'response', 'learner', 'weather']
    country_scores = region_scores.groupby(columns+['country']).mean(numeric_only=True).reset_index()
    country_scores.groupby(columns).mean(numeric_only=True).reset_index().to_csv(out/'summary.csv', index=False)
    for name in ['runtime.json', 'execution_plan.json', 'region_oof_metrics.csv', 'summary.csv']:
        seals[name] = digest(out/name)
    (out/'model_seal.json').write_text(json.dumps(seals, indent=2)+'\n')
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('source/code changed during preparation')
    for name, expected in seals.items():
        if digest(out/name) != expected:
            raise RuntimeError('source model/OOF changed')
    (out/'completion.json').write_text(json.dumps(dict(complete=True, models=48, oof_files=48,
        source_hours=len(frame), source_labels_unchanged=True, target_predictions=False,
        source_calibration=False, new_country_opened=False, novelty_established=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
