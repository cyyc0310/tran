"""Fit ordinary anchored references using exactly the sealed R5 source labels."""
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
from run_resource_experiment import validate_observed_source_rows
from transcif.research import hourly_anchor_baselines as anchor
from transcif.research import hourly_transfer_baselines as direct
from transcif.research import multicountry as mc
from transcif.research.capacity_transfer import load_capacity


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    parent = ROOT / 'results/hourly-direct-baselines-r5'
    completion = json.loads((parent / 'completion.json').read_text())
    if not completion['complete'] or not completion['strict_paired_source_label_budget']:
        raise ValueError('completed paired-label parent required')
    inputs = json.loads((parent / 'input_code_seal.json').read_text())
    for name, expected in json.loads((parent / 'model_seal.json').read_text()).items():
        inputs[str(parent / name)] = expected
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise ValueError('R5 source/code/model seal changed')
    runtime = dict(python=platform.python_version(), **{
        name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    if runtime != json.loads((parent / 'runtime.json').read_text()):
        raise ValueError('runtime must match the frozen absolute-response reference')
    training = ROOT / 'results/source-observed-weather-2022-r1'
    climate_path = ROOT / 'results/resource-saturated-reference-r1/climatology.csv'
    capacity_path = ROOT / 'results/public-capacity-input-r1/capacity.json'
    paths = [Path(__file__), ROOT / 'src/transcif/research/hourly_anchor_baselines.py',
             ROOT / 'tests/test_hourly_anchor_baselines.py', ROOT / 'docs/HOURLY_ANCHOR_BASELINES_PROTOCOL.md',
             parent / 'input_code_seal.json', parent / 'model_seal.json', parent / 'completion.json',
             parent / 'runtime.json', training / 'cohort.csv', climate_path, capacity_path]
    for path in paths:
        name, current = str(path.resolve()), digest(path)
        if name in inputs and inputs[name] != current:
            raise ValueError('inherited input changed before source training')
        inputs[name] = current
    raw, _ = mc.load_countries(ROOT)
    frame = pd.read_csv(training / 'cohort.csv', parse_dates=['clock', 'valid_at'])
    validate_observed_source_rows(frame, raw)
    capacity = load_capacity(capacity_path)
    for region, part in frame.groupby('region'):
        stock = capacity['regions'][region]
        frame.loc[part.index, 'capacity_factor'] = 400. + 600. * stock['coal_mw'] / (stock['coal_mw'] + stock['gas_mw'])
    fuel = canonical_fuel(frame, raw)
    response = 1000. * fuel[:, :4].sum(axis=1) / fuel.sum(axis=1)
    climate = pd.read_csv(climate_path)
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    (output / 'models').mkdir()
    (output / 'input_code_seal.json').write_text(json.dumps(inputs, indent=2) + '\n')
    (output / 'runtime.json').write_text(json.dumps(runtime, indent=2) + '\n')
    sources = [('DE',), ('GB',), ('US',), ('DE', 'GB'), ('DE', 'US'), ('GB', 'US')]
    (output / 'execution_plan.json').write_text(json.dumps(dict(
        source_sets=sources, source_candidate_hours=len(frame), response_kinds=anchor.KINDS,
        epsilon=anchor.EPSILON, options=anchor.OPTIONS, budgets=['monthly', 'capacity'],
        weather=[False, True], target_labels_used_for_training_or_selection=False, source_calibration=False,
        point_definition='inverse encoded squared-loss location; CQR uses fitted q50',
        source_weather='observed ERA5; not source issued GFS', new_country_opened=False), indent=2) + '\n')
    seals = {}
    for source in sources:
        key = '+'.join(source); subset = frame.loc[frame.country.isin(source)].copy()
        y = response[subset.index]
        with np.load(parent / 'models' / f'{key}_monthly_thermal_no_weather.oof.npz', allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['row'], subset.index.to_numpy())
            np.testing.assert_array_equal(archive['response'], y)
            np.testing.assert_array_equal(archive['weights'], mc.weights(subset))
        for budget in ('monthly', 'capacity'):
            for weather in (False, True):
                x = direct.design(subset, climate, weather=weather, capacity=budget == 'capacity')
                weather_name = 'weather' if weather else 'no_weather'
                for kind in anchor.KINDS:
                    name = f'{key}_{budget}_{kind}_{weather_name}'
                    fitted, oof = anchor.fit(subset, y, x, kind=kind, allowed_countries=source,
                        progress=lambda step: print(name, step, flush=True))
                    fitted['audit'].update(budget=budget, weather=weather,
                        parent_response_labels='exact R5 source row/permille/weights verified')
                    model_path = output / 'models' / (name + '.joblib')
                    joblib.dump(fitted, model_path, compress=3)
                    oof_path = model_path.with_suffix('.oof.npz')
                    np.savez_compressed(oof_path, row=subset.index.to_numpy(), response_permille=y,
                        response_encoded=anchor.encode(y, subset.thermal.to_numpy(), kind),
                        anchor=subset.thermal.to_numpy(), oof_encoded=oof['encoded'],
                        oof_permille=oof['permille'], weights=mc.weights(subset))
                    restored = joblib.load(model_path)
                    probe = np.unique(np.r_[np.arange(min(24, len(subset))), np.arange(max(0, len(subset)-24), len(subset))])
                    public = subset.thermal.to_numpy()[probe]; factor = np.full(len(probe), .6)
                    original = anchor.forecast(fitted, x[probe], public, factor)
                    for field, values in anchor.forecast(restored, x[probe], public, factor).items():
                        np.testing.assert_array_equal(values, original[field])
                    for path in (model_path, oof_path):
                        seals[str(path.relative_to(output))] = digest(path)
    seals.update({name: digest(output / name) for name in ['runtime.json', 'execution_plan.json']})
    (output / 'model_seal.json').write_text(json.dumps(seals, indent=2) + '\n')
    for name, expected in inputs.items():
        if digest(Path(name)) != expected:
            raise RuntimeError('source/code changed during training')
    for name, expected in seals.items():
        if digest(output / name) != expected:
            raise RuntimeError('source model or OOF changed')
    (output / 'completion.json').write_text(json.dumps(dict(complete=True, model_families=48,
        oof_files=48, source_candidate_hours=len(frame), source_candidate_days=len(frame)//24,
        source_sets=6, source_labels_unchanged=True, target_predictions=False,
        source_calibration=False, new_country_opened=False, novelty_established=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
