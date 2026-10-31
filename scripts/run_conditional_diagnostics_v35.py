"""Zero-fit, source-only diagnostics on sealed V34 predictions."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import conditional_diagnostics_v35 as model
from scripts import run_coarse_context_v30 as common
from scripts import run_forward_context_v31 as v31

OUT = ROOT / 'results/conditional-diagnostics-v35'
PARENT = ROOT / 'docs/evidence/fleet-information-v34-verification.json'
PARENT_SHA = 'c9e6a6f4f91465407268d5dbe7a0ba56eedfaf8fecc1a5bad5c643ff4259437e'
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
RIDGES = ('Size-Ridge', 'SizeCap-Ridge', 'IntensiveCap-Ridge', 'DirectEF-Ridge')
FEATURES = dict(zip(RIDGES, ('Size', 'SizeCap', 'IntensiveCap', 'DirectEF')))
G5_NAMES = ('POS', 'nuclear', 'hydro', 'wind', 'solar')
CODE = ('scripts/conditional_diagnostics_v35.py', 'scripts/run_conditional_diagnostics_v35.py',
        'tests/test_conditional_diagnostics_v35.py', 'tests/test_run_conditional_diagnostics_v35.py',
        'docs/CONDITIONAL_DIAGNOSTICS_V35_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/conditional-diagnostics-v35-{name}-review.md' for name in ('code', 'math'))
MEMBERS = ('origins', 'keys', 'arms', 'MG_draws', 'CIF_draws', 'EF_draws', 'CIF_point', 'EF_point',
           'query_feature_Size', 'query_feature_SizeCap', 'query_feature_IntensiveCap', 'query_feature_DirectEF')
sha, read, write, now = common.sha, common.read, common.write, common.now


def verify(mapping: dict) -> None:
    for path in mapping:
        if not Path(path).is_file() or Path(path).is_symlink():
            raise ValueError('Regular non-symlink bound file required')
    common.verify(mapping)


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    manifest = read(PARENT)
    if manifest['status'] != 'VERIFICATION_COMPLETE' or manifest['independent_actual_audit_status'] != 'PASS':
        raise ValueError('Independent V34 PASS required')
    inputs = {str(PARENT): PARENT_SHA}
    for field in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for path, digest in manifest[field].items():
            if path in inputs and inputs[path] != digest:
                raise ValueError('Conflicting parent mapping')
            inputs[path] = digest
    verify(inputs)
    old = read(ROOT / 'results/fleet-information-v34/freeze.json')
    seal = read(ROOT / 'results/fleet-information-v34/prediction_seal.json')
    entries = seal['entries']
    support = str(ROOT / 'results/fleet-information-v34/label_support.csv')
    aliases = [old['monthly_path'], support, *old['head_source_paths'].values()]
    aliases += [entry[key] for entry in entries for key in ('path', 'audit_path')]
    if [entry['region'] for entry in entries] != list(SOURCES) or any(p not in inputs for p in aliases):
        raise ValueError('Exact bound seven-source aliases required')
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Final code and Python/math reviews required')
        inputs[str(path)] = sha(path)
    code = {str(ROOT / path): sha(ROOT / path) for path in CODE}
    verify(code)
    return dict(version=35, code_hashes=code, input_hashes=inputs, environment=common.environment(),
                entries=entries, head_source_paths=old['head_source_paths'], monthly_path=old['monthly_path'],
                support_path=support, sources=list(SOURCES), new_fits=0, original_source_months=35,
                hourly_values_read=False, target_values_opened=False, unseen_values_opened=False,
                oracle_G5_for_EF_diagnostic_only=True, deployment_method=False, mechanism_breakthrough=False)


def load_archive(path: str, source: str) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in MEMBERS}
    if (values['keys'].tolist() != [f'{source}/{at:%Y-%m}' for at in v31.DATES] or
            not np.array_equal(values['origins'], v31.DATES.asi8) or len(values['arms']) != 11 or
            len(set(values['arms'].tolist())) != 11 or values['MG_draws'].shape != (6, 128, 5)):
        raise ValueError('Exact V34 archive grid required')
    for kind in ('CIF', 'EF'):
        draws, point = values[kind + '_draws'], values[kind + '_point']
        if (draws.shape != (11, 6, 128) or point.shape != (11, 6) or
                not np.isfinite(draws).all() or not np.isfinite(point).all() or
                not np.allclose(draws.mean(-1), point, atol=1e-9, rtol=0)):
            raise ValueError('Original finite particle mean grid required')
    return values


def aggregate(frame: pd.DataFrame, groups: list[str], rmse: tuple[str, ...] = ()) -> pd.DataFrame:
    grouped = frame.groupby(groups)
    result = grouped.mean(numeric_only=True)
    for prefix in rmse:
        result[prefix + '_rmse'] = np.sqrt(result[prefix + '_sq_error'])
    result['scored_months'] = grouped.size()
    return result.reset_index()


def compute(frozen: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    labels, support = v31.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or not all(y['ef_defined'] for y in labels.values()) or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Exact same 35 defined-EF months and 42 support rows required')
    mg, points, oracle, max_error = [], [], [], 0.
    for entry in frozen['entries']:
        source = entry['region']
        values = load_archive(entry['path'], source)
        audit = read(entry['audit_path'])
        by_arm = {fit['arm']: fit for fit in audit['fits']}
        old = read(frozen['head_source_paths'][source])
        by_arm['Size-Ridge'] = next(f for f in old['fits'] if f['arm'] == 'Size-Ridge')
        predictions = {}
        names = audit['query_feature_names']['Size']
        past_ids = [names.index('lag0_' + name) for name in G5_NAMES]
        for arm in RIDGES:
            fit, family = by_arm[arm], FEATURES[arm]
            feature = values['query_feature_' + family]
            if (fit['excluded_outer'] != source or fit['model'] != 'Ridge' or
                    fit['feature_names'] != audit['query_feature_names'][family] or
                    tuple(fit['feature_names'][:5]) != G5_NAMES or
                    feature.shape != (6, 128, len(fit['feature_names']))):
                raise ValueError('Matching fixed Ridge and named features required')
            rebuilt = model.ridge_ef(fit, feature)
            original = values['EF_draws'][values['arms'].tolist().index(arm)]
            error = float(np.max(np.abs(rebuilt - original)))
            if error > 1e-9:
                raise ValueError('Fixed Ridge reconstruction changed')
            max_error = max(max_error, error)
            predictions[arm] = fit
        for j, key in enumerate(values['keys']):
            if key not in labels:
                continue
            truth = labels[key]
            mg.append(dict(region=source, key=str(key), **model.forecast_case(
                values['MG_draws'][j], truth['g5'], values['query_feature_Size'][j, 0, past_ids])))
            for index, arm in enumerate(values['arms']):
                summaries = model.point_case(values['CIF_draws'][index, j], values['EF_draws'][index, j], truth['cif'], truth['ef'])
                # Preserve the old mean endpoints even if a serialization roundoff exists.
                mean = summaries['mean']
                for kind, target in (('cif', truth['cif']), ('ef', truth['ef'])):
                    mean[kind + '_point'] = float(values[kind.upper() + '_point'][index, j])
                    bias = mean[kind + '_point'] - target
                    mean.update({kind + '_bias': bias, kind + '_abs_error': abs(bias), kind + '_sq_error': bias ** 2})
                points.extend(dict(region=source, key=str(key), arm=str(arm), mode=mode, **record) for mode, record in summaries.items())
            for arm, fit in predictions.items():
                query = values['query_feature_' + FEATURES[arm]][j].copy()
                query[:, :5] = truth['g5']
                ef = model.ridge_ef(fit, query)
                if not np.allclose(ef, ef[0], atol=1e-9, rtol=0):
                    raise ValueError('Realized G plus fixed context must give a constant EF')
                predicted = float(ef[0])
                forecast = float(values['EF_point'][values['arms'].tolist().index(arm), j])
                record = dict(region=source, key=str(key), arm=arm, true_ef=truth['ef'], forecast_ef_point=forecast, oracle_ef_point=predicted)
                for name, point in (('forecast_ef', forecast), ('oracle_ef', predicted)):
                    bias = point - truth['ef']
                    record.update({name + '_bias': bias, name + '_abs_error': abs(bias), name + '_sq_error': bias ** 2})
                oracle.append(record)
    mg, points, oracle = pd.DataFrame(mg), pd.DataFrame(points), pd.DataFrame(oracle)
    mg_region = aggregate(mg, ['region'])
    point_region = aggregate(points, ['region', 'arm', 'mode'], ('cif', 'ef'))
    oracle_region = aggregate(oracle, ['region', 'arm'], ('forecast_ef', 'oracle_ef'))
    tables = dict(mg_case=mg, mg_region=mg_region, mg_macro=mg_region.drop(columns='scored_months').mean(numeric_only=True).to_frame().T,
                  point_case=points, point_region=point_region,
                  point_macro=point_region.drop(columns='scored_months').groupby(['arm', 'mode']).mean(numeric_only=True).reset_index(),
                  oracle_ef_case=oracle, oracle_ef_region=oracle_region,
                  oracle_ef_macro=oracle_region.drop(columns='scored_months').groupby('arm').mean(numeric_only=True).reset_index(), label_support=support)
    expected = (35, 7, 1, 770, 154, 22, 140, 28, 4, 42)
    if tuple(len(t) for t in tables.values()) != expected or any(t.empty for t in tables.values()):
        raise ValueError('Exact registered diagnostic support required')
    return tables, dict(new_fits=0, source_months=35, fixed_ridge_max_reconstruction_error=max_error,
        table_rows={key: len(value) for key, value in tables.items()}, monthly_MG_empirical_coverage_evaluated=True,
        CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False, oracle_deployment_comparison=False,
        mechanism_breakthrough=False, new_holdout_authorized=False)


def run() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve V35 attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    write(OUT / 'analysis_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    bound = {**frozen['input_hashes'], **frozen['code_hashes'], **{str(OUT / name): sha(OUT / name) for name in ('freeze.json', 'analysis_started.json')}}
    verify(bound)
    tables, summary = compute(frozen)
    verify(bound)
    for name, table in tables.items():
        table.to_csv(OUT / f'{name}.csv', mode='x', index=False)
    write(OUT / 'summary.json', summary)
    outputs = {str(path): sha(path) for path in OUT.iterdir() if path.name not in ('freeze.json', 'analysis_started.json')}
    write(OUT / 'source_completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
          analysis_started_sha256=sha(OUT / 'analysis_started.json'), output_hashes=outputs,
          initial_final_hashes_unchanged=True, **summary))
    print('Completed zero-fit diagnostics:', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    run()
