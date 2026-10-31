"""Source-only monthly history controls on fixed archived MG forecasts."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import history_signature_v32 as model
from scripts import run_coarse_context_v30 as common
from scripts import run_forward_context_v31 as previous

OUT = ROOT/'results/history-signature-v32'
PARENT = ROOT/'docs/evidence/forward-context-v31-verification.json'
PARENT_SHA = 'f6544892fc7bf24256cadd5809b3b9fcdfde60c4e42bde9bfa13db41ea9b46df'
CODE = ('scripts/history_signature_v32.py', 'scripts/run_history_signature_v32.py',
        'tests/test_history_signature_v32.py', 'tests/test_run_history_signature_v32.py',
        'docs/HISTORY_SIGNATURE_V32_PROTOCOL.md')
REVIEWS = tuple(ROOT/f'results/history-signature-v32-{k}-review.md' for k in ('code', 'python', 'science'))
DATES = previous.DATES
BASE, SECONDARY, METRICS, OUTPUTS = previous.BASE, previous.SECONDARY, previous.METRICS, previous.OUTPUTS
sha, read, write, verify, now = common.sha, common.read, common.write, common.verify, common.now


def expected_contract() -> dict:
    if sha(PARENT) != PARENT_SHA:
        raise ValueError('Fixed audited V31 parent required')
    evidence = read(PARENT)
    if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_actual_audit_status'] != 'PASS':
        raise ValueError('Independent actual V31 PASS required')
    inputs = {str(PARENT): PARENT_SHA}
    for field in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for path, digest in evidence[field].items():
            if path in inputs and inputs[path] != digest:
                raise ValueError('Conflicting parent mappings')
            inputs[path] = digest
    verify(inputs)
    old = read(ROOT/'results/forward-context-v31/freeze.json')
    legacy_region = str(ROOT/'results/forward-context-v31/region_metrics.csv')
    legacy_support = str(ROOT/'results/forward-context-v31/label_support.csv')
    aliases = [old['monthly_path'], legacy_region, legacy_support, *old['forecast_paths'].values()]
    if (old['sources'] != list(model.SOURCES) or old['origins'] != DATES.asi8.tolist()
            or set(old['forecast_paths']) != set(model.SOURCES) or any(p not in inputs for p in aliases)):
        raise ValueError('Bound same monthly/forecast/origin/Legacy aliases required')
    for path in REVIEWS:
        if not path.is_file() or not path.read_text().strip():
            raise ValueError('Three final prelaunch reviews required')
        inputs[str(path)] = sha(path)
    return dict(version=32, sources=list(model.SOURCES), arms=list(model.ARMS),
        feature_dimensions={f: len(v) for f, v in model.FEATURE_NAMES.items()},
        origins=DATES.asi8.tolist(), samples=128, fit_count=126, history_rows=12,
        monthly_path=old['monthly_path'], forecast_paths=old['forecast_paths'],
        legacy_region_path=legacy_region, legacy_support_path=legacy_support,
        code_hashes={str(ROOT/p): sha(ROOT/p) for p in CODE}, input_hashes=inputs,
        environment=common.environment(), deployment_cutoff=model.CUTOFF.isoformat(),
        score_cutoff=previous.SCORE_CUTOFF.isoformat(), label_minimum_coverage=.9,
        oracle_current_group=False, training_current_group_realized=True, forward_forecast=True,
        models_retrained_on_matched_cohort=True, new_training_labels_used=False,
        hourly_values_read=False, probability_coverage_evaluated=False, target_values_opened=False,
        unseen_values_opened=False, custom_method=False, china_category_mapping_verified=False,
        real_publication_vintage_verified=False)


def prepare() -> None:
    if OUT.exists():
        raise FileExistsError('Preserve V32 attempt')
    contract = expected_contract()
    OUT.mkdir()
    write(OUT/'freeze.json', dict(created_utc=now(), **contract))
    print('Prepared', len(contract['code_hashes']), 'code', len(contract['input_hashes']), 'input hashes', flush=True)


def check() -> dict:
    frozen, expected = read(OUT/'freeze.json'), expected_contract()
    if (set(frozen) != set(expected)|{'created_utc'} or any(frozen[k] != v for k, v in expected.items())
            or any(frozen[k] is not v for k, v in expected.items() if isinstance(v, bool))):
        raise ValueError('Frozen V32 contract changed')
    return frozen


def predict() -> None:
    frozen = check()
    initial, folder = sha(OUT/'freeze.json'), OUT/'predictions'
    if folder.exists() or (OUT/'prediction_seal.json').exists():
        raise FileExistsError('Preserve V32 prediction attempt')
    folder.mkdir()
    rows = common.monthly_rows(frozen)
    cases = model.build_cases(rows)
    if set(cases['regions']) != set(model.SOURCES):
        raise ValueError('All seven common source cohorts required')
    entries = []
    for region in model.SOURCES:
        draws = previous.forecast_inputs(frozen['forecast_paths'][region])
        result = model.fit_predict(rows, cases, outer=region, origins=DATES, mg_draws=draws)
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        arrays.update(arms=np.asarray(model.ARMS), origins=DATES.asi8,
                      keys=np.asarray([f'{region}/{at:%Y-%m}' for at in DATES]))
        path = folder/f'{region}.npz'
        with path.open('xb') as f:
            np.savez_compressed(f, **arrays)
        audit = path.with_suffix('.json')
        write(audit, {**result['audit'], 'region': region, 'forecast_path': frozen['forecast_paths'][region],
            'fits': result['fits'], 'cohort': cases['audit'],
            'all_case_keys': cases['keys'], 'all_case_regions': cases['regions']})
        entries.append(dict(region=region, path=str(path), sha256=sha(path),
                            audit_path=str(audit), audit_sha256=sha(audit)))
        print('Predicted', region, 'origins', len(DATES), 'fits', result['audit']['fit_count'], flush=True)
    check()
    verify({str(OUT/'freeze.json'): initial})
    write(OUT/'prediction_seal.json', dict(created_utc=now(), freeze_sha256=initial,
        entries=entries, complete_grid=True, archives=7, arms=18, origins_per_source=6, fit_count=126,
        oracle_current_group=False, forward_forecast=True, hourly_values_read=False,
        new_training_labels_used=False, target_values_opened=False, unseen_values_opened=False))


def predictions() -> dict:
    seal = read(OUT/'prediction_seal.json')
    closed = ('oracle_current_group', 'hourly_values_read', 'new_training_labels_used',
              'target_values_opened', 'unseen_values_opened')
    if (seal['freeze_sha256'] != sha(OUT/'freeze.json') or seal['complete_grid'] is not True
            or seal['forward_forecast'] is not True or any(seal[k] is not False for k in closed)
            or seal['archives'] != 7 or seal['arms'] != 18 or seal['origins_per_source'] != 6
            or seal['fit_count'] != 126 or [e['region'] for e in seal['entries']] != list(model.SOURCES)):
        raise ValueError('Complete seven-source V32 seal required before labels or Legacy')
    folder = OUT/'predictions'
    if (not folder.is_dir() or folder.is_symlink()
            or {p.name for p in folder.iterdir()} != {f'{s}.{e}' for s in model.SOURCES for e in ('npz', 'json')}
            or any(not p.is_file() or p.is_symlink() for p in folder.iterdir())):
        raise ValueError('Exact fourteen regular prediction files required')
    for entry in seal['entries']:
        path = folder/f"{entry['region']}.npz"
        if entry['path'] != str(path) or entry['audit_path'] != str(path.with_suffix('.json')):
            raise ValueError('Registered prediction aliases required')
        verify({entry['path']: entry['sha256'], entry['audit_path']: entry['audit_sha256']})
        with np.load(path, allow_pickle=False) as a:
            if (a['arms'].tolist() != list(model.ARMS) or not np.array_equal(a['origins'], DATES.asi8)
                    or a['keys'].tolist() != [f"{entry['region']}/{at:%Y-%m}" for at in DATES]
                    or any(a[k].shape != (18, 6, 128, 4) or not np.isfinite(a[k]).all() for k in ('ratios_raw', 'ratios'))
                    or any(a[k].shape != (18, 6, 128) or not np.isfinite(a[k]).all() for k in ('EF_draws', 'CIF_draws', 'projection_l1'))
                    or any(a[k].shape != (18, 6) or not np.isfinite(a[k]).all() for k in ('EF_point', 'CIF_point'))):
                raise ValueError('Complete eighteen-arm V32 forecast grid required')
    return seal


def matched_pairs() -> list[tuple[str, str, str]]:
    pairs = [(f'{a}-{m}', f'{b}-{m}', 'feature') for m in model.MODELS
             for a, b in [('Trend', 'G5'), ('Size', 'G5'), ('Size', 'Trend')]]
    pairs += [(f'{f}-ET', f'{f}-Ridge', 'model') for f in ('G5', 'Trend', 'Size')]
    pairs += [(f'JointShape-{m}', f'{f}-{m}', 'feature') for m in model.MODELS
              for f in ('Size', 'LongTotal', 'JointLevel', 'JointTrend', 'JointAmplitude', 'PhaseScramble')]
    pairs += [(f'JointShape-{m}', f'LegacySize-{m}', 'legacy') for m in model.MODELS]
    if len(pairs) != 23 or len(set(pairs)) != 23:
        raise ValueError('Fixed twenty-three comparisons required')
    return pairs


def legacy_values(frozen: dict, support: pd.DataFrame) -> pd.DataFrame:
    old = pd.read_csv(frozen['legacy_support_path']).sort_values('key').reset_index(drop=True)
    new = support.sort_values('key').reset_index(drop=True)
    if len(old) != 42 or old.key.duplicated().any() or set(old) != set(new):
        raise ValueError('Exact Legacy support grid required')
    for column in ('region', 'key', 'origin', 'status', 'available_at', 'ef_defined'):
        if not old[column].fillna('').equals(new[column].fillna('')):
            raise ValueError('Legacy support differs: '+column)
    for column in ('coverage', 'total_mwh'):
        if not np.allclose(old[column], new[column], atol=1e-12, rtol=0, equal_nan=True):
            raise ValueError('Legacy quality support differs: '+column)
    region = pd.read_csv(frozen['legacy_region_path'])
    if (len(region) != 42 or region.duplicated(['region', 'arm']).any()
            or set(region.region) != set(model.SOURCES) or set(region.arm) != set(previous.model.ARMS)):
        raise ValueError('Fixed six-arm Legacy region grid required')
    legacy = region[region.arm.isin(['Size-Ridge', 'Size-ET'])].copy()
    counts = support[support.status.eq('scored')].groupby('region').size()
    if any(row.scored_months != counts.get(row.region, 0) for row in legacy.itertuples()):
        raise ValueError('Legacy scored-month denominators differ')
    ef_counts = support[support.status.eq('scored') & support.ef_defined].groupby('region').size()
    if any(row.internal_EF_months != ef_counts.get(row.region, 0) for row in legacy.itertuples()):
        raise ValueError('Legacy internal-EF denominators differ')
    legacy['arm'] = 'Legacy'+legacy.arm
    return legacy


def summarize(case: pd.DataFrame, legacy: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if (case.duplicated(['region', 'key', 'arm']).any() or set(case.region) != set(model.SOURCES)
            or set(case.arm) != set(model.ARMS) or not np.isfinite(case[list(BASE)]).all().all()
            or not np.isfinite(case.loc[case.ef_defined, list(SECONDARY)]).all().all()
            or not case.loc[~case.ef_defined, list(SECONDARY)].isna().all().all()):
        raise ValueError('Complete common V32 score cohort required')
    groups = case.groupby(['region', 'arm'])
    region = groups[list(METRICS)].mean().reset_index()
    region['cif_rmse'], region['ef_rmse'] = np.sqrt(region.cif_sq_error), np.sqrt(region.ef_sq_error)
    counts = groups.agg(scored_months=('key', 'count'), internal_EF_months=('ef_defined', 'sum')).reset_index()
    region = region.merge(counts, on=['region', 'arm'], validate='one_to_one')
    macro = region.groupby('arm')[[*METRICS, 'cif_rmse', 'ef_rmse']].mean().reset_index()
    macro['scored_region_months'] = region.groupby('arm').scored_months.sum().reindex(macro.arm).to_numpy()
    macro['internal_EF_region_months'] = region.groupby('arm').internal_EF_months.sum().reindex(macro.arm).to_numpy()
    all_region = pd.concat([region, legacy], ignore_index=True)
    comparisons, contrasts = [], []
    for candidate, reference, kind in matched_pairs():
        c = all_region[all_region.arm.eq(candidate)].set_index('region').reindex(model.SOURCES)
        b = all_region[all_region.arm.eq(reference)].set_index('region').reindex(model.SOURCES)
        a, r = c.cif_abs_error, b.cif_abs_error
        if kind != 'model':
            passed = (r.mean() > 1e-12 and a.mean() <= .95*r.mean() and int((a < r-1e-12).sum()) >= 4
                      and bool((a <= 1.1*r+1e-12).all()) and all(a[s] <= r[s]+1e-12 for s in ('MISO', 'ERCO')))
            comparisons.append(dict(candidate=candidate, reference=reference, kind=kind,
                signal=bool(passed), improved_regions=int((a < r-1e-12).sum()), macro_cif_mae_delta=float(a.mean()-r.mean())))
        for source in model.SOURCES:
            contrasts.append(dict(region=source, candidate=candidate, reference=reference, kind=kind,
                **{k+'_delta': float(c.loc[source, k]-b.loc[source, k]) for k in (*METRICS, 'cif_rmse', 'ef_rmse')}))
    signals = {(c['candidate'], c['reference']): c['signal'] for c in comparisons}
    phase = [m for m in model.MODELS if all(signals[(f'JointShape-{m}', f'{f}-{m}')]
             for f in ('Size', 'LongTotal', 'JointLevel', 'JointTrend', 'JointAmplitude', 'PhaseScramble'))]
    practical = [m for m in phase if signals[(f'JointShape-{m}', f'LegacySize-{m}')]]
    if len(region) != 126 or len(macro) != 18 or len(contrasts) != 161 or len(comparisons) != 20:
        raise ValueError('Exact126region/18macro/161contrast/20signal grid required')
    choice = dict(comparisons=comparisons, phase_association_models=phase, practical_forward_signal_models=practical,
        oracle_current_group=False, forward_forecast=True, probability_coverage_evaluated=False,
        hourly_stage_authorized=False, new_holdout_authorized=False, mechanism_breakthrough=False, custom_method=False)
    return region, macro, pd.DataFrame(contrasts), choice


def score() -> None:
    if any((OUT/n).exists() for n in (*OUTPUTS, 'source_completed.json')):
        raise FileExistsError('Preserve V32 score attempt')
    frozen, seal = check(), predictions()
    initial = {str(OUT/n): sha(OUT/n) for n in ('freeze.json', 'prediction_seal.json')}
    labels, support = previous.scoring_labels(common.monthly_rows(frozen))
    legacy = legacy_values(frozen, support)
    records = []
    for entry in seal['entries']:
        with np.load(entry['path'], allow_pickle=False) as a:
            for j, key in enumerate(a['keys']):
                if key not in labels:
                    continue
                y, g = labels[key], a['MG_draws'][j].mean(0)
                for i, arm in enumerate(model.ARMS):
                    error, e = float(a['CIF_point'][i, j]-y['cif']), float(a['EF_point'][i, j]-y['ef'])
                    records.append(dict(region=entry['region'], key=str(key), arm=arm, ef_defined=y['ef_defined'],
                        cif_abs_error=abs(error), cif_sq_error=error**2, cif_signed_error=error,
                        ef_abs_error=abs(e), ef_sq_error=e**2, ef_signed_error=e,
                        g5_l1=float(abs(g-y['g5']).sum()), pos_abs_error=float(abs(g[0]-y['g5'][0])),
                        projection_l1=float(a['projection_l1'][i, j].mean())))
        print('Scored source monthly targets', entry['region'], flush=True)
    case = pd.DataFrame(records)
    if len(support) != 42 or len(case) != len(labels)*18:
        raise ValueError('Full shared label support required')
    region, macro, contrasts, choice = summarize(case, legacy)
    for name, frame in [('case_metrics', case), ('region_metrics', region), ('macro_metrics', macro),
                        ('matched_contrasts', contrasts), ('label_support', support)]:
        frame.to_csv(OUT/f'{name}.csv', mode='x', index=False)
    write(OUT/'source_choice.json', choice)
    check()
    predictions()
    verify(initial)
    write(OUT/'source_completed.json', dict(version=32, created_utc=now(),
        freeze_sha256=initial[str(OUT/'freeze.json')], prediction_seal_sha256=initial[str(OUT/'prediction_seal.json')],
        output_hashes={str(OUT/n): sha(OUT/n) for n in OUTPUTS}, oracle_current_group=False, forward_forecast=True,
        hourly_values_read=False, new_training_labels_used=False, probability_coverage_evaluated=False,
        target_values_opened=False, unseen_values_opened=False, mechanism_breakthrough=False, custom_method=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=('prepare', 'predict', 'score'))
    args = parser.parse_args()
    {'prepare': prepare, 'predict': predict, 'score': score}[args.stage]()
