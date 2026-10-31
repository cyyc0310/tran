"""Exclusive zero-fit V36 exploration from the fixed V35 evidence closure."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import numpy as np
import pandas as pd
from scripts import capacity_routing_v36 as model
from scripts import fleet_information_v34 as v34
from scripts import fleet_inventory_v33 as inventory
from scripts import run_conditional_diagnostics_v35 as v35

OUT = ROOT/'results/capacity-routing-v36'
PARENT = ROOT/'docs/evidence/conditional-diagnostics-v35-verification.json'
PARENT_SHA = '7518b714451d50fc6404b5c784e67ab827c583dd27516d4cc531786935f6e5a9'
SOURCES, DATES, v31, common = v35.SOURCES, v35.v31.DATES, v35.v31, v35.common
CODE = ('scripts/capacity_routing_v36.py', 'scripts/run_capacity_routing_v36.py',
        'tests/test_capacity_routing_v36.py', 'tests/test_run_capacity_routing_v36.py',
        'docs/CAPACITY_ROUTING_V36_PROTOCOL.md')
REVIEWS = tuple(ROOT/f'results/capacity-routing-v36-{name}-review.md' for name in ('code', 'math'))
sha, read, write, now, verify = v35.sha, v35.read, v35.write, v35.now, v35.verify
aggregate, point_case, load_archive = v35.aggregate, v35.model.point_case, v35.load_archive


def contract() -> dict:
    if len(PARENT_SHA) != 64 or any(c not in '0123456789abcdef' for c in PARENT_SHA):
        raise ValueError('A unique fixed V35 parent pin is required')
    verify({str(PARENT): PARENT_SHA})
    manifest = read(PARENT)
    if manifest.get('status') != 'VERIFICATION_COMPLETE' or manifest.get('independent_formula_review_status') != 'PASS':
        raise ValueError('Independent V35 formula review PASS required')
    inputs = {str(PARENT): PARENT_SHA}
    for field in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for path, digest in manifest[field].items():
            if path in inputs and inputs[path] != digest:
                raise ValueError('Conflicting parent mapping')
            inputs[path] = digest
    verify(inputs)
    parent_freeze = str(ROOT/'results/conditional-diagnostics-v35/freeze.json')
    if parent_freeze not in inputs:
        raise ValueError('Bound V35 freeze required')
    old = read(parent_freeze)
    aliases = [old['monthly_path'], old['support_path']]
    aliases += [entry[key] for entry in old['entries'] for key in ('path', 'audit_path')]
    if [e['region'] for e in old['entries']] != list(SOURCES) or any(p not in inputs for p in aliases):
        raise ValueError('Exact bound seven-source V34 aliases required')
    for path in REVIEWS:
        if not path.is_file() or path.is_symlink() or not path.read_text().strip():
            raise ValueError('Two final code/math prelaunch reviews required')
        inputs[str(path)] = sha(path)
    code = {str(ROOT/path): sha(ROOT/path) for path in CODE}
    verify(code)
    return dict(version=36, code_hashes=code, input_hashes=inputs, environment=common.environment(),
        entries=old['entries'], monthly_path=old['monthly_path'], support_path=old['support_path'],
        recipes=list(model.RECIPES), sources=list(SOURCES), new_fits=0,
        exploratory_after_V34_V35=True, method_innovation_claim=False, new_holdout_authorized=False)


def source_inputs(audit: dict, values: dict[str, np.ndarray], source: str
                  ) -> tuple[np.ndarray, np.ndarray, list[dict], dict]:
    fits = audit['fits']; by_arm = {fit['arm']: fit for fit in fits}
    if (audit['region'] != source or audit['fit_count'] != 7 or len(fits) != 7
            or set(by_arm) != set(v34.ARMS[4:]) or audit['arms'] != list(v34.ARMS)
            or values['arms'].tolist() != list(v34.ARMS)):
        raise ValueError('Matching complete frozen V34 outer audit required')
    fit = by_arm['SizeCap-Ridge']
    v34._training_metadata(fit, source)  # Date/source checks before reading capacity numbers.
    names = list(v34.FEATURE_NAMES['SizeCap'])
    if (fit['excluded_outer'] != source or fit['family'] != 'SizeCap' or fit['model'] != 'Ridge'
            or fit['parameters'] != v34.PARAMETERS['Ridge'] or fit['feature_names'] != names
            or audit['query_feature_names']['SizeCap'] != names):
        raise ValueError('Named raw SizeCap-Ridge training/query features required')
    regions = np.asarray(fit['training_regions']); starts = fit['training_period_start_ns']; n = len(starts)
    weight = np.asarray(fit['sample_weights'], float)
    expected = np.array([n/(6*np.count_nonzero(regions == region)) for region in regions])
    if (n != (156 if source == 'FPL' else 154) or weight.shape != (n,)
            or not np.allclose(weight, expected, atol=1e-12, rtol=0)):
        raise ValueError('Original source-balanced 180-case-pool training cohort required')
    train_inventory = [dict(key=key, region=region, origin_ns=int(at), **inventory.inventory_for_origin(
        pd.Timestamp(int(at), unit='ns', tz='UTC'))) for key, region, at in zip(fit['training_keys'], regions, starts)]
    query_inventory = [dict(key=f'{source}/{at:%Y-%m}', region=source, origin_ns=int(at.value),
        **inventory.inventory_for_origin(at)) for at in DATES]
    if audit['training_inventory'] != train_inventory or audit['query_inventory'] != query_inventory:
        raise ValueError('Inherited inventory origin/year/release identity changed')
    x, query = np.asarray(fit['training_X'], float), np.asarray(values['query_feature_SizeCap'], float)
    if x.shape != (n, 36) or query.shape != (6, 128, 36) or not np.isfinite(x).all() or not np.isfinite(query).all():
        raise ValueError('Finite original raw capacity feature grids required')
    ids = [names.index(name) for name in v34.CAPACITY_FEATURE_NAMES[:4]]
    train_q, query_particles = x[:, ids], query[..., ids]
    if (not np.array_equal(query_particles, np.broadcast_to(query_particles[:, :1], query_particles.shape))
            or not np.array_equal(query_particles[:, 0], np.repeat(query_particles[:1, 0], 6, axis=0))):
        raise ValueError('Fixed annual query capacity must not vary across particles or six origins')
    query_q = query_particles[:, 0].copy()
    limit = dict(region=source, train_cases=n,
        train_source_year_count=len({(r['region'], r['year']) for r in train_inventory}))
    for i, fuel in enumerate(model.FUELS):
        limit.update({f'q_{fuel}_min': float(train_q[:, i].min()), f'q_{fuel}_max': float(train_q[:, i].max())})
    limit['cap_ef_max'] = float((400+600*train_q[:, 0]+400*train_q[:, 2]).max())
    return train_q, query_q, query_inventory, limit


def ordered(frame: pd.DataFrame) -> pd.DataFrame:
    if 'recipe' not in frame: return frame
    frame = frame.assign(_order=frame.recipe.map(dict(zip(model.RECIPES, range(10)))))
    keys = [k for k in ('omitted_source', 'region', 'key', '_order', 'mode') if k in frame]
    return frame.sort_values(keys).drop(columns='_order').reset_index(drop=True)


def compute(frozen: dict) -> tuple[dict[str, pd.DataFrame], dict]:
    if [e['region'] for e in frozen['entries']] != list(SOURCES):
        raise ValueError('Exact seven source order required')
    routes, limits, prepared = [], [], []
    for entry in frozen['entries']:
        source = entry['region']; values = load_archive(entry['path'], source)
        train, query, inventories, limit = source_inputs(read(entry['audit_path']), values, source)
        w = model.weights(train, query); limits.append(limit)
        base, anchor = [values['arms'].tolist().index(a) for a in ('Size-Ridge', 'Capacity-OP')]
        pred = model.blend(w, values['EF_draws'][base], values['EF_draws'][anchor], values['MG_draws'])
        if not np.allclose(values['CIF_draws'][[base, anchor]], values['EF_draws'][[base, anchor]]*values['MG_draws'][None, ..., 0], atol=1e-9, rtol=0):
            raise ValueError('Original baseline particle correspondence changed')
        for i, recipe in enumerate(model.RECIPES):
            for endpoint, old in ((0, base), (1, anchor)):
                mask = w[i] == endpoint
                for kind in ('EF', 'CIF'): pred[kind+'_draws'][i, mask] = values[kind+'_draws'][old, mask]
            routes.extend(dict(region=source, key=str(key), origin_ns=int(DATES[j].value), recipe=recipe,
                weight=float(w[i, j]), inventory_year=inventories[j]['year'],
                **{f'q_{f}': float(query[j, k]) for k, f in enumerate(model.FUELS)}) for j, key in enumerate(values['keys']))
        prepared.append((source, values, w, pred, base, anchor))
    # Every route is fixed before any scoring labels are accessed.
    labels, support = v31.scoring_labels(common.monthly_rows(frozen))
    if len(labels) != 35 or not all(y['ef_defined'] for y in labels.values()) or support.to_csv(index=False) != Path(frozen['support_path']).read_text():
        raise ValueError('Exact same 35 defined-EF months and 42 support rows required')
    points = []
    for source, values, w, pred, base, anchor in prepared:
        for j, key in enumerate(values['keys']):
            if key not in labels: continue
            truth = labels[key]
            for i, recipe in enumerate(model.RECIPES):
                summaries = point_case(pred['CIF_draws'][i, j], pred['EF_draws'][i, j], truth['cif'], truth['ef'])
                if w[i, j] in (0, 1):
                    old = base if w[i, j] == 0 else anchor
                    for kind in ('cif', 'ef'):
                        point = float(values[kind.upper()+'_point'][old, j]); bias = point-truth[kind]
                        summaries['mean'].update({kind+'_point': point, kind+'_bias': bias,
                            kind+'_abs_error': abs(bias), kind+'_sq_error': bias**2})
                points.extend(dict(region=source, key=str(key), recipe=recipe, mode=mode, **record)
                              for mode, record in summaries.items())
    cases = pd.DataFrame(points)
    region = aggregate(cases, ['region', 'recipe', 'mode'], ('cif', 'ef'))
    macro = region.drop(columns='scored_months').groupby(['recipe', 'mode']).mean(numeric_only=True).reset_index()
    leave = [region.loc[region.region.ne(source)].drop(columns='scored_months').groupby(['recipe', 'mode']).mean(numeric_only=True).reset_index().assign(omitted_source=source) for source in SOURCES]
    tables = dict(route_table=pd.DataFrame(routes), training_limits=pd.DataFrame(limits), case_metrics=cases,
        region_metrics=region, macro_metrics=macro, leave_one_source_out_macro=pd.concat(leave, ignore_index=True), label_support=support)
    tables = {k: ordered(t) for k, t in tables.items()}
    if tuple(len(t) for t in tables.values()) != (420, 7, 700, 140, 20, 140, 42):
        raise ValueError('Exact registered routing and scoring support required')
    return tables, dict(new_fits=0, source_months=35, table_rows={k: len(t) for k, t in tables.items()},
        recipes=list(model.RECIPES), exploratory_after_V34_V35=True, method_innovation_claim=False,
        routing_uses_truth=False, risk_identification_claim=False, new_holdout_authorized=False,
        source_level_routing=True, leave_one_source_out_used_for_selection=False,
        CIF_interval_coverage_evaluated=False, hourly_coverage_evaluated=False,
        hourly_values_read=False, target_values_opened=False, unseen_values_opened=False)


def run() -> None:
    if OUT.exists() or OUT.is_symlink(): raise FileExistsError('Preserve V36 attempt')
    frozen = contract(); OUT.mkdir()
    write(OUT/'freeze.json', dict(created_utc=now(), **frozen))
    write(OUT/'analysis_started.json', dict(created_utc=now(), freeze_sha256=sha(OUT/'freeze.json')))
    bound = {**frozen['input_hashes'], **frozen['code_hashes'],
        **{str(OUT/name): sha(OUT/name) for name in ('freeze.json', 'analysis_started.json')}}
    verify(bound); tables, summary = compute(frozen); verify(bound)
    for name, table in tables.items(): table.to_csv(OUT/f'{name}.csv', mode='x', index=False)
    write(OUT/'summary.json', summary); verify(bound)
    expected = {'freeze.json', 'analysis_started.json', 'summary.json'} | {f'{name}.csv' for name in tables}
    if {p.name for p in OUT.iterdir()} != expected or any(not p.is_file() or p.is_symlink() for p in OUT.iterdir()):
        raise ValueError('Exact regular V36 output files required before completion')
    outputs = {str(p): sha(p) for p in OUT.iterdir() if p.name not in ('freeze.json', 'analysis_started.json')}
    write(OUT/'source_completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT/'freeze.json'),
        analysis_started_sha256=sha(OUT/'analysis_started.json'), output_hashes=outputs,
        initial_final_hashes_unchanged=True, **summary))
    print('Completed exploratory zero-fit routes:', summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args(); run()
