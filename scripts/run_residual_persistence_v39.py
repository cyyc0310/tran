"""One source-only nested-profile persistence diagnostic; no future score labels."""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import residual_persistence_v39 as model
from scripts import coarse_inverse_v38 as profile
from scripts import run_coarse_inverse_v38 as parent

SOURCES = parent.SOURCES
sha, read, write, verify, now = parent.sha, parent.read, parent.write, parent.verify, parent.now
OUT = ROOT / 'results/residual-persistence-v39'
PARENT = ROOT / 'docs/evidence/coarse-inverse-v38-verification.json'
PARENT_SHA = '1a3b73183848a5a0b06bfbad1e00d9bc5a9e46f97dfefb42c6e2ec28bc042f36'
CODE = ('scripts/residual_persistence_v39.py', 'scripts/run_residual_persistence_v39.py',
        'tests/test_residual_persistence_v39.py', 'tests/test_run_residual_persistence_v39.py',
        'docs/RESIDUAL_PERSISTENCE_V39_PROTOCOL.md')
REVIEWS = tuple(ROOT / f'results/residual-persistence-v39-{kind}-review.md' for kind in ('code', 'math'))
OUTPUTS = ('profiles.json', 'monthly_residuals.csv', 'source_pairs.csv', 'outer_summary.csv', 'summary.json')


def mappings(evidence: dict) -> dict:
    result = {}
    for name in ('producer_code_hashes', 'input_hashes', 'auditor_code_hashes', 'artifact_hashes'):
        for p, digest in evidence[name].items():
            if p in result and result[p] != digest:
                raise ValueError('Conflicting evidence mapping')
            result[p] = digest
    return result


def contract() -> dict:
    verify({str(PARENT): PARENT_SHA})
    evidence = read(PARENT)
    if evidence['status'] != 'VERIFICATION_COMPLETE' or evidence['independent_formula_review_status'] != 'PASS':
        raise ValueError('Verified V38 required')
    closure = mappings(evidence)
    verify(closure)
    inherited = evidence['parent_manifest']
    verify({inherited: evidence['parent_manifest_sha256']})
    inherited_map = mappings(read(inherited))
    verify(inherited_map)
    paths = [str(ROOT / f'results/coarse-inverse-v38/predictions/{source}.json') for source in SOURCES]
    paths.append(str(ROOT / 'scripts/coarse_inverse_v38.py'))
    if any(p not in closure for p in paths):
        raise ValueError('Bound seven-source audits and pure profile required')
    inputs = {p: closure[p] for p in paths}
    inputs[str(PARENT)] = PARENT_SHA
    for p in REVIEWS:
        if not p.is_file() or p.is_symlink() or not p.read_text().strip():
            raise ValueError('Two prelaunch reviews required')
        inputs[str(p)] = sha(p)
    return dict(version=39, code_hashes={str(ROOT / p): sha(ROOT / p) for p in CODE},
        input_hashes=inputs, environment=parent.common.environment(), sources=list(SOURCES),
        source_audits=dict(zip(SOURCES, paths[:7])), profile_implementation=paths[-1],
        split='2023-07-01', nested_profiles=42, original_source_pool_keys=180,
        monthly_case_count=1080, new_fine_label_keys=False, future_2025_score_labels_accessed=False,
        future_forecast_fields_used=False,
        hourly_values_read=False, unseen_values_opened=False, method_innovation_claim=False)


def check() -> dict:
    frozen, expected = read(OUT / 'freeze.json'), contract()
    if set(frozen) != set(expected) | {'created_utc'} or any(frozen[k] != v for k, v in expected.items()):
        raise ValueError('Frozen inputs/code/environment changed')
    return frozen


def source_data(audit: dict, outer: str) -> dict:
    """Validate source identities/dates before converting training numbers."""
    if outer not in SOURCES or audit['region'] != outer:
        raise ValueError('Matching registered outer audit required')
    keys = audit['training_keys']
    if len(keys) != (156 if outer == 'FPL' else 154) or len(set(keys)) != len(keys):
        raise ValueError('Original exact outer training cohort required')
    sources, dates = [], []
    for key in keys:
        match = re.fullmatch(r'([A-Z]+)/([0-9]{4})-([0-9]{2})', key)
        if match is None or match[1] not in SOURCES or match[1] == outer:
            raise ValueError('Outer-excluded original source keys required')
        at = pd.Timestamp(year=int(match[2]), month=int(match[3]), day=1, tz='UTC')
        if not parent.v30.START <= at < parent.v30.END:
            raise ValueError('Original source supervision dates required')
        sources.append(match[1]); dates.append(at)
    if set(sources) != set(SOURCES)-{outer}:
        raise ValueError('Exactly six non-outer source regions required')
    for source in set(sources):
        if sources.count(source) != (24 if source == 'FPL' else 26):
            raise ValueError('Original per-source support required')
        selected = [at for s, at in zip(sources, dates) if s == source]
        split = pd.Timestamp('2023-07-01', tz='UTC')
        if not any(at < split for at in selected) or not any(at >= split for at in selected):
            raise ValueError('Both fixed time halves required')
    g, c = np.asarray(audit['training_generation_MW'], float), np.asarray(audit['training_capacity_MW'], float)
    months = np.asarray(audit['training_months'])
    if (g.shape != (len(keys), 4) or c.shape != g.shape or not np.isfinite(g).all() or
            not np.isfinite(c).all() or np.any(g < 0) or np.any(c < 0) or
            not np.array_equal(months, np.array([d.month for d in dates]))):
        raise ValueError('Finite aligned source generation/capacity/month arrays required')
    return dict(keys=keys, sources=np.array(sources), dates=dates, generation=g, capacities=c, months=months)


def nested(data: dict, outer: str) -> tuple[list[dict], list[dict]]:
    records, fits = [], []
    for held in SOURCES:
        if held == outer:
            continue
        train = data['sources'] != held
        held_ids = np.flatnonzero(~train)
        regions = data['sources'][train]
        if set(regions) != set(SOURCES)-{outer, held}:
            raise ValueError('Exactly five inner training source regions required')
        n = int(train.sum())
        weight = np.array([n/(5*np.count_nonzero(regions == s)) for s in regions])
        fitted = profile.fit_profiles(data['generation'][train], data['capacities'][train],
            data['months'][train], weight, [k for k, keep in zip(data['keys'], train) if keep])
        u = profile.profile_predict(fitted['month'], data['months'][~train])
        result = model.residuals(data['generation'][~train], data['capacities'][~train], u)
        if not np.allclose(result['identity_rhs'], result['EF_error_true_minus_pred'], atol=1e-9, rtol=0):
            raise ValueError('Exact carbon identity failed')
        fits.append(dict(outer=outer, held_source=held, training_keys=[k for k, keep in zip(data['keys'], train) if keep],
            training_regions=regions, sample_weights=weight, profiles=fitted,
            held_keys=[data['keys'][i] for i in held_ids]))
        for j, i in enumerate(held_ids):
            row = dict(outer=outer, source=held, key=data['keys'][i], period_start=data['dates'][i].isoformat(),
                d=result['d'][j], k=result['k'][j], EF_true=result['EF_true'][j], EF_pred=result['EF_pred'][j],
                EF_error_true_minus_pred=result['EF_error_true_minus_pred'][j], identity_rhs=result['identity_rhs'][j])
            for f, fuel in enumerate(('coal', 'gas', 'oil', 'other')):
                for field, values in (('generation_MW', data['generation'][i]), ('capacity_MW', data['capacities'][i]),
                                      ('profile_u', u[j]), ('x', result['x'][j]), ('xhat', result['xhat'][j]), ('r', result['r'][j])):
                    row[field+'_'+fuel] = values[f]
            records.append(row)
    return records, fits


def run() -> None:
    if OUT.exists() or OUT.is_symlink():
        raise FileExistsError('Preserve V39 actual attempt')
    frozen = contract()
    OUT.mkdir()
    write(OUT / 'freeze.json', dict(created_utc=now(), **frozen))
    write(OUT / 'started.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json')))
    records, fits = [], []
    for source in SOURCES:
        data = source_data(read(frozen['source_audits'][source]), source)
        rows, models = nested(data, source)
        records.extend(rows); fits.extend(models)
        print('Nested profiles', source, len(models), 'residual months', len(rows), flush=True)
    summaries = model.summarize_pairs(records)
    if len(records) != 1080 or len(fits) != 42 or len(summaries['pairs']) != 42 or len(summaries['outers']) != 7:
        raise ValueError('Exact original nested diagnostic grid required')
    write(OUT / 'profiles.json', dict(fits=fits))
    for name, rows in (('monthly_residuals', records), ('source_pairs', summaries['pairs']), ('outer_summary', summaries['outers'])):
        pd.DataFrame(rows).to_csv(OUT / (name+'.csv'), mode='x', index=False)
    summary = dict(nested_profiles=42, inner_training_sizes=sorted({len(f['training_keys']) for f in fits}),
        monthly_residuals=1080, source_pairs=42, outer_summaries=7, independent_regions_per_outer=6,
        split=frozen['split'], fine_carry_is_deployable=False, future_forecast_tested=False,
        method_innovation_claim=False, new_holdout_authorized=False, CIF_interval_coverage_evaluated=False)
    write(OUT / 'summary.json', summary)
    check()
    write(OUT / 'completed.json', dict(created_utc=now(), freeze_sha256=sha(OUT / 'freeze.json'),
        started_sha256=sha(OUT / 'started.json'), output_hashes={str(OUT / p): sha(OUT / p) for p in OUTPUTS}, **summary))
    print('Source-only persistence complete; no future/monthly score, hourly or unknown labels read', flush=True)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
