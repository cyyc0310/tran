"""Synthetic exclusive receipts, inherited metadata and balanced V36 tables."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import fleet_information_v34 as v34
from scripts import fleet_inventory_v33 as inventory
from scripts import run_capacity_routing_v36 as r


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def synthetic_inputs(tmp_path, monkeypatch):
    entries, labels, support_rows, originals = [], {}, [], {}
    for index, source in enumerate(r.SOURCES):
        regions, starts, ends, keys, x = [], [], [], [], []
        for si, other in enumerate(r.SOURCES):
            if other == source: continue
            for j, at in enumerate(pd.date_range('2022-01-01', periods=24 if other == 'FPL' else 26, freq='MS', tz='UTC')):
                q = np.array([.15+.02*si+.005*(at.year-2022), .5-.02*si-.005*(at.year-2022), .1, .25])
                row = np.zeros(36); row[31:35] = q; row[35] = 1
                regions.append(other); starts.append(at.value); ends.append((at+pd.offsets.MonthBegin(1)).value)
                keys.append(f'{other}/{at:%Y-%m}'); x.append(row.tolist())
        n = len(keys)
        weights = [n/(6*regions.count(other)) for other in regions]
        fit = dict(arm='SizeCap-Ridge', family='SizeCap', model='Ridge', excluded_outer=source,
            parameters=v34.PARAMETERS['Ridge'], feature_names=list(v34.FEATURE_NAMES['SizeCap']),
            training_keys=keys, training_regions=regions, training_X=x, training_y='MUST_NOT_READ',
            sample_weights=weights, training_period_start_ns=starts, training_period_end_ns=ends,
            training_available_ns=[t+5*86400*10**9 for t in ends])
        train_inventory = [dict(key=key, region=region, origin_ns=at,
            **inventory.inventory_for_origin(pd.Timestamp(at, unit='ns', tz='UTC')))
            for key, region, at in zip(keys, regions, starts)]
        query_inventory = [dict(key=f'{source}/{at:%Y-%m}', region=source, origin_ns=at.value,
            **inventory.inventory_for_origin(at)) for at in r.DATES]
        audit = dict(region=source, arms=list(v34.ARMS), fit_count=7,
            fits=[fit]+[dict(arm=a) for a in v34.ARMS[4:] if a != 'SizeCap-Ridge'],
            query_feature_names={'SizeCap': list(v34.FEATURE_NAMES['SizeCap'])},
            training_inventory=train_inventory, query_inventory=query_inventory)
        q = np.array([.2+.01*index, .4-.01*index, .1, .3])
        pos = np.linspace(.2, .8, 128)
        mg = np.tile(np.c_[pos, np.repeat(((1-pos)/4)[:, None], 4, axis=1)][None], (6, 1, 1))
        base = np.tile(480+12*index+np.linspace(-20, 20, 128), (6, 1))
        ef = np.tile(base[None], (11, 1, 1)); ef[2] = 400+600*q[0]+400*q[2]
        cif = ef*mg[None, ..., 0]+4e-10  # Allowed archived roundoff exposes accidental endpoint recomputation.
        feature = np.zeros((6, 128, 36)); feature[..., :5] = mg; feature[..., 31:35] = q; feature[..., 35] = 1
        values = dict(origins=r.DATES.asi8, keys=np.array([f'{source}/{at:%Y-%m}' for at in r.DATES]),
            arms=np.array(v34.ARMS), MG_draws=mg, EF_draws=ef, CIF_draws=cif,
            EF_point=ef.mean(-1)+5e-10, CIF_point=cif.mean(-1)+5e-10,
            query_feature_Size=feature[..., :31], query_feature_SizeCap=feature,
            query_feature_IntensiveCap=feature[..., :35], query_feature_DirectEF=feature.copy())
        npz, metadata = tmp_path/f'{source}.npz', tmp_path/f'{source}.json'
        np.savez_compressed(npz, **values); put(metadata, audit)
        entries.append(dict(region=source, path=str(npz), audit_path=str(metadata)))
        originals[source] = values
        scored = {'ISNE': 1, 'MISO': 5, 'NYIS': 5}.get(source, 6)
        for j, key in enumerate(values['keys']):
            support_rows.append(dict(region=source, key=str(key)))
            if j < scored:
                labels[str(key)] = dict(cif=210.+index+j, ef=420.+index+j, ef_defined=True)
    support = pd.DataFrame(support_rows); support_path = tmp_path/'support.csv'; support.to_csv(support_path, index=False)
    monkeypatch.setattr(r.common, 'monthly_rows', lambda _: [])
    monkeypatch.setattr(r.v31, 'scoring_labels', lambda _: (deepcopy(labels), support.copy()))
    frozen = dict(entries=entries, support_path=str(support_path), monthly_path='UNOPENED',
                  input_hashes={}, code_hashes={})
    return frozen, labels, support, originals


def test_existing_or_symlink_attempt_rejected_before_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'contract', lambda: pytest.fail('must preserve attempt first'))
    with pytest.raises(FileExistsError): r.run()


def test_empty_pin_is_fail_closed_before_any_parent_read(monkeypatch):
    monkeypatch.setattr(r, 'PARENT_SHA', '')
    monkeypatch.setattr(r, 'read', lambda _: pytest.fail('empty pin must fail before reads'))
    with pytest.raises(ValueError, match='pin'): r.contract()


def test_parent_gate_uses_formula_review_and_checks_conflicting_closure(tmp_path, monkeypatch):
    parent = tmp_path/'parent.json'
    manifest = dict(status='VERIFICATION_COMPLETE', independent_formula_review_status='FAIL',
                    producer_code_hashes={}, input_hashes={}, auditor_code_hashes={}, artifact_hashes={})
    put(parent, manifest); monkeypatch.setattr(r, 'PARENT', parent); monkeypatch.setattr(r, 'PARENT_SHA', r.sha(parent))
    with pytest.raises(ValueError, match='formula'): r.contract()
    manifest.update(independent_formula_review_status='PASS', input_hashes={str(parent): '0'*64})
    put(parent, manifest); monkeypatch.setattr(r, 'PARENT_SHA', r.sha(parent))
    with pytest.raises(ValueError, match='Conflicting'): r.contract()


def test_parent_formula_pass_accepts_empty_auditor_mapping_and_binds_five_files(tmp_path, monkeypatch):
    entries = []
    bound = {}
    for source in r.SOURCES:
        paths = [tmp_path/f'{source}.npz', tmp_path/f'{source}.json']
        for path in paths:
            path.write_text('synthetic bound bytes'); bound[str(path)] = r.sha(path)
        entries.append(dict(region=source, path=str(paths[0]), audit_path=str(paths[1])))
    monthly, support = tmp_path/'monthly', tmp_path/'support'
    for path in (monthly, support):
        path.write_text('synthetic'); bound[str(path)] = r.sha(path)
    freeze = tmp_path/'results/conditional-diagnostics-v35/freeze.json'
    put(freeze, dict(entries=entries, monthly_path=str(monthly), support_path=str(support)))
    bound[str(freeze)] = r.sha(freeze)
    parent = tmp_path/'parent.json'
    put(parent, dict(status='VERIFICATION_COMPLETE', independent_formula_review_status='PASS',
        producer_code_hashes={}, input_hashes={}, auditor_code_hashes={}, artifact_hashes=bound))
    for name in r.CODE:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('synthetic code')
    reviews = [tmp_path/'code-review', tmp_path/'math-review']
    for path in reviews: path.write_text('PASS synthetic')
    monkeypatch.setattr(r, 'ROOT', tmp_path); monkeypatch.setattr(r, 'PARENT', parent)
    monkeypatch.setattr(r, 'PARENT_SHA', r.sha(parent)); monkeypatch.setattr(r, 'REVIEWS', reviews)
    monkeypatch.setattr(r.common, 'environment', lambda: {'synthetic': True})
    result = r.contract()
    assert len(result['code_hashes']) == 5
    assert result['input_hashes'][str(parent)] == r.sha(parent)
    assert set(map(str, reviews)) <= result['input_hashes'].keys()
    assert result['new_fits'] == 0 and result['method_innovation_claim'] is False


def test_synthetic_tables_same_support_exact_endpoints_and_six_source_macros(tmp_path, monkeypatch):
    frozen, _, support, originals = synthetic_inputs(tmp_path, monkeypatch)
    tables, summary = r.compute(frozen)
    assert {k: len(v) for k, v in tables.items()} == dict(route_table=420, training_limits=7,
        case_metrics=700, region_metrics=140, macro_metrics=20, leave_one_source_out_macro=140, label_support=42)
    assert tables['label_support'].to_csv(index=False) == support.to_csv(index=False)
    assert summary['new_fits'] == 0 and summary['method_innovation_claim'] is False
    assert summary['new_holdout_authorized'] is False and summary['routing_uses_truth'] is False
    limits = tables['training_limits'].set_index('region')
    assert limits.loc['FPL', 'train_cases'] == 156
    assert limits.drop(index='FPL').train_cases.tolist() == [154]*6
    route = tables['route_table']
    assert route.groupby(['region', 'key']).size().tolist() == [10]*42
    assert set(('q_coal', 'q_gas', 'q_oil', 'q_other')) <= set(route)
    points = tables['case_metrics']; means = points[points['mode'].eq('mean')]
    route_weights = route.set_index(['region', 'key', 'recipe']).weight
    for row in points.itertuples():
        weight = route_weights.loc[(row.region, row.key, row.recipe)]
        if weight not in (0, 1): continue
        idx = 0 if weight == 0 else 2
        j = originals[row.region]['keys'].tolist().index(row.key)
        for kind in ('cif', 'ef'):
            expected = (originals[row.region][kind.upper()+'_point'][idx, j] if row.mode == 'mean'
                        else np.median(originals[row.region][kind.upper()+'_draws'][idx, j]))
            assert getattr(row, kind+'_point') == expected
    region = tables['region_metrics']; macro = tables['macro_metrics']
    for row in macro.itertuples():
        subset = region[region.recipe.eq(row.recipe) & region['mode'].eq(row.mode)]
        assert row.cif_abs_error == pytest.approx(subset.cif_abs_error.mean())
        assert row.cif_rmse == pytest.approx(subset.cif_rmse.mean())
    for row in tables['leave_one_source_out_macro'].itertuples():
        subset = region[region.region.ne(row.omitted_source) & region.recipe.eq(row.recipe) & region['mode'].eq(row.mode)]
        assert len(subset) == 6
        assert row.ef_abs_error == pytest.approx(subset.ef_abs_error.mean())


def test_all_routes_are_fixed_before_scoring_labels_are_accessed(tmp_path, monkeypatch):
    frozen, labels, support, _ = synthetic_inputs(tmp_path, monkeypatch)
    calls = []
    original = r.model.weights
    def weights(*args):
        calls.append('route'); return original(*args)
    def scoring(_):
        assert calls == ['route']*7
        return labels, support
    monkeypatch.setattr(r.model, 'weights', weights)
    monkeypatch.setattr(r.v31, 'scoring_labels', scoring)
    r.compute(frozen)


@pytest.mark.parametrize('kind', ['outer', 'future-release', 'feature-names', 'query-particle', 'inventory-year', 'weight'])
def test_metadata_named_columns_and_inventory_identity_guards(tmp_path, monkeypatch, kind):
    frozen, _, _, _ = synthetic_inputs(tmp_path, monkeypatch)
    path = Path(frozen['entries'][0]['audit_path']); audit = json.loads(path.read_text()); fit = audit['fits'][0]
    if kind == 'outer': fit['training_regions'][0] = 'BPAT'
    elif kind == 'future-release': fit['training_available_ns'][0] = pd.Timestamp('2025-01-02', tz='UTC').value
    elif kind == 'feature-names': fit['feature_names'] = list(reversed(fit['feature_names']))
    elif kind == 'inventory-year': audit['training_inventory'][0]['year'] = 2023
    elif kind == 'weight': fit['sample_weights'][0] *= 2
    else:
        file = Path(frozen['entries'][0]['path'])
        with np.load(file, allow_pickle=False) as z: values = {k: z[k] for k in z.files}
        values['query_feature_SizeCap'][0, 1, 31] += .01
        np.savez_compressed(file, **values)
    put(path, audit)
    with pytest.raises(ValueError): r.compute(frozen)


def test_changed_support_is_rejected(tmp_path, monkeypatch):
    frozen, labels, support, _ = synthetic_inputs(tmp_path, monkeypatch)
    labels.pop(next(iter(labels)))
    monkeypatch.setattr(r.v31, 'scoring_labels', lambda _: (labels, support))
    with pytest.raises(ValueError, match='35'): r.compute(frozen)


def test_failed_compute_keeps_exclusive_start_and_blocks_retry(tmp_path, monkeypatch):
    out = tmp_path/'out'; monkeypatch.setattr(r, 'OUT', out)
    monkeypatch.setattr(r, 'contract', lambda: dict(input_hashes={}, code_hashes={}))
    monkeypatch.setattr(r, 'compute', lambda _: (_ for _ in ()).throw(ValueError('synthetic failure')))
    with pytest.raises(ValueError, match='synthetic failure'): r.run()
    receipt = (out/'analysis_started.json').read_bytes()
    assert {p.name for p in out.iterdir()} == {'freeze.json', 'analysis_started.json'}
    with pytest.raises(FileExistsError): r.run()
    assert (out/'analysis_started.json').read_bytes() == receipt


def test_successful_synthetic_run_seals_all_seven_tables(tmp_path, monkeypatch):
    frozen, _, _, _ = synthetic_inputs(tmp_path, monkeypatch)
    out = tmp_path/'out'; monkeypatch.setattr(r, 'OUT', out); monkeypatch.setattr(r, 'contract', lambda: frozen)
    r.run()
    result = json.loads((out/'source_completed.json').read_text())
    assert len(result['output_hashes']) == 8
    assert result['initial_final_hashes_unchanged'] is True
    r.verify(result['output_hashes'])
    assert len(list(out.iterdir())) == 11


def test_changed_bound_input_rejected_before_tables_written(tmp_path, monkeypatch):
    source = tmp_path/'input'; source.write_text('before'); out = tmp_path/'out'
    monkeypatch.setattr(r, 'OUT', out)
    monkeypatch.setattr(r, 'contract', lambda: dict(input_hashes={str(source): r.sha(source)}, code_hashes={}))
    def changed(_):
        source.write_text('after'); return {}, {}
    monkeypatch.setattr(r, 'compute', changed)
    with pytest.raises(ValueError): r.run()
    assert {p.name for p in out.iterdir()} == {'freeze.json', 'analysis_started.json'}
