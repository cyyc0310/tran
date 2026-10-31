"""Synthetic support, execution-boundary and fixed comparison regressions."""
import numpy as np
import pandas as pd
import pytest
from scripts import run_history_signature_v32 as r


def label(region, date, zero=False):
    start = pd.Timestamp(date, tz='UTC')
    end = start+pd.offsets.MonthBegin(1)
    return dict(region=region, period_start=start.isoformat(), period_end=end.isoformat(),
        available_at=(end+pd.Timedelta(days=30)).isoformat(), coverage='1', total_mwh='10000',
        **{'fuelshare_'+f: str(float(f == ('hydro' if zero else 'gas'))) for f in r.previous.model.parent.FUELS})


def legacy_table(support):
    counts = support[support.status.eq('scored')].groupby('region').size()
    ef_counts = support[support.status.eq('scored') & support.ef_defined].groupby('region').size()
    return pd.DataFrame([dict(region=s, arm=a, **{k: 100. for k in r.METRICS},
        cif_rmse=10., ef_rmse=10., scored_months=counts.get(s, 0), internal_EF_months=ef_counts.get(s, 0))
        for s in r.model.SOURCES for a in r.previous.model.ARMS])


def legacy_files(tmp_path, support):
    paths = dict(legacy_support_path=str(tmp_path/'legacy-support.csv'),
                 legacy_region_path=str(tmp_path/'legacy-region.csv'))
    support.to_csv(paths['legacy_support_path'], index=False)
    legacy_table(support).to_csv(paths['legacy_region_path'], index=False)
    return paths


def test_labels_and_legacy_wait_for_complete_seal(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'check', lambda: {})
    def missing():
        raise ValueError('Incomplete V32 seal')
    monkeypatch.setattr(r, 'predictions', missing)
    monkeypatch.setattr(r.common, 'monthly_rows', lambda _: pytest.fail('Labels before seal'))
    monkeypatch.setattr(r, 'legacy_values', lambda *_: pytest.fail('Legacy before seal'))
    with pytest.raises(ValueError, match='Incomplete V32 seal'):
        r.score()


def test_prediction_writer_accepts_pure_region_audit_and_seals_all_sources(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    (tmp_path/'freeze.json').write_text('{}')
    frozen = dict(forecast_paths={s: '/synthetic/'+s for s in r.model.SOURCES})
    monkeypatch.setattr(r, 'check', lambda: frozen)
    monkeypatch.setattr(r.common, 'monthly_rows', lambda _: [])
    monkeypatch.setattr(r.model, 'build_cases', lambda _: dict(regions=np.asarray(r.model.SOURCES),
        keys=list(r.model.SOURCES), audit={'synthetic': True}))
    draws = np.tile([.6, 0, .4, 0, 0], (6, 128, 1))
    monkeypatch.setattr(r.previous, 'forecast_inputs', lambda _: draws)
    def fitted(*args, outer, **kwargs):
        ratios = np.zeros((18, 6, 128, 4))
        ratios[..., 1] = 1.
        return dict(ratios_raw=ratios, ratios=ratios.copy(), EF_draws=np.full((18, 6, 128), 400.),
            CIF_draws=np.full((18, 6, 128), 240.), EF_point=np.full((18, 6), 400.),
            CIF_point=np.full((18, 6), 240.), projection_l1=np.zeros((18, 6, 128)), MG_draws=draws,
            audit={'region': outer, 'fit_count': 18}, fits=[{'synthetic': True} for _ in r.model.ARMS])
    monkeypatch.setattr(r.model, 'fit_predict', fitted)
    r.predict()
    seal = r.predictions()
    assert len(seal['entries']) == 7 and seal['fit_count'] == 126
    for entry in seal['entries']:
        stored = r.read(entry['audit_path'])
        assert stored['region'] == entry['region'] and len(stored['fits']) == 18


@pytest.mark.parametrize('defect', ['extra', 'missing', 'symlink'])
def test_prediction_inventory_checked_before_label_and_legacy(tmp_path, monkeypatch, defect):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    folder = tmp_path/'predictions'
    folder.mkdir()
    for source in r.model.SOURCES:
        for ext in ('npz', 'json'):
            (folder/f'{source}.{ext}').touch()
    if defect == 'extra':
        (folder/'orphan.npz').touch()
    else:
        (folder/'BPAT.npz').unlink()
        if defect == 'symlink':
            (folder/'BPAT.npz').symlink_to(folder/'BPAT.json')
    seal = dict(freeze_sha256='f', complete_grid=True, forward_forecast=True,
        oracle_current_group=False, hourly_values_read=False, new_training_labels_used=False,
        target_values_opened=False, unseen_values_opened=False, archives=7, arms=18,
        origins_per_source=6, fit_count=126, entries=[dict(region=s) for s in r.model.SOURCES])
    monkeypatch.setattr(r, 'read', lambda _: seal)
    monkeypatch.setattr(r, 'sha', lambda _: 'f')
    monkeypatch.setattr(r, 'check', lambda: {})
    monkeypatch.setattr(r.common, 'monthly_rows', lambda _: pytest.fail('Labels before inventory'))
    monkeypatch.setattr(r, 'legacy_values', lambda *_: pytest.fail('Legacy before inventory'))
    with pytest.raises(ValueError, match='fourteen'):
        r.score()


@pytest.mark.parametrize('field,value', [('history_rows', 3), ('arms', ['G5-Ridge']),
    ('legacy_support_path', '/foreign'), ('forecast_paths', {}), ('forward_forecast', False),
    ('hourly_values_read', 0)])
def test_frozen_contract_cannot_self_authorize(monkeypatch, field, value):
    expected = dict(history_rows=12, arms=list(r.model.ARMS), legacy_support_path='/fixed',
                    forecast_paths={'s': '/bound'}, forward_forecast=True, hourly_values_read=False)
    frozen = dict(created_utc='fixed', **expected)
    monkeypatch.setattr(r, 'read', lambda _: frozen)
    monkeypatch.setattr(r, 'expected_contract', lambda: expected)
    assert r.check() == frozen
    frozen[field] = value
    with pytest.raises(ValueError, match='contract changed'):
        r.check()


@pytest.mark.parametrize('defect', ['same_count_wrong_key', 'status', 'quality', 'denominator', 'ef_denominator'])
def test_legacy_comparison_requires_exact_support(tmp_path, defect):
    rows = [label(s, at.strftime('%Y-%m-%d')) for s in r.model.SOURCES for at in r.DATES]
    _, support = r.previous.scoring_labels(rows)
    paths = legacy_files(tmp_path, support)
    old = pd.read_csv(paths['legacy_support_path'])
    if defect == 'same_count_wrong_key':
        old.loc[0, 'key'] = 'BPAT/2024-12'
    elif defect == 'status':
        old.loc[0, 'status'] = 'missing'
    elif defect == 'quality':
        old.loc[0, 'coverage'] = .95
    else:
        region = pd.read_csv(paths['legacy_region_path'])
        region.loc[region.arm.eq('Size-Ridge'), 'scored_months' if defect == 'denominator' else 'internal_EF_months'] = 5
        region.to_csv(paths['legacy_region_path'], index=False)
    old.to_csv(paths['legacy_support_path'], index=False)
    with pytest.raises(ValueError, match='Legacy'):
        r.legacy_values(paths, support)


def score_fixture():
    records = []
    for source in r.model.SOURCES:
        for arm in r.model.ARMS:
            error = .8 if arm == 'JointShape-Ridge' else 1.
            records.append(dict(region=source, key=f'{source}/2025-01', arm=arm, ef_defined=True,
                **{**{k: 1. for k in r.METRICS}, 'cif_abs_error': error, 'cif_sq_error': error**2}))
    case = pd.DataFrame(records)
    legacy = pd.DataFrame([dict(region=s, arm=f'LegacySize-{m}', **{k: 1. for k in r.METRICS},
                               cif_rmse=1., ef_rmse=1.) for s in r.model.SOURCES for m in r.model.MODELS])
    return case, legacy


def test_layered_signal_needs_all_six_controls_and_legacy():
    case, legacy = score_fixture()
    region, macro, contrasts, choice = r.summarize(case, legacy)
    assert (len(region), len(macro), len(contrasts), len(choice['comparisons'])) == (126, 18, 161, 20)
    assert choice['phase_association_models'] == choice['practical_forward_signal_models'] == ['Ridge']
    assert choice['mechanism_breakthrough'] is choice['hourly_stage_authorized'] is choice['new_holdout_authorized'] is False
    legacy.loc[legacy.arm.eq('LegacySize-Ridge'), 'cif_abs_error'] = .7
    _, _, _, choice = r.summarize(case, legacy)
    assert choice['phase_association_models'] == ['Ridge'] and not choice['practical_forward_signal_models']
    case.loc[case.arm.eq('JointAmplitude-Ridge'), 'cif_abs_error'] = .7
    _, _, _, choice = r.summarize(case, legacy)
    assert not choice['phase_association_models']


def test_zero_pos_primary_and_case_counts_in_complete_score(tmp_path, monkeypatch):
    monkeypatch.setattr(r, 'OUT', tmp_path)
    folder = tmp_path/'predictions'
    folder.mkdir()
    (tmp_path/'freeze.json').write_text('{}')
    (tmp_path/'prediction_seal.json').write_text('{}')
    rows, entries = [], []
    for source in r.model.SOURCES:
        path = folder/f'{source}.npz'
        np.savez(path, keys=np.array([f'{source}/{at:%Y-%m}' for at in r.DATES]),
            MG_draws=np.tile([.6, 0, .4, 0, 0], (6, 128, 1)), CIF_point=np.full((18, 6), 240.),
            EF_point=np.full((18, 6), 400.), projection_l1=np.zeros((18, 6, 128)))
        entries.append(dict(region=source, path=str(path)))
        rows += [label(source, at.strftime('%Y-%m-%d'), zero=source == 'BPAT' and at.month == 1) for at in r.DATES]
    _, support = r.previous.scoring_labels(rows)
    paths = legacy_files(tmp_path, support)
    monkeypatch.setattr(r, 'check', lambda: paths)
    monkeypatch.setattr(r, 'predictions', lambda: dict(entries=entries))
    monkeypatch.setattr(r.common, 'monthly_rows', lambda _: rows)
    r.score()
    case = pd.read_csv(tmp_path/'case_metrics.csv')
    macro = pd.read_csv(tmp_path/'macro_metrics.csv')
    zero = case[case.key.eq('BPAT/2025-01')]
    assert len(case) == 756 and len(zero) == 18 and zero.cif_abs_error.eq(240.).all()
    assert not zero.ef_defined.any() and zero[list(r.SECONDARY)].isna().all().all()
    assert macro.scored_region_months.eq(42).all() and macro.internal_EF_region_months.eq(41).all()
    assert len(pd.read_csv(tmp_path/'matched_contrasts.csv')) == 161
    with pytest.raises(FileExistsError):
        r.score()


def test_grid_and_attempt_preservation(tmp_path, monkeypatch):
    pairs = r.matched_pairs()
    assert len(pairs) == len(set(pairs)) == 23
    assert sum(k != 'model' for _, _, k in pairs) == 20
    assert sum(k == 'legacy' for _, _, k in pairs) == 2
    monkeypatch.setattr(r, 'OUT', tmp_path)
    monkeypatch.setattr(r, 'expected_contract', lambda: pytest.fail('Do not replace an existing attempt'))
    with pytest.raises(FileExistsError):
        r.prepare()
