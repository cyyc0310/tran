"""Two immutable source-first rounds of native weather-delta transport.

V11 archives supply the exact native baseline; regenerated paths must match
before any phase can be sealed. No frozen V7--V11 code is modified.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import pandas as pd
from scripts import run_weather_recomposition_v11 as v11
from scripts.weather_shapes_v11 import ScaleModel, residual_paths, split_profiles
from scripts.weather_increment_v12 import delta_transport, JointQuantileForest
from transcif.research.dynamic_v2 import EF
from transcif.research.fuel_transport_v7 import recompose_paths

v10 = v11.v10
SOURCES, TARGETS = v11.SOURCES, v11.TARGETS
SEED, N_SAMPLES, FACTORS = v11.SEED, v11.N_SAMPLES, v11.FACTORS
METRICS, DAY_KEYS, REGION_KEYS = v11.METRICS, v11.DAY_KEYS, v11.REGION_KEYS
sha, read, write, now = v11.sha, v11.read, v11.write, v11.now
allowed_sources, _bank, _coarse = v11.allowed_sources, v11._bank, v11._coarse
_load_training, _indices = v11._load_training, v11._indices
PROTOCOL = ROOT / 'docs/WEATHER_INCREMENT_V12_PROTOCOL.md'
V11_ROOT = ROOT / 'results/weather-recomposition-v11'
ALPHAS = {'donor_a025': .25, 'donor_a050': .5, 'donor_a100': 1.}


def configurations(round_number: int) -> list[dict[str, Any]]:
    """Return the fixed 11/13-arm grid, with weather-only primary eligibility."""
    if round_number not in (1, 2):
        raise ValueError('V12 round must be 1 or 2')
    result = [dict(budget='B', model=m, primary_eligible=False)
              for m in ('native_recompose', 'simple_anchor')]
    result += [dict(budget='W', model=m, primary_eligible=True)
               for m in (ALPHAS if round_number == 1 else ('donor',))]
    refs = ('global_total', 'hetero_total', 'qrf_0') if round_number == 1 else (
        'global_total', 'hetero_total', 'qrf_0', 'qrf_25', 'qrf_50')
    result += [dict(budget=b, model=m, primary_eligible=round_number == 2 and b == 'W')
               for b in ('B', 'W') for m in refs]
    return result


def _code_paths() -> list[Path]:
    return sorted(set([*map(Path, read(V11_ROOT / 'prefit_freeze.json')['code_hashes']),
                       Path(__file__).resolve(), PROTOCOL,
                       ROOT / 'scripts/weather_increment_v12.py',
                       ROOT / 'scripts/analyze_weather_increment_v12.py']))


def _native_paths() -> dict[str, str]:
    paths = {}
    for stage, regions in [('source', SOURCES), ('target', TARGETS)]:
        seal = read(V11_ROOT / f'{stage}_prediction_seal.json')
        for region in regions:
            for model in ('native_recompose', 'simple_anchor'):
                matches = [e for e in seal['entries'] if e['region'] == region and
                           e['budget'] == 'B' and e['model'] == model]
                if len(matches) != 1:
                    raise ValueError('Missing unique V11 baseline archive')
                path = V11_ROOT / matches[0]['path']
                if sha(path) != matches[0]['sha256']:
                    raise ValueError('Frozen V11 native archive changed')
                paths[f'{stage}/{region}/{model}'] = str(path.resolve())
    return paths


def freeze(output: Path, data: Path, *, round_number: int,
           prior_round: Path | None = None, weather_csv: Path | None = None,
           points_json: Path | None = None) -> dict[str, Any]:
    """Freeze complete executable/input closure before new source predictions."""
    output, data = Path(output).resolve(), Path(data).resolve()
    if output.exists():
        raise FileExistsError('Use a fresh immutable V12 output directory')
    configurations(round_number)
    old = v11.verify(V11_ROOT)
    if (data != Path(old['data']) or
            (weather_csv is not None and Path(weather_csv).resolve() != Path(old['weather_csv'])) or
            (points_json is not None and Path(points_json).resolve() != Path(old['points_json']))):
        raise ValueError('V12 must reuse frozen V11 inputs')
    prior_paths: list[Path] = []
    alpha = 1.
    if round_number == 2:
        if prior_round is None:
            raise ValueError('Round 2 requires --prior-round source choices/locks')
        prior_round = Path(prior_round).resolve()
        verify_phase(prior_round, 'source')
        if verify(prior_round)['round'] != 1 or any((prior_round / n).exists() for n in
                ('target_prediction_seal.json','target_scores','target_completed.json')):
            raise ValueError('Round 2 source must precede every target score')
        alpha = float(read(prior_round / 'source_choices.json')['locked_alpha'])
        prior_paths = [prior_round / n for n in ('prefit_freeze.json', 'source_choices.json',
                       'source_prediction_seal.json', 'source_completed.json',
                       'source_scores/score_seal.json', 'source_scores/day_metrics.csv',
                       'source_scores/region_metrics.csv', 'source_scores/macro_metrics.csv')]
        prior_paths += [prior_round / e['path'] for e in read(prior_round / 'source_prediction_seal.json')['entries']]
    elif prior_round is not None:
        raise ValueError('Round 1 source does not use a prior round')
    natives = _native_paths()
    frozen = deepcopy(old)
    # Hash frozen V11 manifests themselves, and retain every one of its 42 code paths.
    inputs = [V11_ROOT / n for n in ('prefit_freeze.json', 'source_choices.json',
              'source_prediction_seal.json', 'target_prediction_seal.json',
              'source_completed.json', 'target_completed.json')]
    inputs += [*map(Path, natives.values()), *prior_paths, Path(sys.executable).resolve()]
    frozen['input_hashes'].update({str(p): sha(p) for p in inputs})
    frozen.update(created_utc=now(), version=12, round=round_number, alpha=alpha,
                  prior_round=str(prior_round) if prior_round else None,
                  native_paths=natives, configurations=configurations(round_number),
                  models=sorted({c['model'] for c in configurations(round_number)}),
                  code_hashes={str(p): sha(p) for p in _code_paths()},
                  environment=v10.environment(), target_scores_used_for_round_lock=False)
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    write(output / 'prefit_freeze.json', frozen)
    return frozen


def verify(output: Path) -> dict[str, Any]:
    """Verify byte closure, frozen grid and current executable environment."""
    frozen = read(Path(output) / 'prefit_freeze.json')
    for path, digest in {**frozen['input_hashes'], **frozen['code_hashes']}.items():
        if sha(Path(path)) != digest:
            raise ValueError('Frozen V12 bytes changed: ' + path)
    if (set(frozen['code_hashes']) != {str(p) for p in _code_paths()} or
            frozen['version'] != 12 or frozen['round'] not in (1, 2) or
            frozen['configurations'] != configurations(frozen['round']) or
            frozen['sources'] != list(SOURCES) or frozen['targets'] != list(TARGETS) or
            frozen['samples'] != N_SAMPLES or frozen['seed'] != SEED or
            frozen['factors'] != list(FACTORS) or frozen['proper_start'] != '2024-01-25' or
            frozen['environment'] != v10.environment() or
            frozen['alpha'] not in (.25, .5, 1.) or
            frozen.get('target_scores_used_for_round_lock') is not False):
        raise ValueError('V12 frozen grid/environment changed')
    if frozen['round'] == 2:
        prior = Path(frozen['prior_round'])
        if frozen['alpha'] != read(prior / 'source_choices.json')['locked_alpha']:
            raise ValueError('Round 2 strength detached from round 1 source lock')
    return frozen


class ShapeCache(v11.ShapeCache):
    """V11 cache with explicit inner-calibration query registrations.

    Keep all outer-donor proper queries explicitly registered before the first
    fit; inherited requests also cover each inner calibration donor fold.
    """
    def __init__(self, records: dict, subsets: list[tuple[str, ...]],
                 queries: dict[tuple[str, ...], list[str]]) -> None:
        super().__init__(records, subsets, queries)
        for subset in subsets:
            for donor in subset:
                self.request(tuple(r for r in subset if r != donor), donor, 'proper')


def _donor_predictions(bank: Any, shapes: ShapeCache, budget: str) -> np.ndarray:
    return v11._donor_predictions(bank, shapes, budget)


def center_paths(bank: Any, mixes: np.ndarray, days: pd.DatetimeIndex,
                 query_b: np.ndarray, query_w: np.ndarray,
                 donor_b: np.ndarray, donor_w: np.ndarray, *, alpha: float,
                 diagnostics: dict[str, Any] | None = None) -> np.ndarray:
    """Ratio of weighted mean corrected q; alpha zero uses untouched q exactly."""
    result, amplitude_error, change, negative, count = [], 0., 0., 0, 0
    zero_count = np.zeros(8,dtype=np.int64)
    created_count = np.zeros(8,dtype=np.int64)
    created_amount = np.zeros(8)
    amplitudes, original_shapes = split_profiles(bank.q_)
    valid = np.broadcast_to((bank.fuel_support_ & (amplitudes > 0))[:, None, :], bank.q_.shape)
    for mix, day, qb, qw in zip(mixes, days, query_b, query_w):
        weights, *_ = bank._weights(mix, day, 'recompose_joint', .3, np.ones(len(bank.days_), bool))
        corrected = (bank.q_ if alpha == 0. else delta_transport(
            bank.q_, qw, qb, donor_w, donor_b, bank.fuel_support_, strength=alpha))
        result.append(recompose_paths(mix, np.einsum('d,dhf->hf', weights, corrected)))
        if diagnostics is not None:
            before = original_shapes + alpha * ((qw-qb) - (donor_w-donor_b))
            after = np.divide(corrected, amplitudes[:,None,:], out=np.ones_like(corrected), where=amplitudes[:,None,:] > 0)
            change += float(np.abs(after-before)[valid].sum())
            negative += int((before[valid] < 0).sum()); count += int(valid.sum())
            amplitude_error = max(amplitude_error, float(np.abs(corrected.mean(1)-bank.q_.mean(1)).max()))
            original_zero = (bank.q_ == 0) & valid
            created = original_zero & (corrected > 1e-12)
            zero_count += original_zero.sum((0,1))
            created_count += created.sum((0,1))
            created_amount += np.where(created,corrected,0.).sum((0,1))
    if diagnostics is not None:
        diagnostics.update(mean_absolute_projection_change=change/max(count,1),
                           fraction_negative_before_projection=negative/max(count,1),
                           projection_supported_active_values=count,
                           fullbank_q_mean_max_error=amplitude_error,
                           supported_original_zero_count_by_fuel=zero_count.tolist(),
                           created_positive_from_zero_count_by_fuel=created_count.tolist(),
                           created_positive_from_zero_fraction_by_fuel=np.divide(created_count,zero_count,
                               out=np.zeros(8),where=zero_count>0).tolist(),
                           created_positive_from_zero_q_amount_by_fuel=created_amount.tolist())
    return np.stack(result)


def _calibration(frames: dict, records: dict, shapes: ShapeCache,
                 subset: tuple[str, ...], budget: str, alpha: float) -> dict[str, Any]:
    errors, features, regions, dates, audits = [], [], [], [], []
    for region in sorted(subset):
        inner = tuple(r for r in subset if r != region)
        bank, record = _bank(frames, inner), records[region, 'calibration']
        qb, qw = [shapes.get(inner, b, region, 'calibration') for b in ('B','W')]
        db, dw = [_donor_predictions(bank, shapes, b) for b in ('B','W')]
        center = center_paths(bank, record['features']['B'][:, :8], record['days'],
                              qb, qw, db, dw, alpha=alpha if budget == 'W' else 0.)
        errors.append(record['truth']-center); features.append(record['features'][budget])
        regions.extend([region]*len(center)); dates.extend(record['days'])
        audits.append(dict(calibration_region=region, training_regions=list(inner),
                           donor_regions=list(bank.source_regions_),
                           donor_oof_training_regions={d: [r for r in inner if r != d] for d in inner}))
    errors, features = np.concatenate(errors), np.concatenate(features)
    regions, days = np.array(regions), pd.DatetimeIndex(dates)
    scales = np.empty((len(errors),2))
    for region in sorted(subset):
        keep = regions != region
        scale = ScaleModel().fit(features[keep], errors[keep], regions[keep],
                                 exclude_region=region, days=days[keep])
        scales[~keep] = scale.predict(features[~keep])
    scale = ScaleModel().fit(features, errors, regions, days=days)
    qrf = JointQuantileForest().fit(features, errors, regions, days=days)
    return dict(errors=errors, features=features, regions=regions, days=days,
                scales=scales, model=scale, qrf=qrf, audit=audits)


def _uniforms(bank: Any, days: pd.DatetimeIndex) -> np.ndarray:
    return np.stack([bank._rng(SEED, day).random((N_SAMPLES,8))[:,1] for day in days])


def _predict_region(frames: dict, records: dict, shapes: ShapeCache,
                    subset: tuple[str, ...], region: str, mixes: np.ndarray,
                    draws: np.ndarray, calibrations: dict, frozen: dict,
                    stage: str) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    bank, record = _bank(frames, subset), records[region, 'evaluation']
    days = record['days']; donors, residuals = _indices(bank, mixes, days, calibrations['B']['regions'])
    qb, qw = [shapes.get(subset,b,region,'evaluation') for b in ('B','W')]
    db, dw = [_donor_predictions(bank,shapes,b) for b in ('B','W')]
    result, projection = {}, {}
    sampled = bank.q_[donors]
    # Identity is numerical, not merely statistically equivalent.
    native = np.stack([recompose_paths(draws[i],sampled[i]) for i in range(len(days))])
    simple = bank.raw_cif_[donors] + ((draws-bank.donor_mix_[donors]) @ EF)[...,None]
    for mode, regenerated in [('native_recompose',native),('simple_anchor',simple)]:
        archive = Path(frozen['native_paths'][f'{stage}/{region}/{mode}'])
        with np.load(archive,allow_pickle=False) as a:
            ensemble = np.clip(regenerated,0,1000)
            checks = {'raw_ensemble': regenerated, 'ensemble': ensemble,
                      'point': ensemble.mean(1), 'donor_indices': donors,
                      'origins': days.as_unit('ns').asi8}
            if any(not np.array_equal(a[k],value) for k,value in checks.items()):
                raise ValueError('Native V11 exact identity failed: '+region+'/'+mode)
            result['B',mode] = dict(raw=a['raw_ensemble'].copy(), center=a['center'].copy(),
                                   native_identity=True)
    for mode in (ALPHAS if frozen['round']==1 else ('donor',)):
        alpha = ALPHAS[mode] if mode in ALPHAS else frozen['alpha']
        corrected = delta_transport(sampled, qw[:,None], qb[:,None], dw[donors],
                                    db[donors],bank.fuel_support_[donors],strength=alpha)
        raw = np.stack([recompose_paths(draws[i],corrected[i]) for i in range(len(days))])
        valid = np.broadcast_to((bank.fuel_support_[donors] & (sampled.mean(2)>0))[:,:,None,:],sampled.shape)
        original_zero = (sampled==0) & valid
        created = original_zero & (corrected>1e-12)
        zero_count = original_zero.sum((0,1,2)); created_count = created.sum((0,1,2))
        zero_audit = dict(supported_original_zero_count_by_fuel=zero_count.tolist(),
            created_positive_from_zero_count_by_fuel=created_count.tolist(),
            created_positive_from_zero_fraction_by_fuel=np.divide(created_count,zero_count,
                out=np.zeros(8),where=zero_count>0).tolist(),
            created_positive_from_zero_q_amount_by_fuel=np.where(created,corrected,0.).sum((0,1,2)).tolist())
        result['W',mode] = dict(raw=raw,center=raw.mean(1),sampled_zero_creation=zero_audit,
             amplitude_error=float(np.max(np.abs(corrected.mean(2)-sampled.mean(2)))))
    uniforms = _uniforms(bank,days)
    for budget in ('B','W'):
        cal, diag = calibrations[budget], {}
        center = center_paths(bank,mixes,days,qb,qw,db,dw,
                              alpha=frozen['alpha'] if budget=='W' else 0.,diagnostics=diag)
        projection[budget] = diag
        for mode in ('global_total','hetero_total'):
            args = {} if mode=='global_total' else dict(scales=cal['model'].predict(record['features'][budget]),calibration_scales=cal['scales'])
            result[budget,mode] = dict(raw=residual_paths(center,cal['errors'],residuals,**args),center=center)
        for fraction in ((0.,) if frozen['round']==1 else (0.,.25,.5)):
            mode = {0.:'qrf_0',.25:'qrf_25',.5:'qrf_50'}[fraction]
            weights = cal['qrf'].predict_weights(record['features'][budget],global_fraction=fraction)
            cdf = np.cumsum(weights,axis=1)
            error_ids = np.empty(uniforms.shape,dtype=np.intp)
            for i,(c,u,w) in enumerate(zip(cdf,uniforms,weights)):
                last_positive = np.flatnonzero(w>0)[-1]
                c[last_positive:] = 1.
                error_ids[i] = np.minimum(np.searchsorted(c,u,side='right'),last_positive)
            sampled_errors = cal['errors'][error_ids]
            result[budget,mode] = dict(raw=center[:,None]+sampled_errors,center=center,
                                      conditional_residual_indices=error_ids,
                                      conditional_weights=weights,
                                      calibration_errors=cal['errors'],
                                      calibration_regions=np.asarray(cal['regions'],dtype='U'),
                                      calibration_origins=cal['days'].as_unit('ns').asi8,
                                      residual_uniforms=uniforms,
                                      raw_conditional_quantiles=center[:,None]+cal['qrf'].predict_quantiles(
                                          record['features'][budget],[.05,.5,.95],global_fraction=fraction),
                                      quantile_levels=np.array([.05,.5,.95]))
    audit = dict(training_regions=list(subset),donor_regions=list(bank.source_regions_),
                 excluded_evaluation_region=region if region in SOURCES else None,
                 source_calibration_regions=list(subset),proper_start='2024-01-25',
                 projection_by_budget=projection,calibration=calibrations['B']['audit'],
                 calibration_model_audits={b:dict(scale=calibrations[b]['model'].audit_,
                   qrf=calibrations[b]['qrf'].audit_) for b in ('B','W')},
                 calibration_includes_coarse_error=True,total_risk_uses_coarse_particles=False,
                 native_v11_exact_identity=True,donor_amplitudes_preserved=True,
                 donor_indices_shared=True,residual_level_shape_paired=True,
                 residual_uniforms_shared=True,conditional_residual_indices_shared=False,
                 qrf_residual_sampling='one complete 24h calibration day per shared uniform',
                 qrf_archive_calibration_source_only=True,
                 qrf_archive_calibration_period='2024H2',
                 qrf_archive_audit_arrays=['conditional_weights','calibration_errors',
                    'calibration_regions','calibration_origins','residual_uniforms'],
                 cif_daily_mean_preserved=False,target_hourly_used=False)
    return result,donors,residuals,audit


def _grid(stage: str, frozen: dict) -> set[tuple[str,str,str]]:
    return {(r,c['budget'],c['model']) for r in (SOURCES if stage=='source' else TARGETS)
            for c in configurations(frozen['round'])}



def _check_qrf_archive(a: Any, n_days: int, expected: tuple[str, ...]) -> None:
    """Validate the persisted source calibration bank and its actual day weights."""
    errors, regions, origins = a['calibration_errors'],a['calibration_regions'],a['calibration_origins']
    weights, uniforms, ids = a['conditional_weights'],a['residual_uniforms'],a['conditional_residual_indices']
    n_cal = len(errors)
    if (n_cal==0 or errors.shape!=(n_cal,24) or not np.isfinite(errors).all() or
            regions.shape!=(n_cal,) or regions.dtype.kind!='U' or set(regions)!=set(expected) or
            origins.shape!=(n_cal,) or not np.issubdtype(origins.dtype,np.integer) or
            np.any(origins<pd.Timestamp('2024-07-01',tz='UTC').value) or
            np.any(origins>=pd.Timestamp('2025-01-01',tz='UTC').value) or
            weights.shape!=(n_days,n_cal) or not np.isfinite(weights).all() or np.any(weights<0) or
            not np.allclose(weights.sum(1),1.,rtol=0.,atol=1e-12) or
            uniforms.shape!=(n_days,N_SAMPLES) or not np.isfinite(uniforms).all() or
            np.any((uniforms<0)|(uniforms>=1)) or
            ids.shape!=(n_days,N_SAMPLES) or not np.issubdtype(ids.dtype,np.integer) or
            np.any(ids<0) or np.any(ids>=n_cal)):
        raise ValueError('Invalid QRF conditional weights/source calibration bank')
    if not np.array_equal(a['raw_ensemble'],a['center'][:,None]+errors[ids]):
        raise ValueError('QRF paths detached from archived calibration day IDs')


def check_entries(output: Path, frozen: dict, stage: str, seal: dict) -> None:
    entries = seal['entries']
    actual = {(e['region'],e['budget'],e['model']) for e in entries}
    if len(entries)!=len(_grid(stage,frozen)) or actual!=_grid(stage,frozen):
        raise ValueError('V12 prediction grid incomplete/duplicated')
    days = v10._evaluation_days(stage,True).as_unit('ns').asi8
    common: dict[str, tuple[np.ndarray,np.ndarray]] = {}
    common_uniforms: dict[str,np.ndarray] = {}
    for e in entries:
        expected = allowed_sources(e['region'] if stage=='source' else None)
        audit = e['audit']
        if set(audit['training_regions'])!=set(expected) or set(audit['donor_regions'])!=set(expected):
            raise ValueError('Outer region leaked into fit or donor bank')
        for fold in audit['calibration']:
            inner = set(expected)-{fold['calibration_region']}
            if set(fold['training_regions'])!=inner or set(fold['donor_regions'])!=inner:
                raise ValueError('Inner calibration region leaked')
            for donor, training in fold['donor_oof_training_regions'].items():
                if set(training)!=inner-{donor}:
                    raise ValueError('Donor OOF region leaked')
        path = Path(output)/e['path']
        if sha(path)!=e['sha256']:
            raise ValueError('V12 prediction bytes changed')
        with np.load(path,allow_pickle=False) as a:
            if (not np.array_equal(a['origins'],days) or a['ensemble'].shape!=(len(days),N_SAMPLES,24)
                    or a['point'].shape!=(len(days),24) or a['center'].shape!=(len(days),24)
                    or not np.isfinite(a['raw_ensemble']).all() or not np.isfinite(a['center']).all()
                    or not np.array_equal(a['ensemble'],np.clip(a['raw_ensemble'],0,1000))
                    or not np.array_equal(a['point'],a['ensemble'].mean(1))):
                raise ValueError('Invalid V12 prediction archive')
            pair = a['donor_indices'],a['residual_indices']
            if e['region'] in common and any(not np.array_equal(x,y) for x,y in zip(pair,common[e['region']])):
                raise ValueError('B/W donor/residual indices differ')
            common[e['region']] = tuple(x.copy() for x in pair)
            if e['model'].startswith('qrf_'):
                _check_qrf_archive(a,len(days),expected)
                if (e['region'] in common_uniforms and
                        not np.array_equal(a['residual_uniforms'],common_uniforms[e['region']])):
                    raise ValueError('B/W QRF residual uniforms differ')
                common_uniforms[e['region']] = a['residual_uniforms'].copy()
                if (a['raw_conditional_quantiles'].shape!=(len(days),3,24) or
                        not np.isfinite(a['raw_conditional_quantiles']).all() or
                        not np.array_equal(a['quantile_levels'],[.05,.5,.95]) or
                        np.any(np.diff(a['raw_conditional_quantiles'],axis=1)<0) or
                        a['conditional_residual_indices'].shape!=(len(days),N_SAMPLES)):
                    raise ValueError('Invalid QRF residual day/quantile archive')
            if e['budget']=='B' and e['model'] in ('native_recompose','simple_anchor'):
                with np.load(frozen['native_paths'][f"{stage}/{e['region']}/{e['model']}"],allow_pickle=False) as old:
                    if any(not np.array_equal(a[k],old[k]) for k in ('origins','point','ensemble','raw_ensemble','donor_indices','center')):
                        raise ValueError('Archived native V11 exact identity changed')
def shape_energy_score(paths: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Multivariate energy score for centered paths, normalized L2 / sqrt(24)."""
    paths, truth = np.asarray(paths, float), np.asarray(truth, float)
    if paths.ndim != 3 or paths.shape[2] != 24 or truth.shape != (len(paths), 24):
        raise ValueError('Shape energy score needs days by samples by 24 paths')
    output = np.empty(len(paths))
    for start in range(0, len(paths), 32):
        x, y = paths[start:start+32], truth[start:start+32]
        first = np.sqrt(np.mean((x-y[:,None])**2, axis=2)).mean(1)
        norms = np.sum(x*x, axis=2)
        square = norms[:,:,None] + norms[:,None,:] - 2*np.einsum('dsh,dth->dst', x, x)
        distance = np.sqrt(np.maximum(square, 0)/24)
        output[start:start+32] = first - .5*distance.mean((1,2))
    return output


def _extra_metrics(point, ensemble, truth, center):
    def crps(draw, target):
        s = draw.shape[1]
        ordered = np.sort(draw, axis=1)
        coeff = 2 * np.arange(s) - s + 1
        return (np.abs(draw - target[:, None]).mean(1) -
                np.einsum('s,ds...->d...', coeff, ordered) / s**2).reshape(len(draw), -1).mean(1)
    level, target_level = ensemble.mean(2), truth.mean(1)
    shape, target_shape = ensemble - level[:, :, None], truth - target_level[:, None]
    lo, hi = int(np.ceil(.05 * ensemble.shape[1]))-1, int(np.ceil(.95 * ensemble.shape[1]))-1
    output = dict(level_crps=crps(level, target_level), shape_crps=crps(shape, target_shape),
                  center_level_mae=np.abs(center.mean(1)-truth.mean(1)),
                  center_shape_mae=np.abs((center-center.mean(1)[:,None])-target_shape).mean(1),
                  centered_shape_energy_score=shape_energy_score(shape, target_shape))
    for name, draw, target in [('level', level, target_level), ('shape', shape, target_shape)]:
        ordered = np.sort(draw, axis=1); lower, upper = ordered[:, lo], ordered[:, hi]
        output[name+'_coverage90'] = ((target >= lower) & (target <= upper)).reshape(len(draw), -1).mean(1)
        output[name+'_width90'] = (upper-lower).reshape(len(draw), -1).mean(1)
    return output


def score_phase(output: Path, frozen: dict[str, Any], stage: str,
                seal: dict[str, Any]) -> pd.DataFrame:
    output = Path(output)
    if not (output / f'{stage}_prediction_seal.json').is_file() or read(output / f'{stage}_prediction_seal.json') != seal:
        raise ValueError('Complete prediction seal required before evaluation labels')
    check_entries(output, frozen, stage, seal)
    if stage == 'target':
        _target_prediction_gate(output, frozen, seal)
    rows = []
    data = Path(frozen[stage])
    for region in SOURCES if stage == 'source' else TARGETS:
        truth_days, truth = v10._truth_lookup(data, stage, region)
        for e in [e for e in seal['entries'] if e['region'] == region]:
            with np.load(output / e['path'], allow_pickle=False) as a:
                keep, aligned = v10._match_truth(a['origins'], truth_days, truth)
                point, ensemble, center = a['point'][keep], a['ensemble'][keep], a['center'][keep]
                for factor in FACTORS:
                    scaled = np.clip(ensemble.mean(1, keepdims=True)+factor*(ensemble-ensemble.mean(1, keepdims=True)), 0, 1000)
                    metrics = {**v10.score_arrays(point, scaled, aligned), **_extra_metrics(point, scaled, aligned, center)}
                    rows.extend(dict(stage=stage, region=region, origin=pd.Timestamp(int(day), tz='UTC').isoformat(),
                                     lag=7, budget=e['budget'], model=e['model'], factor=factor, semester='H1',
                                     **{k: float(v[i]) for k, v in metrics.items()})
                                for i, day in enumerate(a['origins'][keep]))
    day = pd.DataFrame(rows)
    if day.empty or day.duplicated(list(DAY_KEYS)).any() or not np.isfinite(day[list(METRICS)]).all().all():
        raise ValueError('Invalid V12 scores')
    folder = output / f'{stage}_scores'; folder.mkdir()
    day.to_csv(folder / 'day_metrics.csv', index=False)
    grouped = day.groupby(list(REGION_KEYS), sort=True)
    region = grouped[list(METRICS)].mean().reset_index(); region['days'] = grouped.size().to_numpy()
    region.to_csv(folder / 'region_metrics.csv', index=False)
    region.groupby(['stage', 'lag', 'budget', 'model', 'factor', 'semester'])[list(METRICS)].mean().reset_index().to_csv(folder / 'macro_metrics.csv', index=False)
    write(folder / 'score_seal.json', {p.name: sha(p) for p in sorted(folder.iterdir()) if p.is_file()})
    return pd.read_csv(folder / 'region_metrics.csv')  # canonical serialized values for selection



def selection(table: pd.DataFrame, round_number: int) -> dict[str, Any]:
    """Select on canonical source CSV values only; round 1 locks donor strength."""
    table = table.loc[table.stage.eq('source') & table.semester.eq('H1') & table.lag.eq(7)]
    expected = {(r,c['budget'],c['model'],f) for r in SOURCES
                for c in configurations(round_number) for f in FACTORS}
    actual = {(r.region,r.budget,r.model,float(r.factor)) for _,r in table.iterrows()}
    if len(table)!=len(expected) or actual!=expected:
        raise ValueError('Incomplete source selection grid')
    grouped = table.groupby(['budget','model','factor'],sort=True)
    ranks = grouped[list(METRICS)].mean().reset_index()
    ranks['worst_coverage'] = grouped.empirical_coverage90.min().to_numpy()
    ranks['violation'] = (np.maximum(.90-ranks.empirical_coverage90,0)+
                          np.maximum(ranks.empirical_coverage90-.95,0)+
                          np.maximum(.85-ranks.worst_coverage,0))
    def choose(part: pd.DataFrame, *, crps_only: bool = False) -> dict[str,Any]:
        passed = part.loc[part.violation <= 1e-12]
        candidates = part if crps_only or passed.empty else passed
        keys = ([] if crps_only or not passed.empty else ['violation'])+['crps','budget','model','factor']
        row = candidates.sort_values(keys).iloc[0]
        return dict(budget=str(row.budget),model=str(row.model),factor=float(row.factor),
                    status='passed' if row.violation<=1e-12 else 'failed',
                    source_metrics={k:float(row[k]) for k in (*METRICS,'worst_coverage','violation')})
    selections = {f"{c['budget']}--{c['model']}":choose(ranks.loc[
        ranks.budget.eq(c['budget']) & ranks.model.eq(c['model'])]) for c in configurations(round_number)}
    eligible = [c['model'] for c in configurations(round_number) if c['primary_eligible']]
    primary = choose(ranks.loc[ranks.budget.eq('W') & ranks.model.isin(eligible)])
    result = dict(selections=selections,primary=primary,
                  native=selections['B--native_recompose'],simple=selections['B--simple_anchor'],
                  baseline=choose(ranks.loc[ranks.budget.eq('B')]),
                  diagnostic_crps_best=choose(ranks.loc[ranks.budget.eq('W')],crps_only=True))
    if round_number == 1:
        result['locked_alpha'] = ALPHAS[primary['model']]
    return result


def validate_choices(output: Path, frozen: dict) -> dict[str, Any]:
    output = Path(output); choice = read(output/'source_choices.json')
    expected = selection(pd.read_csv(output/'source_scores/region_metrics.csv'),frozen['round'])
    if (any(choice.get(k)!=v for k,v in expected.items()) or choice.get('target_scores_used') is not False
            or choice.get('source_score_seal_sha256')!=sha(output/'source_scores/score_seal.json')
            or choice.get('source_prediction_seal_sha256')!=sha(output/'source_prediction_seal.json')):
        raise ValueError('V12 source choice does not reproduce canonical source scores')
    return choice
def check_score_grid(output: Path, frozen: dict[str, Any], stage: str) -> None:
    folder = Path(output) / f'{stage}_scores'
    day, region, macro = [pd.read_csv(folder / name) for name in ('day_metrics.csv','region_metrics.csv','macro_metrics.csv')]
    expected = {(stage, r, 7, b, m, f, 'H1') for r, b, m in _grid(stage, frozen) for f in FACTORS}
    actual = {tuple(row[k] for k in REGION_KEYS) for _, row in region.iterrows()}
    if len(region) != len(expected) or actual != expected or day.duplicated(list(DAY_KEYS)).any():
        raise ValueError('V12 score grid incomplete/duplicated')
    for r in SOURCES if stage == 'source' else TARGETS:
        groups = list(day.loc[day.region.eq(r)].groupby(['budget','model','factor']))
        support = set(groups[0][1].origin)
        if not support or any(set(g.origin) != support for _, g in groups):
            raise ValueError('V12 unequal scoring support')
    for table, keys, calculated in [
        (region, list(REGION_KEYS), day.groupby(list(REGION_KEYS))[list(METRICS)].mean().reset_index()),
        (macro, ['stage','lag','budget','model','factor','semester'], region.groupby(['stage','lag','budget','model','factor','semester'])[list(METRICS)].mean().reset_index())]:
        joined = table.merge(calculated, on=keys, validate='one_to_one', suffixes=('_saved','_computed'))
        if len(joined) != len(table) or not np.allclose(joined[[m+'_saved' for m in METRICS]], joined[[m+'_computed' for m in METRICS]], atol=1e-10):
            raise ValueError('V12 score aggregation mismatch')



def verify_phase(output: Path, stage: str) -> dict[str, Any]:
    """Validate a completed immutable source or target phase independently."""
    if stage not in ('source','target'):
        raise ValueError('Unknown V12 phase')
    output = Path(output); frozen = verify(output)
    done, seal = read(output/f'{stage}_completed.json'),read(output/f'{stage}_prediction_seal.json')
    for field,path in [('prefit_freeze_sha256','prefit_freeze.json'),
                       ('prediction_seal_sha256',f'{stage}_prediction_seal.json'),
                       ('score_seal_sha256',f'{stage}_scores/score_seal.json'),
                       ('source_choices_sha256','source_choices.json')]:
        if done.get(field)!=sha(output/path):
            raise ValueError('V12 completion detached from '+field)
    if (seal.get('round')!=frozen['round'] or seal.get('evaluation_hourly_labels_read_preseal') is not False
            or seal.get('complete_grid') is not True or done.get('target_training_performed') is not False
            or done.get('target_selection_performed') is not False):
        raise ValueError('V12 information boundary changed')
    check_entries(output,frozen,stage,seal)
    folder = output/f'{stage}_scores'; scoreseal = read(folder/'score_seal.json')
    if set(scoreseal)!={'day_metrics.csv','region_metrics.csv','macro_metrics.csv'}:
        raise ValueError('V12 score manifest incomplete')
    for name,digest in scoreseal.items():
        if sha(folder/name)!=digest:
            raise ValueError('V12 score bytes changed')
    check_score_grid(output,frozen,stage); choice = validate_choices(output,frozen)
    if stage=='target':
        verify_phase(output,'source')
        peer = Path(seal['other_round_source'])
        verify_phase(peer,'source')
        if (done.get('other_round_target_prediction_seal_sha256')!=sha(peer/'target_prediction_seal.json') or
                seal.get('other_round_source_completed_sha256')!=sha(peer/'source_completed.json') or
                seal.get('source_choices_sha256')!=sha(output/'source_choices.json') or
                pd.Timestamp(choice['created_utc'])>pd.Timestamp(seal['created_utc'])):
            raise ValueError('Targets precede both frozen source choices/prediction seals')
        _target_prediction_gate(output,frozen,seal)
    return seal


def _target_source_gate(output: Path, frozen: dict,
                        prior_round: Path | None) -> Path:
    verify_phase(output,'source')
    if frozen['round']==2:
        peer = Path(frozen['prior_round'])
    else:
        peer = Path(prior_round).resolve() if prior_round else output.with_name('weather-increment-v12-round2')
    verify_phase(peer,'source')
    other = verify(peer)
    if other['round']!=3-frozen['round']:
        raise ValueError('Both V12 source rounds must finish before targets')
    if frozen['round']==1 and Path(other['prior_round'])!=output:
        raise ValueError('Round 2 was not locked from this round 1 source')
    return peer



def _target_prediction_gate(output: Path, frozen: dict, seal: dict) -> Path:
    """Require both complete target prediction grids before either label reader."""
    peer = _target_source_gate(output,frozen,Path(seal['other_round_source']))
    if Path(seal['other_round_source'])!=peer:
        raise ValueError('Target prediction seal linked to wrong source round')
    own_path = output/'target_prediction_seal.json'
    peer_path = peer/'target_prediction_seal.json'
    if not own_path.is_file() or read(own_path)!=seal or not peer_path.is_file():
        raise ValueError('Both target prediction seals required before evaluation labels')
    other = verify(peer)
    peer_seal = read(peer_path)
    for folder,manifest,prediction in ((output,frozen,seal),(peer,other,peer_seal)):
        if (prediction.get('round')!=manifest['round'] or
                prediction.get('complete_grid') is not True or
                prediction.get('evaluation_hourly_labels_read_preseal') is not False or
                prediction.get('target_training_performed') is not False or
                prediction.get('target_selection_performed') is not False or
                pd.Timestamp(prediction['created_utc']) < pd.Timestamp(read(folder/'source_choices.json')['created_utc']) or
                prediction.get('source_choices_sha256')!=sha(folder/'source_choices.json')):
            raise ValueError('Invalid complete target prediction seal')
        check_entries(folder,manifest,'target',prediction)
    return peer


def _finish_phase(output: Path, frozen: dict, stage: str, seal: dict,
                  table: pd.DataFrame) -> None:
    if stage=='source':
        write(output/'source_choices.json',dict(created_utc=now(),**selection(table,frozen['round']),
              target_scores_used=False,source_h2_used_for_selection=False,
              source_prediction_seal_sha256=sha(output/'source_prediction_seal.json'),
              source_score_seal_sha256=sha(output/'source_scores/score_seal.json')))
    peer = Path(seal['other_round_source']) if stage=='target' else None
    write(output/f'{stage}_completed.json',dict(created_utc=now(),
          prefit_freeze_sha256=sha(output/'prefit_freeze.json'),
          prediction_seal_sha256=sha(output/f'{stage}_prediction_seal.json'),
          score_seal_sha256=sha(output/f'{stage}_scores/score_seal.json'),
          source_choices_sha256=sha(output/'source_choices.json'),
          other_round_target_prediction_seal_sha256=sha(peer/'target_prediction_seal.json') if peer else None,
          target_training_performed=False,target_selection_performed=False))
    verify_phase(output,stage)


def run_phase(output: Path, data: Path, stage: str, *, round_number: int,
              prior_round: Path | None = None, weather_csv: Path | None = None,
              points_json: Path | None = None, predict_only: bool = False,
              score_only: bool = False) -> None:
    """Run source, or explicitly predict/score targets behind both round seals."""
    if stage not in ('source','target'):
        raise ValueError('Unknown V12 phase')
    if stage=='source' and (predict_only or score_only):
        raise ValueError('Prediction/score mode flags are target-only')
    if stage=='target' and predict_only == score_only:
        raise ValueError('Target requires exactly one of --predict-only or --score-only')
    output,data = Path(output).resolve(),Path(data).resolve()
    if not output.exists():
        if stage!='source':
            raise ValueError('Source phase must precede target')
        freeze(output,data,round_number=round_number,prior_round=prior_round,
               weather_csv=weather_csv,points_json=points_json)
    frozen = verify(output)
    if (frozen['round']!=round_number or Path(frozen['data'])!=data or
            (weather_csv is not None and Path(weather_csv).resolve()!=Path(frozen['weather_csv'])) or
            (points_json is not None and Path(points_json).resolve()!=Path(frozen['points_json']))):
        raise ValueError('Input paths/round differ from freeze')
    peer = _target_source_gate(output,frozen,prior_round) if stage=='target' else None
    if score_only:
        if any((output/f'target_{suffix}').exists() for suffix in ('scores','completed.json')):
            raise FileExistsError('V12 target scores immutable')
        seal = read(output/'target_prediction_seal.json')
        _target_prediction_gate(output,frozen,seal)
        table = score_phase(output,frozen,stage,seal)
        _finish_phase(output,frozen,stage,seal,table)
        return
    if any((output/f'{stage}_{suffix}').exists() for suffix in ('prediction_seal.json','scores','completed.json')):
        raise FileExistsError('V12 stage immutable')
    frames,records = _load_training(frozen,stage)
    queries = ({tuple(sorted(allowed_sources(r))):[r] for r in SOURCES} if stage=='source'
               else {tuple(sorted(SOURCES)):list(TARGETS)})
    shapes = ShapeCache(records,list(queries),queries)
    entries = []
    for subset,regions in queries.items():
        calibrations = {b:_calibration(frames,records,shapes,subset,b,frozen['alpha']) for b in ('B','W')}
        for region in regions:
            record = records[region,'evaluation']
            mixes,draws = _coarse(frozen,stage,region,record['days'])
            results,donors,residuals,audit = _predict_region(frames,records,shapes,subset,
                region,mixes,draws,calibrations,frozen,stage)
            for (budget,mode),prediction in results.items():
                path = output/'predictions'/f'{stage}-{region}-{budget}-{mode}.npz'
                if path.exists():
                    raise FileExistsError(path)
                raw = prediction['raw']; ensemble = np.clip(raw,0,1000)
                arrays = dict(origins=record['days'].as_unit('ns').asi8,point=ensemble.mean(1),
                              ensemble=ensemble,center=prediction['center'],raw_ensemble=raw,
                              donor_indices=donors,residual_indices=residuals)
                if 'conditional_residual_indices' in prediction:
                    arrays['conditional_residual_indices'] = prediction['conditional_residual_indices']
                    arrays['raw_conditional_quantiles'] = prediction['raw_conditional_quantiles']
                    arrays['quantile_levels'] = prediction['quantile_levels']
                    for key in ('conditional_weights','calibration_errors','calibration_regions',
                                'calibration_origins','residual_uniforms'):
                        arrays[key] = prediction[key]
                np.savez_compressed(path,**arrays)
                entries.append(dict(stage=stage,region=region,lag=7,budget=budget,model=mode,
                    path=str(path.relative_to(output)),sha256=sha(path),audit={**audit,
                    'donor_q_mean_max_error':prediction.get('amplitude_error',0.),
                    'sampled_zero_creation':prediction.get('sampled_zero_creation'),
                    'raw_clip_fraction':float(np.mean(raw!=ensemble)),
                    'clip_point_shift_mae':float(np.abs(raw.mean(1)-ensemble.mean(1)).mean())}))
            print('Predicted V12 round',round_number,stage,region,flush=True)
    verify(output)
    seal = dict(created_utc=now(),round=round_number,entries=entries,complete_grid=True,
                evaluation_hourly_labels_read_preseal=False,source_2024_labels_only_preseal=True,
                target_training_performed=False,target_selection_performed=False,
                source_choices_sha256=sha(output/'source_choices.json') if stage=='target' else None,
                shape_fit_audits=shapes.audit,
                other_round_source=str(peer) if peer else None,
                other_round_source_completed_sha256=sha(peer/'source_completed.json') if peer else None)
    check_entries(output,frozen,stage,seal)
    write(output/f'{stage}_prediction_seal.json',seal)
    if predict_only:
        return
    table = score_phase(output,frozen,stage,seal)
    _finish_phase(output,frozen,stage,seal,table)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--round1',action='store_true')
    group.add_argument('--round2',action='store_true')
    parser.add_argument('--phase',choices=['source','target'],required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--predict-only',action='store_true')
    mode.add_argument('--score-only',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--prior-round',type=Path)
    parser.add_argument('--data',type=Path,default=ROOT/'results/information-value-v10-data')
    parser.add_argument('--weather-csv',type=Path)
    parser.add_argument('--points-json',type=Path)
    args = parser.parse_args()
    run_phase(args.output,args.data,args.phase,round_number=1 if args.round1 else 2,
              prior_round=args.prior_round,weather_csv=args.weather_csv,points_json=args.points_json,
              predict_only=args.predict_only,score_only=args.score_only)


if __name__=='__main__':
    main()
