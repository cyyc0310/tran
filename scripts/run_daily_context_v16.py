"""Source-only delayed aggregate-context pilot, isolated from hourly scoring."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]

import numpy as np
import pandas as pd

from scripts.daily_context_v16 import (
    ARMS, EF, FORECASTERS, FUELS, DailyHistory, analog_weights, conditional_ids,
    daily_law, normalized_shapes, particle_analog_weights, ratio, sample_indices,
)
from scripts.run_information_value_v10 import score_arrays
from transcif.research.dynamic_v2 import daily_profiles, read_year
from transcif.research.fuel_transport_v7 import FuelTrajectoryBank, recompose_paths

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
LAGS, FACTORS, SEED, SAMPLES = (7, 30), (.75, 1., 1.25, 1.5, 2.), 17023, 128
OUT = ROOT/'results/daily-context-v16'
OLD = ROOT/'results/weather-coarse-v14-round1/prefit_freeze.json'
OLD_MANIFEST = ROOT/'docs/evidence/weather-coarse-v14-verification.json'
METRICS = ('crps', 'empirical_coverage90', 'empirical_width90', 'empirical_interval_score90',
           'empirical_lower_tail_miss', 'empirical_upper_tail_miss', 'point_level_mae', 'point_shape_mae')
BASELINES = ('additive', 'monthly_q', 'daily_native_ids', 'daily_analog', 'forecast_analog', 'particle_analog')
CANDIDATES = ('innovation_carbon', 'innovation_full')


def sha(path: str | Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write(path: Path, value: dict[str, Any]) -> None:
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def verify(hashes: dict[str, str]) -> None:
    for name, expected in hashes.items():
        if sha(name) != expected:
            raise ValueError('Frozen bytes changed: '+name)


def read_proper(path: Path) -> pd.DataFrame:
    """Filter proper timestamps as text before any numeric hourly parsing."""
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [row for row in reader if '2024-01-25' <= row[0] < '2024-07-01']
    buffer = StringIO()
    writer = csv.writer(buffer); writer.writerow(header); writer.writerows(rows)
    buffer.seek(0)
    frame = pd.read_csv(buffer)
    frame['time'] = pd.to_datetime(frame.time,utc=True,errors='raise')
    return frame


def prepare() -> None:
    """Freeze code before isolating source daily aggregates from hourly labels."""
    old = json.loads(OLD.read_text())
    if old['sources'] != list(SOURCES):
        raise ValueError('Source registry changed')
    code = dict(old['code_hashes'])
    for rel in ('scripts/daily_context_v16.py', 'scripts/run_daily_context_v16.py',
                'tests/test_daily_context_v16.py', 'docs/DAILY_CONTEXT_V16_PROTOCOL.md'):
        p = ROOT/rel
        code[str(p)] = sha(p)
    prior_manifest = json.loads(OLD_MANIFEST.read_text())
    paths = [OLD, OLD_MANIFEST]
    for region in SOURCES:
        paths += [Path(old['source'])/f'{region}_hourly.csv',
                  Path(old['coarse_paths'][f'source/{region}']),
                  ROOT/f'results/weather-coarse-v14-round1/predictions/source-{region}-B-native_recompose.npz']
    inputs = {str(p): sha(p) for p in paths}
    verify(code)
    for region in SOURCES:
        p = str(Path(old['source'])/f'{region}_hourly.csv')
        if inputs[p] != old['input_hashes'][p]:
            raise ValueError('Changed inherited source data')
        p = old['coarse_paths'][f'source/{region}']
        if inputs[p] != old['input_hashes'][p]:
            raise ValueError('Changed inherited coarse particles')
        rel = f'results/weather-coarse-v14-round1/predictions/source-{region}-B-native_recompose.npz'
        if inputs[str(ROOT/rel)] != prior_manifest['artifact_hashes'][rel]:
            raise ValueError('Changed sealed V14 native control')
    OUT.mkdir(exist_ok=True)
    frozen = dict(created_utc=now(), version=16, kind='source_only_daily_aggregate_pilot',
        old_freeze=str(OLD), source=old['source'], sources=list(SOURCES), lags=list(LAGS),
        forecasters=list(FORECASTERS), arms=list(ARMS), factors=list(FACTORS), samples=SAMPLES,
        source_calendar='2025H1; proper donors 2024Jan25-Jun30; local released daily feedback',
        target_values_opened=False, unseen_values_opened=False,
        additional_information='daily eight-fuel energy totals with simulated 7/30d delay',
        original_publication_vintages_verified=False,
        isolated_aggregator_reads_source_hourly_values=True,
        prediction_module_reads_evaluation_hourly_values=False,
        code_hashes=code, input_hashes=inputs)
    write(OUT/'freeze.json', frozen)
    freeze_sha = sha(OUT/'freeze.json')
    records = []
    for region in SOURCES:
        path = Path(old['source'])/f'{region}_hourly.csv'
        with path.open(newline='') as stream:
            selected = [r for r in csv.DictReader(stream)
                        if '2022-' <= r['time'] < '2025-07' and r['region'] == region]
        frame = pd.DataFrame(selected)
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True, errors='raise'))
        if times.has_duplicates or not times.equals(times.floor('h')) or not set(frame.valid) <= {'True','False'}:
            raise ValueError('Unique complete UTC source labels required')
        for day in times.floor('D').unique():
            take = np.flatnonzero(times.floor('D') == day)
            if len(take) != 24 or not frame.iloc[take].valid.eq('True').all():
                continue
            a = frame.iloc[take][[f'generation_{f}' for f in FUELS]].to_numpy(float)
            if not np.isfinite(a).all() or np.any(a < 0) or np.any(a.sum(1) <= 0):
                raise ValueError('Bad source energy for coarse aggregation')
            row = dict(region=region, day=day.isoformat(), observed_hours=24)
            row.update({f'energy_{f}': float(a[:, i].sum()) for i, f in enumerate(FUELS)})
            records.append(row)
    table = pd.DataFrame(records).sort_values(['region','day'])
    table.to_csv(OUT/'daily_fuels.csv', index=False, mode='x')
    verify(code); verify(inputs); verify({str(OUT/'freeze.json'):freeze_sha})
    write(OUT/'data_seal.json', dict(created_utc=now(), freeze_sha256=freeze_sha,
                                   daily_sha256=sha(OUT/'daily_fuels.csv'), rows=len(table),
                                   source_hourly_scoring_not_performed=True))


def check() -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the input stream, producer code and frozen protocol before any phase."""
    frozen = json.loads((OUT/'freeze.json').read_text())
    seal = json.loads((OUT/'data_seal.json').read_text())
    if (frozen['version'] != 16 or frozen['kind'] != 'source_only_daily_aggregate_pilot'
            or frozen['sources'] != list(SOURCES) or frozen['lags'] != list(LAGS)
            or frozen['forecasters'] != list(FORECASTERS) or frozen['arms'] != list(ARMS)
            or frozen['factors'] != list(FACTORS) or frozen['samples'] != SAMPLES
            or frozen['target_values_opened'] or frozen['unseen_values_opened']
            or seal['freeze_sha256'] != sha(OUT/'freeze.json')
            or seal['daily_sha256'] != sha(OUT/'daily_fuels.csv')):
        raise ValueError('Frozen daily-context contract mismatch')
    verify(frozen['code_hashes']); verify(frozen['input_hashes'])
    return frozen, seal


def check_predictions(seal: dict[str, Any]) -> None:
    """Require the exact full grid and its hashes before opening hourly truth."""
    expected = {(r,0,'monthly') for r in SOURCES} | {
        (r,lag,method) for r in SOURCES for lag in LAGS for method in FORECASTERS}
    actual = [(e['region'],e['lag'],e['forecaster']) for e in seal['entries']]
    if (len(actual) != len(expected) or set(actual) != expected or seal['complete_grid'] is not True
            or seal['source_hourly_2025_scoring_read_by_prediction'] is not False
            or seal['isolated_aggregation_hourly_read_disclosed'] is not True
            or seal['target_values_opened'] is not False):
        raise ValueError('Exact complete source prediction grid required')
    for entry in seal['entries']:
        p = Path(entry['path']).resolve()
        if p.parent != (OUT/'predictions').resolve() or entry['arms'] != (['native'] if entry['lag'] == 0 else list(ARMS)):
            raise ValueError('Unregistered prediction path or arm list')
    verify({e['path']:e['sha256'] for e in seal['entries']})


def predict() -> None:
    """Produce the complete source grid from proper labels and delayed summaries."""
    frozen, _ = check()
    initial = {str(OUT/p): sha(OUT/p) for p in ('freeze.json','data_seal.json','daily_fuels.csv')}
    if (OUT/'prediction_seal.json').exists() or (OUT/'predictions').exists():
        raise ValueError('Preserve existing prediction attempt')
    folder = OUT/'predictions'; folder.mkdir()
    old = json.loads(OLD.read_text())
    table = pd.read_csv(OUT/'daily_fuels.csv')
    frames = {}
    for r in SOURCES:
        frames[r] = read_proper(Path(frozen['source'])/f'{r}_hourly.csv')
    histories = {(r, lag): DailyHistory(table.loc[table.region.eq(r)], lag) for r in SOURCES for lag in LAGS}
    entries = []
    diagnostics = []
    for region in SOURCES:
        subset = tuple(r for r in SOURCES if r != region)
        bank = FuelTrajectoryBank(subset).fit({r: frames[r] for r in subset})
        source_native = ROOT/f'results/weather-coarse-v14-round1/predictions/source-{region}-B-native_recompose.npz'
        with np.load(source_native, allow_pickle=False) as z:
            origins, native, native_ids = z['origins'], z['raw_ensemble'], z['donor_indices']
        days = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
        with np.load(old['coarse_paths'][f'source/{region}'], allow_pickle=False) as z:
            idx = pd.Index(z['origins']).get_indexer(origins)
            if np.any(idx < 0): raise ValueError('Missing inherited coarse date')
            old_draws = z['draws'][idx]
        reconstructed = np.stack([recompose_paths(m, bank.q_[d]) for m,d in zip(old_draws,native_ids)])
        if not np.array_equal(native,reconstructed):
            raise ValueError('V14 exact native identity failed: '+region)
        native_copy = folder/f'{region}-native.npz'
        np.savez_compressed(native_copy, origins=origins, raw=native, donor_ids=native_ids)
        entries.append(dict(region=region, lag=0, forecaster='monthly', path=str(native_copy),
                            arms=['native'], sha256=sha(native_copy)))
        q = normalized_shapes(bank.q_)
        amplitude = bank.q_.mean(1)
        true_day_mix = bank.donor_mix_*amplitude
        true_day_mix /= true_day_mix.sum(1,keepdims=True)
        for lag in LAGS:
            for method in FORECASTERS:
                teachers = np.stack([histories[r,lag].forecast(d,method)[0] for r,d in zip(bank.regions_,bank.days_)])
                donor_errors = true_day_mix-teachers
                permuted = donor_errors.copy()
                rng = np.random.default_rng(SEED)
                for donor in subset:
                    for month in range(1,7):
                        pos = np.flatnonzero((bank.regions_ == donor) & (bank.months_ == month))
                        permuted[pos] = donor_errors[rng.permutation(pos)]
                raw = np.empty((len(ARMS),len(days),SAMPLES,24))
                ids_all = np.empty((len(ARMS),len(days),SAMPLES),dtype=np.int64)
                ess_all = np.full((len(ARMS),len(days),SAMPLES),np.nan)
                points, mixtures, error_days, error_release, innovations, original_innovations = [], [], [], [], [], []
                for i, origin in enumerate(days):
                    uniforms = bank._rng(SEED,origin).random((SAMPLES,8))
                    law = daily_law(histories[region,lag],origin,method,uniforms[:,1])
                    m, shock = law['draws'], law['innovations']
                    weights = analog_weights(law['point'],true_day_mix,bank.months_,bank.regions_,origin)
                    state_ids = sample_indices(weights,uniforms[:,0])
                    forecast_weights = analog_weights(law['point'],teachers,bank.months_,bank.regions_,origin)
                    forecast_ids = sample_indices(forecast_weights,uniforms[:,0])
                    particle_weights = particle_analog_weights(m,true_day_mix,bank.months_,bank.regions_,origin)
                    particle_ids = np.array([sample_indices(w,u) for w,u in zip(particle_weights,uniforms[:,0])])
                    carbon_ids, carbon_ess = conditional_ids(forecast_weights,donor_errors,shock,law['local_applied_errors'],uniforms[:,0])
                    full_ids, full_ess = conditional_ids(forecast_weights,donor_errors,shock,law['local_applied_errors'],uniforms[:,0],carbon=False)
                    null_ids, null_ess = conditional_ids(forecast_weights,permuted,shock,law['local_applied_errors'],uniforms[:,0])
                    choices = [native_ids[i],native_ids[i],native_ids[i],state_ids,forecast_ids,particle_ids,
                               carbon_ids,full_ids,null_ids]
                    for ai,(arm,donors) in enumerate(zip(ARMS,choices)):
                        ids_all[ai,i] = donors
                        if arm == 'additive':
                            baseline_shape = native[i]-native[i].mean(1,keepdims=True)
                            raw[ai,i] = baseline_shape+(m@EF)[:,None]
                        elif arm == 'monthly_q':
                            raw[ai,i] = ratio(m,bank.q_[donors])
                        else:
                            raw[ai,i] = ratio(m,q[donors])
                    ess_all[6,i],ess_all[7,i],ess_all[8,i] = carbon_ess,full_ess,null_ess
                    points.append(law['point']); mixtures.append(m)
                    chosen = law['error_indices']
                    error_days.append(law['error_days'][chosen]); error_release.append(law['error_release'][chosen])
                    innovations.append(shock); original_innovations.append(law['raw_innovations'])
                    diagnostics.append(dict(region=region,lag=lag,forecaster=method,origin=origin.isoformat(),
                        point_history_count=law['point_history_count'],released_error_count=len(law['errors']),
                        latest_selected_response_release=int(law['error_release'][chosen].max()),
                        mean_projection_change=float(abs(shock-law['raw_innovations']).mean()),
                        mean_carbon_ess=float(carbon_ess.mean()),mean_full_ess=float(full_ess.mean()),
                        unique_carbon_donors=int(len(np.unique(carbon_ids))),unique_state_donors=int(len(np.unique(state_ids)))))
                path = folder/f'{region}-{lag}-{method}.npz'
                np.savez_compressed(path, origins=origins, arms=np.array(ARMS), raw=raw, donor_ids=ids_all,
                    effective_donors=ess_all, coarse_points=np.stack(points), coarse_draws=np.stack(mixtures),
                    selected_error_days=np.stack(error_days),selected_response_release=np.stack(error_release),
                    applied_innovations=np.stack(innovations),raw_innovations=np.stack(original_innovations),
                    donor_errors=donor_errors,permuted_donor_errors=permuted,
                    proper_donor_regions=np.array(bank.regions_,dtype='U'),proper_donor_days=bank.days_.as_unit('ns').asi8)
                entries.append(dict(region=region,lag=lag,forecaster=method,path=str(path),arms=list(ARMS),sha256=sha(path)))
                print(f'Predicted {region} lag{lag} {method}',flush=True)
    pd.DataFrame(diagnostics).to_csv(OUT/'prediction_diagnostics.csv',index=False,mode='x')
    if len(entries) != 63: raise ValueError('Incomplete source prediction grid')
    check(); verify(initial)
    write(OUT/'prediction_seal.json',dict(created_utc=now(),entries=entries,
        freeze_sha256=initial[str(OUT/'freeze.json')],
        diagnostics_sha256=sha(OUT/'prediction_diagnostics.csv'),
        complete_grid=True,source_hourly_2025_scoring_read_by_prediction=False,
        isolated_aggregation_hourly_read_disclosed=True,target_values_opened=False))


def select(region: pd.DataFrame) -> dict[str, Any]:
    """Source development selection only; retain failure instead of switching gate."""
    rows = []
    for key, part in region.groupby(['lag','forecaster','arm','factor'],sort=True):
        macro = part[list(METRICS)].mean().to_dict()
        macro.update(lag=int(key[0]),forecaster=key[1],arm=key[2],factor=float(key[3]),
                     worst_coverage=float(part.empirical_coverage90.min()))
        macro['passed'] = .9 <= macro['empirical_coverage90'] <= .95 and macro['worst_coverage'] >= .85
        macro['violation'] = max(.9-macro['empirical_coverage90'],0)+max(macro['empirical_coverage90']-.95,0)+max(.85-macro['worst_coverage'],0)
        rows.append(macro)
    macro = pd.DataFrame(rows)
    def best(names: tuple[str,...]) -> dict[str,Any]:
        part = macro.loc[(macro.lag.eq(7)|macro.lag.eq(0)) & macro.arm.isin(names)]
        eligible = part.loc[part.passed]
        choice = (eligible if len(eligible) else part).sort_values(
            ['crps','arm','forecaster','factor'] if len(eligible) else ['violation','crps','arm','forecaster','factor']).iloc[0]
        return choice.to_dict()
    native, baseline, candidate = best(('native',)),best(BASELINES),best(CANDIDATES)
    references = (native,baseline)
    gate = bool(candidate['passed'] and all(candidate['crps'] <= .95*r['crps']
        and candidate['empirical_width90'] <= r['empirical_width90']
        and candidate['empirical_interval_score90'] < r['empirical_interval_score90'] for r in references))
    return dict(created_utc=now(),native=native,strong_baseline=baseline,candidate=candidate,
                source_pilot_gate=gate,target_test_authorized_by_this_pilot=False,
                independent_confirmation=False,full_same_budget_conditional_risk_baselines_not_yet_run=True)


def score() -> None:
    """Only after every source prediction is sealed, read source H1 CIF labels."""
    frozen,_ = check()
    seal = json.loads((OUT/'prediction_seal.json').read_text())
    if (seal['freeze_sha256'] != sha(OUT/'freeze.json') or not seal['complete_grid']
            or seal['diagnostics_sha256'] != sha(OUT/'prediction_diagnostics.csv') or len(seal['entries']) != 63):
        raise ValueError('Complete unchanged prediction seal required')
    check_predictions(seal)
    initial = {str(OUT/p):sha(OUT/p) for p in ('freeze.json','data_seal.json','prediction_seal.json')}
    rows = []
    for region in SOURCES:
        dates,truth = daily_profiles(read_year(Path(frozen['source'])/f'{region}_hourly.csv',2025))
        dates = dates.as_unit('ns').asi8
        for entry in [e for e in seal['entries'] if e['region'] == region]:
            with np.load(entry['path'],allow_pickle=False) as z:
                index = pd.Index(dates).get_indexer(z['origins']); keep = index >= 0
                aligned,origins = truth[index[keep]],z['origins'][keep]
                ensemble = z['raw'][None] if entry['arms'] == ['native'] else z['raw']
                for arm,raw in zip(entry['arms'],ensemble):
                    paths = np.clip(raw[keep],0,1000); point = paths.mean(1)
                    for factor in FACTORS:
                        scaled = np.clip(point[:,None]+factor*(paths-point[:,None]),0,1000)
                        metrics = score_arrays(point,scaled,aligned)
                        for i,origin in enumerate(origins):
                            rows.append(dict(region=region,origin=pd.Timestamp(int(origin),tz='UTC').isoformat(),
                                lag=entry['lag'],forecaster=entry['forecaster'],arm=arm,factor=factor,
                                **{m:float(metrics[m][i]) for m in METRICS}))
        print('Scored',region,flush=True)
    day = pd.DataFrame(rows)
    if (len(day[['region','origin']].drop_duplicates()) != 1145
            or day.duplicated(['region','origin','lag','forecaster','arm','factor']).any()
            or not np.isfinite(day[list(METRICS)].to_numpy()).all()):
        raise ValueError('Invalid/mismatched source scoring cohort')
    day.to_csv(OUT/'day_metrics.csv',index=False,mode='x')
    keys = ['region','lag','forecaster','arm','factor']
    regional = day.groupby(keys)[list(METRICS)].mean().reset_index()
    regional.to_csv(OUT/'region_metrics.csv',index=False,mode='x')
    regional.groupby(keys[1:])[list(METRICS)].mean().reset_index().to_csv(OUT/'macro_metrics.csv',index=False,mode='x')
    selected = select(pd.read_csv(OUT/'region_metrics.csv'))
    write(OUT/'source_choice.json',selected)
    check(); verify(initial)
    write(OUT/'source_completed.json',dict(created_utc=now(),version=16,
        prediction_seal_sha256=sha(OUT/'prediction_seal.json'),
        output_hashes={str(OUT/p):sha(OUT/p) for p in ('day_metrics.csv','region_metrics.csv','macro_metrics.csv','source_choice.json')},
        target_values_opened=False,unseen_values_opened=False,source_pilot_gate=selected['source_pilot_gate']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['prepare','predict','score'],required=True)
    args = parser.parse_args()
    {'prepare':prepare,'predict':predict,'score':score}[args.stage]()
