"""Source-first structural oracle diagnostic; never a forecasting benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.identifiability_v15 import (
    BUDGETS, EF, EF_STRESS, GROUPS, GROUP_ONLY, group_only_bounds, oracle_bounds, read_evaluation,
)

SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
TARGETS = ('AZPS', 'DUK', 'LGEE', 'PACE', 'PSCO', 'SRP')
OUT = ROOT/'results/identifiability-v15'
OLD = ROOT/'results/weather-coarse-v14-round1/prefit_freeze.json'
SINGLE = ('day_coal', 'day_gas', 'day_emitting')
DOUBLE = ('day_coal_gas', 'day_emitting_coal')
CLOSURE = ('scripts/identifiability_v15.py', 'scripts/run_identifiability_v15.py',
           'tests/test_identifiability_v15.py', 'docs/IDENTIFIABILITY_V15_PROTOCOL.md')


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
    for path, expected in hashes.items():
        if sha(path) != expected:
            raise ValueError('Changed frozen file: '+path)


def prepare() -> dict[str, Any]:
    old = json.loads(OLD.read_text())
    if old['sources'] != list(SOURCES) or old['targets'] != list(TARGETS):
        raise ValueError('Wrong inherited region registry')
    paths = [ROOT/p for p in CLOSURE] + [OLD]
    data = {}
    for stage, regions in (('source', SOURCES), ('target', TARGETS)):
        data[stage] = {}
        for region in regions:
            path = Path(old[stage])/f'{region}_hourly.csv'
            if sha(path) != old['input_hashes'][str(path)]:
                raise ValueError('Changed V14 registered input')
            data[stage][region] = str(path)
            paths.append(path)
    frozen = dict(version=15, created_utc=now(), kind='retrospective_structural_oracle',
                  source_year=2025, target_year=2026, data=data,
                  sources=list(SOURCES), targets=list(TARGETS),
                  budgets=['past_only']+[b[0] for b in BUDGETS]+list(GROUP_ONLY),
                  ef={'proxy': EF.tolist(), 'algebraic_stress': EF_STRESS.tolist()},
                  target_selection=False, unseen_region_values_opened=False,
                  input_hashes={str(p): sha(p) for p in paths})
    OUT.mkdir(exist_ok=True)
    write(OUT/'freeze.json', frozen)
    return frozen


def summarize(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ['ef', 'budget', 'region']
    metrics = ['hour_width', 'hour_width_q95', 'resolved_hour_fraction100',
               'daily_mean_width', 'abs_mean_weighted_gap']
    region = daily.groupby(keys, sort=True)[metrics].mean().reset_index()
    counts = daily.groupby(keys).size().rename('days').reset_index()
    region = region.merge(counts, on=keys, validate='one_to_one')
    macro = region.groupby(['ef', 'budget'], sort=True)[metrics].mean().reset_index()
    return region, macro


def choose(region: pd.DataFrame, macro: pd.DataFrame) -> dict[str, Any]:
    primary = macro.loc[macro.ef.eq('proxy')].set_index('budget')
    best = lambda names: min(names, key=lambda n: (float(primary.loc[n, 'hour_width']), n))
    eligible = []
    for name in SINGLE+DOUBLE:
        rows = region.loc[region.ef.eq('proxy') & region.budget.eq(name)]
        if (len(rows) == len(SOURCES) and rows.resolved_hour_fraction100.ge(.95).all()
                and rows.daily_mean_width.le(25).all()):
            eligible.append(name)
    smallest = min(eligible, key=lambda n: (1 if n in SINGLE else 2,
                    float(primary.loc[n, 'hour_width']), n)) if eligible else None
    return dict(created_utc=now(), single_channel=best(SINGLE), two_channels=best(DOUBLE),
                smallest_tested_resolution_budget=smallest, eligible_budgets=sorted(eligible),
                thresholds={'each_region_hour_fraction_width_le100': .95,
                            'each_region_mean_daily_level_width_le': 25},
                selection_scope='Source main EF only; conditional on exact daily fuel totals and hourly G',
                independent_confirmation=False)


def run(stage: str) -> None:
    if stage not in ('source', 'target'):
        raise ValueError('Only registered source/target diagnostic stages are allowed')
    marker = OUT/f'{stage}_completed.json'
    if marker.exists() or (OUT/f'{stage}_daily.csv').exists():
        raise ValueError('Completed/partial output exists; preserve it')
    frozen = prepare() if stage == 'source' else json.loads((OUT/'freeze.json').read_text())
    freeze_sha = sha(OUT/'freeze.json')
    verify(frozen['input_hashes'])
    source_receipts = {}
    if stage == 'target':
        done = json.loads((OUT/'source_completed.json').read_text())
        if (done['stage'] != 'source' or done['version'] != 15
                or done['freeze_sha256'] != freeze_sha
                or done['prediction_performed'] is not False):
            raise ValueError('Source receipt does not bind the current diagnostic freeze')
        verify(done['output_hashes'])
        source_receipts = {str(OUT/'source_completed.json'): sha(OUT/'source_completed.json')}
        locked = json.loads((OUT/'source_choice.json').read_text())
    else:
        locked = None
    rows, outputs, identities = [], [], []
    regions = SOURCES if stage == 'source' else TARGETS
    for region in regions:
        days, a = read_evaluation(Path(frozen['data'][stage][region]), region, frozen[f'{stage}_year'])
        h, n = len(a), len(days)
        truth = (a @ EF)/a.sum(1)
        times = days.repeat(24)+pd.to_timedelta(np.tile(np.arange(24), n), unit='h')
        month = times.strftime('%Y-%m').to_numpy()
        archives = {'days': days.as_unit('ns').asi8, 'generation': a,
                    'ef_names': np.array(list(frozen['ef'])),
                    'budgets': np.array(frozen['budgets'])}
        bounds, mean_bounds = [], []
        for ef_name, ef_values in frozen['ef'].items():
            ef = np.array(ef_values)
            actual = (a @ ef)/a.sum(1)
            hourly, means = [], []
            for name in frozen['budgets']:
                if name == 'past_only':
                    lo, hi = np.full(h, ef.min()), np.full(h, ef.max())
                    ml, mh = np.full(n, ef.min()), np.full(n, ef.max())
                elif name in GROUP_ONLY:
                    lo, hi, ml, mh = group_only_bounds(a, GROUPS[GROUP_ONLY[name]], ef)
                else:
                    _, size, group = next(b for b in BUDGETS if b[0] == name)
                    blocks = month if size == 0 else np.arange(h)//size
                    lo, hi, ml, mh = oracle_bounds(a, blocks, GROUPS[group], ef)
                if (np.any(lo > actual+1e-7) or np.any(hi < actual-1e-7)
                        or np.any(ml > actual.reshape(n,24).mean(1)+1e-7)
                        or np.any(mh < actual.reshape(n,24).mean(1)-1e-7)
                        or np.any(hi < lo-1e-7) or np.any(mh < ml-1e-7)):
                    raise ValueError('Oracle truth containment/ordering failed')
                hourly.append(np.stack([lo, hi]))
                means.append(np.stack([ml, mh]))
                width = np.maximum(hi-lo, 0.).reshape(n, 24)
                tables = a.reshape(n, 24, 8)
                weighted = (tables.sum(1) @ ef)/tables.sum((1, 2))
                arithmetic = actual.reshape(n, 24).mean(1)
                for i, day in enumerate(days):
                    rows.append(dict(stage=stage, region=region, day=day.isoformat(), month=day.strftime('%Y-%m'),
                        ef=ef_name, budget=name, hour_width=float(width[i].mean()),
                        hour_width_q95=float(np.quantile(width[i], .95)),
                        resolved_hour_fraction100=float(np.mean(width[i] <= 100+1e-8)),
                        daily_mean_lower=float(ml[i]), daily_mean_upper=float(mh[i]),
                        daily_mean_width=float(max(mh[i]-ml[i], 0.)),
                        true_arithmetic_mean=float(arithmetic[i]), true_weighted_mean=float(weighted[i]),
                        abs_mean_weighted_gap=float(abs(arithmetic[i]-weighted[i]))))
            bounds.append(np.stack(hourly))
            mean_bounds.append(np.stack(means))
            for value in np.unique(month):
                take = month == value
                weighted = float((a[take].sum(0) @ ef)/a[take].sum())
                identity = float(np.average(actual[take], weights=a[take].sum(1)))
                identities.append(dict(region=region, ef=ef_name, month=value,
                    observed_hours=int(take.sum()), weighted_C=weighted,
                    arithmetic_C=float(actual[take].mean()), identity_error=abs(identity-weighted)))
        archives.update(hour_bounds=np.stack(bounds), daily_mean_bounds=np.stack(mean_bounds), proxy_truth=truth)
        path = OUT/f'{stage}-{region}-bounds.npz'
        if path.exists():
            raise ValueError('Archive already exists')
        np.savez_compressed(path, **archives)
        outputs.append(path)
        print(f'{stage} {region}: {n} days; all budgets and EF sensitivities complete', flush=True)
    daily = pd.DataFrame(rows)
    region, macro = summarize(daily)
    expected = 1145 if stage == 'source' else 1080
    if len(daily[['region', 'day']].drop_duplicates()) != expected:
        raise ValueError('Changed V14 evaluation cohort')
    for name, data in [('daily', daily), ('region', region), ('macro', macro),
                       ('monthly_identity', pd.DataFrame(identities))]:
        path = OUT/f'{stage}_{name}.csv'
        data.to_csv(path, index=False, mode='x')
        outputs.append(path)
    if stage == 'source':
        write(OUT/'source_choice.json', choose(region, macro))
        outputs.append(OUT/'source_choice.json')
    else:
        if locked != json.loads((OUT/'source_choice.json').read_text()):
            raise ValueError('Source choice changed during target diagnosis')
    verify(frozen['input_hashes'])
    verify({str(OUT/'freeze.json'): freeze_sha})
    verify(source_receipts)
    write(marker, dict(created_utc=now(), stage=stage, version=15,
        kind='retrospective_structural_oracle', valid_region_days=expected,
        source_receipt_hashes=source_receipts, freeze_sha256=freeze_sha,
        output_hashes={str(path): sha(path) for path in outputs},
        prediction_performed=False, unseen_region_values_opened=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['source', 'target'], required=True)
    run(parser.parse_args().stage)
