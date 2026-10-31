"""Exact source-only query-budget diagnostic; no forecasts or new-region data."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/local-hourly-query-v23'
V20 = ROOT / 'results/tail-count-v20'
PARENT_MANIFEST = ROOT / 'docs/evidence/tail-count-v20-verification.json'
SOURCES = ('BPAT', 'CISO', 'ERCO', 'FPL', 'ISNE', 'MISO', 'NYIS')
FUELS = ('coal', 'gas', 'oil', 'nuclear', 'hydro', 'wind', 'solar', 'other')
EF = (1000, 400, 800, 0, 0, 0, 0, 400)
TAUS = (300, 100)
BUDGETS = (0, 1, 2, 4)
STRATEGIES = ('uniform_clock', 'seeded_random', 'widest_feasible_CIF', 'one_step_minimax_Kspan')
PRECISIONS = ('exact', '10g_interval')
B = F(1000)
CODE = ('scripts/local_hourly_query_v23.py', 'docs/LOCAL_HOURLY_QUERY_BUDGET_V23_PROTOCOL.md')


def sha(path: Path) -> str:
    return hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json_new(path: Path, obj: dict) -> None:
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parent_check() -> tuple[dict, dict, set[tuple[str, str]]]:
    manifest = read_json(PARENT_MANIFEST)
    if sha(PARENT_MANIFEST) != '7c2311d8dc8224778b45ea41a5386df5c7e5e5bea8c72d80db842d43b6cbf7d2':
        raise ValueError('V20 parent manifest digest changed')
    for mapping in ('frozen_code_hashes', 'frozen_input_hashes', 'artifact_hashes'):
        for name, digest in manifest[mapping].items():
            p = Path(name) if Path(name).is_absolute() else ROOT / name
            if sha(p) != digest:
                raise ValueError('Changed V20 evidence closure: ' + name)
    freeze = read_json(V20 / 'freeze.json')
    done = read_json(V20 / 'completed.json')
    if (done['freeze_sha256'] != sha(V20 / 'freeze.json') or done['region_days'] != 1000
            or done['target_values_opened'] is not False or done['unseen_values_opened'] is not False):
        raise ValueError('Completed V20 source-only parent required')
    rows = pd.read_csv(V20 / 'day_counts.csv', usecols=['region', 'day', 'budget'])
    mask = set(rows[rows.budget.eq('T8')][['region', 'day']].itertuples(index=False, name=None))
    if len(mask) != 1000 or {r for r, _ in mask} != set(SOURCES):
        raise ValueError('Inherited V20 1000-day mask changed')
    chosen = set()
    for region in SOURCES:
        days = sorted(d for r, d in mask if r == region)
        if len(days) < 2:
            raise ValueError('Need first and last inherited V20 day')
        chosen.add((region, days[0])); chosen.add((region, days[-1]))
    if len(chosen) != 14:
        raise ValueError('Expected exactly14 fixed first/last source-days')
    return manifest, freeze, chosen


def input_closure() -> dict[str, str]:
    _, freeze, chosen = parent_check()
    paths = [PARENT_MANIFEST, V20/'freeze.json', V20/'completed.json', V20/'day_counts.csv']
    paths += [Path(freeze['source']) / f'{r}_hourly.csv' for r in SOURCES]
    paths += [ROOT / p for p in CODE]
    closure = {str(p): sha(p) for p in paths}
    parent_inputs = read_json(V20/'freeze.json')['input_hashes']
    for r in SOURCES:
        p = str(Path(freeze['source']) / f'{r}_hourly.csv')
        if closure[p] != parent_inputs[p]:
            raise ValueError('Source changed from V20 parent: ' + p)
    return closure


def contract() -> dict:
    return dict(version=23, source_only=True, selected_region_days=14, selected_days_per_source=2,
        sources=list(SOURCES), thresholds=list(TAUS), budgets=list(BUDGETS), strategies=list(STRATEGIES),
        precisions=list(PRECISIONS), support=1000, emission_factors=list(EF),
        base_information='contemporaneous exact T8 H/G plus exact hourly G24 oracle',
        exact_decimal_source_values=True,  # exact only relative to the stored decimal proxy
        independent_confirmation=False, forecast_gain=False, new_method=False,
        mechanism_breakthrough=False, target_authorized=False, unseen_values_opened=False)


def prepare() -> None:
    if any((OUT / n).exists() for n in ('freeze.json', 'completed.json', 'day_results.csv', 'summary.csv')):
        raise FileExistsError('Preserve existing or partial V23 run')
    _, parent, chosen = parent_check()
    OUT.mkdir(parents=True, exist_ok=True)
    frozen = {**contract(), 'created_utc': now(), 'parent_manifest_sha256': sha(PARENT_MANIFEST),
        'parent_completion_sha256': sha(V20/'completed.json'), 'parent_freeze_sha256': sha(V20/'freeze.json'),
        'chosen_region_days': sorted([list(x) for x in chosen]), 'source_directory': parent['source'],
        'code_hashes': {str(ROOT/p): sha(ROOT/p) for p in CODE}, 'input_hashes': input_closure()}
    write_json_new(OUT/'freeze.json', frozen)


def check_freeze() -> dict:
    f = read_json(OUT/'freeze.json')
    if any(f[k] != v for k, v in contract().items()):
        raise ValueError('V23 contract changed')
    _, parent, chosen = parent_check()
    if (f['parent_manifest_sha256'] != sha(PARENT_MANIFEST)
            or f['parent_completion_sha256'] != sha(V20/'completed.json')
            or f['parent_freeze_sha256'] != sha(V20/'freeze.json')
            or f['chosen_region_days'] != sorted([list(x) for x in chosen])
            or f['source_directory'] != parent['source']
            or f['input_hashes'] != input_closure()
            or f['code_hashes'] != {str(ROOT/p): sha(ROOT/p) for p in CODE}):
        raise ValueError('V23 frozen code/input/cohort binding changed')
    return f


def observations_for_status(g: list[F], lo: list[F], hi: list[F], h: F, tau: F,
                            mask: int) -> bool:
    """Whether a binary event assignment can realize the block's exact H."""
    lower = F(0); upper = F(0); lower_open = False
    for j, gj in enumerate(g):
        is_event = bool(mask & (1 << j))
        if is_event:
            l = max(lo[j], tau); u = min(hi[j], B)
            if l > u or (l == u and l == tau):
                return False
            lower += gj*l; upper += gj*u
            lower_open |= (l == tau)
        else:
            l = max(lo[j], F(0)); u = min(hi[j], tau)
            if l > u:
                return False
            lower += gj*l; upper += gj*u
    # The admissible weighted sum is (lower,upper] iff any selected C must be >tau at its lower endpoint.
    return (lower < h if lower_open else lower <= h) and h <= upper


def block_counts(g: list[F], h: F, obs: dict[int, tuple[F, F]], tau: F) -> tuple[int, ...]:
    lo = [obs.get(i, (F(0), B))[0] for i in range(len(g))]
    hi = [obs.get(i, (F(0), B))[1] for i in range(len(g))]
    possible = set()
    for mask in range(1 << len(g)):
        if observations_for_status(g, lo, hi, h, tau, mask):
            possible.add(mask.bit_count())
    if not possible:
        raise ValueError('Observed response intervals excluded every feasible true path')
    return tuple(sorted(possible))


def daily_counts(g: list[F], h: list[F], obs: dict[int, tuple[F, F]], tau: F) -> tuple[int, ...]:
    possible = {0}
    for block in range(4):
        start = block*6
        local_obs = {i-start: iv for i, iv in obs.items() if start <= i < start+6}
        counts = block_counts(g[start:start+6], h[block], local_obs, tau)
        possible = {a+b for a in possible for b in counts}
    return tuple(sorted(possible))


def marginal_interval(g: list[F], h: list[F], obs: dict[int, tuple[F, F]], hour: int) -> tuple[F, F]:
    block, start = hour//6, (hour//6)*6
    local_g = g[start:start+6]; local_h = h[block]; local = {i-start: iv for i, iv in obs.items() if start <= i < start+6}
    ix = hour-start
    min_other = sum((local_g[j]*iv[0] for j, iv in local.items() if j != ix), F(0))
    max_other = sum((local_g[j]*iv[1] for j, iv in local.items() if j != ix), F(0))
    unobserved = [j for j in range(6) if j != ix and j not in local]
    max_other += B*sum((local_g[j] for j in unobserved), F(0))
    low = max(F(0), (local_h-max_other)/local_g[ix])
    high = min(B, (local_h-min_other)/local_g[ix])
    if low > high:
        raise ValueError('Infeasible marginal interval under current answers')
    return low, high


def minimax_exact_candidates(g: list[F], h: list[F], obs: dict[int, tuple[F, F]],
                             candidate: int, tau: F) -> int:
    """One-step minimax over exact answer cells; all breakpoints are rational."""
    lo, hi = marginal_interval(g, h, obs, candidate)
    block, start = candidate//6, (candidate//6)*6
    local_observed = {i: obs[i] for i in range(start, start+6) if i in obs}
    # Feasible candidate values change only at its event threshold or an exact subset-energy boundary.
    cuts = {lo, hi}
    if lo <= tau <= hi:
        cuts.add(tau)
    block_h = h[block]
    others = [i for i in range(start, start+6) if i != candidate]
    for mask in range(1 << len(others)):
        lower=F(0);upper=F(0);valid=True
        for bit,hour in enumerate(others):
            is_event=bool(mask & (1 << bit));qlo,qhi=local_observed.get(hour,(F(0),B))
            if is_event:
                a=max(qlo,tau);z=min(qhi,B)
                if a>z or (a==z==tau): valid=False;break
                lower+=g[hour]*a;upper+=g[hour]*z
            else:
                a=qlo;z=min(qhi,tau)
                if a>z: valid=False;break
                lower+=g[hour]*a;upper+=g[hour]*z
        if not valid: continue
        for boundary in (lower,upper):
            x=(block_h-boundary)/g[candidate]
            if lo<=x<=hi: cuts.add(x)
    ordered = sorted(cuts)
    points = set(ordered)
    points.update((a+b)/2 for a,b in zip(ordered, ordered[1:]) if a < b)
    worst = 0
    for value in points:
        test = dict(obs);test[candidate] = (value, value)
        counts = block_counts(g[start:start+6], block_h,
            {i-start: iv for i,iv in test.items() if start <= i < start+6}, tau)
        worst = max(worst, max(counts)-min(counts))
    return worst


def possible_bins(interval: tuple[F, F]) -> list[tuple[F, F]]:
    lo, hi = interval
    first = (lo.numerator // lo.denominator)//10
    last = (hi.numerator // hi.denominator)//10
    result = []
    for k in range(first, last+1):
        a, b = F(10*k), min(F(10*(k+1)), B)
        if max(a,lo) <= min(b,hi):
            result.append((a,b))
    return result


def minimax_quant_candidates(g: list[F], h: list[F], obs: dict[int, tuple[F, F]],
                             candidate: int, tau: F) -> int:
    interval = marginal_interval(g,h,obs,candidate)
    start = candidate//6*6
    worst = 0
    for answer in possible_bins(interval):
        test = dict(obs);test[candidate] = answer
        counts = block_counts(g[start:start+6], h[candidate//6],
            {i-start: iv for i,iv in test.items() if start <= i < start+6}, tau)
        worst = max(worst, max(counts)-min(counts))
    return worst


def deterministic_random_order(region: str, day: str) -> list[int]:
    import numpy as np
    seed = int.from_bytes(hashlib.sha256(f'{region}|{day}|V23'.encode()).digest()[:8], 'big')
    return np.random.Generator(np.random.PCG64(seed)).permutation(24).astype(int).tolist()


def choose_hour(strategy: str, g: list[F], h: list[F], obs: dict[int, tuple[F,F]],
                used: set[int], region: str, day: str, precision: str, tau: F) -> int:
    available = [i for i in range(24) if i not in used]
    if strategy == 'uniform_clock':
        order = (12,6,18,0)
        return next(i for i in order if i not in used)
    if strategy == 'seeded_random':
        return next(i for i in deterministic_random_order(region,day) if i not in used)
    if strategy == 'widest_feasible_CIF':
        ranked = [(marginal_interval(g,h,obs,i)[1]-marginal_interval(g,h,obs,i)[0], -i, i) for i in available]
        return max(ranked)[2]
    if strategy == 'one_step_minimax_Kspan':
        scorer = minimax_exact_candidates if precision == 'exact' else minimax_quant_candidates
        return min((scorer(g,h,obs,i,tau), i) for i in available)[1]
    raise ValueError('Unregistered query strategy')


def query_interval(value: F, precision: str) -> tuple[F,F]:
    if precision == 'exact':
        return value,value
    k = (value.numerator // value.denominator)//10
    return F(k*10), min(F((k+1)*10),B)


def source_days(path: Path, region: str, selected_days: set[str]) -> dict[str, tuple[list[F], list[F]]]:
    """Read only selected timestamp rows; filter before converting fuel decimals."""
    by_day: dict[str,list[tuple[pd.Timestamp,dict]]] = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            day_key = row['time'][:10]
            if day_key not in selected_days:
                continue
            stamp = pd.Timestamp(row['time'])
            if stamp.tzinfo is None or stamp != stamp.floor('h') or row['region'] != region:
                raise ValueError('Selected row time/region invalid')
            if row['valid'] != 'True':
                raise ValueError('Inherited selected day includes invalid hour')
            by_day.setdefault(day_key, []).append((stamp,row))
    result = {}
    for day_key, rows in by_day.items():
        rows.sort(key=lambda x:x[0])
        if len(rows)!=24 or [t for t,_ in rows] != list(pd.date_range(day_key,periods=24,freq='h',tz='UTC')):
            raise ValueError('Selected source day not exactly24 complete UTC hours')
        gs=[];hs=[]
        for _,row in rows:
            fuels=[F(row['generation_'+fuel]) for fuel in FUELS]
            if any(v<0 for v in fuels): raise ValueError('Negative source fuel proxy')
            total=sum(fuels,F(0))
            if total<=0: raise ValueError('Nonpositive hourly positive-fuel total')
            gs.append(total);hs.append(sum((v*ef for v,ef in zip(fuels,EF)),F(0)))
        result[day_key]=(gs,hs)
    if set(result)!=selected_days:
        raise ValueError('Selected first/last source-day rows incomplete')
    return result


def one_path(region: str, day: str, g: list[F], hourly_h: list[F], tau: int,
             strategy: str, precision: str) -> list[dict]:
    block_h=[sum(hourly_h[s:s+6],F(0)) for s in range(0,24,6)]
    true_c=[hh/gg for hh,gg in zip(hourly_h,g)]
    true_k=sum(c>tau for c in true_c)
    obs: dict[int,tuple[F,F]]={};used=set();rows=[]
    for budget in BUDGETS:
        if budget>len(used):
            while len(used)<budget:
                hour=choose_hour(strategy,g,block_h,obs,used,region,day,precision,F(tau))
                used.add(hour);obs[hour]=query_interval(true_c[hour],precision)
        possible=daily_counts(g,block_h,obs,F(tau))
        if true_k not in possible:
            raise ValueError('Observed information excluded the realized strict K')
        event0={k==0 for k in possible};event6={k>=6 for k in possible}
        cert0=(len(event0)==1);cert6=(len(event6)==1)
        event_any=(true_k>=1);event_six=(true_k>=6)
        rows.append(dict(region=region,day=day,threshold=tau,strategy=strategy,precision=precision,
            budget=budget,queried_hours='|'.join(map(str,sorted(used))),true_k=true_k,
            possible_k='|'.join(map(str,possible)),k_span=max(possible)-min(possible),
            true_event_any=int(event_any),certified_any=int(cert0),
            correct_any=int(cert0 and next(iter(event0))==(not event_any)),
            wrong_any=int(cert0 and next(iter(event0))!=(not event_any)),unresolved_any=int(not cert0),
            true_event_6=int(event_six),certified_6=int(cert6),
            correct_6=int(cert6 and next(iter(event6))==event_six),
            wrong_6=int(cert6 and next(iter(event6))!=event_six),unresolved_6=int(not cert6)))
    return rows


def run() -> None:
    if any((OUT/n).exists() for n in ('completed.json','day_results.csv','summary.csv')):
        raise FileExistsError('Preserve existing or partial V23 outputs')
    frozen=check_freeze();initial=sha(OUT/'freeze.json')
    chosen={r:{d for rr,d in frozen['chosen_region_days'] if rr==r} for r in SOURCES}
    daily=[]
    for region in SOURCES:
        path=Path(frozen['source_directory'])/f'{region}_hourly.csv'
        data=source_days(path,region,chosen[region])
        for day,(g,hh) in sorted(data.items()):
            for tau in TAUS:
                for precision in PRECISIONS:
                    for strategy in STRATEGIES:
                        daily.extend(one_path(region,day,g,hh,tau,strategy,precision))
        print('Processed',region,flush=True)
    table=pd.DataFrame(daily)
    expected=7*2*len(TAUS)*len(PRECISIONS)*len(STRATEGIES)*len(BUDGETS)
    if len(table)!=expected or table.duplicated(['region','day','threshold','strategy','precision','budget']).any():
        raise ValueError('Changed fixed V23 result grid')
    # Source-first macro: each region contributes its mean over the fixed14 days.
    per_source=table.groupby(['region','threshold','strategy','precision','budget']).agg(
        unresolved_any=('unresolved_any','mean'),unresolved_6=('unresolved_6','mean'),
        mean_k_span=('k_span','mean'),correct_any=('correct_any','mean'),correct_6=('correct_6','mean'),
        positive_days=('true_event_any','sum'),zero_days=('true_event_any',lambda x:int((x==0).sum())),
        tail6_days=('true_event_6','sum'),cases=('day','size')).reset_index()
    sm=per_source.groupby(['threshold','strategy','precision','budget']).agg(
        source_macro_unresolved_any=('unresolved_any','mean'),source_macro_unresolved_6=('unresolved_6','mean'),
        source_macro_k_span=('mean_k_span','mean'),source_macro_correct_any=('correct_any','mean'),
        source_macro_correct_6=('correct_6','mean'),
        regions=('region','nunique')).reset_index()
    wrong=table.groupby(['threshold','strategy','precision','budget']).agg(
        wrong_certificates_any=('wrong_any','sum'),wrong_certificates_6=('wrong_6','sum'),
        days=('day','size')).reset_index()
    summary=sm.merge(wrong,on=['threshold','strategy','precision','budget'],validate='one_to_one')
    out1=OUT/'day_results.csv';out2=OUT/'summary.csv'
    table.to_csv(out1,index=False,mode='x');summary.to_csv(out2,index=False,mode='x')
    check_freeze()
    if sha(OUT/'freeze.json')!=initial: raise ValueError('Freeze changed during V23 run')
    write_json_new(OUT/'completed.json',{**contract(),'created_utc':now(),'freeze_sha256':initial,
        'rows':len(table),'input_hashes':frozen['input_hashes'],
        'output_hashes':{str(out1):sha(out1),str(out2):sha(out2)},
        'forecast_gain':False,'new_method':False,'mechanism_breakthrough':False,
        'target_authorized':False,'unseen_values_opened':False})
    print(summary.to_string(index=False))


def main() -> None:
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=('prepare','run'))
    args=parser.parse_args()
    if args.command=='prepare': prepare()
    else: run()


if __name__=='__main__': main()
