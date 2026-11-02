"""Matched-label-budget source-pool controls for target structural zeros."""
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.multicountry import FINE

POOLS={'base':(), 'nuclear_free':('PACE','LGEE'), 'nuclear_control':('SRP','DUK')}
ZONES={'PACE':'America/Denver','LGEE':'America/New_York','SRP':'America/Phoenix','DUK':'America/New_York'}


def extra_sources(root: Path, overrides: dict | None = None):
    frames,audit={},{}
    for region,zone in ZONES.items():
        path=root/f'results/external-v6-data/{region}_hourly.csv'
        if overrides and region in overrides:path=Path(overrides[region]).resolve()
        raw=pd.read_csv(path)
        time=pd.to_datetime(raw.time,utc=True)
        if time.duplicated().any():raise ValueError('duplicate external source times')
        fuel=raw[[f'generation_{f}' for f in FINE[:-1]]].to_numpy(float)
        valid=(raw.valid.astype(str).str.lower().eq('true').to_numpy()&np.isfinite(fuel).all(axis=1)
               &(fuel>=0).all(axis=1)&(fuel.sum(axis=1)>0))
        frame=pd.DataFrame(fuel[valid],columns=FINE[:-1])
        frame['other_renewable']=0.
        frame['time']=time.loc[valid].reset_index(drop=True)
        frame['clock']=frame.time.dt.tz_convert(zone).dt.tz_localize(None)
        periods=frame.clock.dt.to_period('M')
        hours={p:((p+1).start_time.tz_localize(zone)-p.start_time.tz_localize(zone)).total_seconds()/3600
               for p in periods.unique()}
        frame['expected_hours']=periods.map(hours)
        frame['region'],frame['country']=region,'US'
        frames[region]=frame
        audit[region]={'path':str(path.relative_to(root)) if path.is_relative_to(root) else str(path),
                       'calendar':zone,'role':'source-only expansion'}
    return frames,audit


def matched_counts(features: dict, year: int, half: bool=False):
    counts={}
    for r in ZONES:
        f=features[r]
        f=f.loc[f.year.eq(year)&(f.clock.dt.month.le(6) if half else True)]
        counts[r]=f.groupby(f.clock.dt.month).size().to_dict()
    return {month:min(counts[r].get(month,0) for r in ZONES) for month in range(1,13)}


def match_added(frame: pd.DataFrame, counts: dict) -> pd.DataFrame:
    blocks=[]
    for month,group in frame.groupby(frame.clock.dt.month,sort=True):
        n=counts.get(month,0)
        if n:
            group=group.sort_values('clock')
            if len(group)<n:raise ValueError('matched source budget exceeds support')
            ix=np.linspace(0,len(group)-1,n,dtype=int)
            blocks.append(group.iloc[ix])
    if not blocks:return frame.iloc[:0]
    return pd.concat(blocks)
