"""Lossless SMARD native-category ingestion; no inferred CIF or fuel merging."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CATEGORIES = {'1223':'lignite','1224':'nuclear','1225':'wind_offshore','1226':'hydro',
              '1227':'other_conventional','1228':'other_renewable','4066':'biomass',
              '4067':'wind_onshore','4068':'solar','4069':'hard_coal',
              '4070':'pumped_storage','4071':'gas'}


def load_native(manifests: list[Path], root: Path) -> tuple[pd.DataFrame, dict]:
    values = {code:{} for code in CATEGORIES}
    years = set()
    files = set()
    for manifest in manifests:
        meta = json.loads(manifest.read_text())
        if meta.get('errors'):
            raise ValueError('incomplete acquisition manifest')
        years.add(int(meta['year']))
        for record in meta['records']:
            code = record['filter']
            if code not in CATEGORIES:
                raise ValueError(f'unknown category {code}')
            path = root/record['path']
            content = path.read_bytes()
            if hashlib.sha256(content).hexdigest()!=record['sha256']:
                raise ValueError(f'hash mismatch: {path}')
            if str(path) in files:
                continue
            files.add(str(path))
            for stamp, value in json.loads(content)['series']:
                value = np.nan if value is None else float(value)
                old = values[code].get(stamp)
                if old is not None and not (old==value or (np.isnan(old) and np.isnan(value))):
                    raise ValueError(f'conflicting duplicate: {code} {stamp}')
                values[code][stamp] = value
    if any(not v for v in values.values()):
        raise ValueError('one or more native categories absent')
    frame = pd.DataFrame({CATEGORIES[k]:pd.Series(v) for k,v in values.items()}).sort_index()
    frame.index = pd.to_datetime(frame.index,unit='ms',utc=True)
    frame.index.name = 'time'
    local = frame.index.tz_convert('Europe/Berlin')
    frame = frame.loc[np.isin(local.year,list(years))]
    local = frame.index.tz_convert('Europe/Berlin')
    audit = {'schema':'SMARD_NATIVE_12_NO_CIF', 'timezone':'Europe/Berlin',
             'energy_unit':'MWh per hour', 'unique_raw_files':len(files), 'years':{}}
    for year in sorted(years):
        block = frame.loc[local.year==year]
        start=pd.Timestamp(year=year,month=1,day=1,tz='Europe/Berlin')
        end=pd.Timestamp(year=year+1,month=1,day=1,tz='Europe/Berlin')
        audit['years'][str(year)] = {'hours':len(block),
            'expected_hours':int((end-start).total_seconds()/3600),
            'nonfinite':int((~np.isfinite(block.to_numpy())).sum()),
            'negative':int((block.to_numpy()<0).sum()),
            'category_mwh':{k:float(v) for k,v in block.sum().items()}}
    frame['clock'] = local.tz_localize(None)
    return frame, audit
