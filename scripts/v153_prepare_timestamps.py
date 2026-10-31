"""Recover public UTC calendar aligned exactly with frozen V144 hour arrays."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import v90_coarse_capacity_identification as v90
from v139_physical_reserve_calibration import sha

ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'results/v144-canonical-profile-cache-r1'
OUT=ROOT/'results/v153-calendar-times-r1'

def main():
    if OUT.exists():raise FileExistsError(OUT)
    protocol=ROOT/'docs/V153_CALENDAR_LAW_PROTOCOL.md';protocol_hash=sha(protocol)
    index=pd.read_csv(P/'profile_index.csv');raw=np.load(P/'profiles.npz')['generation']
    data=v90.source_data;data.ALL_REGIONS=tuple(sorted(index.region.unique()))
    data.START,data.STOP=pd.Timestamp('2024-01-01T00:00:00Z'),pd.Timestamp('2026-01-01T00:00:00Z')
    hourly,_,audits,hashes=data.prepare_regions();timestamps=[]
    for row in index.itertuples(index=False):
        frame=hourly[row.region];times=pd.to_datetime(frame.time,utc=True)
        selected=frame.loc[times.dt.strftime('%Y-%m-01').eq(row.month)&frame.valid.astype(bool)]
        values=selected[[f'generation_{f}' for f in v90.FUELS]].to_numpy(float)
        if not np.array_equal(values,raw[row.offset:row.stop]):raise ValueError(f'row alignment mismatch {row.region}/{row.month}')
        time=pd.DatetimeIndex(pd.to_datetime(selected.time,utc=True))
        unix=time.as_unit('s').asi8
        if len(np.unique(unix))!=len(unix) or np.any(unix%3600):raise ValueError('timestamp uniqueness/hour alignment')
        timestamps.extend(unix)
    if len(timestamps)!=len(raw) or sha(protocol)!=protocol_hash:raise ValueError('timestamp denominator/protocol')
    OUT.mkdir(parents=True);np.savez_compressed(OUT/'timestamps.npz',unix_seconds=np.asarray(timestamps,np.int64))
    result={'assertion_failures':0,'profiles':len(index),'verified_hour_rows':len(raw),'raw_source_hashes':hashes,
        'input_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [protocol,P/'profiles.npz',P/'profile_index.csv',Path(__file__)]},
        'timestamp_sha256':sha(OUT/'timestamps.npz')}
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'UTC times recovered; {len(raw)} generation rows exactly reproduced')

if __name__=='__main__':main()
