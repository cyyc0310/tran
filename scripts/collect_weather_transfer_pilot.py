"""Freeze a small seasonal weather plan, audit cohorts, then acquire compact runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile

import pandas as pd

from transcif.data.gfs_compact import compact_window, load_compact_window
from transcif.data.gfs_window import collect_gfs_window
from transcif.research import multicountry as mc
from transcif.research.cross_country import ZONES
from transcif.research.public_horizon import public_horizon_features

ROOT = Path(__file__).resolve().parents[1]
ZONES = {**ZONES, 'GB': 'UTC', 'DE': 'Europe/Berlin'}
COORDINATES = {'BPAT': (45.4, -119.5), 'CISO': (36.8, -119.5),
               'ERCO': (31.8, -99.3), 'FPL': (27.4, -81.2),
               'ISNE': (42.8, -71.5), 'MISO': (41.5, -93.),
               'NYIS': (42.9, -75.3), 'GB': (54., -2.), 'DE': (51., 10.),
               'Shanghai_deployment_probe': (31.25, 121.5)}
DATES = {'train': ['2022-05-15', '2022-06-15', '2022-08-15', '2022-09-15',
                   '2022-11-15', '2022-12-15'],
         'calibration': ['2023-01-15', '2023-03-15', '2023-05-15'],
         'development': ['2023-07-15', '2023-09-15', '2023-11-15']}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plan() -> dict:
    """Dates and coordinates fixed independently of label errors or weather."""
    return {'schema': 'WEATHER_TRANSFER_SEASONAL_PILOT_V1', 'dates': DATES,
            'points': [{'id': key, 'latitude': lat, 'longitude': lon}
                       for key, (lat, lon) in COORDINATES.items()],
            'first_lead': 24, 'horizon': 48,
            'clock_zones': ZONES,
            'gb_clock_source': 'https://www.neso.energy/data-portal/historic-generation-mix/historic_gb_generation_mix',
            'day_selection': 'first complete 24-clock-hour day beginning at or after UTC origin; fixed dates avoid DST transitions',
            'availability': 'archive object writes before origin; original public availability assumed, not certified',
            'sampling': 'one fixed geographic proxy per region, not area/fleet-weighted weather',
            'confirmation': False}


def cohort(raw: dict, protocol: dict) -> tuple[pd.DataFrame, list]:
    """Select using input availability and complete labels, never error values."""
    frames, audit = [], []
    for split, dates in protocol['dates'].items():
        for date in dates:
            origin = pd.Timestamp(date, tz='UTC')
            for region, frame in raw.items():
                zone = protocol['clock_zones'][region]
                local_origin = origin.tz_convert(zone).tz_localize(None)
                start = local_origin.ceil('D')
                clocks = pd.Series(pd.date_range(start, periods=24, freq='h'))
                record = dict(split=split, origin=str(origin), region=region,
                              day=str(start.date()), eligible=False)
                day_rows = frame.loc[frame.clock.ge(start) & frame.clock.lt(start+pd.Timedelta(days=1))].copy()
                if day_rows.clock.duplicated().any():
                    record['reason'] = 'ambiguous DST day'
                    audit.append(record)
                    continue
                day_rows['label_row'] = day_rows.index
                match = day_rows.set_index('clock').reindex(clocks)
                if match['time'].isna().any() or match[list(mc.FINE)].isna().any().any():
                    record['reason'] = 'incomplete hourly labels'
                else:
                    try:
                        public = public_horizon_features(mc.public_months(frame), clocks, local_origin)
                    except ValueError as error:
                        record['reason'] = str(error)
                    else:
                        public['valid_at'] = clocks.dt.tz_localize(zone).dt.tz_convert('UTC')
                        if not (public.valid_at.ge(origin).all() and public.valid_at.lt(origin+pd.Timedelta(hours=48)).all()):
                            raise ValueError('selected day outside fixed weather horizon')
                        public['split'], public['origin'] = split, str(origin)
                        public['region'], public['country'] = region, frame.country.iloc[0]
                        public['label_row'] = match.label_row.to_numpy(dtype=int)
                        frames.append(public)
                        record['eligible'] = True
                audit.append(record)
    if not frames:
        raise ValueError('no eligible cohorts')
    return pd.concat(frames, ignore_index=True), audit


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    protocol = plan(); path = output/'plan.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError('existing plan differs')
    else:
        path.write_text(json.dumps(protocol, indent=2)+'\n')
    seal_path = output/'cohort_seal.json'
    if not seal_path.exists():
        raw, _ = mc.load_countries(ROOT)
        selected, audit = cohort(raw, protocol)
        selected.to_csv(output/'cohort.csv', index=False)
        (output/'cohort_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
        seals = {p.name: digest(p) for p in [path, output/'cohort.csv', output/'cohort_audit.json']}
        seal_path.write_text(json.dumps(seals, indent=2)+'\n')
    for name, expected in json.loads(seal_path.read_text()).items():
        if digest(output/name) != expected:
            raise ValueError('frozen cohort/plan changed')
    if args.plan_only:
        return
    for split, dates in protocol['dates'].items():
        for date in dates:
            destination = output/f'window-{date}'
            if destination.exists():
                weather = load_compact_window(destination, protocol['points'])
                origin = pd.Timestamp(date, tz='UTC')
                expected_keys = {(p['id'], str(t)) for p in protocol['points']
                                 for t in pd.date_range(origin, periods=48, freq='h')}
                actual_keys = [(r['point'], str(pd.Timestamp(r['valid_at']))) for r in weather['rows']]
                if (pd.Timestamp(weather['initialization_at']) != origin-pd.Timedelta(days=1)
                        or weather['first_lead'] != 24 or weather['end_lead'] != 72
                        or len(actual_keys) != len(expected_keys) or set(actual_keys) != expected_keys):
                    raise ValueError('cached weather window differs from plan')
                print(date, 'verified compact checkpoint', flush=True)
                continue
            # Only this process's disposable raw cache is removed. Completed
            # compact windows survive interrupts; historical raw pilots remain.
            with tempfile.TemporaryDirectory(prefix='.weather-transfer-', dir=output) as temporary:
                scratch = Path(temporary)
                collect_gfs_window(pd.Timestamp(date, tz='UTC')-pd.Timedelta(days=1),
                    protocol['first_lead'], protocol['horizon'], protocol['points'],
                    scratch/'native', scratch/'window', workers=4, retries=3,
                    solar_negative_policy='packing_resolution',
                    progress=lambda row: print(date, 'lead', row['lead'], flush=True))
                compact_window(scratch/'window', destination)
            print(date, split, 'compact committed', flush=True)


if __name__ == '__main__':
    main()
