"""Independently recompute native-point weather from archived GFS pilot messages."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import eccodes
import numpy as np
import pandas as pd

SIGNATURES = {'temperature_2m': (167, 'heightAboveGround', 2, 'K'),
              'wind_u_100m': (131, 'heightAboveGround', 100, 'm s**-1'),
              'wind_v_100m': (132, 'heightAboveGround', 100, 'm s**-1'),
              'shortwave_surface': (260087, 'surface', 0, 'W m**-2')}


def validate_window(meta: dict, weather: dict, points: list) -> tuple:
    run, origin = pd.Timestamp(meta['initialization_at']), pd.Timestamp(meta['forecast_origin_at'])
    first, end = weather['first_lead'], weather['end_lead']
    if (run.tz is None or origin.tz is None or pd.Timestamp(weather['initialization_at']) != run
            or end-first != meta['horizon'] or origin != run+pd.Timedelta(hours=first)
            or weather['points'] != points or meta['native_leads'] != list(range(first,end+1))
            or sorted(r['lead'] for r in meta['snapshots']) != list(range(first,end+1))):
        raise ValueError('inconsistent window headers')
    expected = {(p['id'], run+pd.Timedelta(hours=h)) for p in points for h in range(first,end)}
    actual = [(r['point'], pd.Timestamp(r['valid_at'])) for r in weather['rows']]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError('point-hour keys do not cover exactly the declared window')
    return run, origin, first, end


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve previous audit')
    hashes, windows = {}, []
    points_path = args.pilot/'points.json'
    points = json.loads(points_path.read_text())
    hashes[str(points_path.resolve())] = sha(points_path)
    hashes[str(Path(__file__).resolve())] = sha(Path(__file__))
    for folder in sorted(args.pilot.glob('window-*')):
        metadata_path, weather_path = folder/'metadata.json', folder/'hourly_points.json'
        for p in (metadata_path, weather_path):
            hashes[str(p.resolve())] = sha(p)
        meta, weather = json.loads(metadata_path.read_text()), json.loads(weather_path.read_text())
        if sha(weather_path) != meta['weather_sha256']:
            raise ValueError('window weather hash mismatch')
        run, origin, first, end = validate_window(meta, weather, points)
        if run + pd.Timedelta(hours=6) > origin:
            raise ValueError('pilot must leave at least six hours after initialization')
        native, latest, total_bytes, max_distance = {}, [], 0, 0.
        for record in meta['snapshots']:
            snapshot = Path(record['snapshot']); p = snapshot/'metadata.json'
            if sha(p) != record['metadata_sha256']:
                raise ValueError('snapshot metadata hash mismatch')
            hashes[str(p.resolve())] = sha(p)
            data = json.loads(p.read_text()); lead = data['lead_hours']
            if lead != record['lead'] or set(data['messages']) != set(SIGNATURES):
                raise ValueError('native lead or field population mismatch')
            for kind in ('grib_object', 'index_object'):
                written = pd.Timestamp(data[kind]['last_modified'])
                if not run <= written <= origin:
                    raise ValueError('object write outside pre-origin interval')
                latest.append(written)
            index_path = snapshot/'source.idx'
            if sha(index_path) != data['index_sha256']:
                raise ValueError('index changed')
            hashes[str(index_path.resolve())] = sha(index_path)
            native[lead] = {}
            for field, item in data['messages'].items():
                path = snapshot/item['file']
                if sha(path) != item['sha256']:
                    raise ValueError('native field changed')
                hashes[str(path.resolve())] = sha(path); total_bytes += path.stat().st_size
                with path.open('rb') as stream:
                    handle = eccodes.codes_grib_new_from_file(stream)
                    try:
                        if handle is None:
                            raise ValueError('missing GRIB handle')
                        if (eccodes.codes_get(handle, 'dataDate') != int(run.strftime('%Y%m%d'))
                                or eccodes.codes_get(handle, 'dataTime') != run.hour*100
                                or eccodes.codes_get(handle, 'endStep') != lead
                                or eccodes.codes_get(handle, 'stepUnits') != 1):
                            raise ValueError('encoded run/lead mismatch')
                        signature = tuple(eccodes.codes_get(handle,k) for k in ['paramId','typeOfLevel','level','units'])
                        solar = field == 'shortwave_surface'
                        start = int(eccodes.codes_get(handle,'startStep'))
                        if (signature != SIGNATURES[field]
                                or eccodes.codes_get(handle,'stepType') != ('avg' if solar else 'instant')
                                or (not solar and start != lead) or (solar and not 0 <= start < lead)):
                            raise ValueError('native field semantics mismatch')
                        quantum = None
                        if solar:
                            packing = eccodes.codes_get(handle,'packingType')
                            quantum = 2.**eccodes.codes_get(handle,'binaryScaleFactor') * 10.**(-eccodes.codes_get(handle,'decimalScaleFactor'))
                            if packing not in ('grid_simple','grid_complex','grid_complex_spatial_differencing') or not math.isfinite(quantum) or quantum <= 0:
                                raise ValueError('unreviewed native solar resolution')
                        values = []
                        for point in points:
                            nearest = eccodes.codes_grib_find_nearest(handle, point['latitude'], point['longitude'])[0]
                            values.append(float(nearest['value'])); max_distance = max(max_distance, nearest['distance'])
                        native[lead][field] = {'values': np.array(values),
                            'start': start, 'resolution': quantum}
                    finally:
                        if handle is not None: eccodes.codes_release(handle)
        rows = {(r['point'], pd.Timestamp(r['valid_at'])): r for r in weather['rows']}
        expected_rows = meta['horizon']*len(points)
        if len(rows) != expected_rows or len(weather['rows']) != expected_rows:
            raise ValueError('missing or duplicate point hours')
        deltas = []
        for h in range(first, end):
            current, nxt = native[h], native[h+1]
            solar = nxt['shortwave_surface']; start = solar['start']
            duration = h+1-start
            raw_energy = solar['values']*duration
            tolerance = duration*solar['resolution']
            if start < h:
                if current['shortwave_surface']['start'] != start:
                    raise ValueError('solar reset-window mismatch')
                raw_energy -= current['shortwave_surface']['values']*(duration-1)
                tolerance += (duration-1)*current['shortwave_surface']['resolution']
            for i, point in enumerate(points):
                actual = rows[point['id'], run+pd.Timedelta(hours=h)]
                expected = [current['temperature_2m']['values'][i]-273.15,
                            math.hypot(current['wind_u_100m']['values'][i], current['wind_v_100m']['values'][i]),
                            raw_energy[i], max(0., raw_energy[i])]
                got = [actual[k] for k in ['temperature_c', 'wind_speed_ms', 'shortwave_raw_w_per_m2', 'shortwave_w_per_m2']]
                np.testing.assert_allclose(expected, got, atol=1e-10, rtol=0.)
                allowed = tolerance if weather['solar_negative_policy']=='packing_resolution' else 0.
                if raw_energy[i] < -allowed or bool(actual['shortwave_projected_within_resolution']) != bool(raw_energy[i] < 0):
                    raise ValueError('invalid solar projection')
                np.testing.assert_allclose(actual['shortwave_resolution_tolerance_w_per_m2'],allowed,atol=1e-10,rtol=0.)
                deltas.extend(abs(a-b) for a,b in zip(expected,got))
        frame = pd.DataFrame(weather['rows'])
        windows.append({'window': str(folder), 'run': str(run), 'origin': str(origin),
                        'point_hours': len(rows), 'native_snapshots': len(native), 'native_field_bytes': total_bytes,
                        'latest_object_write': str(max(latest)), 'hours_from_last_write_to_origin': (origin-max(latest)).total_seconds()/3600,
                        'max_probe_distance_km': float(max_distance), 'max_independent_value_delta': float(max(deltas)),
                        'projected_solar_point_hours': int(frame.shortwave_projected_within_resolution.sum()),
                        'temperature_c_range': [float(frame.temperature_c.min()),float(frame.temperature_c.max())],
                        'wind_ms_range': [float(frame.wind_speed_ms.min()),float(frame.wind_speed_ms.max())],
                        'shortwave_raw_range': [float(frame.shortwave_raw_w_per_m2.min()),float(frame.shortwave_raw_w_per_m2.max())],
                        'original_issued_public_flag': meta['eligible_for_issued_public']})
    if not windows:
        raise ValueError('no complete windows')
    for name, expected in hashes.items():
        if sha(Path(name)) != expected: raise RuntimeError('pilot inputs changed')
    result = {'status': 'NATIVE_POINT_WINDOW_VALUES_VERIFIED_NOT_REGIONAL_MODEL_INPUTS',
              'windows': windows, 'input_hashes': hashes, 'eccodes_version': eccodes.codes_get_api_version(),
              'latency_check_hours': 6, 'availability_limit': 'pre-origin object writes and latency check; not certified historical public availability',
              'new_cif_model_trained': False, 'limits': ['probe points are not regional fleet-weighted weather',
              'two runs do not establish full-archive coverage', 'existing issued-public flags are preserved']}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(windows,indent=2))


if __name__ == '__main__':
    main()
