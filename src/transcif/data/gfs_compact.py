"""Compact verified GFS point windows, with source identities but no GRIB copies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from transcif.data.gfs_decode import decode_gfs_points, hourly_gfs_points
from transcif.data.gfs_window import _cached_metadata


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compact_window(source: Path, output: Path) -> dict:
    """Re-decode a complete raw window, preserve evidence and copy only point data.

    Raw snapshots remain untouched. Original metadata/index files preserve byte
    ranges, ETags and message SHA values for upstream reacquisition; this bundle
    is deliberately not a raw cache accepted by the original GRIB decoder.
    """
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError('compact output must be new')
    meta_path, weather_path = source/'metadata.json', source/'hourly_points.json'
    initial = {str(p.resolve()): digest(p) for p in (meta_path, weather_path)}
    meta, weather = json.loads(meta_path.read_text()), json.loads(weather_path.read_text())
    if initial[str(weather_path.resolve())] != meta['weather_sha256']:
        raise ValueError('source weather changed')
    run, origin = pd.Timestamp(meta['initialization_at']), pd.Timestamp(meta['forecast_origin_at'])
    first, end = weather['first_lead'], weather['end_lead']
    if (origin != run+pd.Timedelta(hours=first) or end-first != meta['horizon']
            or meta['native_leads'] != list(range(first,end+1))
            or [r['lead'] for r in meta['snapshots']] != list(range(first,end+1))):
        raise ValueError('inconsistent source window')
    decoded, copies, raw_bytes = [], [], 0
    for record in meta['snapshots']:
        path = Path(record['snapshot']); lead = record['lead']
        p = path/'metadata.json'
        if digest(p) != record['metadata_sha256']:
            raise ValueError('source snapshot metadata changed')
        native = _cached_metadata(path, run, lead)
        for key in ('grib_object','index_object'):
            if not run <= pd.Timestamp(native[key]['last_modified']) <= origin:
                raise ValueError('source object not written before origin')
        for name in ['metadata.json','source.idx']:
            p = path/name; initial[str(p.resolve())] = digest(p)
            copies.append((p, f'evidence/f{lead:03d}_{name}'))
        for field in native['messages'].values():
            p = path/field['file']; initial[str(p.resolve())] = digest(p); raw_bytes += p.stat().st_size
        decoded.append(decode_gfs_points(path,weather['points']))
    expected = hourly_gfs_points(decoded,first,end,meta['solar_negative_policy'])
    if expected != weather:
        raise ValueError('source weather differs from independent native re-decode')
    output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.compact-gfs-',dir=output.parent) as temporary:
        staged = Path(temporary)/'complete'; (staged/'evidence').mkdir(parents=True)
        copies += [(meta_path,'source_window_metadata.json'),(weather_path,'hourly_points.json')]
        files = []
        for original, name in copies:
            shutil.copyfile(original,staged/name)
            files.append({'file':name,'sha256':digest(staged/name)})
        for name, expected_hash in initial.items():
            if digest(Path(name)) != expected_hash:
                raise RuntimeError('source window changed during compaction')
        report = {'schema':'GFS_COMPACT_POINT_WINDOW_V1','initialization_at':str(run),
                  'forecast_origin_at':str(origin),'horizon':meta['horizon'],
                  'files':files,'source_raw_bytes':raw_bytes,'raw_grib_copied':False,
                  'source_files_unchanged':True,'point_hours':len(weather['rows']),
                  'eligible_for_issued_public':meta['eligible_for_issued_public'],
                  'available_at':meta['available_at'],
                  'limits':['native metadata filenames describe upstream GRIB; those files are not in this bundle',
                            'upstream reacquisition with recorded URL/range/ETag/SHA is required for full GRIB re-decoding',
                            'point coordinates and temporal support remain unchanged; not regional weather']}
        report['compact_payload_bytes'] = sum(p.stat().st_size for p in staged.rglob('*') if p.is_file())
        (staged/'manifest.json').write_text(json.dumps(report,indent=2))
        staged.rename(output)
    return report


def load_compact_window(path: Path, expected_points: list | None = None) -> dict:
    """Load only integrity-checked point data; never upgrade availability flags."""
    path = Path(path); report = json.loads((path/'manifest.json').read_text())
    if report['schema'] != 'GFS_COMPACT_POINT_WINDOW_V1' or report['raw_grib_copied']:
        raise ValueError('not a compact point bundle')
    names = [r['file'] for r in report['files']]
    if len(names) != len(set(names)) or not {'hourly_points.json','source_window_metadata.json'} <= set(names):
        raise ValueError('compact bundle must bind weather and source metadata')
    resolved = []
    for record in report['files']:
        relative = Path(record['file'])
        if relative.is_absolute() or relative.as_posix() != record['file'] or any(x in ('.','..','') for x in record['file'].split('/')):
            raise ValueError('compact paths must be canonical relative paths')
        item = path/record['file']
        if not item.resolve().is_relative_to(path.resolve()) or not item.is_file() or digest(item) != record['sha256']:
            raise ValueError('compact evidence hash/path mismatch')
        resolved.append(item.resolve())
    if len(set(resolved)) != len(resolved):
        raise ValueError('compact paths must not alias the same file')
    source = json.loads((path/'source_window_metadata.json').read_text())
    if digest(path/'hourly_points.json') != source['weather_sha256']:
        raise ValueError('compact weather differs from original window seal')
    references = source['snapshots']
    if len({r['lead'] for r in references}) != len(references):
        raise ValueError('duplicate source lead evidence')
    expected = {'hourly_points.json','source_window_metadata.json'}
    for reference in references:
        prefix = f"evidence/f{reference['lead']:03d}_"
        expected.update([prefix+'metadata.json',prefix+'source.idx'])
    if set(names) != expected:
        raise ValueError('compact bundle must preserve all native metadata/index evidence')
    for reference in references:
        prefix = f"evidence/f{reference['lead']:03d}_"
        p = path/(prefix+'metadata.json')
        if digest(p) != reference['metadata_sha256']:
            raise ValueError('native metadata does not match source window')
        native = json.loads(p.read_text())
        if (native['lead_hours'] != reference['lead'] or native['initialization_at'] != source['initialization_at']
                or digest(path/(prefix+'source.idx')) != native['index_sha256']):
            raise ValueError('native lead/index evidence mismatch')
    weather = json.loads((path/'hourly_points.json').read_text())
    if weather['input_metadata_sha256'] != [r['metadata_sha256'] for r in references]:
        raise ValueError('point values do not bind the native lead evidence')
    if expected_points is not None and weather['points'] != expected_points:
        raise ValueError('compact cache belongs to a different point plan')
    return weather
