"""Acquire a fixed, expanded development calendar without changing pilot seals."""
from __future__ import annotations

import argparse
from copy import deepcopy
from http.client import IncompleteRead
import json
from pathlib import Path
import shutil
import ssl
import tempfile
import time
from urllib.error import URLError

import pandas as pd

from collect_weather_transfer_pilot import ROOT, cohort, digest, plan as pilot_plan
from run_weather_transfer_pipeline import retry_collection
from transcif.data.gfs_compact import compact_window, load_compact_window
from transcif.data.gfs_window import collect_gfs_window
from transcif.research import multicountry as mc


def plan() -> dict:
    """Calendar-only expansion within the three previously opened countries."""
    protocol = deepcopy(pilot_plan())
    protocol.update(
        schema='WEATHER_TRANSFER_EXPANDED_DEVELOPMENT_V1',
        dates={split: [f'2023-{month:02d}-{day:02d}' for month in months for day in (15, 25)]
               for split, months in [('calibration', range(1, 7)), ('development', range(7, 13))]},
        source_training='unchanged source-observed ERA5 35448-hour cohort; no new source labels',
        adaptive_development=True,
        new_country_opened=False,
    )
    return protocol


def write_checked(path: Path, content: bytes) -> None:
    """Resume an interrupted initialization only if existing bytes agree."""
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError(f'existing initialization differs: {path.name}')
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.initializing-', delete=False) as stream:
        stream.write(content)
        temporary = Path(stream.name)
    temporary.replace(path)


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()


def verify_files(files: dict) -> None:
    for name, expected in files.items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'frozen producer/source changed: {name}')


def initialize(output: Path, pilot: Path) -> dict:
    """Bind unchanged source bytes, calendar, producers and label-eligibility audit."""
    protocol = plan()
    inherited_path = pilot / 'pre_model_source_receipt.json'
    inherited = json.loads(inherited_path.read_text())['files']
    verify_files(inherited)
    output.mkdir(parents=True, exist_ok=True)
    write_checked(output / 'plan.json', json_bytes(protocol))
    seal_path = output / 'cohort_seal.json'
    if not seal_path.exists():
        raw, _ = mc.load_countries(ROOT)
        selected, audit = cohort(raw, protocol)
        write_checked(output / 'cohort.csv', selected.to_csv(index=False).encode())
        write_checked(output / 'cohort_audit.json', json_bytes(audit))
        seals = {name: digest(output / name) for name in ['plan.json', 'cohort.csv', 'cohort_audit.json']}
        write_checked(seal_path, json_bytes(seals))
    for name, expected in json.loads(seal_path.read_text()).items():
        if digest(output / name) != expected:
            raise ValueError('expanded cohort/plan changed')
    paths = [Path(__file__).resolve(), ROOT / 'tests/test_weather_transfer_expansion.py',
             ROOT / 'docs/WEATHER_TRANSFER_EXPANSION_PROTOCOL.md', inherited_path,
             output / 'plan.json', output / 'cohort.csv', output / 'cohort_audit.json', seal_path,
             ROOT / 'scripts/run_weather_transfer_pipeline.py']
    paths.extend(sorted((ROOT / 'src/transcif/data').glob('gfs*.py')))
    files = {**inherited, **{str(path.relative_to(ROOT)): digest(path) for path in paths}}
    receipt = dict(stage='cohort and calendar frozen before new acquisition or model fitting', files=files)
    write_checked(output / 'pre_model_source_receipt.json', json_bytes(receipt))
    verify_files(files)
    return protocol


def verify_window(path: Path, date: str, protocol: dict) -> None:
    """A valid compact file must also be the exact planned run, leads and hours."""
    weather = load_compact_window(path, protocol['points'])
    origin = pd.Timestamp(date, tz='UTC')
    expected = {(point['id'], instant) for point in protocol['points']
                for instant in pd.date_range(origin, periods=48, freq='h')}
    actual = [(row['point'], pd.Timestamp(row['valid_at'])) for row in weather['rows']]
    manifest = json.loads((path / 'manifest.json').read_text())
    if (pd.Timestamp(weather['initialization_at']) != origin - pd.Timedelta(days=1)
            or weather['first_lead'] != 24 or weather['end_lead'] != 72
            or len(actual) != len(expected) or set(actual) != expected
            or manifest['eligible_for_issued_public'] is not False or manifest['available_at'] is not None):
        raise ValueError('compact window does not satisfy expanded plan')


def copy_window(source: Path, destination: Path, date: str, protocol: dict) -> None:
    """Copy an already verified pilot window atomically, without editing it."""
    verify_window(source, date, protocol)
    with tempfile.TemporaryDirectory(prefix='.reuse-weather-', dir=destination.parent) as temporary:
        staged = Path(temporary) / 'complete'
        shutil.copytree(source, staged)
        verify_window(staged, date, protocol)
        staged.rename(destination)


def collect_window(output: Path, date: str, protocol: dict) -> None:
    """Keep per-lead checkpoints on failure; remove only this window's owned raw cache."""
    destination = output / f'window-{date}'
    work = output / 'raw-checkpoints' / date
    owner = dict(date=date, plan_sha256=digest(output / 'plan.json'))
    if work.is_symlink() or work.parent.is_symlink():
        raise ValueError('refusing a symlinked raw checkpoint')
    if work.exists() and (not (work / 'owner.json').is_file()
                          or json.loads((work / 'owner.json').read_text()) != owner):
        raise ValueError('refusing an unowned raw checkpoint')
    if not destination.exists():
        work.mkdir(parents=True, exist_ok=True)
        write_checked(work / 'owner.json', json_bytes(owner))
        assembled = work / 'assembled'
        if not assembled.exists():
            collect_gfs_window(pd.Timestamp(date, tz='UTC') - pd.Timedelta(days=1),
                24, 48, protocol['points'], work / 'native', assembled,
                workers=4, retries=3, solar_negative_policy='packing_resolution',
                progress=lambda row: print(date, 'lead', row['lead'], 'cached', row['cached'], flush=True))
        compact_window(assembled, destination)
    verify_window(destination, date, protocol)
    if work.exists():
        if work.is_symlink() or any(path.is_symlink() for path in work.rglob('*')):
            raise ValueError('refusing symlinked raw checkpoint cleanup')
        if set(path.name for path in work.iterdir()) - {'owner.json', 'native', 'assembled'}:
            raise ValueError('unexpected file in owned raw checkpoint')
        shutil.rmtree(work)


def retry_window(output: Path, date: str, protocol: dict, *, pause=time.sleep) -> None:
    """Retry bounded transport failures, retaining completed native lead caches."""
    def acquire() -> None:
        try:
            collect_window(output, date, protocol)
        except (IncompleteRead, ConnectionError, ssl.SSLError) as error:
            if isinstance(error, ssl.SSLCertVerificationError):
                raise
            raise URLError(error) from error

    retry_collection(acquire,
        lambda attempt, error: print(date, 'retry', attempt, type(error).__name__, str(error), flush=True),
        pause=pause)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=ROOT / 'results/weather-transfer-seasonal-r1')
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    output, pilot = args.output.resolve(), args.pilot.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('expanded cache must remain under the repository for relative receipts')
    protocol = initialize(output, pilot)
    if args.plan_only:
        return
    for dates in protocol['dates'].values():
        for date in dates:
            destination = output / f'window-{date}'
            if destination.exists():
                verify_window(destination, date, protocol)
                if (output / 'raw-checkpoints' / date).exists():
                    collect_window(output, date, protocol)
            elif (pilot / f'window-{date}').exists():
                copy_window(pilot / f'window-{date}', destination, date, protocol)
                print(date, 'verified pilot window reused', flush=True)
            else:
                retry_window(output, date, protocol)
                print(date, 'new native window complete', flush=True)
    windows = {date: digest(output / f'window-{date}/manifest.json')
               for dates in protocol['dates'].values() for date in dates}
    verify_files(json.loads((output / 'pre_model_source_receipt.json').read_text())['files'])
    write_checked(output / 'collection_complete.json', json_bytes(dict(
        complete=True, windows=windows, model_fitting=False, new_country_opened=False,
        eligible_for_issued_public=False, available_at=None)))


if __name__ == '__main__':
    main()
