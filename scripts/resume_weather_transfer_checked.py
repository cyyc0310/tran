"""Resume the frozen calendar, retrying evidenced short reads and network errors."""
from __future__ import annotations

import argparse
from http.client import IncompleteRead
import json
from pathlib import Path
import ssl
import tempfile
import time
from urllib.error import URLError

import collect_weather_transfer_expansion as collection
from collect_weather_transfer_pilot import ROOT, digest
from run_weather_transfer_pipeline import retry_collection
from transcif.data.gfs_transport import truncated_response


def retry_checked_window(output: Path, date: str, protocol: dict, *, pause=time.sleep) -> None:
    """Bad ETag/range/hash remains fatal; a rejected, verified short read can retry."""
    def acquire() -> None:
        try:
            collection.collect_window(output, date, protocol)
        except ValueError as error:
            evidence = truncated_response(error)
            if evidence is None:
                raise
            directory = output / 'transport-events'; directory.mkdir(exist_ok=True)
            if directory.is_symlink():
                raise ValueError('transport event directory must not be symlinked')
            with tempfile.NamedTemporaryFile(mode='wb', prefix=f'{date}-', suffix='.json',
                                             dir=directory, delete=False) as stream:
                stream.write(collection.json_bytes(evidence))
            print(date, 'rejected short read', evidence['field'], evidence['received_bytes'],
                  '/', evidence['expected_bytes'], 'evidence', Path(stream.name).name, flush=True)
            raise URLError('GFS response truncated after verified range/version/GRIB header') from error
        except (IncompleteRead, ConnectionError, ssl.SSLError) as error:
            if isinstance(error, ssl.SSLCertVerificationError):
                raise
            raise URLError(error) from error

    retry_collection(acquire,
        lambda attempt, error: print(date, 'bounded network retry', attempt, str(error), flush=True), pause=pause)


def resume_protocol(output: Path) -> dict:
    """Verify the closed original receipt, without rebuilding its old file glob."""
    receipt = json.loads((output / 'pre_model_source_receipt.json').read_text())
    collection.verify_files(receipt['files'])
    protocol = json.loads((output / 'plan.json').read_text())
    if protocol != collection.plan():
        raise ValueError('fixed original calendar changed')
    for name, expected in json.loads((output / 'cohort_seal.json').read_text()).items():
        if digest(output / name) != expected:
            raise ValueError('original cohort/plan seal changed')
    return protocol


def verify_completion(output: Path, completed: dict, windows: dict) -> None:
    if not completed['complete'] or completed['windows'] != windows:
        raise ValueError('existing completion does not bind the fixed windows')
    if 'transport_recovery_receipt_sha256' in completed:
        if digest(output / 'transport_recovery_receipt.json') != completed['transport_recovery_receipt_sha256']:
            raise ValueError('transport recovery receipt changed')
        events = completed['transport_events']
        if (output / 'transport-events').is_symlink():
            raise ValueError('transport event directory must not be symlinked')
        actual = {str(path.relative_to(output)) for path in (output / 'transport-events').glob('*.json')}
        if actual != set(events):
            raise ValueError('transport event inventory changed')
        for name, expected in events.items():
            path = output / name
            if (Path(name).is_absolute() or len(Path(name).parts) != 2
                    or Path(name).parts[0] != 'transport-events' or path.is_symlink()
                    or not path.is_file() or digest(path) != expected):
                raise ValueError('transport event missing or changed')


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pilot', type=Path, default=ROOT / 'results/weather-transfer-seasonal-r1')
    args = parser.parse_args()
    output, pilot = args.output.resolve(), args.pilot.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError('cache must remain under the repository')
    protocol = resume_protocol(output)
    paths = [Path(__file__).resolve(), ROOT / 'src/transcif/data/gfs_transport.py',
             ROOT / 'tests/test_gfs_transport_recovery.py', ROOT / 'docs/GFS_TRANSPORT_RECOVERY_PROTOCOL.md',
             output / 'pre_model_source_receipt.json']
    receipt = dict(mode='original validators unchanged; evidenced read truncation is a transport retry',
        window_attempts=3, retries_wrong_etag=False, retries_wrong_range=False, retries_hash_errors=False,
        files={str(path.relative_to(ROOT)): digest(path) for path in paths})
    recovery_receipt = output / 'transport_recovery_receipt.json'
    collection.write_checked(recovery_receipt, collection.json_bytes(receipt))
    collection.verify_files(receipt['files'])
    for dates in protocol['dates'].values():
        for date in dates:
            destination = output / f'window-{date}'
            if destination.exists():
                collection.verify_window(destination, date, protocol)
                if (output / 'raw-checkpoints' / date).exists():
                    collection.collect_window(output, date, protocol)
                print(date, 'verified complete checkpoint', flush=True)
            elif (pilot / f'window-{date}').exists():
                collection.copy_window(pilot / f'window-{date}', destination, date, protocol)
                print(date, 'verified pilot checkpoint reused', flush=True)
            else:
                retry_checked_window(output, date, protocol)
                print(date, 'strict native window complete', flush=True)
    windows = {date: digest(output / f'window-{date}/manifest.json')
               for dates in protocol['dates'].values() for date in dates}
    collection.verify_files(receipt['files'])
    collection.verify_files(json.loads((output / 'pre_model_source_receipt.json').read_text())['files'])
    completed = output / 'collection_complete.json'
    if completed.exists():
        verify_completion(output, json.loads(completed.read_text()), windows)
        return
    events = {str(path.relative_to(output)): digest(path) for path in sorted((output / 'transport-events').glob('*.json'))}
    collection.write_checked(completed, collection.json_bytes(dict(
        complete=True, windows=windows, model_fitting=False, new_country_opened=False,
        eligible_for_issued_public=False, available_at=None,
        transport_recovery_receipt_sha256=digest(recovery_receipt), transport_events=events)))


if __name__ == '__main__':
    main()
