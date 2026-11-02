"""Transport recovery requires exact range/version and a matching GRIB header."""
import importlib
import json
from pathlib import Path

import pandas as pd
import pytest

from transcif.data import gfs
from transcif.data.gfs_transport import truncated_response


def rejected_snapshot(monkeypatch, tmp_path, *, bad_etag=False, bad_range=False, full_body=False, bad_magic=False):
    rows = ['1:0:d=2023010100:TMP:2 m above ground:1 hour fcst:',
            '2:40:d=2023010100:UGRD:100 m above ground:1 hour fcst:',
            '3:80:d=2023010100:VGRD:100 m above ground:1 hour fcst:',
            '4:120:d=2023010100:DSWRF:surface:0-1 hour ave fcst:']

    def request(url, *, method='GET', headers=None, **kwargs):
        if method == 'HEAD' or url.endswith('.idx'):
            metadata = {'content-length': '160', 'etag': '"fixed"',
                        'last-modified': 'Sun, 01 Jan 2023 00:30:00 GMT'}
            return metadata, b'' if method == 'HEAD' else '\n'.join(rows).encode()
        start, end = map(int, headers['Range'].removeprefix('bytes=').split('-'))
        response = {'content-length': '40', 'etag': '"wrong"' if bad_etag else '"fixed"',
                    'content-range': 'bytes 1-40/160' if bad_range else f'bytes {start}-{end}/160'}
        body = (b'HTML' if bad_magic else b'GRIB') + b'\0\0\0\x02' + (40).to_bytes(8, 'big')
        body += b'x' * (24 if full_body else 4)
        return response, body

    monkeypatch.setattr(gfs, '_request', request)
    path = tmp_path / 'snapshot'
    with pytest.raises(ValueError) as caught:
        gfs.collect_gfs(pd.Timestamp('2023-01-01', tz='UTC'), 1, path)
    assert not path.exists()
    return caught.value


def test_verified_short_response_is_transport_evidence_not_accepted_data(monkeypatch, tmp_path):
    evidence = truncated_response(rejected_snapshot(monkeypatch, tmp_path))
    assert evidence['received_bytes'] == 20 and evidence['expected_bytes'] == 40
    assert evidence['partial_snapshot_accepted'] is False


@pytest.mark.parametrize('options', [{'bad_etag': True}, {'bad_range': True}, {'full_body': True}, {'bad_magic': True}])
def test_identity_or_grib_failures_remain_fatal(monkeypatch, tmp_path, options):
    assert truncated_response(rejected_snapshot(monkeypatch, tmp_path, **options)) is None


def test_recovery_retries_only_verified_truncation_and_records_it(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('resume_weather_transfer_checked')
    error = rejected_snapshot(monkeypatch, tmp_path)
    calls = []

    def collect(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error

    monkeypatch.setattr(module.collection, 'collect_window', collect)
    pauses = []
    module.retry_checked_window(tmp_path, '2023-01-25', {}, pause=pauses.append)
    assert len(calls) == 2 and pauses == [10.]
    assert len(list((tmp_path / 'transport-events').glob('*.json'))) == 1


def test_unrecognized_integrity_error_is_never_retried(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('resume_weather_transfer_checked')
    calls = []

    def collect(*args):
        calls.append(args)
        raise ValueError('cache hash mismatch')

    monkeypatch.setattr(module.collection, 'collect_window', collect)
    with pytest.raises(ValueError, match='hash mismatch'):
        module.retry_checked_window(tmp_path, '2023-01-25', {}, pause=lambda _: None)
    assert len(calls) == 1


@pytest.mark.parametrize('delete', [False, True])
def test_completed_transport_events_are_reverified(monkeypatch, tmp_path, delete):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('resume_weather_transfer_checked')
    receipt = tmp_path / 'transport_recovery_receipt.json'; receipt.write_text('{}')
    events = tmp_path / 'transport-events'; events.mkdir()
    event = events / 'read.json'; event.write_text('original')
    completed = dict(complete=True, windows={}, transport_recovery_receipt_sha256=module.digest(receipt),
                     transport_events={'transport-events/read.json': module.digest(event)})
    module.verify_completion(tmp_path, completed, {})
    if delete:
        event.unlink()
    else:
        event.write_text('changed')
    with pytest.raises(ValueError, match='transport event'):
        module.verify_completion(tmp_path, completed, {})


def test_resume_verifies_old_closed_receipt_without_rebuilding_glob(monkeypatch, tmp_path):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'scripts'))
    module = importlib.import_module('resume_weather_transfer_checked')
    monkeypatch.setattr(module.collection, 'ROOT', tmp_path)
    monkeypatch.setattr(module.collection, 'initialize', lambda *args: pytest.fail('old initialization rerun'))
    source = tmp_path / 'input.txt'; source.write_text('fixed')
    receipt = tmp_path / 'pre_model_source_receipt.json'
    receipt.write_text(json.dumps({'files': {'input.txt': module.digest(source)}}))
    original = receipt.read_bytes()
    (tmp_path / 'plan.json').write_text(json.dumps(module.collection.plan()))
    (tmp_path / 'cohort.csv').write_text('original cohort')
    seal = {name: module.digest(tmp_path / name) for name in ['plan.json', 'cohort.csv']}
    (tmp_path / 'cohort_seal.json').write_text(json.dumps(seal))
    assert module.resume_protocol(tmp_path) == module.collection.plan()
    assert receipt.read_bytes() == original
