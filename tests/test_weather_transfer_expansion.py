"""Expanded calendar and resumable raw cache integrity contracts."""
import json
from http.client import IncompleteRead
from pathlib import Path
import ssl
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import collect_weather_transfer_expansion as expansion  # noqa: E402


def test_expanded_calendar_is_fixed_and_does_not_mutate_pilot():
    before = expansion.pilot_plan()
    protocol = expansion.plan()
    assert len(protocol['dates']['calibration']) == len(protocol['dates']['development']) == 12
    assert protocol['dates']['calibration'][0] == '2023-01-15'
    assert protocol['dates']['development'][-1] == '2023-12-25'
    assert before == expansion.pilot_plan()
    assert protocol['points'] == before['points']
    assert protocol['new_country_opened'] is False and protocol['confirmation'] is False
    assert not set(protocol['dates']['calibration']) & set(protocol['dates']['development'])


def test_existing_initialization_is_never_silently_overwritten(tmp_path):
    path = tmp_path / 'state'
    expansion.write_checked(path, b'fixed')
    expansion.write_checked(path, b'fixed')
    with pytest.raises(ValueError, match='initialization differs'):
        expansion.write_checked(path, b'changed')
    assert path.read_bytes() == b'fixed'


def test_window_requires_exact_run_hour_grid_and_uncertified_flags(monkeypatch, tmp_path):
    protocol = expansion.plan(); date = '2023-07-25'
    origin = pd.Timestamp(date, tz='UTC')
    weather = dict(initialization_at=str(origin-pd.Timedelta(days=1)), first_lead=24, end_lead=72,
        rows=[dict(point=point['id'], valid_at=str(instant)) for point in protocol['points']
              for instant in pd.date_range(origin, periods=48, freq='h')])
    monkeypatch.setattr(expansion, 'load_compact_window', lambda *args: weather)
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps(dict(eligible_for_issued_public=False, available_at=None)))
    expansion.verify_window(tmp_path, date, protocol)
    weather['rows'][-1] = weather['rows'][0]
    with pytest.raises(ValueError, match='expanded plan'):
        expansion.verify_window(tmp_path, date, protocol)


def test_unowned_checkpoint_is_preserved(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    work = tmp_path / 'raw-checkpoints/2023-07-25'; work.mkdir(parents=True)
    private = work / 'do-not-delete'; private.write_text('keep')
    with pytest.raises(ValueError, match='unowned'):
        expansion.collect_window(tmp_path, '2023-07-25', expansion.plan())
    assert private.read_text() == 'keep'


def test_failed_window_reuses_completed_lead_cache_on_retry(monkeypatch, tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    attempts = []

    def acquire(run, first, horizon, points, cache, assembled, **kwargs):
        attempts.append(cache)
        if len(attempts) == 1:
            cache.mkdir(parents=True); (cache / 'one-lead').write_text('sealed')
            raise TimeoutError('transient')
        assert (cache / 'one-lead').read_text() == 'sealed'
        assembled.mkdir()

    monkeypatch.setattr(expansion, 'collect_gfs_window', acquire)
    monkeypatch.setattr(expansion, 'compact_window', lambda source, destination: destination.mkdir())
    monkeypatch.setattr(expansion, 'verify_window', lambda *args: None)
    with pytest.raises(TimeoutError):
        expansion.collect_window(tmp_path, '2023-07-25', expansion.plan())
    expansion.collect_window(tmp_path, '2023-07-25', expansion.plan())
    assert attempts[0] == attempts[1]
    assert (tmp_path / 'window-2023-07-25').is_dir()
    assert not (tmp_path / 'raw-checkpoints/2023-07-25').exists()


@pytest.mark.parametrize('error', [IncompleteRead(b'partial', 10), ConnectionResetError('reset'),
                                 ssl.SSLEOFError('closed')])
def test_read_transport_failure_is_retried_without_real_sleep(monkeypatch, tmp_path, error):
    calls = []

    def collect(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error

    monkeypatch.setattr(expansion, 'collect_window', collect)
    pauses = []
    expansion.retry_window(tmp_path, '2023-07-25', expansion.plan(), pause=pauses.append)
    assert len(calls) == 2 and pauses == [10.]


def test_completed_compact_can_cleanup_owned_raw_without_network(monkeypatch, tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    work = tmp_path / 'raw-checkpoints/2023-07-25'; work.mkdir(parents=True)
    (work / 'owner.json').write_bytes(expansion.json_bytes(dict(
        date='2023-07-25', plan_sha256=expansion.digest(tmp_path / 'plan.json'))))
    (work / 'native').mkdir(); (work / 'assembled').mkdir()
    destination = tmp_path / 'window-2023-07-25'; destination.mkdir()
    monkeypatch.setattr(expansion, 'verify_window', lambda *args: None)
    monkeypatch.setattr(expansion, 'collect_gfs_window', lambda *args, **kwargs: pytest.fail('network called'))
    expansion.collect_window(tmp_path, '2023-07-25', expansion.plan())
    assert destination.is_dir() and not work.exists()


def test_symlinked_checkpoint_parent_is_never_written_or_cleaned(tmp_path):
    (tmp_path / 'plan.json').write_text('{}')
    outside = tmp_path / 'outside'; outside.mkdir()
    (tmp_path / 'raw-checkpoints').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='symlinked'):
        expansion.collect_window(tmp_path, '2023-07-25', expansion.plan())
    assert not list(outside.iterdir())
