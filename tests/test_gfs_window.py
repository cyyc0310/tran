from copy import deepcopy
import hashlib
import json
from urllib.error import HTTPError, URLError

import pytest

from transcif.data import gfs_window as window
from tests.test_gfs import decoded_windows


@pytest.fixture
def mocked_window(tmp_path, monkeypatch):
    pytest.importorskip("eccodes")
    calls = []
    records = {row["lead_hours"]: row for row in decoded_windows()}
    points = records[23]["points"]
    def collect(run, lead, output, timeout=30):
        calls.append(lead)
        output.mkdir(parents=True)
        fields = {}
        for name in records[lead]["fields"]:
            data = name.encode()
            file = name + ".grib2"
            (output / file).write_bytes(data)
            fields[name] = {"file": file, "sha256": hashlib.sha256(data).hexdigest()}
        (output / "source.idx").write_bytes(b"index")
        report = {"initialization_at": str(run), "lead_hours": lead,
            "source_url": window.BASE + f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}",
            "index_sha256": hashlib.sha256(b"index").hexdigest(), "messages": fields,
            "grib_object": {"last_modified": "2023-01-01T03:39:00Z"},
            "index_object": {"last_modified": "2023-01-01T03:40:00Z"}}
        (output / "metadata.json").write_text(json.dumps(report))
    def decode(path, points):
        lead = int(path.name[1:])
        result = deepcopy(records[lead])
        result["snapshot_metadata_sha256"] = hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest()
        return result
    monkeypatch.setattr(window, "collect_gfs", collect)
    monkeypatch.setattr(window, "decode_gfs_points", decode)
    monkeypatch.setattr(window.time, "sleep", lambda _: None)
    return tmp_path, points, calls, collect


def test_complete_window_resume_reuses_hash_verified_snapshots(mocked_window):
    path, points, calls, _ = mocked_window
    cache, out = path / "cache", path / "output"
    result = window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, cache, out)
    assert sorted(calls) == [23, 24, 25]
    assert result["native_leads"] == [23, 24, 25] and result["all_objects_written_by_origin"]
    assert result["available_at"] is None and not result["eligible_for_issued_public"]
    weather = out / "hourly_points.json"
    assert hashlib.sha256(weather.read_bytes()).hexdigest() == result["weather_sha256"]
    assert [row["shortwave_w_per_m2"] for row in json.loads(weather.read_text())["rows"]] == [220, 50]
    result = window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, cache, path / "resumed")
    assert len(calls) == 3 and all(row["cached"] for row in result["acquisitions"])
    with pytest.raises(ValueError, match="new output"):
        window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, cache, out)


def test_corrupt_cache_rejected_without_network_or_replacement(mocked_window):
    path, points, calls, collect = mocked_window
    run = window.run_timestamp("2023-01-01T00:00Z")
    cache = path / "cache"
    snapshot = window.snapshot_path(cache, run, 23)
    collect(run, 23, snapshot)
    raw = snapshot / "source.idx"
    raw.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash/path mismatch"):
        window.collect_gfs_window(run, 23, 2, points, cache, path / "out")
    assert calls == [23] and raw.read_bytes() == b"corrupt"
    assert not (path / "out").exists()


def test_partial_downloads_survive_failure_and_only_missing_leads_are_resumed(mocked_window, monkeypatch):
    path, points, calls, collect = mocked_window
    def fail_middle(run, lead, output, **kwargs):
        if lead == 24:
            raise URLError("temporary network failure")
        return collect(run, lead, output, **kwargs)
    monkeypatch.setattr(window, "collect_gfs", fail_middle)
    cache, out = path / "cache", path / "out"
    with pytest.raises(URLError):
        window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, cache, out, workers=1, retries=1)
    assert not out.exists()
    completed = set(calls)
    assert 23 in completed and completed <= {23, 25}
    monkeypatch.setattr(window, "collect_gfs", collect)
    result = window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, cache, out, workers=1)
    assert sorted(calls) == [23, 24, 25]
    assert {row["lead"] for row in result["acquisitions"] if not row["cached"]} == {23, 24, 25} - completed


@pytest.mark.parametrize("kind", ["late", "before_run", "naive"])
def test_unusable_object_write_vintage_rejected_before_final_window(mocked_window, kind):
    path, points, calls, collect = mocked_window
    run = window.run_timestamp("2023-01-01T00:00Z")
    cache = path / "cache"
    for lead in [23, 24, 25]:
        collect(run, lead, window.snapshot_path(cache, run, lead))
    metadata = window.snapshot_path(cache, run, 24) / "metadata.json"
    report = json.loads(metadata.read_text())
    report["grib_object"]["last_modified"] = {"late": "2023-01-02T00:00Z",
        "before_run": "2022-12-31T23:00Z", "naive": "2023-01-01T03:00"}[kind]
    metadata.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="object-write evidence"):
        window.collect_gfs_window(run, 23, 2, points, cache, path / "out")
    assert not (path / "out").exists() and len(calls) == 3


@pytest.mark.parametrize("code,attempts", [(503, 2), (429, 2), (404, 1)])
def test_only_transient_http_errors_are_retried(mocked_window, monkeypatch, code, attempts):
    path, points, calls, collect = mocked_window
    failures = []
    def flaky(run, lead, output, **kwargs):
        if lead == 23 and not failures:
            failures.append(lead)
            raise HTTPError("https://archive/", code, "test", {}, None)
        return collect(run, lead, output, **kwargs)
    monkeypatch.setattr(window, "collect_gfs", flaky)
    if code == 404:
        with pytest.raises(HTTPError):
            window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, path / "cache", path / "out", workers=1)
        assert 23 not in calls
    else:
        result = window.collect_gfs_window("2023-01-01T00:00Z", 23, 2, points, path / "cache", path / "out", workers=1)
        assert result["acquisitions"][0]["attempts"] == attempts
