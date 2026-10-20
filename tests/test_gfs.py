import hashlib
import json
from copy import deepcopy

import pandas as pd
import pytest

from transcif.data import gfs
from transcif.data.gfs_decode import hourly_gfs_points


def fake_archive():
    descriptors = [("UGRD", "100 mb", "24 hour fcst"),
        ("TMP", "2 m above ground", "24 hour fcst"),
        ("UGRD", "100 m above ground", "24 hour fcst"),
        ("VGRD", "100 m above ground", "24 hour fcst"),
        ("DSWRF", "surface", "18-24 hour ave fcst")]
    lines, messages = [], []
    for i, (variable, level, step) in enumerate(descriptors):
        lines.append(f"{i+1}:{i*64}:d=2023010100:{variable}:{level}:{step}:")
        messages.append(b"GRIB" + bytes([0, 0, 0, 2]) + (64).to_bytes(8, "big") + bytes([i]) * 44 + b"7777")
    return "\n".join(lines).encode(), b"".join(messages)


def mocked_request(index, content, corruption=None):
    evidence = {"etag": '"version-1"', "last-modified": "Sun, 01 Jan 2023 03:39:34 GMT",
                "content-length": str(len(content))}

    def request(url, *, method="GET", headers=None, **options):
        if method == "HEAD":
            return evidence, b""
        if url.endswith(".idx"):
            return {**evidence, "content-length": str(len(index))}, index
        assert headers["If-Match"] == evidence["etag"]
        assert options["expected_status"] == 206
        start, end = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        result = {**evidence, "content-range": f"bytes {start}-{end}/{len(content)}"}
        body = content[start:end + 1]
        if corruption == "range":
            result["content-range"] = "bytes 0-0/1"
        elif corruption == "version":
            result["etag"] = '"different"'
        elif corruption == "truncated":
            body = body[:-1]
        elif corruption == "grib":
            body = b"WRONG" + body[5:]
        return result, body
    return request


def test_native_index_disambiguates_height_and_retains_six_hour_solar_window():
    index, content = fake_archive()
    result = gfs.select_messages(index.decode(), "2023-01-01T00:00Z", 24, len(content))
    assert result["wind_u_100m"]["start_byte"] == 128  # Not 100 mb pressure-level wind.
    solar = result["shortwave_surface"]
    assert solar["measurement_kind"] == "interval_mean"
    assert pd.Timestamp(solar["averaging_start_at"]) == pd.Timestamp("2023-01-01T18:00Z")
    assert pd.Timestamp(solar["forecast_end_at"]) == pd.Timestamp("2023-01-02T00:00Z")
    assert result["temperature_2m"]["averaging_start_at"] is None


@pytest.mark.parametrize("corruption", ["run", "level", "offset", "step", "duplicate", "missing"])
def test_unreviewed_or_misaligned_index_is_rejected(corruption):
    index, content = fake_archive()
    text = index.decode()
    if corruption == "run":
        text = text.replace("d=2023010100", "d=2023010106")
    elif corruption == "level":
        text = text.replace("100 m above ground", "10 m above ground")
    elif corruption == "offset":
        text = text.replace("2:64:", "2:0:")
    elif corruption == "step":
        text = text.replace("18-24 hour ave fcst", "24 hour fcst")
    elif corruption == "duplicate":
        text += "\n6:320:d=2023010100:TMP:2 m above ground:24 hour fcst:"
        content += bytes(64)
    else:
        text = "\n".join(text.splitlines()[:-1])
    with pytest.raises(ValueError):
        gfs.select_messages(text, "2023-01-01T00:00Z", 24, len(content))


def test_collection_pins_versions_hashes_exact_messages_and_keeps_availability_unset(tmp_path, monkeypatch):
    index, content = fake_archive()
    monkeypatch.setattr(gfs, "_request", mocked_request(index, content))
    output = tmp_path / "archive"
    report = gfs.collect_gfs("2023-01-01T08:00+08:00", 24, output)
    assert report["source_url"].endswith("gfs.20230101/00/atmos/gfs.t00z.pgrb2.0p25.f024")
    assert report["available_at"] is None and not report["eligible_for_issued_public"]
    assert report["grib_object"]["last_modified"] == "2023-01-01T03:39:34+00:00"
    assert report["index_sha256"] == hashlib.sha256((output / "source.idx").read_bytes()).hexdigest()
    for row in report["messages"].values():
        raw = (output / row["file"]).read_bytes()
        assert raw == content[row["start_byte"]:row["end_byte"]+1]
        assert row["sha256"] == hashlib.sha256(raw).hexdigest()
    assert json.loads((output / "metadata.json").read_text()) == report
    with pytest.raises(ValueError, match="new output"):
        gfs.collect_gfs("2023-01-01T00:00Z", 24, output)


@pytest.mark.parametrize("corruption", ["range", "version", "truncated", "grib"])
def test_bad_archive_ranges_never_become_complete_output(tmp_path, monkeypatch, corruption):
    index, content = fake_archive()
    monkeypatch.setattr(gfs, "_request", mocked_request(index, content, corruption))
    output = tmp_path / "archive"
    with pytest.raises(ValueError):
        gfs.collect_gfs("2023-01-01T00:00Z", 24, output)
    assert not output.exists() and not list(tmp_path.glob(".gfs-stage-*"))


@pytest.mark.parametrize("run,lead", [("2023-01-01", 24), ("2023-01-01T01:00Z", 24),
    ("2023-01-01T00:01Z", 24), ("2023-01-01T00:00Z", 0), ("2023-01-01T00:00Z", 121)])
def test_invalid_cycles_fail_before_network(tmp_path, monkeypatch, run, lead):
    def fail(*args, **kwargs):
        pytest.fail("invalid requests should not contact archive")
    monkeypatch.setattr(gfs, "_request", fail)
    with pytest.raises(ValueError):
        gfs.collect_gfs(run, lead, tmp_path / "archive")


def test_server_ignoring_range_is_rejected_before_body_read(monkeypatch):
    class Response:
        status = 200
        headers = {"Content-Length": "534412062"}
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, *args):
            pytest.fail("must not download whole GRIB when Range is ignored")
    monkeypatch.setattr(gfs, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="expected HTTP 206"):
        gfs._request(gfs.BASE + "test", headers={"Range": "bytes=0-63"}, expected_status=206, max_bytes=64)


def decoded_windows():
    records = []
    run = pd.Timestamp("2023-01-01T00:00Z")
    for lead, start, mean in [(23, 18, 100), (24, 18, 120), (25, 24, 50)]:
        fields = {}
        for name, value, units in [("temperature_2m", 280, "K"), ("wind_u_100m", 3, "m s**-1"),
                                  ("wind_v_100m", 4, "m s**-1"), ("shortwave_surface", mean, "W m**-2")]:
            fields[name] = {"units": units, "measurement_kind": "interval_mean" if name == "shortwave_surface" else "instantaneous",
                "forecast_end_at": str(run + pd.Timedelta(hours=lead)),
                "averaging_start_at": str(run + pd.Timedelta(hours=start)) if name == "shortwave_surface" else None,
                "samples": [{"id": "A", "value": value}]}
        records.append({"initialization_at": str(run), "lead_hours": lead,
            "points": [{"id": "A", "latitude": 40, "longitude": -100}],
            "fields": fields, "snapshot_metadata_sha256": str(lead)})
    return records


def test_hourly_shortwave_deaccumulates_native_window_and_handles_reset():
    report = hourly_gfs_points(decoded_windows(), 23, 25)
    assert [row["shortwave_w_per_m2"] for row in report["rows"]] == [220, 50]
    assert report["rows"][0]["temperature_c"] == pytest.approx(6.85)
    assert report["rows"][0]["wind_speed_ms"] == 5
    assert pd.Timestamp(report["rows"][0]["valid_at"]) == pd.Timestamp("2023-01-01T23:00Z")
    assert report["available_at"] is None and not report["eligible_for_issued_public"]


@pytest.mark.parametrize("change", ["missing", "cycle", "points", "window", "negative", "units", "duplicate"])
def test_hourly_weather_never_stitches_cycles_interpolates_or_clamps(change):
    records = deepcopy(decoded_windows())
    if change == "missing":
        records.pop(0)
    elif change == "cycle":
        records[1]["initialization_at"] = "2023-01-01T06:00Z"
    elif change == "points":
        records[1]["points"][0]["latitude"] = 45
    elif change == "window":
        records[0]["fields"]["shortwave_surface"]["averaging_start_at"] = "2023-01-01T19:00Z"
    elif change == "negative":
        records[0]["fields"]["shortwave_surface"]["samples"][0]["value"] = 1000
    elif change == "units":
        records[0]["fields"]["temperature_2m"]["units"] = "C"
    else:
        records.append(deepcopy(records[0]))
    with pytest.raises(ValueError):
        hourly_gfs_points(records, 23, 25)
