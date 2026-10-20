import hashlib
import json

import numpy as np
import pytest

from transcif.data.gfs_decode import EXPECTED, decode_gfs_points


@pytest.fixture
def encoded_snapshot(tmp_path):
    eccodes = pytest.importorskip("eccodes")
    report = {"initialization_at": "2023-01-01T00:00Z", "lead_hours": 24,
              "grid": "0.25 degree", "messages": {}}
    for name, (param, level_type, level, _) in EXPECTED.items():
        handle = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
        try:
            settings = {"centre": "kwbc", "paramId": param, "typeOfLevel": level_type, "level": level,
                "dataDate": 20230101, "dataTime": 0, "Ni": 1440, "Nj": 721,
                "latitudeOfFirstGridPointInDegrees": 90, "latitudeOfLastGridPointInDegrees": -90,
                "longitudeOfFirstGridPointInDegrees": 0, "longitudeOfLastGridPointInDegrees": 359.75,
                "iDirectionIncrementInDegrees": .25, "jDirectionIncrementInDegrees": .25,
                "stepType": "avg" if name == "shortwave_surface" else "instant",
                "startStep": 18 if name == "shortwave_surface" else 24, "endStep": 24}
            for key, value in settings.items():
                eccodes.codes_set(handle, key, value)
            eccodes.codes_set_values(handle, np.full(1440 * 721, 10.))
            raw = eccodes.codes_get_message(handle)
        finally:
            eccodes.codes_release(handle)
        filename = name + ".grib2"
        (tmp_path / filename).write_bytes(raw)
        report["messages"][name] = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "forecast_end_at": "2023-01-02T00:00Z", "native_step": "18-24 hour ave fcst" if name == "shortwave_surface" else "24 hour fcst",
            "measurement_kind": "interval_mean" if name == "shortwave_surface" else "instantaneous",
            "averaging_start_at": "2023-01-01T18:00Z" if name == "shortwave_surface" else None}
    (tmp_path / "source.idx").write_bytes(b"fixture-index")
    report["index_sha256"] = hashlib.sha256(b"fixture-index").hexdigest()
    (tmp_path / "metadata.json").write_text(json.dumps(report))
    return tmp_path, report


def test_decode_verifies_real_encoding_and_keeps_native_temporal_support(encoded_snapshot):
    path, _ = encoded_snapshot
    result = decode_gfs_points(path, [{"id": "A", "latitude": 40, "longitude": -100}])
    assert result["fields"]["temperature_2m"]["samples"][0]["value"] == 10
    assert result["fields"]["temperature_2m"]["units"] == "K"
    assert result["fields"]["shortwave_surface"]["averaging_start_at"] == "2023-01-01T18:00Z"
    assert not result["eligible_for_issued_public"] and result["available_at"] is None


@pytest.mark.parametrize("change", ["hash", "lead", "window", "field", "kind"])
def test_decode_rejects_mutation_or_index_encoding_disagreement(encoded_snapshot, change):
    path, report = encoded_snapshot
    if change == "hash":
        with (path / "temperature_2m.grib2").open("ab") as stream:
            stream.write(b"changed")
    elif change == "lead":
        report["lead_hours"] = 25
    elif change == "window":
        report["messages"]["shortwave_surface"]["averaging_start_at"] = "2023-01-01T19:00Z"
    elif change == "field":
        report["messages"]["temperature_2m"] = report["messages"]["wind_u_100m"]
    else:
        report["messages"]["shortwave_surface"]["measurement_kind"] = "instantaneous"
    (path / "metadata.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        decode_gfs_points(path, [{"id": "A", "latitude": 40, "longitude": -100}])
