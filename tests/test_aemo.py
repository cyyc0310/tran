import csv
import gzip
import json

import pandas as pd
import pytest

from transcif.data.aemo import normalize_scada, sha256, stage_mms


def snapshot(path, rows, identity=("DISPATCH", "UNIT_SCADA", "1"), columns=None):
    columns = columns or ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["C", "SETP.WORLD", "DVD_TEST", "AEMO", "PUBLIC", "2023/02/07", "00:00:00"])
        writer.writerow(["I", *identity, *columns])
        writer.writerows(["D", *identity, *row] for row in rows)
        writer.writerow(["C", "END OF REPORT", len(rows) + 3])
    return path


def test_scada_retains_native_interval_key_negative_and_missing_in_fixed_nem_time():
    frame = pd.DataFrame([
        ["2023/01/01 00:05:00", "BATTERY", "-4.50"],
        ["2023/07/01 00:05:00", "SOLAR", ""],
    ], columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"])
    result = normalize_scada(frame)
    assert result.dispatch_interval_timestamp_utc.tolist() == [
        pd.Timestamp("2022-12-31T14:05Z"), pd.Timestamp("2023-06-30T14:05Z")]
    assert result.scada_mw.tolist() == ["-4.50", ""]
    assert result.SETTLEMENTDATE.tolist() == frame.SETTLEMENTDATE.tolist()
    assert not {"valid_at", "available_at", "cif_g_per_kwh"}.intersection(result)


@pytest.mark.parametrize("time,duid,value", [
    ("2023/01/01 00:06:00", "A", "3"), ("", "A", "3"),
    ("2023/01/01 00:05:00", "", "3"), ("2023/01/01 00:05:00", "A", "inf"),
    ("2023/01/01 00:05:00", "A", "bad"),
])
def test_scada_rejects_malformed_measurements(time, duid, value):
    with pytest.raises((ValueError, TypeError)):
        normalize_scada(pd.DataFrame([[time, duid, value]], columns=["SETTLEMENTDATE", "DUID", "SCADAVALUE"]))


def test_complete_staging_keeps_snapshot_identity_and_verifies_output_hashes(tmp_path):
    paths = [snapshot(tmp_path / f"v{i}.csv", [["2023/01/01 00:05:00", "A", value]])
             for i, value in enumerate(["-5", ""])]
    out = tmp_path / "output"
    report = stage_mms(paths, out, chunk_size=1)
    assert not report["eligible_for_issued_public"]
    assert len(report["snapshots"]) == 2  # Revisions remain separate, not silently overwritten.
    for path, row in zip(paths, report["snapshots"]):
        assert row["rows"] == 1 and row["available_at"] is None
        assert row["input_sha256"] == sha256(path)
        assert row["output_sha256"] == sha256(out / row["output"])
    assert report["snapshots"][0]["negative_readings"] == 1
    assert report["snapshots"][1]["missing_readings"] == 1
    assert json.loads((out / "metadata.json").read_text()) == report
    with pytest.raises(ValueError, match="already exist"):
        stage_mms(paths, out)


def test_duplicate_keys_across_chunks_fail_atomically_and_do_not_touch_raw(tmp_path):
    row = ["2023/01/01 00:05:00", "A", "4"]
    path = snapshot(tmp_path / "raw.csv", [row, row])
    before = path.read_bytes()
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate AEMO primary key"):
        stage_mms([path], output, chunk_size=1)
    assert not output.exists() and path.read_bytes() == before
    assert not list(tmp_path.glob(".aemo-stage-*"))


@pytest.mark.parametrize("corruption", ["truncate", "count", "mixed", "version", "trailing"])
def test_corrupt_reports_never_become_complete_staging(tmp_path, corruption):
    path = snapshot(tmp_path / "raw.csv", [["2023/01/01 00:05:00", "A", "4"]])
    content = path.read_text()
    if corruption == "truncate":
        content = content[:content.rfind("C,END OF REPORT")]
    elif corruption == "count":
        content = content.replace("END OF REPORT,4", "END OF REPORT,99")
    elif corruption == "mixed":
        content = content.replace("D,DISPATCH,UNIT_SCADA,1", "D,DISPATCH,OTHER,1")
    elif corruption == "version":
        content = content.replace("UNIT_SCADA,1", "UNIT_SCADA,9")
    else:
        content += "C,TRAILING\n"
    path.write_text(content)
    with pytest.raises(ValueError):
        stage_mms([path], tmp_path / "out", chunk_size=1)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("version", ["2", "3"])
def test_registration_preserves_ef_boundary_and_native_text(tmp_path, version):
    columns = ["GENSETID", "LASTCHANGED", "CO2E_EMISSIONS_FACTOR", "CO2E_ENERGY_SOURCE"]
    values = ["001", "2023/12/01 00:00:00", "", "Other, unspecified"]
    path = snapshot(tmp_path / "register.csv", [values],
                    ("PARTICIPANT_REGISTRATION", "GENUNITS", version), columns)
    report = stage_mms([path], tmp_path / "output")
    record = report["snapshots"][0]
    assert record["native_columns"] == columns
    assert record["available_at"] is None  # LASTCHANGED is not publication evidence.
    with gzip.open(tmp_path / "output" / record["output"], "rt", newline="") as stream:
        assert list(csv.reader(stream)) == [columns, values]


def test_empty_snapshot_and_registration_compound_keys(tmp_path):
    path = snapshot(tmp_path / "empty.csv", [])
    report = stage_mms([path], tmp_path / "empty-output")
    assert report["snapshots"][0]["rows"] == 0
    columns = ["DUID", "EFFECTIVEDATE", "GENSETID", "VERSIONNO", "LASTCHANGED"]
    rows = [["A", "2023/01/01 00:00:00", "G1", "1", ""],
            ["A", "2023/01/01 00:00:00", "G2", "1", ""]]
    path = snapshot(tmp_path / "allocation.csv", rows, ("PARTICIPANT_REGISTRATION", "DUALLOC", "1"), columns)
    report = stage_mms([path], tmp_path / "allocation-output", chunk_size=1)
    assert report["snapshots"][0]["rows"] == 2  # A DUID can map to multiple physical gensets.
