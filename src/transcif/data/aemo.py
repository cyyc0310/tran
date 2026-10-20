"""Versioned, lossless staging of AEMO DVD snapshots, not a label builder."""

import csv
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile

import numpy as np
import pandas as pd


SCADA_REFERENCE = "https://visualisations.aemo.com.au/aemo/nemweb/MMSDataModelReport/Electricity/MMS%20Data%20Model%20Report_files/MMS_123.htm"
REGISTRATION_REFERENCE = "https://visualisations.aemo.com.au/aemo/nemweb/mmsdatamodelreport/electricity/mms%20data%20model%20report_files/MMS_242.htm"
TABLES = {
    ("DISPATCH", "UNIT_SCADA", "1"): ("SETTLEMENTDATE", "DUID"),
    ("PARTICIPANT_REGISTRATION", "DUALLOC", "1"): ("DUID", "EFFECTIVEDATE", "GENSETID", "VERSIONNO"),
    ("PARTICIPANT_REGISTRATION", "DUDETAILSUMMARY", "6"): ("DUID", "START_DATE"),
    ("PARTICIPANT_REGISTRATION", "GENUNITS", "2"): ("GENSETID",),
    ("PARTICIPANT_REGISTRATION", "GENUNITS", "3"): ("GENSETID",),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_scada(frame):
    """Retain the native interval key; do not assign an energy-interval start.

    NEM timestamps use fixed UTC+10, including southern summer. Documentation
    describes the MW reading as instantaneous at dispatch start, but identifies
    SETTLEMENTDATE only as the dispatch interval. No five-minute shift or energy
    integration is inferred here.
    """
    if list(frame.columns) != ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]:
        raise ValueError("unreviewed UNIT_SCADA columns")
    if frame.DUID.isna().any() or frame.DUID.eq("").any():
        raise ValueError("empty SCADA DUID")
    times = pd.to_datetime(frame.SETTLEMENTDATE, format="%Y/%m/%d %H:%M:%S", errors="raise")
    if times.isna().any() or not times.eq(times.dt.floor("5min")).all():
        raise ValueError("finite five-minute-aligned NEM interval timestamps required")
    values = pd.to_numeric(frame.SCADAVALUE.replace("", np.nan), errors="raise")
    if np.isinf(values.to_numpy(dtype=float)).any():
        raise ValueError("infinite SCADA reading")
    result = frame.rename(columns={"DUID": "duid", "SCADAVALUE": "scada_mw"}).copy()
    result.insert(0, "dispatch_interval_timestamp_utc", times.dt.tz_localize("Etc/GMT-10").dt.tz_convert("UTC"))
    return result


def _stage_snapshot(path, destination, chunk_size):
    """Validate the complete report and primary keys with bounded RAM."""
    input_hash = sha256(path)
    database = destination.with_suffix(".keys.sqlite")
    count, negative, missing, units = 0, 0, 0, set()
    first, last = None, None
    try:
        with path.open(newline="", encoding="utf-8-sig") as source, sqlite3.connect(database) as keys:
            reader = csv.reader(source)
            envelope, header = next(reader, []), next(reader, [])
            if len(envelope) < 7 or envelope[:1] != ["C"] or envelope[1] == "END OF REPORT":
                raise ValueError("missing AEMO report envelope")
            identity = tuple(header[1:4])
            if header[:1] != ["I"] or identity not in TABLES:
                raise ValueError(f"unreviewed AEMO table/version: {identity}")
            columns, primary = header[4:], TABLES[identity]
            if len(set(columns)) != len(columns) or not set(primary).issubset(columns):
                raise ValueError("invalid AEMO columns/primary key")
            if identity[1] == "UNIT_SCADA" and columns != ["SETTLEMENTDATE", "DUID", "SCADAVALUE"]:
                raise ValueError("unreviewed UNIT_SCADA columns")
            indices = [columns.index(name) for name in primary]
            key_columns = ",".join(f"k{i} TEXT NOT NULL" for i in range(len(primary)))
            key_names = ",".join(f"k{i}" for i in range(len(primary)))
            keys.execute(f"CREATE TABLE seen ({key_columns}, PRIMARY KEY ({key_names})) WITHOUT ROWID")
            insert = "INSERT INTO seen VALUES (" + ",".join("?" for _ in primary) + ")"
            # A temporary on-disk index avoids retaining millions of tuples in RAM.
            keys.execute("PRAGMA journal_mode=OFF")
            keys.execute("PRAGMA synchronous=OFF")
            with gzip.open(destination, "wt", newline="", encoding="utf-8", compresslevel=1) as target:
                writer = csv.writer(target)
                written_header = False

                def flush(rows):
                    nonlocal count, negative, missing, first, last, written_header
                    key_rows = [tuple(row[i] for i in indices) for row in rows]
                    if any(not all(key) for key in key_rows):
                        raise ValueError("empty AEMO primary key")
                    try:
                        keys.executemany(insert, key_rows)
                    except sqlite3.IntegrityError as error:
                        raise ValueError("duplicate AEMO primary key within snapshot") from error
                    keys.commit()
                    frame = pd.DataFrame(rows, columns=columns)
                    if identity[1] == "UNIT_SCADA":
                        frame = normalize_scada(frame)
                        times = frame.dispatch_interval_timestamp_utc
                        first = times.min() if first is None else min(first, times.min())
                        last = times.max() if last is None else max(last, times.max())
                        numbers = pd.to_numeric(frame.scada_mw.replace("", np.nan))
                        negative += int(numbers.lt(0).sum())
                        missing += int(numbers.isna().sum())
                        units.update(frame.duid)
                    if not written_header:
                        writer.writerow(frame.columns)
                        written_header = True
                    writer.writerows(frame.itertuples(index=False, name=None))
                    count += len(rows)

                pending, footer = [], None
                for number, row in enumerate(reader, start=3):
                    if row[:2] == ["C", "END OF REPORT"]:
                        if len(row) != 3 or row[2] != str(number) or next(reader, None) is not None:
                            raise ValueError("invalid AEMO footer record count or trailing records")
                        footer = row
                        break
                    if row[:4] != ["D", *identity] or len(row) != len(columns) + 4:
                        raise ValueError(f"invalid or mixed AEMO record at {number}")
                    pending.append(row[4:])
                    if len(pending) >= chunk_size:
                        flush(pending)
                        pending = []
                if footer is None:
                    raise ValueError("missing AEMO report footer (possibly truncated download)")
                if pending:
                    flush(pending)
                if not written_header:
                    writer.writerow(normalize_scada(pd.DataFrame(columns=columns)).columns
                                    if identity[1] == "UNIT_SCADA" else columns)
        if sha256(path) != input_hash:
            raise ValueError("AEMO input changed during staging")
        result = {"input": str(path.resolve()), "input_sha256": input_hash,
            "output": destination.name, "output_sha256": sha256(destination),
            "table": list(identity), "native_columns": columns, "primary_key": list(primary),
            "rows": count, "envelope": envelope, "available_at": None,
            "publication_note": "report creation and LASTCHANGED are not original public availability"}
        if identity[1] == "UNIT_SCADA":
            result.update({"first_native_interval_utc": str(first) if first is not None else None,
                "last_native_interval_utc": str(last) if last is not None else None,
                "duids": sorted(units), "negative_readings": negative, "missing_readings": missing,
                "measurement": "instantaneous MW, not five-minute energy or an interval average",
                "interval_start_mapping": "unresolved; no shift or integration applied"})
        return result
    finally:
        database.unlink(missing_ok=True)


def stage_mms(paths, output, chunk_size=100_000, progress=None):
    """Keep snapshots independent, including all registration versions and fuel text.

    A whole output is committed atomically after validation. Invalid or truncated
    inputs cannot leave a seemingly complete staging directory.
    """
    paths, output = [Path(path) for path in paths], Path(output)
    if not paths or len(set(path.resolve() for path in paths)) != len(paths) or chunk_size < 1:
        raise ValueError("unique input snapshots and positive chunk size required")
    if output.exists():
        raise ValueError("AEMO output must not already exist")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".aemo-stage-", dir=output.parent) as directory:
        staged = Path(directory) / "complete"
        staged.mkdir()
        snapshots = []
        for index, path in enumerate(paths):
            record = _stage_snapshot(path, staged / f"{index:03d}-{path.stem}.csv.gz", chunk_size)
            snapshots.append(record)
            if progress is not None:
                progress(record)
        report = {"provider": "AEMO DVD", "schema_version": 1,
            "references": [SCADA_REFERENCE, REGISTRATION_REFERENCE],
            "eligible_for_issued_public": False, "snapshots": snapshots,
            "time_convention": "native NEM interval key converted from fixed UTC+10, without DST",
            "registration_note": "native records retained; no retrospective as-of join, fuel merge or EF conversion",
            "missing_evidence": ["original forecast/config/label publication vintages",
                "SCADA interval-key to measurement-instant mapping and energy integration method",
                "effective unit-to-genset/region joins and allocation of multi-genset dispatch units",
                "complete generation boundary including non-SCADA and distributed generation",
                "CO2e versus direct CO2 EF boundary, storage and net interchange accounting"]}
        (staged / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.rename(output)
    return report
