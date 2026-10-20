"""Collect native GFS single-run GRIB messages with archive-time evidence."""

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path
import re
import tempfile
from urllib.request import Request, urlopen

import pandas as pd


BASE = "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
REFERENCE = "https://registry.opendata.aws/noaa-gfs-bdp-pds/"
FIELDS = {"temperature_2m": ("TMP", "2 m above ground"),
          "wind_u_100m": ("UGRD", "100 m above ground"),
          "wind_v_100m": ("VGRD", "100 m above ground"),
          "shortwave_surface": ("DSWRF", "surface")}


def _request(url, *, method="GET", headers=None, max_bytes=2_000_000,
             expected_status=200, timeout=30):
    request = Request(url, method=method, headers=headers or {})
    with urlopen(request, timeout=timeout) as response:
        if response.status != expected_status:
            raise ValueError(f"expected HTTP {expected_status}, got {response.status}")
        metadata = {key.lower(): value for key, value in response.headers.items()}
        if method == "HEAD":
            return metadata, b""
        if "content-length" in metadata and int(metadata["content-length"]) > max_bytes:
            raise ValueError("archive response exceeds requested byte budget")
        content = response.read(max_bytes + 1)
        if len(content) > max_bytes:
            raise ValueError("archive response exceeds requested byte budget")
        return metadata, content


def run_timestamp(run):
    run = pd.Timestamp(run)
    if pd.isna(run) or run.tz is None:
        raise ValueError("explicit timezone-aware GFS run required")
    run = run.tz_convert("UTC")
    if run != run.floor("h") or run.hour % 6:
        raise ValueError("GFS initialization must be a 00/06/12/18 UTC cycle")
    return run


def select_messages(index, run, lead, file_size):
    """Select exact levels, preserving instantaneous versus averaged support.

    In particular, 100 mb is not 100 m above ground, and the solar field at
    lead 24 can describe an 18–24 hour average rather than a one-hour mean.
    """
    run = run_timestamp(run)
    if not isinstance(lead, int) or isinstance(lead, bool) or not 1 <= lead <= 120 or file_size < 1:
        raise ValueError("hourly forecast lead 1..120 and positive file size required")
    records = [line.split(":") for line in index.splitlines() if line.strip()]
    if not records or any(len(row) < 7 or row[2] != f"d={run:%Y%m%d%H}" for row in records):
        raise ValueError("invalid GFS index or mixed initialization times")
    try:
        offsets = [int(row[1]) for row in records] + [file_size]
    except ValueError as error:
        raise ValueError("invalid GFS byte offsets") from error
    if offsets[0] != 0 or any(a >= b for a, b in zip(offsets, offsets[1:])):
        raise ValueError("ordered unique GFS byte offsets within the object required")
    result = {}
    for name, (variable, level) in FIELDS.items():
        matches = [i for i, row in enumerate(records) if tuple(row[3:5]) == (variable, level)]
        if len(matches) != 1:
            raise ValueError(f"exactly one GFS message required for {name}")
        i = matches[0]
        step = records[i][5]
        instant = re.fullmatch(r"(\d+) hour fcst", step)
        average = re.fullmatch(r"(\d+)-(\d+) hour ave fcst", step)
        if name == "shortwave_surface":
            if not average or int(average[2]) != lead or not 0 <= int(average[1]) < lead:
                raise ValueError("explicit shortwave averaging window ending at forecast lead required")
            start = str(run + pd.Timedelta(hours=int(average[1])))
            kind = "interval_mean"
        else:
            if not instant or int(instant[1]) != lead:
                raise ValueError(f"instantaneous forecast at requested lead required for {name}")
            start, kind = None, "instantaneous"
        result[name] = {"variable": variable, "level": level, "native_step": step,
            "measurement_kind": kind, "averaging_start_at": start,
            "forecast_end_at": str(run + pd.Timedelta(hours=lead)),
            "start_byte": offsets[i], "end_byte": offsets[i + 1] - 1,
            "index_record": ":".join(records[i])}
    return result


def _object_evidence(headers):
    if not headers.get("etag") or not headers.get("last-modified"):
        raise ValueError("object ETag and Last-Modified evidence required")
    modified = parsedate_to_datetime(headers["last-modified"])
    if modified.tzinfo is None:
        raise ValueError("timezone-aware object modification evidence required")
    return {"etag": headers["etag"], "last_modified": modified.astimezone(timezone.utc).isoformat(),
            "response_headers": headers}


def collect_gfs(run, lead, output, timeout=30):
    """Fetch four global weather fields, without downloading the whole GRIB.

    Range responses must be HTTP 206 with exact Content-Range and GRIB length.
    If-Match ties every message to the HEAD object version. Original byte
    messages and index are retained. No point extraction, hourly solar
    conversion, public-availability certification or model CSV is fabricated.
    """
    run = run_timestamp(run)
    if not isinstance(lead, int) or isinstance(lead, bool) or not 1 <= lead <= 120:
        raise ValueError("hourly forecast lead 1..120 required")
    output = Path(output)
    if output.exists() or timeout <= 0:
        raise ValueError("new output directory and positive timeout required")
    key = f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}"
    url = BASE + key
    headers, _ = _request(url, method="HEAD", timeout=timeout)
    grib_evidence = _object_evidence(headers)
    size = int(headers["content-length"])
    index_headers, index = _request(url + ".idx", timeout=timeout)
    index_evidence = _object_evidence(index_headers)
    messages = select_messages(index.decode("utf-8"), run, lead, size)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gfs-stage-", dir=output.parent) as directory:
        staged = Path(directory) / "complete"
        staged.mkdir()
        (staged / "source.idx").write_bytes(index)
        for name, record in messages.items():
            start, end = record["start_byte"], record["end_byte"]
            length = end - start + 1
            if length > 32_000_000:
                raise ValueError("unreviewed GFS field exceeds 32 MB budget")
            part_headers, content = _request(url, headers={"Range": f"bytes={start}-{end}",
                "If-Match": grib_evidence["etag"]}, max_bytes=length, expected_status=206, timeout=timeout)
            if (part_headers.get("content-range") != f"bytes {start}-{end}/{size}"
                    or part_headers.get("etag") != grib_evidence["etag"] or len(content) != length):
                raise ValueError("GFS range length/version evidence mismatch")
            if (len(content) < 20 or content[:4] != b"GRIB" or content[7] != 2
                    or int.from_bytes(content[8:16], "big") != length or content[-4:] != b"7777"):
                raise ValueError("complete single GRIB2 message required")
            filename = name + ".grib2"
            (staged / filename).write_bytes(content)
            record.update({"file": filename, "sha256": hashlib.sha256(content).hexdigest(),
                           "response_headers": part_headers})
        report = {"provider": "NOAA GFS public S3 archive", "reference": REFERENCE,
            "schema_version": 1, "source_url": url, "initialization_at": str(run),
            "lead_hours": lead, "grid": "0.25 degree", "member": "deterministic operational GFS",
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "grib_object": grib_evidence, "index_object": index_evidence,
            "index_sha256": hashlib.sha256(index).hexdigest(), "messages": messages,
            "eligible_for_issued_public": False, "available_at": None,
            "availability_note": "S3 object write timestamps retained as evidence; neither initialization nor today's retrieval is certified original public availability",
            "missing_evidence": ["predeclared availability/latency protocol for the original public archive",
                "spatial aggregation with public pre-origin coordinates and fleet/demand weights",
                "instantaneous/hour-mean alignment and solar accumulation-window conversion",
                "matched demand forecasts and published grid configurations"]}
        (staged / "metadata.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        staged.rename(output)
    return report
