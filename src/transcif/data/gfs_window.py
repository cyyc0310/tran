"""Resumable acquisition of complete single-run weather windows."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from urllib.error import HTTPError, URLError

import pandas as pd

from transcif.data.gfs import BASE, collect_gfs, run_timestamp
from transcif.data.gfs_decode import decode_gfs_points, hourly_gfs_points


def snapshot_path(cache, run, lead):
    run = run_timestamp(run)
    return Path(cache) / f"{run:%Y%m%d}" / f"{run:%H}" / f"f{lead:03d}"


def _cached_metadata(path, run, lead):
    """Check immutable cached bytes before reusing a snapshot, without refetching."""
    metadata_path = path / "metadata.json"
    report = json.loads(metadata_path.read_text())
    expected_url = BASE + f"gfs.{run:%Y%m%d}/{run:%H}/atmos/gfs.t{run:%H}z.pgrb2.0p25.f{lead:03d}"
    if (run_timestamp(report["initialization_at"]) != run or report["lead_hours"] != lead
            or report["source_url"] != expected_url):
        raise ValueError("cached GFS run/lead/source mismatch; cache is never overwritten")
    records = [("source.idx", report["index_sha256"])] + [
        (row["file"], row["sha256"]) for row in report["messages"].values()]
    for name, digest in records:
        file = path / name
        if file.parent.resolve() != path.resolve() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise ValueError("cached GFS hash/path mismatch; cache is never overwritten")
    return report


def collect_gfs_window(run, first_lead, horizon, points, cache, output,
                       workers=3, retries=3, timeout=30, progress=None, solar_negative_policy="error"):
    """Collect H+1 native leads to construct H complete point-weather hours.

    Completed snapshots survive a failed/interrupted window and are hash-checked
    on resume. Corrupt cache entries are rejected, never silently replaced.
    ecCodes decoding is sequential, separate from concurrent network I/O.
    The final window is atomically committed only after full validation.
    """
    run, output = run_timestamp(run), Path(output)
    integers = (first_lead, horizon, workers, retries)
    if (any(not isinstance(x, int) or isinstance(x, bool) for x in integers)
            or not (1 <= first_lead < first_lead + horizon <= 120)
            or not 1 <= workers <= 8 or not 1 <= retries <= 5
            or not math.isfinite(timeout) or timeout <= 0 or output.exists()):
        raise ValueError("valid lead/horizon, workers 1..8, retries 1..5, timeout and new output required")
    if solar_negative_policy not in ("error", "packing_resolution"):
        raise ValueError("valid solar_negative_policy required")
    if not points or len({p["id"] for p in points}) != len(points):
        raise ValueError("nonempty unique coordinate points required")
    for point in points:
        lat, lon = point["latitude"], point["longitude"]
        if not point["id"] or not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 360):
            raise ValueError("valid explicit coordinate points required")
    # Fail before any network work if the optional decoder is unavailable.
    try:
        import eccodes  # noqa: F401
    except ImportError as error:
        raise ImportError("install transcif[weather] to construct GFS weather windows") from error

    origin = run + pd.Timedelta(hours=first_lead)
    leads = list(range(first_lead, first_lead + horizon + 1))
    for lead in leads:
        path = snapshot_path(cache, run, lead)
        if path.exists():
            _cached_metadata(path, run, lead)

    def acquire(lead):
        path = snapshot_path(cache, run, lead)
        if path.exists():
            _cached_metadata(path, run, lead)
            return {"lead": lead, "cached": True, "attempts": 0}
        for attempt in range(1, retries + 1):
            try:
                collect_gfs(run, lead, path, timeout=timeout)
                return {"lead": lead, "cached": False, "attempts": attempt}
            except (URLError, TimeoutError) as error:
                if (isinstance(error, HTTPError) and error.code != 429 and error.code < 500) or attempt == retries:
                    raise
                time.sleep(min(4., .5 * 2 ** (attempt - 1)))

    acquisitions = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(acquire, lead) for lead in leads]
        try:
            for future in as_completed(futures):
                record = future.result()
                acquisitions.append(record)
                if progress is not None:
                    progress(record)
        except BaseException:
            for future in futures:
                future.cancel()
            raise

    decoded, references, writes = [], [], []
    for lead in leads:
        path = snapshot_path(cache, run, lead)
        report = _cached_metadata(path, run, lead)
        for name in ("grib_object", "index_object"):
            written = pd.Timestamp(report[name]["last_modified"])
            if pd.isna(written) or written.tz is None or not run <= written <= origin:
                raise ValueError(f"GFS lead {lead} object-write evidence is outside initialization/origin bounds")
            writes.append(written.tz_convert("UTC"))
        decoded.append(decode_gfs_points(path, points))
        references.append({"lead": lead, "snapshot": str(path.resolve()),
            "metadata_sha256": hashlib.sha256((path / "metadata.json").read_bytes()).hexdigest()})
    hourly = hourly_gfs_points(decoded, first_lead, first_lead + horizon, solar_negative_policy)
    result = {"schema_version": 1, "initialization_at": str(run), "forecast_origin_at": str(origin),
        "horizon": horizon, "native_leads": leads, "snapshots": references,
        "acquisitions": sorted(acquisitions, key=lambda x: x["lead"]),
        "latest_object_write_at": str(max(writes)), "all_objects_written_by_origin": True,
        "solar_negative_policy": solar_negative_policy,
        "shortwave_adjusted_point_hours": hourly["shortwave_adjusted_point_hours"],
        "available_at": None, "eligible_for_issued_public": False,
        "interpretation": "complete single-run point-weather window; object-write evidence is necessary but is not certified original public availability",
        "missing_evidence": ["predeclared original archive availability protocol",
            "public pre-origin spatial/fleet/demand weights", "matched demand forecasts and configurations"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".gfs-window-", dir=output.parent) as directory:
        staged = Path(directory) / "complete"
        staged.mkdir()
        weather = staged / "hourly_points.json"
        weather.write_text(json.dumps(hourly, indent=2, allow_nan=False) + "\n")
        result["weather_sha256"] = hashlib.sha256(weather.read_bytes()).hexdigest()
        (staged / "metadata.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        staged.rename(output)
    return result
