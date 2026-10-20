"""Optional ecCodes verification and native point extraction, without grid aggregation."""

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from transcif.data.gfs import run_timestamp


EXPECTED = {"temperature_2m": (167, "heightAboveGround", 2, "K"),
    "wind_u_100m": (131, "heightAboveGround", 100, "m s**-1"),
    "wind_v_100m": (132, "heightAboveGround", 100, "m s**-1"),
    "shortwave_surface": (260087, "surface", 0, "W m**-2")}


def decode_gfs_points(snapshot, points):
    """Verify encoded fields/run/windows, then extract the nearest grid points.

    Points are explicit dicts with id/latitude/longitude. They are not inferred
    from target telemetry or treated as fleet/demand-weighted regional weather.
    Native units and solar averaging support are retained; a six-hour average
    is never silently relabeled as hourly shortwave.
    """
    try:
        import eccodes
    except ImportError as error:
        raise ImportError("install transcif[weather] to decode GRIB messages") from error
    if not points or len({p["id"] for p in points}) != len(points):
        raise ValueError("nonempty points with unique IDs required")
    for point in points:
        lat, lon = point["latitude"], point["longitude"]
        if not point["id"] or not all(math.isfinite(x) for x in (lat, lon)) or not (-90 <= lat <= 90 and -180 <= lon <= 360):
            raise ValueError("finite latitude/longitude and nonempty point ID required")
    snapshot = Path(snapshot)
    metadata_path = snapshot / "metadata.json"
    report = json.loads(metadata_path.read_text())
    run, lead = run_timestamp(report["initialization_at"]), report["lead_hours"]
    if report["grid"] != "0.25 degree" or set(report["messages"]) != set(EXPECTED):
        raise ValueError("reviewed GFS grid and four native weather fields required")
    index = snapshot / "source.idx"
    if hashlib.sha256(index.read_bytes()).hexdigest() != report["index_sha256"]:
        raise ValueError("GFS index hash mismatch")
    fields, geometry = {}, None
    for name, expected in EXPECTED.items():
        record = report["messages"][name]
        path = snapshot / record["file"]
        if path.parent.resolve() != snapshot.resolve():
            raise ValueError("GRIB file must be inside snapshot")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("GFS message hash mismatch")
        with path.open("rb") as stream:
            handle = eccodes.codes_grib_new_from_file(stream)
            if handle is None:
                raise ValueError("empty GRIB message")
            try:
                get = lambda key: eccodes.codes_get(handle, key)
                signature = tuple(get(k) for k in ("paramId", "typeOfLevel", "level", "units"))
                if signature != expected:
                    raise ValueError(f"encoded GFS field/height/units mismatch for {name}")
                if (get("dataDate") != int(run.strftime("%Y%m%d")) or get("dataTime") != run.hour * 100
                        or get("stepUnits") != 1 or get("endStep") != lead or stream.read(1)):
                    raise ValueError("encoded GFS cycle/lead mismatch or trailing messages")
                solar = name == "shortwave_surface"
                start = ((pd.Timestamp(record["averaging_start_at"]) - run) / pd.Timedelta(hours=1)
                         if solar else lead)
                if (record["measurement_kind"] != ("interval_mean" if solar else "instantaneous")
                        or get("stepType") != ("avg" if solar else "instant") or get("startStep") != start
                        or pd.Timestamp(record["forecast_end_at"]) != run + pd.Timedelta(hours=lead)):
                    raise ValueError("encoded GFS temporal averaging support mismatch")
                shape = tuple(get(k) for k in ("gridType", "Ni", "Nj", "latitudeOfFirstGridPointInDegrees",
                    "longitudeOfFirstGridPointInDegrees", "latitudeOfLastGridPointInDegrees",
                    "longitudeOfLastGridPointInDegrees", "scanningMode", "iDirectionIncrementInDegrees",
                    "jDirectionIncrementInDegrees"))
                if shape[:3] != ("regular_ll", 1440, 721) or (geometry is not None and shape != geometry):
                    raise ValueError("reviewed matching global 0.25 degree GFS grids required")
                geometry = shape
                samples = []
                for point in points:
                    nearest = eccodes.codes_grib_find_nearest(handle, point["latitude"], point["longitude"])[0]
                    value = float(nearest["value"])
                    if not math.isfinite(value) or value == get("missingValue"):
                        raise ValueError("missing GFS forecast at requested point")
                    samples.append({"id": point["id"], "latitude": float(nearest["lat"]),
                        "longitude": float(nearest["lon"]), "distance_km": float(nearest["distance"]), "value": value})
                fields[name] = {"units": get("units"), "native_step": record["native_step"],
                    "measurement_kind": record["measurement_kind"], "forecast_end_at": record["forecast_end_at"],
                    "averaging_start_at": record["averaging_start_at"], "samples": samples}
                if solar:
                    fields[name]["packing"] = {"type": get("packingType"),
                        "binary_scale": get("binaryScaleFactor"), "decimal_scale": get("decimalScaleFactor"),
                        "reference_value": get("referenceValue"),
                        "scaling_step_native": 2. ** get("binaryScaleFactor") * 10. ** (-get("decimalScaleFactor"))}
            finally:
                eccodes.codes_release(handle)
    return {"initialization_at": str(run), "lead_hours": lead, "points": points, "fields": fields,
        "snapshot_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "decoder": "ecCodes", "decoder_version": eccodes.codes_get_api_version(),
        "eligible_for_issued_public": False, "available_at": None,
        "interpretation": "native point forecasts, not regional averages or model-ready hourly weather"}


def hourly_gfs_points(decoded, first_lead, end_lead, solar_negative_policy="error"):
    """Form [first_lead,end_lead) point hours from one verified forecast run.

    Temperature and vector wind are sampled at the hour start. Shortwave is
    the hour mean obtained by differencing cumulative-window mean * duration.
    A reset window needs no earlier anchor. A missing anchor, changing points,
    mixed run or negative implied energy is rejected by default. Explicit
    packing_resolution policy projects only small negative values within a
    duration-weighted encoded scaling-step tolerance. Raw values and all such
    adjustments are retained. This tolerance is not a meteorological error
    interval or a certified bound on pre-packing model values.
    """
    if (not decoded or any(not isinstance(x, int) or isinstance(x, bool) for x in (first_lead, end_lead))
            or not (1 <= first_lead < end_lead <= 120)):
        raise ValueError("nonempty single-run forecasts and positive hourly lead range required")
    if solar_negative_policy not in ("error", "packing_resolution"):
        raise ValueError("solar_negative_policy must be error or packing_resolution")
    by_lead = {record["lead_hours"]: record for record in decoded}
    run = run_timestamp(decoded[0]["initialization_at"])
    points = decoded[0]["points"]
    if (len(by_lead) != len(decoded) or any(run_timestamp(x["initialization_at"]) != run
            or x["points"] != points for x in decoded)):
        raise ValueError("unique leads from one initialization and unchanged points required")

    point_locations = None
    def values(record, field):
        nonlocal point_locations
        samples = record["fields"][field]["samples"]
        if [x["id"] for x in samples] != [p["id"] for p in points]:
            raise ValueError("aligned native forecast point IDs required")
        data = [x["value"] for x in samples]
        if not all(math.isfinite(x) for x in data):
            raise ValueError("finite native forecast values required")
        locations = [(x["latitude"], x["longitude"]) for x in samples]
        if point_locations is not None and locations != point_locations:
            raise ValueError("unchanged sampled grid coordinates across fields/leads required")
        point_locations = locations
        return data

    def resolution(field):
        packing = field.get("packing", {})
        value = packing.get("scaling_step_native", float("nan"))
        if (packing.get("type") not in ("grid_simple", "grid_complex", "grid_complex_spatial_differencing")
                or not math.isfinite(value) or value <= 0):
            raise ValueError("reviewed GRIB scaling-step evidence required for packing_resolution policy")
        return value

    rows, adjusted = [], 0
    for lead in range(first_lead, end_lead):
        if lead not in by_lead or lead + 1 not in by_lead:
            raise ValueError("complete consecutive forecast leads required, without cycle stitching")
        current, next_hour = by_lead[lead], by_lead[lead + 1]
        for field, unit in (("temperature_2m", "K"), ("wind_u_100m", "m s**-1"), ("wind_v_100m", "m s**-1")):
            native = current["fields"][field]
            if (native["units"] != unit or native["measurement_kind"] != "instantaneous"
                    or pd.Timestamp(native["forecast_end_at"]) != run + pd.Timedelta(hours=lead)):
                raise ValueError("reviewed instantaneous weather units and valid time required")
        temperature, u, v = (values(current, key) for key in
                            ("temperature_2m", "wind_u_100m", "wind_v_100m"))
        solar = next_hour["fields"]["shortwave_surface"]
        start = (pd.Timestamp(solar["averaging_start_at"]) - run) / pd.Timedelta(hours=1)
        if (solar["units"] != "W m**-2" or solar["measurement_kind"] != "interval_mean"
                or pd.Timestamp(solar["forecast_end_at"]) != run + pd.Timedelta(hours=lead + 1)
                or not float(start).is_integer() or not 0 <= start <= lead):
            raise ValueError("reviewed shortwave window ending at next hour required")
        duration = lead + 1 - start
        energy = [value * duration for value in values(next_hour, "shortwave_surface")]
        tolerance = duration * resolution(solar) if solar_negative_policy == "packing_resolution" else 0.
        if start < lead:
            previous = current["fields"]["shortwave_surface"]
            if (previous["units"] != "W m**-2" or previous["measurement_kind"] != "interval_mean"
                    or pd.Timestamp(previous["averaging_start_at"]) != pd.Timestamp(solar["averaging_start_at"])
                    or pd.Timestamp(previous["forecast_end_at"]) != run + pd.Timedelta(hours=lead)):
                raise ValueError("matching prior accumulation window required for hourly shortwave")
            energy = [total - value * (duration - 1) for total, value in
                      zip(energy, values(current, "shortwave_surface"))]
            if solar_negative_policy == "packing_resolution":
                tolerance += (duration - 1) * resolution(previous)
        for i, point in enumerate(points):
            raw = energy[i]
            if raw < -tolerance:
                raise ValueError("negative implied hourly shortwave; resolve packing/forecast inconsistency explicitly")
            projected = raw < 0
            adjusted += int(projected)
            rows.append({"point": point["id"], "valid_at": str(run + pd.Timedelta(hours=lead)),
                "temperature_c": temperature[i] - 273.15,
                "wind_speed_ms": math.hypot(u[i], v[i]), "shortwave_w_per_m2": max(0., raw),
                "shortwave_raw_w_per_m2": raw, "shortwave_resolution_tolerance_w_per_m2": tolerance,
                "shortwave_projected_within_resolution": projected})
    return {"initialization_at": str(run), "first_lead": first_lead, "end_lead": end_lead,
        "points": points, "rows": rows, "available_at": None, "eligible_for_issued_public": False,
        "solar_negative_policy": solar_negative_policy, "shortwave_adjusted_point_hours": adjusted,
        "interpretation": "hourly point weather; instantaneous start-of-hour temperature/wind and deaccumulated hour-mean shortwave; not regional fleet/demand averages",
        "input_metadata_sha256": [x["snapshot_metadata_sha256"] for x in decoded]}
