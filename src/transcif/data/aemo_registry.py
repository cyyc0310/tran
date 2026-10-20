"""Retrospective registration diagnostics; never a target configuration builder."""

from datetime import datetime

import numpy as np
import pandas as pd


PROCEDURE = "https://www.aemo.com.au/-/media/files/stakeholder_consultation/consultations/nem-consultations/2019/cdeii/carbon-dioxide-equivalent-intensity-index-procedure-v40final.pdf"


def _native_dates(values):
    for value in values:
        if not isinstance(value, str) or datetime.strptime(value, "%Y/%m/%d %H:%M:%S").strftime("%Y/%m/%d %H:%M:%S") != value:
            raise ValueError("canonical nonmissing native NEM registration timestamps required")


def registry_at(duids, at, details, allocations, gensets):
    """Audit effective records at an explicit instant, with no publication claim.

    [START_DATE, END_DATE) is the stated effective-window convention. Latest
    allocation EFFECTIVEDATE is used, but all native version-specific genset
    sets at that date remain candidates; ambiguous version semantics are not
    silently resolved. Genset EF snapshots do not contain an effective-time
    history. The result is retrospective evidence, not pre-origin metadata.

    Each DUID stays one record, even if it maps to multiple physical gensets.
    If any candidate genset EF is absent, no partial weighted EF is invented.
    For complete candidate EFs, min/max only describe possible convex mixture
    factors, with no generation shares assumed and no CO2e-to-CO2 conversion.
    """
    at = pd.Timestamp(at)
    if pd.isna(at) or at.tz is None or not duids or len(set(duids)) != len(duids):
        raise ValueError("aware audit instant and unique nonempty DUID inventory required")
    native_at = at.tz_convert("Etc/GMT-10").strftime("%Y/%m/%d %H:%M:%S")
    required = [(details, ["DUID", "START_DATE", "END_DATE", "REGIONID", "DISPATCHTYPE"], ["DUID", "START_DATE"]),
        (allocations, ["DUID", "EFFECTIVEDATE", "VERSIONNO", "GENSETID"], ["DUID", "EFFECTIVEDATE", "VERSIONNO", "GENSETID"]),
        (gensets, ["GENSETID", "CO2E_EMISSIONS_FACTOR", "CO2E_ENERGY_SOURCE"], ["GENSETID"])]
    for frame, columns, primary in required:
        if not set(columns).issubset(frame) or frame.duplicated(primary).any():
            raise ValueError("complete unique native registration table keys required")
        if frame[primary].isna().any().any() or frame[primary].eq("").any().any():
            raise ValueError("nonempty native registration primary keys required")
    _native_dates(details.START_DATE)
    _native_dates(details.END_DATE)
    _native_dates(allocations.EFFECTIVEDATE)
    if (details.START_DATE >= details.END_DATE).any():
        raise ValueError("ordered registration effective windows required")
    versions = pd.to_numeric(allocations.VERSIONNO, errors="raise")
    if not np.isfinite(versions).all() or (versions < 1).any() or (versions % 1 != 0).any():
        raise ValueError("positive integer native allocation versions required")
    allocations = allocations.assign(VERSIONNO=versions.astype(int))
    if allocations.duplicated(["DUID", "EFFECTIVEDATE", "VERSIONNO", "GENSETID"]).any():
        raise ValueError("duplicate normalized native allocation keys")
    factors = pd.to_numeric(gensets.CO2E_EMISSIONS_FACTOR.replace("", np.nan), errors="raise")
    if np.isinf(factors).any() or (factors < 0).any():
        raise ValueError("finite nonnegative native EFs or explicit missing values required")
    gensets = gensets.assign(_factor=factors).set_index("GENSETID")
    active = details[(details.START_DATE <= native_at) & (native_at < details.END_DATE)]
    eligible = allocations[allocations.EFFECTIVEDATE <= native_at]
    detail_groups = {name: rows for name, rows in active.groupby("DUID")}
    allocation_groups = {name: rows for name, rows in eligible.groupby("DUID")}
    records = []
    for duid in sorted(duids):
        detail = detail_groups.get(duid)
        mapping = allocation_groups.get(duid)
        candidates, effective = [], None
        if mapping is not None:
            effective = mapping.EFFECTIVEDATE.max()
            mapping = mapping[mapping.EFFECTIVEDATE == effective]
            candidates = [{"version": int(version), "gensets": sorted(rows.GENSETID.unique())}
                for version, rows in mapping.groupby("VERSIONNO")]
            candidates.sort(key=lambda x: x["version"])
        all_ids = sorted({name for candidate in candidates for name in candidate["gensets"]})
        missing = [name for name in all_ids if name not in gensets.index]
        known = gensets.reindex(all_ids)
        missing_ef = [name for name in all_ids if name in gensets.index and pd.isna(gensets.loc[name, "_factor"])]
        missing_fuel = [name for name in all_ids if name in gensets.index and
                        (pd.isna(gensets.loc[name, "CO2E_ENERGY_SOURCE"]) or gensets.loc[name, "CO2E_ENERGY_SOURCE"] == "")]
        complete_ef = bool(all_ids) and not missing and not missing_ef
        low = float(known._factor.min()) if complete_ef else None
        high = float(known._factor.max()) if complete_ef else None
        versions_differ = len({tuple(x["gensets"]) for x in candidates}) > 1
        records.append({"duid": duid, "region_record_count": 0 if detail is None else len(detail),
            "region_candidates": [] if detail is None else sorted(set(detail.REGIONID.dropna()) - {""}),
            "dispatch_type_candidates": [] if detail is None else sorted(set(detail.DISPATCHTYPE.dropna()) - {""}),
            "allocation_effective_at_native": effective, "allocation_candidates": candidates,
            "allocation_version_ambiguous": versions_differ, "candidate_gensets": all_ids,
            "multi_genset": any(len(x["gensets"]) > 1 for x in candidates),
            "missing_gensets": missing, "missing_ef_gensets": missing_ef,
            "missing_fuel_gensets": missing_fuel,
            "fuel_candidates": sorted(set(known.CO2E_ENERGY_SOURCE.dropna()) - {""}),
            "native_ef_min": low, "native_ef_max": high,
            "single_native_ef": complete_ef and low == high})
    count = lambda condition: sum(bool(condition(row)) for row in records)
    return {"audit_at": str(at.tz_convert("UTC")), "native_effective_key": native_at,
        "eligible_for_issued_public": False, "retrospective_only": True, "records": records,
        "counts": {"duids": len(records), "missing_region_record": count(lambda r: r["region_record_count"] == 0),
            "overlapping_region_records": count(lambda r: r["region_record_count"] > 1),
            "missing_allocation": count(lambda r: not r["allocation_candidates"]),
            "ambiguous_allocation_versions": count(lambda r: r["allocation_version_ambiguous"]),
            "multi_genset": count(lambda r: r["multi_genset"]),
            "generator_inventory": count(lambda r: r["dispatch_type_candidates"] == ["GENERATOR"]),
            "load_inventory": count(lambda r: r["dispatch_type_candidates"] == ["LOAD"]),
            "incomplete_candidate_ef": count(lambda r: r["native_ef_min"] is None),
            "generator_incomplete_candidate_ef": count(lambda r: r["dispatch_type_candidates"] == ["GENERATOR"] and r["native_ef_min"] is None),
            "nondegenerate_native_ef_range": count(lambda r: r["native_ef_min"] is not None and not r["single_native_ef"]),
            "multiple_native_fuels": count(lambda r: len(r["fuel_candidates"]) > 1)},
        "limitations": ["effective dates are not publication dates; EF snapshot has no complete effective history",
            "allocation versions kept as alternatives, without asserting their revision/deletion semantics",
            "native CO2e factors are not harmonized direct CO2; CDEII procedure includes fugitive emissions",
            "no instantaneous-MW integration, fuel generation labels or CIF labels are produced"],
        "accounting_reference": PROCEDURE}
