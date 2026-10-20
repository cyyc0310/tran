import pandas as pd
import pytest

from transcif.data.aemo_registry import registry_at


def frames():
    details = pd.DataFrame([
        ["A", "2020/01/01 00:00:00", "2023/07/01 00:00:00", "NSW1", "GENERATOR"],
        ["A", "2023/07/01 00:00:00", "2999/12/31 00:00:00", "QLD1", "GENERATOR"],
        ["B", "2020/01/01 00:00:00", "2999/12/31 00:00:00", "SA1", "LOAD"],
    ], columns=["DUID", "START_DATE", "END_DATE", "REGIONID", "DISPATCHTYPE"])
    allocations = pd.DataFrame([
        ["A", "2020/01/01 00:00:00", "1", "G0"],
        ["A", "2023/01/01 00:00:00", "1", "G1"],
        ["A", "2023/01/01 00:00:00", "1", "G2"],
        ["A", "2023/01/01 00:00:00", "2", "G1"],
        ["B", "2020/01/01 00:00:00", "1", "LOAD"],
    ], columns=["DUID", "EFFECTIVEDATE", "VERSIONNO", "GENSETID"])
    gensets = pd.DataFrame([["G0", "1.2", "Diesel oil"], ["G1", "0.4", "Gas"],
        ["G2", "0.8", "Coal"], ["LOAD", "", ""]],
        columns=["GENSETID", "CO2E_EMISSIONS_FACTOR", "CO2E_ENERGY_SOURCE"])
    return details, allocations, gensets


def test_multi_genset_does_not_multiply_unit_records_or_invent_a_weighted_ef():
    result = registry_at(["A", "B", "MISSING"], "2023-03-01T00:00Z", *frames())
    assert len(result["records"]) == 3
    a, b, missing = result["records"]
    assert a["candidate_gensets"] == ["G1", "G2"] and a["multi_genset"]
    assert a["allocation_version_ambiguous"]
    assert a["native_ef_min"] == .4 and a["native_ef_max"] == .8
    assert "weighted_ef" not in a and not a["single_native_ef"]
    assert b["native_ef_min"] is None and b["dispatch_type_candidates"] == ["LOAD"]
    assert missing["region_record_count"] == 0 and not missing["allocation_candidates"]
    assert result["counts"]["generator_inventory"] == 1
    assert result["retrospective_only"] and not result["eligible_for_issued_public"]


def test_effective_boundaries_use_fixed_nem_time_and_preserve_out_of_bounds_dates():
    before = registry_at(["A"], "2023-06-30T13:55Z", *frames())["records"][0]
    after = registry_at(["A"], "2023-06-30T14:00Z", *frames())["records"][0]
    assert before["region_candidates"] == ["NSW1"]
    assert after["region_candidates"] == ["QLD1"]


def test_partial_missing_factors_never_become_zero_or_average_of_available_gensets():
    details, allocations, gensets = frames()
    gensets.loc[gensets.GENSETID == "G2", "CO2E_EMISSIONS_FACTOR"] = ""
    a = registry_at(["A"], "2023-03-01T00:00Z", details, allocations, gensets)["records"][0]
    assert a["missing_ef_gensets"] == ["G2"]
    assert a["native_ef_min"] is None and a["native_ef_max"] is None
    gensets = gensets[gensets.GENSETID != "G2"]
    a = registry_at(["A"], "2023-03-01T00:00Z", details, allocations, gensets)["records"][0]
    assert a["missing_gensets"] == ["G2"] and a["native_ef_min"] is None


def test_overlapping_region_histories_are_explicitly_flagged():
    details, allocations, gensets = frames()
    row = details.iloc[0].copy()
    row.START_DATE = "2022/01/01 00:00:00"
    row.REGIONID = "VIC1"
    details = pd.concat([details, row.to_frame().T], ignore_index=True)
    result = registry_at(["A"], "2023-03-01T00:00Z", details, allocations, gensets)
    assert result["records"][0]["region_candidates"] == ["NSW1", "VIC1"]
    assert result["counts"]["overlapping_region_records"] == 1


@pytest.mark.parametrize("change", ["duplicate", "bad_date", "negative_ef", "bad_version", "naive"])
def test_invalid_native_evidence_is_rejected(change):
    details, allocations, gensets = frames()
    at = "2023-03-01T00:00Z"
    if change == "duplicate":
        gensets = pd.concat([gensets, gensets.iloc[:1]])
    elif change == "bad_date":
        details.loc[0, "END_DATE"] = "2023/02/30 00:00:00"
    elif change == "negative_ef":
        gensets.loc[0, "CO2E_EMISSIONS_FACTOR"] = "-1"
    elif change == "bad_version":
        allocations.loc[0, "VERSIONNO"] = "1.5"
    else:
        at = "2023-03-01"
    with pytest.raises(ValueError):
        registry_at(["A"], at, details, allocations, gensets)
