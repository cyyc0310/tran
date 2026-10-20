from copy import deepcopy
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from transcif.data.asof import public_config_asof, forecast_asof, utc
from transcif.data.calendar import calendar_features
from transcif.data.dataset import load_bundle, source_names, validate_values


def test_config_uses_publication_time_across_year_and_revisions():
    rows = pd.DataFrame({
        "period_end": ["2022-11-30Z", "2022-12-31Z", "2022-11-30Z"],
        "available_at": ["2022-12-15Z", "2023-02-01Z", "2023-01-10Z"],
        "value": [1., 999., 2.]})
    # Use explicit complete ISO timestamps, never implicit local dates.
    for col in ("period_end", "available_at"):
        rows[col] = rows[col].str.replace("Z", "T00:00:00Z")
    assert public_config_asof(rows, "2023-01-01T00:00Z", ["value"]).value == 1
    assert public_config_asof(rows, "2023-01-11T00:00Z", ["value"]).value == 2
    assert public_config_asof(rows, "2023-02-01T00:00Z", ["value"]).value == 999
    with pytest.raises(ValueError, match="no config"):
        public_config_asof(rows, "2022-12-01T00:00Z", ["value"])


def test_forecast_requires_one_complete_cycle_and_ignores_late_revisions():
    origin = pd.Timestamp("2023-01-01T00:00Z")
    hours = pd.date_range(origin, periods=2, freq="h")
    rows = []
    for age, complete in ((6, True), (3, False)):
        for member in (0, 1):
            for t in range(2):
                if not complete and member == 1 and t == 1:
                    continue
                rows.append(dict(issued_at=origin - pd.Timedelta(hours=age),
                    available_at=origin - pd.Timedelta(hours=age - 1),
                    valid_at=hours[t], member=member, value=age))
    frame = pd.DataFrame(rows)
    late = frame.iloc[0].copy()
    late.available_at = origin + pd.Timedelta(hours=1)
    late.value = 999
    frame = pd.concat([frame, late.to_frame().T], ignore_index=True)
    selected = forecast_asof(frame, origin, hours, (0, 1), ["value"])
    assert (selected.value == 6).all()
    # Without that old complete run, stitching the two cycles cannot rescue it.
    frame = frame.drop(index=3)
    with pytest.raises(ValueError, match="no complete"):
        forecast_asof(frame, origin, hours, (0, 1), ["value"])


def test_reject_ambiguous_revisions(bundle):
    _, doc, grids = bundle
    grid = grids["target_mixed"]
    grid.forecasts = pd.concat([grid.forecasts, grid.forecasts.iloc[:1]])
    with pytest.raises(ValueError, match="ambiguous"):
        grid.example(doc["origins"][0])


def test_public_example_works_without_any_label_file(bundle):
    _, doc, grids = bundle
    grid = grids["target_mixed"]
    before = grid.example(doc["origins"][0])
    grid.labels_path.unlink()
    after = grid.example(doc["origins"][0])
    for key in before.values:
        np.testing.assert_array_equal(before.values[key], after.values[key])
    values = deepcopy(after.values)
    values["actual_demand"] = values["demand"]
    with pytest.raises(ValueError, match="exactly"):
        validate_values(values)


def test_source_label_publication_cutoff(bundle):
    _, doc, grids = bundle
    grid = grids["source_coal"]
    example = grid.example(doc["origins"][0])
    end = example.origin + pd.Timedelta(hours=24)
    assert grid.labels(example, available_by=end)["generation"].shape == (24, 6)
    with pytest.raises(ValueError, match="missing, late"):
        grid.labels(example, available_by=end - pd.Timedelta(seconds=1))


def test_label_cache_keeps_revision_cutoffs_and_invalidates_changed_or_deleted_files(bundle, monkeypatch):
    _, doc, grids = bundle
    grid = grids["source_coal"]
    example = grid.example(doc["origins"][0])
    end = example.origin + pd.Timedelta(hours=24)
    original_read = pd.read_csv
    reads = []

    def read(*args, **kwargs):
        reads.append(args[0])
        return original_read(*args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", read)
    before = grid.labels(example, available_by=end)["cif"].copy()
    grid.labels(example, available_by=end)
    with pytest.raises(ValueError, match="missing, late"):
        grid.labels(example, available_by=end - pd.Timedelta(seconds=1))
    assert reads == [grid.labels_path]
    frame = original_read(grid.labels_path)
    revision = frame.iloc[[0]].copy()
    revision["available_at"] = str(end + pd.Timedelta(hours=1))
    revision["cif_g_per_kwh"] += 100
    pd.concat([frame, revision], ignore_index=True).to_csv(grid.labels_path, index=False)
    np.testing.assert_array_equal(before, grid.labels(example, available_by=end)["cif"])
    assert grid.labels(example)["cif"][0] == pytest.approx(before[0] + 100, abs=1e-4)
    assert reads == [grid.labels_path, grid.labels_path]
    grid.labels_path.unlink()
    with pytest.raises(FileNotFoundError):
        grid.labels(example)


def test_coverage_and_jurisdiction_exclusion(bundle):
    _, _, grids = bundle
    held = grids["target_mixed"]
    grids["aggregate"] = replace(held, name="aggregate", coverage=held.coverage | {"other"})
    grids["same_country"] = replace(grids["source_gas"], name="same_country",
                                    jurisdiction=held.jurisdiction)
    assert "aggregate" not in source_names(grids, held.name)
    assert "same_country" in source_names(grids, held.name)
    assert "same_country" not in source_names(grids, held.name, True)


def test_cannot_promote_synthetic_or_oracle_to_public(bundle):
    manifest, doc, _ = bundle
    doc["track"] = "issued_public"
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="published configs"):
        load_bundle(manifest)
    for grid in doc["regions"]:
        grid.update(config_kind="published", forecast_kind="issued")
    manifest.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="synthetic labels"):
        load_bundle(manifest)


def test_timezone_required_and_dst_is_civil_time():
    with pytest.raises(ValueError, match="timezone-aware"):
        utc(["2023-01-01"])
    hours = pd.date_range("2023-03-12T06:00Z", periods=3, freq="h")
    features = calendar_features(hours, "America/New_York")
    # New York jumps from 01:00 to 03:00 at 07:00 UTC.
    phases = (np.array([1, 3, 4]) + .5) * 2 * np.pi / 24
    np.testing.assert_allclose(features[:, 0], np.sin(phases), atol=1e-6)


def test_origin_must_be_hour_aligned(bundle):
    _, doc, grids = bundle
    with pytest.raises(ValueError, match="hour-aligned"):
        grids["target_mixed"].example(pd.Timestamp(doc["origins"][0]) + pd.Timedelta(minutes=5))
