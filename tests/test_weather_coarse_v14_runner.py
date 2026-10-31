"""V14 source policy locks, exact controls, lawful cohort and label-read gates."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import pytest

from scripts import run_weather_coarse_v14 as r


def selection_table(round_number: int) -> pd.DataFrame:
    rows = []
    for region in r.SOURCES:
        for c in r.configurations(round_number):
            for factor in r.FACTORS:
                metrics = {m: 1. for m in r.METRICS}
                metrics.update(empirical_coverage90=.925, crps=20+abs(factor-1))
                rows.append(dict(stage="source", region=region, semester="H1", lag=7,
                                 budget=c["budget"], model=c["model"], factor=factor, **metrics))
    return pd.DataFrame(rows)


def calibration() -> dict[str, np.ndarray]:
    origins = pd.date_range("2024-07-01", periods=2, tz="UTC").as_unit("ns").asi8
    end = np.full(2, pd.Timestamp("2024-08-01", tz="UTC").value, dtype=np.int64)
    available = np.full(2, pd.Timestamp("2024-08-31", tz="UTC").value, dtype=np.int64)
    return dict(calibration_regions=np.array(["BPAT", "CISO"], dtype="U"),
                calibration_origins=origins, calibration_response_period_end=end,
                calibration_response_available_at=available)


def region_results(round_number: int, n: int = 2) -> dict:
    results = {(c["budget"], c["model"]): dict(raw=np.full((n, 128, 24), 100.), center=np.full((n, 24), 100.))
               for c in r.configurations(round_number)}
    for (_, model), value in results.items():
        if model == "native_recompose" or model in (*r.POLICIES, "selected_donor"):
            value.update(amplitude_error=0., sampled_zero_creation={k: [0.]*8 for k in (
                "supported_original_zero_count_by_fuel", "created_positive_from_zero_count_by_fuel",
                "created_positive_from_zero_fraction_by_fuel", "created_positive_from_zero_q_amount_by_fuel")})
    return results


def baseline_fixture(tmp_path: Path, days: pd.DatetimeIndex, region: str, stage: str) -> dict:
    paths = {}
    for model in ("native_recompose", "simple_anchor", "fixed_donor"):
        path = tmp_path / f"old-{stage}-{region}-{model}.npz"
        raw = np.full((len(days), 128, 24), 100.)
        np.savez(path, origins=days.as_unit("ns").asi8, point=raw.mean(1), ensemble=raw,
                 raw_ensemble=raw, center=raw.mean(1), donor_indices=np.zeros((len(days), 128), dtype=int))
        paths[f"{stage}/{region}/{model}"] = str(path)
    return paths


def factor_archive(model: str) -> dict[str, np.ndarray]:
    mix = np.array([.1, 0, 0, .9, 0, 0, 0, 0])
    data = {**calibration(), "calibration_errors": np.zeros((2, 24)),
        "residual_uniforms": np.full((2, 128), .1), "calibration_fine_errors": np.zeros((2, 24)),
        "calibration_z": np.zeros(2), "calibration_true_mix": np.tile(mix, (2, 1)),
        "calibration_forecast_mix": np.tile(mix, (2, 1)), "calibration_qbar": np.ones((2, 24, 8)),
        "query_qbar": np.ones((2, 24, 8)), "query_coarse_mix": np.tile(mix, (2, 1)),
        "query_coarse_draws": np.tile(mix, (2, 128, 1)), "query_coarse_features": np.zeros((2, 25)),
        "structural_center": np.full((2, 24), 100.), "center": np.full((2, 24), 100.)}
    if model == "location_scale":
        data["location_prediction"] = np.zeros((2, 24))
    if model.startswith(("level_", "physical_")):
        data.update(fine_residual_indices=np.zeros((2, 128), int),
            independent_residual_indices=np.zeros((2, 128), int), calibration_rank_indices=np.zeros((2, 128), int))
    return data


def test_predeclared_grids_and_primary_eligibility() -> None:
    assert len(r.configurations(1)) == 6
    assert len(r.configurations(2)) == 22
    assert {c["model"] for c in r.configurations(1) if c["primary_eligible"]} == set(r.POLICIES)
    assert {c["model"] for c in r.configurations(2) if c["primary_eligible"]} == {"selected_donor", *r.RISK_FAMILIES}
    assert not next(c for c in r.configurations(2) if c["model"] == "fixed_donor")["primary_eligible"]
    with pytest.raises(ValueError):
        r.configurations(3)


def test_round1_source_policy_and_spread_lock_ignore_target_rows(tmp_path: Path) -> None:
    table = selection_table(1)
    table.loc[table.model.eq("gate_ridge"), "crps"] = 1+abs(table.factor-1)
    targets = table.assign(stage="target", model="gate_forest", crps=-1000)
    path = tmp_path / "canonical.csv"
    pd.concat([table, targets]).to_csv(path, index=False)
    choice = r.selection(pd.read_csv(path), 1)
    assert choice["locked_policy"] == "gate_ridge"
    assert choice["primary"]["factor"] == 1.
    assert choice == r.selection(table, 1)


def test_round2_ignores_fixed_diagnostic_as_primary_and_preserves_policy() -> None:
    table = selection_table(2)
    table.loc[table.model.eq("fixed_donor"), "crps"] = 0
    table.loc[table.model.eq("selected_donor"), "crps"] = 1+abs(table.factor-1)
    choice = r.selection(table, 2, "gate_forest")
    assert choice["primary"]["model"] == "selected_donor"
    assert choice["diagnostic_crps_best"]["model"] == "fixed_donor"
    assert choice["locked_policy"] == "gate_forest"
    with pytest.raises(ValueError, match="source-locked"):
        r.selection(table, 2)


def test_coverage_gate_worst_region_and_failed_fallback() -> None:
    table = selection_table(2)
    table.loc[table.model.eq("qrf_50"), "crps"] = 0
    table.loc[table.model.eq("qrf_50") & table.region.eq(r.SOURCES[0]), "empirical_coverage90"] = .84
    choice = r.selection(table, 2, "fixed_donor")
    assert choice["primary"]["model"] != "qrf_50"
    table["empirical_coverage90"] = .5
    table.loc[table.model.eq("selected_donor"), "empirical_coverage90"] = .8
    choice = r.selection(table, 2, "fixed_donor")
    assert choice["primary"]["model"] == "selected_donor"
    assert choice["primary"]["status"] == "failed"


@pytest.mark.parametrize("kind", ["missing", "duplicate", "nonfinite"])
def test_selection_rejects_incomplete_duplicate_or_nonfinite_grid(kind: str) -> None:
    table = selection_table(1)
    if kind == "missing":
        table = table.iloc[:-1]
    elif kind == "duplicate":
        table = pd.concat([table, table.iloc[:1]])
    else:
        table.loc[0, "crps"] = np.nan
    with pytest.raises(ValueError, match="source selection grid"):
        r.selection(table, 1)


def test_round2_requires_prior_before_reading_upstream_or_creating_output(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r.v12, "verify", lambda _: pytest.fail("upstream opened before required prior"))
    with pytest.raises(ValueError, match="requires --prior-round"):
        r.freeze(tmp_path / "new", tmp_path / "data", round_number=2)
    assert not (tmp_path / "new").exists()


def test_target_modes_are_explicit_and_flags_are_target_only(tmp_path: Path) -> None:
    for flags in ({}, dict(predict_only=True, score_only=True)):
        with pytest.raises(ValueError, match="exactly one"):
            r.run_phase(tmp_path / "new", tmp_path, "target", round_number=1, **flags)
    with pytest.raises(ValueError, match="target-only"):
        r.run_phase(tmp_path / "new", tmp_path, "source", round_number=1, predict_only=True)
    assert not (tmp_path / "new").exists()


def test_no_evaluation_label_read_before_own_complete_prediction_seal(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(r.v10, "_truth_lookup", lambda *args: pytest.fail("labels opened"))
    with pytest.raises(ValueError, match="seal required"):
        r.score_phase(tmp_path, {}, "target", {})


def test_no_target_label_read_before_peer_prediction_seal(tmp_path: Path, monkeypatch) -> None:
    peer = tmp_path / "peer"
    peer.mkdir()
    seal = dict(other_round_source=str(peer))
    r.write(tmp_path / "target_prediction_seal.json", seal)
    monkeypatch.setattr(r, "check_entries", lambda *args: None)
    monkeypatch.setattr(r, "_target_source_gate", lambda *args: peer)
    monkeypatch.setattr(r.v10, "_truth_lookup", lambda *args: pytest.fail("labels opened"))
    with pytest.raises(ValueError, match="Both target prediction seals"):
        r.score_phase(tmp_path, {}, "target", seal)
    assert not (tmp_path / "target_scores").exists()


def test_target_source_gate_checks_both_linked_rounds(tmp_path: Path, monkeypatch) -> None:
    calls = []
    r1, r2 = tmp_path / "r1", tmp_path / "r2"
    monkeypatch.setattr(r, "verify_phase", lambda p, s: calls.append((p, s)))
    monkeypatch.setattr(r, "verify", lambda _: dict(round=2, prior_round=str(r1)))
    assert r._target_source_gate(r1, dict(round=1), r2) == r2
    assert calls == [(r1, "source"), (r2, "source")]
    monkeypatch.setattr(r, "verify", lambda _: dict(round=2, prior_round=str(tmp_path / "unrelated")))
    with pytest.raises(ValueError, match="linked source rounds"):
        r._target_source_gate(r1, dict(round=1), r2)


def test_source_calibration_cutoff_is_response_only_and_inclusive() -> None:
    a = calibration()
    r._check_calibration(a, ("BPAT", "CISO"))
    a["calibration_response_available_at"][:] = r.CALIBRATION_CUTOFF.value
    r._check_calibration(a, ("BPAT", "CISO"))
    a["calibration_response_available_at"][0] += 1
    with pytest.raises(ValueError, match="lawful cutoff"):
        r._check_calibration(a, ("BPAT", "CISO"))


@pytest.mark.parametrize("kind", ["target", "future", "duplicate", "floatdate", "response_before_end"])
def test_invalid_source_response_cohort_is_rejected(kind: str) -> None:
    a = calibration()
    if kind == "target":
        a["calibration_regions"] = np.array(["AZPS", "CISO"])
    elif kind == "future":
        a["calibration_origins"][0] = r.CALIBRATION_CUTOFF.value
    elif kind == "duplicate":
        a["calibration_regions"][:] = "BPAT"
        a["calibration_origins"][:] = a["calibration_origins"][0]
    elif kind == "floatdate":
        a["calibration_origins"] = a["calibration_origins"].astype(float)
    else:
        a["calibration_response_available_at"][:] = a["calibration_response_period_end"]-1
    with pytest.raises(ValueError):
        r._check_calibration(a, ("BPAT", "CISO"))


def test_exact_control_check_catches_raw_or_center_changes(tmp_path: Path) -> None:
    days = pd.date_range("2025-01-01", periods=2, tz="UTC")
    paths = baseline_fixture(tmp_path, days, "BPAT", "source")
    path = Path(paths["source/BPAT/fixed_donor"])
    with np.load(path) as data:
        a = {k: data[k] for k in data.files}
    r._check_baseline(a, path)
    for key in ("raw_ensemble", "center", "donor_indices"):
        changed = {**a, key: a[key]+1}
        with pytest.raises(ValueError, match="Exact V12"):
            r._check_baseline(changed, path)


def test_profile_audit_must_be_measured_and_consistent() -> None:
    prediction = region_results(1)["W", "gate_ridge"]
    audit = dict(donor_q_mean_max_error=prediction["amplitude_error"], sampled_zero_creation=prediction["sampled_zero_creation"])
    r._check_profile_audit(audit)
    with pytest.raises(ValueError, match="measured structural"):
        r._check_profile_audit({**audit, "donor_q_mean_max_error": None})
    audit["sampled_zero_creation"]["created_positive_from_zero_count_by_fuel"][0] = 1
    with pytest.raises(ValueError, match="counts/fractions"):
        r._check_profile_audit(audit)


@pytest.mark.parametrize("kind", ["simplex", "decomposition", "rank_multiset", "independent_weights", "index_range", "location_center"])
def test_factor_and_pairing_guards_reject_broken_saved_identities(kind: str) -> None:
    model = "location_scale" if kind == "location_center" else "physical_rank"
    a = factor_archive(model)
    r._check_factor_archive(a, 2, model)
    if kind == "simplex":
        a["calibration_true_mix"][0, 0] += .1
    elif kind == "decomposition":
        a["calibration_fine_errors"][0, 0] += 1
    elif kind == "rank_multiset":
        a["calibration_rank_indices"][:] = 1
    elif kind == "independent_weights":
        for key in ("fine_residual_indices", "calibration_rank_indices", "independent_residual_indices"):
            a[key][:] = 1
    elif kind == "index_range":
        a["fine_residual_indices"][:] = 2
    else:
        a["location_prediction"][:] = 1
    with pytest.raises(ValueError):
        r._check_factor_archive(a, 2, model)


def test_region_serializer_cannot_replace_protected_arrays_or_pickle_objects(tmp_path: Path) -> None:
    (tmp_path / "predictions").mkdir()
    days = pd.date_range("2025-01-01", periods=2, tz="UTC")
    results = region_results(1)
    results["B", "native_recompose"]["archive_arrays"] = {"origins": np.zeros(2)}
    with pytest.raises(ValueError, match="protected prediction keys"):
        r._save_region(tmp_path, dict(round=1), "source", "BPAT", days, results,
                       np.zeros((2, 128), int), None, {})
    results["B", "native_recompose"]["archive_arrays"] = {"unsafe": np.array([{}], dtype=object)}
    with pytest.raises(ValueError, match="pickled objects"):
        r._save_region(tmp_path, dict(round=1), "source", "BPAT", days, results,
                       np.zeros((2, 128), int), None, {})


def test_complete_region_serialization_and_exact_control_validation(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "predictions").mkdir()
    days = pd.date_range("2026-01-01", periods=2, tz="UTC")
    monkeypatch.setattr(r, "TARGETS", ("AZPS",))
    monkeypatch.setattr(r.v10, "_evaluation_days", lambda *args: days)
    frozen = dict(round=1, baseline_paths=baseline_fixture(tmp_path, days, "AZPS", "target"))
    audit = dict(training_regions=list(r.SOURCES), donor_regions=list(r.SOURCES), target_hourly_used=False)
    entries = r._save_region(tmp_path, frozen, "target", "AZPS", days, region_results(1),
                             np.zeros((2, 128), int), None, audit)
    r.check_entries(tmp_path, frozen, "target", dict(entries=entries))
    assert len(entries) == 6
    with np.load(tmp_path / entries[0]["path"]) as a:
        assert "residual_indices" not in a
    bad = [*entries, entries[0]]
    with pytest.raises(ValueError, match="incomplete/duplicated"):
        r.check_entries(tmp_path, frozen, "target", dict(entries=bad))
    entries[0]["path"] = "../outside.npz"
    with pytest.raises(ValueError, match="path/hash"):
        r.check_entries(tmp_path, frozen, "target", dict(entries=entries))


def test_all_risk_references_require_identical_lawful_response_cohort(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "predictions").mkdir()
    days = pd.date_range("2026-01-01", periods=2, tz="UTC")
    monkeypatch.setattr(r, "TARGETS", ("AZPS",))
    monkeypatch.setattr(r, "allowed_sources", lambda _: ("BPAT", "CISO"))
    monkeypatch.setattr(r.v10, "_evaluation_days", lambda *args: days)
    frozen = dict(round=2, calibration_sources=["BPAT", "CISO"],
                  baseline_paths=baseline_fixture(tmp_path, days, "AZPS", "target"))
    results = region_results(2)
    for (budget, model), prediction in results.items():
        if model in r.RISK_FAMILIES:
            extra = factor_archive(model)
            del extra["center"]
            if model.startswith("qrf_"):
                extra.update(conditional_weights=np.full((2, 2), .5),
                             conditional_residual_indices=np.zeros((2, 128), int),
                             raw_conditional_quantiles=np.full((2, 3, 24), 100.), quantile_levels=np.array([.05, .5, .95]))
            prediction["archive_arrays"] = extra
    audit = dict(training_regions=["BPAT", "CISO"], donor_regions=["BPAT", "CISO"],
                 source_calibration_regions=["BPAT", "CISO"], target_hourly_used=False)
    entries = r._save_region(tmp_path, frozen, "target", "AZPS", days, results,
                             np.zeros((2, 128), int), np.zeros((2, 128), int), audit)
    r.check_entries(tmp_path, frozen, "target", dict(entries=entries))
    selected = next(e for e in entries if e["budget"] == "W" and e["model"] == "physical_rank")
    path = tmp_path / selected["path"]
    with np.load(path) as a:
        changed = {k: a[k] for k in a.files}
    changed["calibration_origins"][1] += pd.Timedelta(days=1).value
    np.savez_compressed(path, **changed)
    selected["sha256"] = r.sha(path)
    with pytest.raises(ValueError, match="same calibration cohort"):
        r.check_entries(tmp_path, frozen, "target", dict(entries=entries))


def test_prediction_only_harness_seals_all_arms_without_scoring(tmp_path: Path, monkeypatch) -> None:
    import scripts
    from types import SimpleNamespace
    days = pd.date_range("2026-01-01", periods=2, tz="UTC")
    data, peer = tmp_path / "data", tmp_path / "peer"
    (tmp_path / "predictions").mkdir()
    peer.mkdir()
    r.write(tmp_path / "source_choices.json", dict(created_utc="2026-01-01T00:00:00+00:00"))
    r.write(peer / "source_completed.json", {})
    frozen = dict(round=1, data=str(data), weather_policy=None,
                  baseline_paths=baseline_fixture(tmp_path, days, "AZPS", "target"))
    monkeypatch.setattr(r, "TARGETS", ("AZPS",))
    monkeypatch.setattr(r, "verify", lambda _: frozen)
    monkeypatch.setattr(r, "_target_source_gate", lambda *args: peer)
    monkeypatch.setattr(r, "_load_training", lambda *args: ({}, {("AZPS", "evaluation"): dict(days=days)}))
    monkeypatch.setattr(r, "_coarse", lambda *args: (np.full((2, 8), 1/8), np.full((2, 128, 8), 1/8)))
    monkeypatch.setattr(r.v10, "_evaluation_days", lambda *args: days)
    monkeypatch.setattr(r, "score_phase", lambda *args: pytest.fail("prediction-only opened scorer"))
    calls = []
    class Context:
        shape_fit_audits = {}
        gate_fit_audits = {}
        def predict_region(self, subset, region, mixes, draws, **kwargs):
            calls.append((subset, region, kwargs))
            audit = dict(training_regions=list(subset), donor_regions=list(subset), target_hourly_used=False)
            return region_results(1), np.zeros((2, 128), int), None, audit
    def prepare(*args, **kwargs):
        assert kwargs == dict(stage="target", round_number=1, weather_policy=None)
        return Context()
    monkeypatch.setattr(scripts, "weather_coarse_v14", SimpleNamespace(prepare=prepare), raising=False)
    r.run_phase(tmp_path, data, "target", round_number=1, predict_only=True)
    seal = r.read(tmp_path / "target_prediction_seal.json")
    assert len(seal["entries"]) == 6
    assert seal["evaluation_hourly_labels_read_preseal"] is False
    assert calls[0][1] == "AZPS"
    assert not (tmp_path / "target_completed.json").exists()
    assert not (tmp_path / "target_scores").exists()


def test_monthly_preflight_freezes_six_sources_without_dropping_isne_proper(tmp_path: Path) -> None:
    rows = []
    for region in r.SOURCES:
        if region == "ISNE":
            continue
        rows.append(dict(region=region, period_start="2024-07-01", period_end="2024-08-01",
                         available_at="2024-08-31", **{c: 1/8 for c in r.COLUMNS}))
    # Unreleased response content must not affect the approved set.
    rows.append(dict(region="ISNE", period_start="2024-12-01", period_end="2025-01-01",
                     available_at="2025-01-31", **{c: np.nan for c in r.COLUMNS}))
    pd.DataFrame(rows).to_csv(tmp_path / "monthly.csv", index=False)
    approved = r.calibration_sources(tmp_path)
    assert approved == [s for s in r.SOURCES if s != "ISNE"]
    frozen = dict(calibration_sources=approved)
    assert "ISNE" in r.allowed_sources()
    assert len(r.allowed_sources("ISNE")) == 6
    assert len(r.allowed_calibration_sources(frozen, r.allowed_sources("ISNE"))) == 6
    assert len(r.allowed_calibration_sources(frozen, r.allowed_sources("BPAT"))) == 5


def test_approved_missing_isne_calibration_is_accepted_but_unexpected_missing_is_rejected() -> None:
    approved = tuple(s for s in r.SOURCES if s != "ISNE")
    frozen = dict(calibration_sources=list(approved))
    expected = r.allowed_calibration_sources(frozen, r.allowed_sources())
    n = len(approved)
    origins = pd.date_range("2024-07-01", periods=n, tz="UTC").as_unit("ns").asi8
    a = dict(calibration_regions=np.array(approved, dtype="U"), calibration_origins=origins,
        calibration_response_period_end=np.full(n, pd.Timestamp("2024-08-01", tz="UTC").value),
        calibration_response_available_at=np.full(n, pd.Timestamp("2024-08-31", tz="UTC").value))
    r._check_calibration(a, expected)
    for key in a:
        a[key] = a[key][:-1]
    with pytest.raises(ValueError, match="Wrong source calibration"):
        r._check_calibration(a, expected)


def test_freeze_pins_monthly_preflight_and_verify_recomputes_it(tmp_path: Path, monkeypatch) -> None:
    source, upstream, data = tmp_path / "source", tmp_path / "upstream", tmp_path / "data"
    source.mkdir()
    upstream.mkdir()
    table = pd.DataFrame([dict(region=s, period_start="2024-07-01", period_end="2024-08-01",
        available_at="2024-08-31", **{c: 1/8 for c in r.COLUMNS}) for s in r.SOURCES if s != "ISNE"])
    table.to_csv(source / "monthly.csv", index=False)
    for name in ("prefit_freeze.json", "source_choices.json", "source_prediction_seal.json",
                 "target_prediction_seal.json", "source_completed.json", "target_completed.json"):
        (upstream / name).write_text("{}")
    old = dict(data=str(data), source=str(source), weather_csv="weather", points_json="points", alpha=.5,
               input_hashes={}, sources=list(r.SOURCES), targets=list(r.TARGETS), samples=r.N_SAMPLES,
               seed=r.SEED, factors=list(r.FACTORS), proper_start="2024-01-25")
    monkeypatch.setattr(r, "V12_ROOT", upstream)
    monkeypatch.setattr(r.v12, "verify", lambda _: old)
    monkeypatch.setattr(r, "_baseline_paths", lambda: {})
    monkeypatch.setattr(r, "_code_paths", lambda: [])
    output = tmp_path / "new"
    frozen = r.freeze(output, data, round_number=1)
    assert frozen["version"] == 14
    assert frozen["calibration_sources"] == [s for s in r.SOURCES if s != "ISNE"]
    assert r.verify(output) == frozen
    # This test's synthetic upstream deliberately omits monthly hashes, so the
    # new recomputation check, rather than the generic byte guard, must reject.
    table.loc[table.region.eq("CISO"), "available_at"] = "2025-01-31"
    table.to_csv(source / "monthly.csv", index=False)
    with pytest.raises(ValueError, match="frozen grid/environment"):
        r.verify(output)


def repair_fixture(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    old, new = tmp_path / "old-v13", tmp_path / "new-v14"
    monkeypatch.setattr(r, "SOURCES", ("BPAT",))
    monkeypatch.setattr(r, "V13_REPAIR_ROOT", old)
    for folder in (old, new):
        (folder / "predictions").mkdir(parents=True)
        (folder / "source_scores").mkdir()
        entries = []
        for i, c in enumerate(r.configurations(1)):
            path = folder / "predictions" / f"{i}.npz"
            np.savez(path, origins=np.arange(2, dtype=np.int64), raw_ensemble=np.full((2, 3), float(i)))
            entries.append(dict(region="BPAT", budget=c["budget"], model=c["model"],
                                path=str(path.relative_to(folder)), sha256=r.sha(path)))
        r.write(folder / "source_prediction_seal.json", dict(round=1, entries=entries, complete_grid=True,
            evaluation_hourly_labels_read_preseal=False))
        r.write(folder / "prefit_freeze.json", dict(version=13 if folder == old else 14, round=1))
        choice = {k: {"identity": k} for k in ("selections", "primary", "native", "simple", "baseline", "diagnostic_crps_best")}
        r.write(folder / "source_choices.json", dict(**choice, locked_policy="gate_ridge", target_scores_used=False))
        hashes = {}
        for name in ("day_metrics.csv", "region_metrics.csv", "macro_metrics.csv"):
            path = folder / "source_scores" / name
            path.write_text("value\n1\n")
            hashes[name] = r.sha(path)
        r.write(folder / "source_scores/score_seal.json", hashes)
        r.write(folder / "source_completed.json", dict(prefit_freeze_sha256=r.sha(folder / "prefit_freeze.json"),
            prediction_seal_sha256=r.sha(folder / "source_prediction_seal.json"),
            score_seal_sha256=r.sha(folder / "source_scores/score_seal.json"),
            source_choices_sha256=r.sha(folder / "source_choices.json")))
    return old, new


def test_repair_identity_binds_reference_manifests_scores_and_every_array(tmp_path: Path, monkeypatch) -> None:
    old, new = repair_fixture(tmp_path, monkeypatch)
    paths = r._verify_repair_identity(new)
    assert len(paths) == 8+len(r.configurations(1))
    assert old / "prefit_freeze.json" in paths
    assert all(p.is_relative_to(old) for p in paths)
    assert len([p for p in paths if p.suffix == ".npz"]) == 6


@pytest.mark.parametrize("kind", ["array", "keys", "score", "choice", "old_hash", "incomplete"])
def test_repair_identity_fails_closed_on_any_source_drift(kind: str, tmp_path: Path, monkeypatch) -> None:
    old, new = repair_fixture(tmp_path, monkeypatch)
    manifest = r.read(new / "source_prediction_seal.json")
    if kind in ("array", "keys", "old_hash"):
        folder = old if kind == "old_hash" else new
        path = folder / manifest["entries"][0]["path"]
        with np.load(path) as a:
            changed = {k: a[k] for k in a.files}
        if kind == "keys":
            changed["extra"] = np.zeros(1)
        else:
            changed["raw_ensemble"] += 1
        np.savez(path, **changed)
        if kind != "old_hash":
            manifest["entries"][0]["sha256"] = r.sha(path)
            (new / "source_prediction_seal.json").write_text(json.dumps(manifest))
    elif kind == "score":
        (new / "source_scores/region_metrics.csv").write_text("value\n2\n")
    elif kind == "choice":
        choice = r.read(new / "source_choices.json")
        choice["locked_policy"] = "gate_forest"
        (new / "source_choices.json").write_text(json.dumps(choice))
    else:
        manifest["entries"] = manifest["entries"][:-1]
        (new / "source_prediction_seal.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        r._verify_repair_identity(new)
