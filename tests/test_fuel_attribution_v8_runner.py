"""Small, label-free checks of V8 selection, oracle isolation, and seals."""
import copy
import importlib.util
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("fuel_attribution_v8_runner_tests",
                                               ROOT / "scripts/run_fuel_attribution_v8.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def score_table():
    rows = []
    identities = [
        (1, "latest-recompose_joint-b1.0"),
        (2, "adaptive_mixture-t0.03-recompose_joint-b0.3"),
    ]
    for round_, primary in identities:
        for mode in runner.MODES:
            name = primary.replace("recompose_joint", mode)
            for factor in (1., 1.25):
                for region in ("A", "B"):
                    for delay in (0, 30):
                        for semester in ("H1", "H2"):
                            rows.append({"round": round_, "name": name, "mode": mode,
                                         "factor": factor, "region": region, "delay": delay,
                                         "semester": semester, "days": 10,
                                         "crps": (0 if mode == "recompose_joint" else 1) + (0 if factor == 1.25 else 2),
                                         "empirical_coverage90": .92 if factor == 1.25 else .8,
                                         "empirical_width90": 20.,
                                         "empirical_interval_score90": 25.})
    return pd.DataFrame(rows)


def test_source_selection_uses_only_h1_and_keeps_exact_factor():
    table = score_table()
    selected, _ = runner.selection(table)
    altered = table.copy()
    stress = altered.semester.eq("H2")
    altered.loc[stress, "crps"] = np.where(altered.loc[stress, "factor"].eq(1.), -100., 1e6)
    altered.loc[stress, "empirical_coverage90"] = np.where(altered.loc[stress, "factor"].eq(1.), .92, 0.)
    assert runner.selection(altered)[0] == selected
    for scope in ("1", "2", "joint"):
        primary = selected[scope]["primary"]["configuration"]
        for role, mode in (("exact_additive", "additive_anchor"), ("exact_direct", "direct_analog")):
            exact = selected[scope][role]["configuration"]
            assert exact["factor"] == primary["factor"] == 1.25
            assert exact["name"] == primary["name"].replace("recompose_joint", mode)
    oracle = table.iloc[:1].copy()
    oracle["mode"] = "oracle_recompose_joint"
    with pytest.raises(ValueError, match="Only lawful"):
        runner.selection(pd.concat([table, oracle], ignore_index=True))


def choices_fixture(tmp_path):
    output = tmp_path / "v8"
    scores = output / "source_scores"
    scores.mkdir(parents=True)
    table = score_table()
    table.to_csv(scores / "region_metrics.csv", index=False)
    for name in runner.A7.SCORE_FILES - {"region_metrics.csv"}:
        (scores / name).write_text("synthetic\n")
    table.to_csv(scores / "ranking.csv", index=False)
    runner.write(scores / "score_seal.json", {name: runner.sha(scores / name)
                                                for name in runner.A7.SCORE_FILES})
    selected, _ = runner.selection(table)
    runner.write(output / "source_prediction_seal.json", {"entries": [], "complete_grid": True})
    choice = {"created_utc": "2026-10-23T00:00:00+00:00", "selections": selected,
              "source_score_hashes": {path.name: runner.sha(path) for path in scores.iterdir()},
              "source_seal_sha256": runner.sha(output / "source_prediction_seal.json"),
              "oracle_used": False, "target_scores_used": False}
    runner.write(output / "source_choices.json", choice)
    return output, choice


def test_source_choices_reject_tampering_and_oracle_claim(tmp_path):
    output, choice = choices_fixture(tmp_path)
    runner.validate_choices(output, choice)
    changed = copy.deepcopy(choice)
    changed["selections"]["1"]["primary"]["configuration"]["factor"] = 1.
    with pytest.raises(ValueError, match="does not reproduce"):
        runner.validate_choices(output, changed)
    changed = copy.deepcopy(choice)
    changed["oracle_used"] = True
    with pytest.raises(ValueError, match="source-only"):
        runner.validate_choices(output, changed)
    score_path = output / "source_scores/region_metrics.csv"
    score_path.write_bytes(score_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source scores changed"):
        runner.validate_choices(output, choice)


def monthly_row(region, start, **shares):
    month = pd.Timestamp(start, tz="UTC")
    values = {"region": region, "period_start": str(month),
              "period_end": str(month + pd.offsets.MonthBegin(1))}
    values.update({"fuelshare_" + fuel: shares.get(fuel, 0.) for fuel in runner.FUELS})
    return values


def test_oracle_mix_uses_true_target_month_and_rejects_bad_rows():
    rows = pd.DataFrame([monthly_row("T", "2026-01-01", gas=.2, wind=.8),
                         monthly_row("T", "2026-02-01", gas=.7, wind=.3),
                         monthly_row("OTHER", "2026-01-01", gas=1.)])
    days = pd.DatetimeIndex(["2026-01-15", "2026-02-01", "2026-01-31"], tz="UTC").as_unit("us")
    actual = runner.oracle_mix(rows, "T", days)
    np.testing.assert_allclose(actual[:, runner.FUELS.index("gas")], [.2, .7, .2])
    with pytest.raises(ValueError, match="Missing oracle month"):
        runner.oracle_mix(rows.iloc[[0]], "T", days)
    with pytest.raises(ValueError, match="Duplicate"):
        runner.oracle_mix(pd.concat([rows, rows.iloc[[0]]]), "T", days)
    bad = rows.copy()
    bad.loc[0, "fuelshare_gas"] = .5
    with pytest.raises(ValueError, match="Invalid actual composition"):
        runner.oracle_mix(bad, "T", days)
    bad = rows.copy()
    bad.loc[0, "period_end"] = "2026-03-01T00:00:00+00:00"
    with pytest.raises(ValueError, match="one calendar month"):
        runner.oracle_mix(bad, "T", days)


def test_saved_origins_parity_is_ns_us_invariant(tmp_path):
    output = tmp_path / "v8"
    (output / "predictions").mkdir(parents=True)
    days = pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC").as_unit("us")
    point = np.full((2, 24), 100.)
    prediction = {"point": point, "ensemble": np.repeat(point[:, None], 128, axis=1)}
    base = {"round": 1, "name": "latest-recompose_joint-b1.0", "mode": "recompose_joint",
            "forecast": "latest", "bandwidth": 1., "temperature": .03}
    result = {"direct_analog": prediction, "audit": {"draw_sha256": "synthetic"}}
    entry = runner.save(output, "source", "A", 0, days, base, result, ("direct_analog",))[0]
    assert runner.parity(output / entry["path"], prediction, days.as_unit("ns")) == {
        "point": 0., "ensemble": 0.}


class BankIdentity:
    def __init__(self, regions):
        self.source_regions_ = tuple(regions)


def test_load_fit_enforces_excluded_region_and_frozen_hash(tmp_path, monkeypatch):
    monkeypatch.setattr(runner.V7, "SOURCES", ("A", "B", "HELD"))
    upstream = tmp_path / "v7"
    fits = upstream / "fits"
    fits.mkdir(parents=True)
    path = fits / "lodo-HELD-30.pkl"
    path.write_bytes(pickle.dumps({"training_regions": ("A", "B"),
                                   "bank": BankIdentity(("A", "B"))}))
    frozen = {"upstream": str(upstream), "dependencies": {str(path): runner.sha(path)}}
    assert runner.load_fit(frozen, "HELD", 30, "source")["training_regions"] == ("A", "B")
    path.write_bytes(pickle.dumps({"training_regions": ("A", "B", "HELD"),
                                   "bank": BankIdentity(("A", "B", "HELD"))}))
    frozen["dependencies"][str(path)] = runner.sha(path)
    with pytest.raises(ValueError, match="leave-one-domain-out"):
        runner.load_fit(frozen, "HELD", 30, "source")
    path.write_bytes(path.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="Fitted model changed"):
        runner.load_fit(frozen, "HELD", 30, "source")


def test_oracle_cannot_read_actual_mix_until_legal_is_verified(tmp_path, monkeypatch):
    output = tmp_path / "v8"
    output.mkdir()
    frozen = {"source": str(tmp_path / "source"), "target": str(tmp_path / "target")}
    monkeypatch.setattr(runner, "verify", lambda path: frozen)
    seen = []

    def verify_phase(path, stage):
        seen.append(stage)
        if stage == "legal":
            raise ValueError("legal completion invalid")
        return {}

    monkeypatch.setattr(runner, "verify_phase", verify_phase)
    monkeypatch.setattr(runner, "oracle_mix", lambda *args: pytest.fail("oracle mix read before legal validation"))
    with pytest.raises(ValueError, match="legal completion invalid"):
        runner.run_phase(output, tmp_path / "v7", "oracle")
    assert seen == ["source", "legal"]


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("stage", ["source", "legal"])
def test_scores_wait_for_complete_hashed_prediction_grid(tmp_path, monkeypatch, corrupt, stage):
    upstream = tmp_path / "v7"
    output = tmp_path / "v8"
    (upstream / "predictions").mkdir(parents=True)
    source, target = tmp_path / "source", tmp_path / "target"
    source.mkdir()
    target.mkdir()
    pd.DataFrame({"region": ["A"]}).to_csv(source / "monthly.csv", index=False)
    pd.DataFrame({"region": ["T"]}).to_csv(target / "monthly.csv", index=False)
    base = {"round": 1, "name": "latest-recompose_joint-b1.0",
            "mode": "recompose_joint", "forecast": "latest", "bandwidth": 1., "temperature": .03}
    year = 2025 if stage == "source" else 2026
    region = "A" if stage == "source" else "T"
    prior_stage = "source" if stage == "source" else "target"
    days = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01" if stage == "source" else "2026-07-01",
                         inclusive="left", tz="UTC")
    point = np.full((len(days), 24), 100.)
    prediction = {"point": point, "ensemble": point[:, None, :]}
    old_entries = []
    for mode in ("direct_analog", "recompose_joint"):
        name = runner.config(base, mode)["name"]
        path = upstream / "predictions" / f"{prior_stage}-{region}-0-r1-{name}.npz"
        np.savez_compressed(path, origins=days.as_unit("ns").asi8, **prediction)
        old_entries.append({"region": region, "delay": 0, "round": 1, "name": name,
                            "path": str(path.relative_to(upstream)), "sha256": runner.sha(path)})
    runner.write(upstream / f"{prior_stage}_prediction_seal.json", {"entries": old_entries})
    frozen = {"source": str(source), "target": str(target), "upstream": str(upstream),
              "identities": [base]}
    monkeypatch.setattr(runner.V7, "SOURCES", ("A",))
    monkeypatch.setattr(runner.V7, "TARGETS", ("T",))
    monkeypatch.setattr(runner.V7, "DELAYS", (0,))
    monkeypatch.setattr(runner.V7, "make_features", lambda *args: {})
    monkeypatch.setattr(runner, "load_fit", lambda *args: {"bank": object()})
    monkeypatch.setattr(runner, "verify", lambda *args: frozen)
    monkeypatch.setattr(runner, "finish", lambda *args: None)

    def freeze(path, prior):
        (path / "predictions").mkdir(parents=True)
        return frozen

    monkeypatch.setattr(runner, "freeze", freeze)
    if stage == "legal":
        (output / "predictions").mkdir(parents=True)
        runner.write(output / "source_choices.json", {"synthetic": True})
        monkeypatch.setattr(runner, "verify_phase", lambda path, phase: {})

    def generate(bank, coarse, origins, config, oracle_mix=None):
        assert oracle_mix is None
        assert origins.equals(days)
        return {**{mode: prediction for mode in runner.MODES},
                "audit": {"draw_sha256": "synthetic", "sampled_donor_indices": np.zeros((1, 1), dtype=int)}}

    monkeypatch.setattr(runner.PATHS, "generate_matched", generate)
    if corrupt:
        original_save = runner.save

        def save(*args):
            entries = original_save(*args)
            path = output / entries[0]["path"]
            path.write_bytes(path.read_bytes() + b"tampered")
            return entries

        monkeypatch.setattr(runner, "save", save)

    score_calls = []

    def score(path, entries, data, year, folder):
        score_calls.append(True)
        seal = runner.read(output / f"{stage}_prediction_seal.json")
        assert seal["complete_grid"] and not seal["hourly_labels_read_in_this_prediction_phase"]
        assert len(seal["entries"]) == len(runner.MODES)
        assert {e["name"] for e in seal["entries"]} == {
            runner.config(base, mode)["name"] for mode in runner.MODES}
        assert all(runner.sha(output / e["path"]) == e["sha256"] for e in seal["entries"])
        folder.mkdir()
        score_table().to_csv(folder / "region_metrics.csv", index=False)
        for name in runner.A7.SCORE_FILES - {"region_metrics.csv"}:
            (folder / name).write_text("synthetic\n")
        runner.write(folder / "score_seal.json", {name: runner.sha(folder / name)
                                                 for name in runner.A7.SCORE_FILES})

    monkeypatch.setattr(runner.V7, "score_entries", score)
    monkeypatch.setattr(runner, "selection", lambda table: ({"synthetic": True}, table))
    if corrupt:
        with pytest.raises(ValueError, match="prediction grid"):
            runner.run_phase(output, upstream, stage)
        assert score_calls == []
        assert not (output / f"{stage}_prediction_seal.json").exists()
    else:
        runner.run_phase(output, upstream, stage)
        assert score_calls == [True]
        assert (output / "source_choices.json").exists()
