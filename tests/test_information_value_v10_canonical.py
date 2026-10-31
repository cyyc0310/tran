"""Versioned W rerun uses the sealed CSV for source choice reproduction."""

from pathlib import Path

import pandas as pd

from scripts import run_information_value_v10 as original
from scripts import run_information_value_v10_canonical as canonical


def test_canonical_return_uses_exact_sealed_csv_values(tmp_path, monkeypatch):
    folder = tmp_path / "source_scores"
    folder.mkdir()
    in_memory = pd.DataFrame({"empirical_coverage90": [0.9444945177913981],
                              "budget": ["R"], "model": ["ridge"]})

    def original_scorer(output, frozen, stage, seal):
        in_memory.to_csv(folder / "region_metrics.csv", index=False)
        return in_memory

    monkeypatch.setattr(canonical, "ORIGINAL_SCORE_PHASE", original_scorer)
    returned = canonical.canonical_score_phase(tmp_path, {}, "source", {})
    sealed = pd.read_csv(folder / "region_metrics.csv")
    pd.testing.assert_frame_equal(returned, sealed, check_exact=True)
    assert returned is not in_memory
    assert returned.empirical_coverage90.iloc[0] != in_memory.empirical_coverage90.iloc[0]


def test_new_code_closure_adds_only_versioned_entry_point():
    old = canonical.ORIGINAL_CODE_PATHS()
    new = canonical.canonical_code_paths()
    assert new == [*old, Path(canonical.__file__).resolve()]
    assert Path(original.__file__).resolve() in old
    assert len(new) == len(set(new))


def test_main_installs_hooks_then_delegates_once(monkeypatch):
    observed = []

    def delegated_main():
        observed.append((original.score_phase, original._code_paths))

    monkeypatch.setattr(original, "main", delegated_main)
    with monkeypatch.context() as local:
        local.setattr(original, "score_phase", canonical.ORIGINAL_SCORE_PHASE)
        local.setattr(original, "_code_paths", canonical.ORIGINAL_CODE_PATHS)
        canonical.main()
        assert observed == [(canonical.canonical_score_phase,
                             canonical.canonical_code_paths)]
