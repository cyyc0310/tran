import numpy as np
import pytest

from transcif.research.benchmark_v2 import _trim, joint_scores, run


def test_joint_scores_zero_for_deterministic_correct_trajectory():
    curve = np.linspace(100, 800, 24)[None, :]
    result = joint_scores(curve, np.ones(1), np.repeat(curve, 4, axis=0))
    assert result["energy_score_mc"] == pytest.approx(0)
    assert result["variogram_score"] == pytest.approx(0)


def test_joint_variogram_detects_wrong_time_dependence():
    pattern = np.tile([-100, 100], 12)
    truth = np.stack([500 + pattern, 500 - pattern])
    # Same hourly marginals, different trajectory dependence.
    forecast = np.stack([np.full(24, 400), np.full(24, 600)])
    correct = joint_scores(truth, np.array([.5, .5]), truth)
    wrong = joint_scores(forecast, np.array([.5, .5]), truth)
    assert wrong["variogram_score"] > correct["variogram_score"] + 1


def test_trim_records_and_bounds_dropped_probability():
    trimmed = _trim({"weights": np.array([1e-16, .4, .6-1e-16]),
                     "predictions": np.zeros((3, 24)), "v": np.arange(3),
                     "quadrature_evaluations": 3})
    assert trimmed["removed_mass"] == pytest.approx(1e-16)
    assert trimmed["weights"].sum() == pytest.approx(1)
    assert trimmed["quadrature_evaluations"] == 3
    assert len(trimmed["v"]) == 2


def test_runner_refuses_to_overwrite_existing_evidence(tmp_path):
    (tmp_path / "existing.json").write_text("{}")
    with pytest.raises(ValueError, match="immutable"):
        run(tmp_path, tmp_path, "stress")
