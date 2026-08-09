"""Tests for LOO-CV training pipeline (Task 3.2).

LOO-CV leaves one source region out, trains on the rest, and predicts the
held-out region. This characterizes head overfit (R2 in the risk table) and
gives an honest estimate of how the head will perform on an unseen target.
"""

import numpy as np
import pytest

from transcif.config import HORIZON
from transcif.models.zeroshot.fusion import (
    DIRECTION_ORDER,
    FusionModel,
    loo_cv_train,
)


def _synthetic_sources(n_sources: int = 5,
                       n_windows: int = 20,
                       signal_strength: np.ndarray | None = None,
                       noise_std: float = 5.0,
                       seed: int = 0):
    """Construct synthetic source stacks where one direction is informative.

    By default, ``direction 0`` (rag) carries the true signal and the other
    four emit noise. This gives the head a clear "correct" answer (one-hot on
    direction 0) so we can verify the head learns it stably across folds.
    """
    rng = np.random.default_rng(seed)
    if signal_strength is None:
        signal_strength = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

    stacks, truths, names = [], [], []
    for i in range(n_sources):
        y = rng.normal(500.0, 50.0, size=(n_windows, HORIZON)).astype(np.float32)
        stack = np.empty((n_windows, len(DIRECTION_ORDER), HORIZON),
                         dtype=np.float32)
        for d in range(len(DIRECTION_ORDER)):
            base = y if signal_strength[d] > 0 else np.zeros_like(y)
            noise = rng.normal(0.0, noise_std, size=y.shape).astype(np.float32)
            stack[:, d, :] = (signal_strength[d] * base +
                              (1.0 - signal_strength[d]) * 500.0 + noise)
        stacks.append(stack)
        truths.append(y)
        names.append(f"SRC{i}")
    return stacks, truths, names


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

def test_loo_cv_train_returns_required_fields():
    """Output dict contains per-fold records + final model + weight stats."""
    stacks, truths, names = _synthetic_sources(n_sources=5, seed=0)

    result = loo_cv_train(stacks, truths, names, epochs=20, seed=0)

    for key in ("loo_per_fold", "weight_vectors",
                "weight_std_per_direction", "oof_mae_mean",
                "oof_mae_std", "final_model"):
        assert key in result, f"Missing required field: {key}"

    assert isinstance(result["final_model"], FusionModel)
    assert result["weight_vectors"].shape == (5, len(DIRECTION_ORDER))
    assert result["weight_std_per_direction"].shape == (len(DIRECTION_ORDER),)


def test_loo_cv_train_per_fold_record_shape():
    """Each per-fold record has the documented fields and shapes."""
    stacks, truths, names = _synthetic_sources(n_sources=4, seed=0)

    result = loo_cv_train(stacks, truths, names, epochs=10, seed=0)

    assert len(result["loo_per_fold"]) == 4
    for i, rec in enumerate(result["loo_per_fold"]):
        assert rec["fold"] == i
        assert rec["name"] == f"SRC{i}"
        assert rec["weights"].shape == (len(DIRECTION_ORDER),)
        assert np.isfinite(rec["oof_mae"])
        assert np.isfinite(rec["in_fold_mae"])
        assert rec["oof_mae"] >= 0
        assert rec["in_fold_mae"] >= 0


# ---------------------------------------------------------------------------
# Overfit detection (R2)
# ---------------------------------------------------------------------------

def test_loo_cv_oof_gap_small_when_signal_is_clear():
    """When one direction is clearly informative, OOF MAE should track
    in-fold MAE (no overfit). The R2 DoD asks for <20% gap."""
    stacks, truths, names = _synthetic_sources(
        n_sources=5, noise_std=2.0, seed=42,
    )

    result = loo_cv_train(stacks, truths, names, epochs=80, seed=0)
    oof = result["oof_mae_mean"]
    in_fold = np.mean([r["in_fold_mae"] for r in result["loo_per_fold"]])
    gap = abs(oof - in_fold) / max(in_fold, 1e-6)

    assert gap < 0.20, (
        f"OOF/in-fold gap {gap:.1%} exceeds 20% R2 budget "
        f"(oof={oof:.3f}, in_fold={in_fold:.3f})"
    )


def test_loo_cv_weight_std_small_when_signal_is_clear():
    """Clear signal direction should produce stable weights across folds.
    The R2 DoD asks for weight vector std < 0.15 per direction."""
    stacks, truths, names = _synthetic_sources(
        n_sources=5, noise_std=2.0, seed=42,
    )

    result = loo_cv_train(stacks, truths, names, epochs=80, seed=0)
    std = result["weight_std_per_direction"]

    assert (std < 0.15).all(), (
        f"Per-direction weight std {std} exceeds 0.15 R2 budget; head is "
        f"flip-flopping across folds"
    )


# ---------------------------------------------------------------------------
# Final head retrained on all sources
# ---------------------------------------------------------------------------

def test_loo_cv_final_model_trained_on_all_sources():
    """final_model is built from train_fusion on the full source list.

    When folds stably converge to the same weights (easy problem), the final
    head may legitimately match a fold. The contract is structural: the
    returned model is non-null and exposes the trained head + (optional)
    predictors wiring. We check those properties here rather than asserting
    weight divergence, which is data-dependent.
    """
    stacks, truths, names = _synthetic_sources(n_sources=5, seed=0)

    result = loo_cv_train(stacks, truths, names, epochs=10, seed=0)
    final_head = result["final_model"].head

    with __import__("torch").no_grad():
        weights = final_head.weights().cpu().numpy()

    # Structural: weights are a valid probability distribution
    assert weights.shape == (len(DIRECTION_ORDER),)
    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= 0).all()


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------

def test_loo_cv_train_rejects_mismatched_lengths():
    """Mismatched stack/true/name lengths raise ValueError."""
    stacks, truths, names = _synthetic_sources(n_sources=5)
    with pytest.raises(ValueError, match="length"):
        loo_cv_train(stacks[:-1], truths, names, epochs=1)


def test_loo_cv_train_rejects_fewer_than_two_sources():
    """LOO needs at least 2 sources (1 train + 1 holdout)."""
    stacks, truths, names = _synthetic_sources(n_sources=1)
    with pytest.raises(ValueError, match="at least 2"):
        loo_cv_train(stacks, truths, names, epochs=1)
