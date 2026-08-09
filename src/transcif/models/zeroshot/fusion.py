"""Five-direction fusion model for zero-shot CIF forecasting.

This module is the public contract between (a) the per-direction predictors
under :mod:`transcif.models.zeroshot.{rag,phys_irm,causal,icl,hier}`,
(b) the source-region stack collector (Task 1.3), and (c) the test-time
calibration pipeline :func:`transcif.calibration.zs_plus.zs_plus_predict`,
which consumes a ``share_fn`` callable.

Task 1.1 establishes the interface only. The BasisMix head (non-negative
mixture with diversity regularization) lands in Task 3.1; the LOO-CV training
pipeline lands in Task 3.2. The default head shipped here is a plain
softmax-weight fusion so the interface can be exercised end-to-end.
"""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transcif.config import HORIZON, SEQ_LEN

DIRECTION_ORDER: tuple[str, ...] = ("rag", "phys", "causal", "icl", "hier")

PredictorFn = Callable[
    [np.ndarray, np.ndarray, float, float],
    np.ndarray,
]
"""Per-direction predictor contract.

A predictor takes ``(x_rs, config, ef_r, ef_nr)`` and returns CIF predictions:

    x_rs   : ``(B, SEQ_LEN)`` RenewShare windows (batched).
    config : ``(config_dim,)`` target config vector.
    ef_r   : renewable emission factor (gCO2/kWh).
    ef_nr  : non-renewable emission factor (gCO2/kWh).
    returns: ``(B, HORIZON)`` CIF predictions.

Batched semantics match the underlying direction modules
(``predict_rag_zs``, ``predict_phys_irm``, etc.) which all take ``(B, SEQ_LEN)``
input and return ``(B, HORIZON)``.
"""


class FusionHead(nn.Module):
    """5 -> 1 softmax-weight fusion over CIF predictions.

    Forward input : ``(n, 5, HORIZON)`` tensor or ndarray.
    Forward output: ``(n, HORIZON)`` tensor.
    """

    def __init__(self, n_directions: int = len(DIRECTION_ORDER)):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(n_directions))

    def forward(self, cif_stack):
        if isinstance(cif_stack, np.ndarray):
            cif_stack = torch.as_tensor(cif_stack, dtype=torch.float32)
        weights = torch.softmax(self.logit, dim=0)
        return (cif_stack * weights.view(1, -1, 1)).sum(dim=1)

    def weights(self) -> torch.Tensor:
        """Return the learned softmax weights as a (5,) tensor that sums to 1."""
        return torch.softmax(self.logit, dim=0).detach()


class EqualWeightFusion(nn.Module):
    """Equal-weight fusion baseline (Task 2.1).

    Forward input : ``(n, 5, HORIZON)`` tensor or ndarray.
    Forward output: ``(n, HORIZON)`` tensor computed as mean over 5 directions.

    No learnable parameters. Useful as a sanity baseline to compare against
    learned fusion heads.
    """

    def __init__(self, n_directions: int = len(DIRECTION_ORDER)):
        super().__init__()
        self.n_directions = n_directions

    def forward(self, cif_stack):
        if isinstance(cif_stack, np.ndarray):
            cif_stack = torch.as_tensor(cif_stack, dtype=torch.float32)
        return cif_stack.mean(dim=1)

    def weights(self) -> torch.Tensor:
        """Return uniform weights (1/n, ..., 1/n) for interface consistency."""
        return torch.ones(self.n_directions) / self.n_directions


class MedianFusion(nn.Module):
    """Median fusion baseline (Task 2.1).

    Forward input : ``(n, 5, HORIZON)`` tensor or ndarray.
    Forward output: ``(n, HORIZON)`` tensor computed as elementwise median.

    No learnable parameters. Robust to broken individual predictors (relevant
    since Hier alone has MAE 77.6). Median ignores outliers that would distort
    a mean fusion.
    """

    def __init__(self, n_directions: int = len(DIRECTION_ORDER)):
        super().__init__()
        self.n_directions = n_directions

    def forward(self, cif_stack):
        if isinstance(cif_stack, np.ndarray):
            cif_stack = torch.as_tensor(cif_stack, dtype=torch.float32)
        return cif_stack.median(dim=1).values

    def weights(self) -> torch.Tensor:
        """Return uniform weights (1/n, ..., 1/n) for interface consistency.

        Note: Median fusion doesn't actually use linear weights, but we
        return uniform weights to maintain a consistent interface with other
        fusion heads.
        """
        return torch.ones(self.n_directions) / self.n_directions


class BasisMixFusion(nn.Module):
    """Non-negative basis mixture fusion with diversity regularization (Task 3.1).

    This head extends the softmax-weight fusion (FusionHead) with three
    regularization terms:

    1. **L2 regularization** on logit weights (handled by optimizer weight_decay,
       but exposed via :meth:`l2_penalty` for explicit loss construction).
    2. **Entropy floor** that penalizes weight collapse to one-hot. When all
       5 weights are uniform, entropy = log(5) and loss = 0. When one weight = 1
       and rest = 0, entropy = 0 and loss = log(5)^2.
    3. **Diversity regularization** that penalizes pairwise cosine similarity
       between the 5 directions' CIF predictions. If two directions produce
       nearly identical predictions (cosine ≈ 1), we penalize redundancy.

    Paper framing: each direction = a named basis function
    (knowledge/physics/causality/context/hierarchy), and BasisMixFusion learns
    a non-negative mixture with diversity regularization.

    Forward input : ``(n, 5, HORIZON)`` tensor or ndarray.
    Forward output: ``(n, HORIZON)`` tensor.
    """

    def __init__(self, n_directions: int = len(DIRECTION_ORDER)):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(n_directions))

    def forward(self, cif_stack):
        if isinstance(cif_stack, np.ndarray):
            cif_stack = torch.as_tensor(cif_stack, dtype=torch.float32)
        weights = torch.softmax(self.logit, dim=0)
        return (cif_stack * weights.view(1, -1, 1)).sum(dim=1)

    def weights(self) -> torch.Tensor:
        """Return the learned softmax weights as a (5,) tensor that sums to 1."""
        return torch.softmax(self.logit, dim=0).detach()

    def l2_penalty(self) -> torch.Tensor:
        """Return the L2 penalty on logit weights: ||logit||^2."""
        return (self.logit ** 2).sum()

    def entropy_floor_loss(self) -> torch.Tensor:
        """Return the entropy floor penalty.

        Computes max(0, log(5) - H(w))^2 where H(w) = -sum(w * log(w+eps)).
        This penalizes weight collapse (one-hot). When all 5 weights are uniform,
        entropy = log(5) and loss = 0. When one weight = 1 and rest = 0,
        entropy = 0 and loss = log(5)^2.
        """
        weights = torch.softmax(self.logit, dim=0)
        eps = 1e-8
        entropy = -(weights * torch.log(weights + eps)).sum()
        target_entropy = np.log(len(self.logit))
        gap = target_entropy - entropy.item()
        return torch.tensor(max(0.0, gap) ** 2, dtype=torch.float32)

    def diversity_loss(self, cif_stack: torch.Tensor, threshold: float = 0.9) -> torch.Tensor:
        """Return the diversity penalty based on pairwise cosine similarity.

        Computes the mean off-diagonal cosine similarity between the 5 directions'
        CIF predictions. If two directions produce nearly identical predictions
        (cosine ≈ 1), we penalize redundancy. Only penalizes when cosine > threshold.

        Args:
            cif_stack: ``(n, 5, HORIZON)`` tensor of per-direction CIF predictions.
            threshold: Only penalize cosine similarity above this value.

        Returns:
            Scalar tensor representing the diversity penalty.
        """
        if cif_stack.ndim != 3 or cif_stack.shape[1] != len(self.logit):
            raise ValueError(
                f"cif_stack must be (n, {len(self.logit)}, HORIZON); "
                f"got shape {cif_stack.shape}"
            )

        n, n_directions, horizon = cif_stack.shape

        # Flatten each direction's predictions: (n, 5, HORIZON) -> (n_directions, n*HORIZON)
        # Stack along direction dimension to compute pairwise similarities
        flat_preds = cif_stack.transpose(0, 1).reshape(n_directions, -1)

        # Compute pairwise cosine similarities
        # Normalize vectors
        norms = flat_preds.norm(dim=1, keepdim=True) + 1e-8
        normalized = flat_preds / norms

        # Compute cosine matrix: (n_directions, n_directions)
        cosine_matrix = torch.mm(normalized, normalized.t())

        # Mask diagonal (self-similarity) and lower triangle (duplicates)
        mask = ~torch.eye(n_directions, dtype=torch.bool, device=cif_stack.device)
        upper_triangle = torch.triu(cosine_matrix, diagonal=1)

        # Only penalize when cosine > threshold
        high_similarity = upper_triangle[upper_triangle > threshold]

        if high_similarity.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=cif_stack.device)

        return high_similarity.mean()


class FusionModel:
    """Combines the 5 zero-shot direction predictors into a single CIF output.

    Two evaluation paths:

    1. **From a pre-computed stack**: :meth:`predict_cif_from_stack` is a pure
       combiner -- no predictors needed. Use this when the caller has already
       run the 5 directions and cached their CIF output.
    2. **End-to-end**: :meth:`predict_cif` takes raw RenewShare windows and
       calls each attached predictor per window. Requires ``predictors`` to be
       set at construction time.

    For ZS+ integration, call :meth:`configure_for_target` to bind the target
    region's emission factors, then pass ``fusion_model.share_fn`` directly
    to ``zs_plus_predict(..., share_fn=fusion_model.share_fn)``.
    """

    def __init__(
        self,
        head: FusionHead,
        predictors: Mapping[str, PredictorFn] | None = None,
    ):
        self.head = head
        self.predictors: dict[str, PredictorFn] | None = (
            dict(predictors) if predictors is not None else None
        )
        if self.predictors is not None:
            _validate_predictor_keys(self.predictors)
        self._target_cfg: tuple[np.ndarray, float, float] | None = None

    # ------------------------------------------------------------------
    # Pure combiner
    # ------------------------------------------------------------------

    def predict_cif_from_stack(self, cif_stack: np.ndarray) -> np.ndarray:
        """Fuse a pre-computed 5-direction CIF stack.

        Args:
            cif_stack: ``(n, 5, HORIZON)`` array of per-direction CIF preds.

        Returns:
            ``(n, HORIZON)`` fused CIF predictions.
        """
        if cif_stack.ndim != 3 or cif_stack.shape[1] != len(DIRECTION_ORDER):
            raise ValueError(
                f"cif_stack must be (n, {len(DIRECTION_ORDER)}, HORIZON); "
                f"got shape {cif_stack.shape}"
            )
        with torch.no_grad():
            fused = self.head(cif_stack)
        return fused.cpu().numpy()

    # ------------------------------------------------------------------
    # End-to-end
    # ------------------------------------------------------------------

    def predict_cif(
        self,
        x_rs: np.ndarray,
        config: np.ndarray,
        ef_r: float,
        ef_nr: float,
    ) -> np.ndarray:
        """End-to-end fused CIF prediction from raw RenewShare windows.

        Args:
            x_rs   : ``(n, SEQ_LEN)`` RenewShare windows.
            config : ``(config_dim,)`` target config vector.
            ef_r   : renewable emission factor.
            ef_nr  : non-renewable emission factor.

        Returns:
            ``(n, HORIZON)`` fused CIF predictions.
        """
        if self.predictors is None:
            raise RuntimeError(
                "FusionModel.predict_cif requires predictors to be set at "
                "construction time; got predictors=None."
            )
        if x_rs.ndim != 2 or x_rs.shape[1] != SEQ_LEN:
            raise ValueError(
                f"x_rs must be (n, SEQ_LEN={SEQ_LEN}); got {x_rs.shape}"
            )

        n = x_rs.shape[0]
        stack = np.empty((n, len(DIRECTION_ORDER), HORIZON), dtype=np.float32)
        for d, name in enumerate(DIRECTION_ORDER):
            pred = self.predictors[name](x_rs, config, ef_r, ef_nr)
            pred = np.asarray(pred, dtype=np.float32)
            if pred.shape != (n, HORIZON):
                raise ValueError(
                    f"predictor '{name}' returned shape {pred.shape}, "
                    f"expected ({n}, {HORIZON}). Predictors must accept "
                    f"batched (B, SEQ_LEN) input and return (B, HORIZON)."
                )
            stack[:, d, :] = pred
        return self.predict_cif_from_stack(stack)

    # ------------------------------------------------------------------
    # ZS+ integration
    # ------------------------------------------------------------------

    def configure_for_target(
        self,
        config: np.ndarray,
        ef_r: float,
        ef_nr: float,
    ) -> None:
        """Bind the target region's config + emission factors.

        Required before :meth:`share_fn` can be called by ``zs_plus_predict``.
        """
        self._target_cfg = (
            np.asarray(config, dtype=np.float64),
            float(ef_r),
            float(ef_nr),
        )

    def share_fn(self, x_window_np: np.ndarray) -> np.ndarray:
        """Per-window RenewShare prediction for ``zs_plus_predict``.

        The signature ``(x_window_np) -> (HORIZON,)`` matches the ``share_fn``
        hook in :func:`transcif.calibration.zs_plus.zs_plus_predict`. The
        fused CIF for the window is inverted to a RenewShare via the target's
        emission factors and clipped to ``[0, 1]``.
        """
        if self._target_cfg is None:
            raise RuntimeError(
                "FusionModel.share_fn requires configure_for_target(...) to "
                "bind the target region's emission factors first."
            )
        if self.predictors is None:
            raise RuntimeError(
                "FusionModel.share_fn requires predictors to be set."
            )
        if x_window_np.shape != (SEQ_LEN,):
            raise ValueError(
                f"x_window_np must be (SEQ_LEN={SEQ_LEN},); "
                f"got {x_window_np.shape}"
            )

        config, ef_r, ef_nr = self._target_cfg
        # Predictors take batched (B, SEQ_LEN). Wrap the single window and
        # squeeze the leading axis off each direction's (1, HORIZON) output.
        x_batch = x_window_np[None, :]
        stack = np.empty((len(DIRECTION_ORDER), HORIZON), dtype=np.float32)
        for d, name in enumerate(DIRECTION_ORDER):
            pred = np.asarray(
                self.predictors[name](x_batch, config, ef_r, ef_nr),
                dtype=np.float32,
            )
            stack[d] = pred.reshape(-1)
        # head expects (n, 5, HORIZON); single-window batch of 1.
        fused_cif = self.predict_cif_from_stack(stack[None]).squeeze(0)
        share = (fused_cif - ef_nr) / (ef_r - ef_nr + 1e-8)
        return np.clip(share, 0.0, 1.0)


def _validate_predictor_keys(predictors: Mapping[str, PredictorFn]) -> None:
    missing = set(DIRECTION_ORDER) - set(predictors.keys())
    extra = set(predictors.keys()) - set(DIRECTION_ORDER)
    if missing:
        raise ValueError(
            f"predictors missing required keys: {sorted(missing)}; "
            f"expected exactly {list(DIRECTION_ORDER)}."
        )
    if extra:
        raise ValueError(
            f"predictors has unexpected keys: {sorted(extra)}; "
            f"expected exactly {list(DIRECTION_ORDER)}."
        )


def train_fusion(
    src_cif_stacks: Sequence[np.ndarray],
    src_cif_true: Sequence[np.ndarray],
    predictors: Mapping[str, PredictorFn] | None = None,
    epochs: int = 200,
    lr: float = 1e-2,
    l2: float = 1e-4,
    seed: int = 0,
) -> FusionModel:
    """Train a FusionHead on source-region CIF stacks (zero-shot safe).

    The head is trained to minimize MAE between the fused 5-direction output
    and the source CIF ground truth. Source regions are NOT the target, so
    using their labels does not violate the zero-shot constraint.

    Args:
        src_cif_stacks : list of ``(n_i, 5, HORIZON)`` per-direction CIF
                         predictions on each source region's TEST window.
        src_cif_true   : list of ``(n_i, HORIZON)`` source CIF ground truth.
        predictors     : optional per-direction predictor callables, attached
                         to the returned model for end-to-end evaluation.
        epochs, lr, l2 : training hyperparameters.
        seed           : RNG seed for reproducibility.

    Returns:
        A :class:`FusionModel` with the trained head and (optionally) the
        supplied predictors attached.
    """
    if not src_cif_stacks:
        raise ValueError("src_cif_stacks must contain at least one source.")
    if len(src_cif_stacks) != len(src_cif_true):
        raise ValueError(
            f"len(src_cif_stacks)={len(src_cif_stacks)} must equal "
            f"len(src_cif_true)={len(src_cif_true)}."
        )
    if predictors is not None:
        _validate_predictor_keys(predictors)

    torch.manual_seed(seed)
    np.random.seed(seed)

    head = FusionHead(n_directions=len(DIRECTION_ORDER))
    optimizer = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=l2)

    X = np.concatenate(src_cif_stacks, axis=0).astype(np.float32)
    Y = np.concatenate(src_cif_true, axis=0).astype(np.float32)
    expected_X_shape = (X.shape[0], len(DIRECTION_ORDER), HORIZON)
    if X.shape != expected_X_shape:
        raise ValueError(
            f"each src stack must be (n_i, {len(DIRECTION_ORDER)}, HORIZON); "
            f"got concatenated shape {X.shape}."
        )
    if Y.shape != (X.shape[0], HORIZON):
        raise ValueError(
            f"src_true must concatenate to ({X.shape[0]}, HORIZON); "
            f"got {Y.shape}."
        )

    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    head.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = head(X_t)
        loss = torch.abs(pred - Y_t).mean()
        loss.backward()
        optimizer.step()
    head.eval()

    return FusionModel(head, predictors=predictors)


def _predict_stack_with_head(head: nn.Module,
                             cif_stack: np.ndarray) -> np.ndarray:
    """Run a trained head on a (n, 5, HORIZON) stack → (n, HORIZON)."""
    head.eval()
    with torch.no_grad():
        x = torch.as_tensor(cif_stack, dtype=torch.float32)
        return head(x).cpu().numpy()


def loo_cv_train(
    src_cif_stacks: Sequence[np.ndarray],
    src_cif_true: Sequence[np.ndarray],
    src_names: Sequence[str],
    predictors: Mapping[str, PredictorFn] | None = None,
    epochs: int = 200,
    lr: float = 1e-2,
    l2: float = 1e-4,
    seed: int = 0,
) -> dict:
    """Leave-one-out CV training for the fusion head (Task 3.2).

    For each source region *i*, train a head on all sources except *i*, then
    predict source *i*'s CIF. The resulting out-of-fold (OOF) MAE is an
    honest estimate of how the head will perform on an unseen target region.
    A final head is then retrained on *all* sources for deployment.

    The function also reports per-fold weight vectors and their per-direction
    standard deviation. Large std means the head is flip-flopping across
    folds (R2: meta-overfit signal).

    Args:
        src_cif_stacks : list of ``(n_i, 5, HORIZON)`` per-direction CIF
                         predictions on each source region's TEST window.
        src_cif_true   : list of ``(n_i, HORIZON)`` source CIF ground truth.
        src_names      : list of source region names, parallel to the above.
        predictors     : optional per-direction predictor callables, attached
                         to the final returned model.
        epochs, lr, l2 : training hyperparameters.
        seed           : RNG seed.

    Returns:
        Dict with keys:

            loo_per_fold            : list of ``{fold, name, weights,
                                      oof_mae, in_fold_mae}`` records.
            weight_vectors          : ``(n_sources, 5)`` array of per-fold
                                      softmax weights.
            weight_std_per_direction: ``(5,)`` array of per-direction std.
            oof_mae_mean            : mean OOF MAE across folds.
            oof_mae_std             : std of OOF MAE across folds.
            final_model             : :class:`FusionModel` trained on all
                                      sources (for deployment).
    """
    n = len(src_cif_stacks)
    if not (n == len(src_cif_true) == len(src_names)):
        raise ValueError(
            f"length mismatch: stacks={len(src_cif_stacks)}, "
            f"true={len(src_cif_true)}, names={len(src_names)}"
        )
    if n < 2:
        raise ValueError(
            f"LOO-CV requires at least 2 sources (1 train + 1 holdout); "
            f"got {n}."
        )
    if predictors is not None:
        _validate_predictor_keys(predictors)

    loo_per_fold = []
    weight_rows = []

    for i in range(n):
        train_stacks = [s for j, s in enumerate(src_cif_stacks) if j != i]
        train_true = [s for j, s in enumerate(src_cif_true) if j != i]

        fold_model = train_fusion(
            train_stacks, train_true,
            predictors=None,  # head-only; no need to wire predictors per fold
            epochs=epochs, lr=lr, l2=l2, seed=seed,
        )

        # OOF: predict held-out source
        oof_pred = _predict_stack_with_head(fold_model.head, src_cif_stacks[i])
        oof_mae = float(np.abs(oof_pred - src_cif_true[i]).mean())

        # In-fold MAE: mean over training sources (memorization diagnostic)
        in_fold_preds = [
            _predict_stack_with_head(fold_model.head, s)
            for s in train_stacks
        ]
        in_fold_maes = [
            float(np.abs(p - t).mean())
            for p, t in zip(in_fold_preds, train_true)
        ]
        in_fold_mae = float(np.mean(in_fold_maes))

        with torch.no_grad():
            w = fold_model.head.weights().cpu().numpy()

        weight_rows.append(w)
        loo_per_fold.append({
            "fold": i,
            "name": src_names[i],
            "weights": w,
            "oof_mae": oof_mae,
            "in_fold_mae": in_fold_mae,
        })

    weight_vectors = np.stack(weight_rows, axis=0)
    weight_std_per_direction = weight_vectors.std(axis=0)

    oof_maes = np.array([r["oof_mae"] for r in loo_per_fold])

    final_model = train_fusion(
        list(src_cif_stacks), list(src_cif_true),
        predictors=predictors,
        epochs=epochs, lr=lr, l2=l2, seed=seed,
    )

    return {
        "loo_per_fold": loo_per_fold,
        "weight_vectors": weight_vectors,
        "weight_std_per_direction": weight_std_per_direction,
        "oof_mae_mean": float(oof_maes.mean()),
        "oof_mae_std": float(oof_maes.std()),
        "final_model": final_model,
    }


def basis_mix_loss(
    head: BasisMixFusion,
    cif_stack: torch.Tensor,
    y_true: torch.Tensor,
    lambda_l2: float = 1e-3,
    lambda_entropy: float = 1e-2,
    lambda_diversity: float = 1e-2,
) -> torch.Tensor:
    """Compute the combined BasisMixFusion loss (Task 3.1).

    Combines MAE + L2 + entropy floor + diversity regularization into a single
    differentiable loss. This will be used by Task 3.2's LOO-CV training.

    Args:
        head: The BasisMixFusion head.
        cif_stack: ``(n, 5, HORIZON)`` tensor of per-direction CIF predictions.
        y_true: ``(n, HORIZON)`` tensor of true CIF values.
        lambda_l2: Weight for L2 regularization.
        lambda_entropy: Weight for entropy floor penalty.
        lambda_diversity: Weight for diversity penalty.

    Returns:
        Scalar tensor representing the combined loss.
    """
    # MAE loss
    pred = head(cif_stack)
    mae_loss = torch.abs(pred - y_true).mean()

    # Regularization terms
    l2_loss = head.l2_penalty()
    entropy_loss = head.entropy_floor_loss()
    diversity_loss = head.diversity_loss(cif_stack)

    # Combined loss
    total_loss = (
        mae_loss
        + lambda_l2 * l2_loss
        + lambda_entropy * entropy_loss
        + lambda_diversity * diversity_loss
    )

    return total_loss


__all__ = [
    "DIRECTION_ORDER",
    "FusionHead",
    "EqualWeightFusion",
    "MedianFusion",
    "BasisMixFusion",
    "FusionModel",
    "PredictorFn",
    "train_fusion",
    "basis_mix_loss",
]
