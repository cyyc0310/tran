"""Torch-native zero-shot direction wrappers + learned fusion (Phase 9).

The five direction models (rag/phys_irm/causal/icl/hier) are already
``nn.Module`` instances with differentiable ``forward`` — the only thing that
breaks the gradient graph is the terminal ``cif_from_shares`` call (which forces
``.detach().cpu().numpy()``) plus an outer ``torch.no_grad()``. This module
provides thin wrappers that reuse each model's differentiable internals and
inline the physics conversion (``share*ef_r + (1-share)*ef_nr``) as torch ops,
so CIF predictions carry gradient back to the model parameters.

Three directions (phys_irm / causal / hier) are "easy" — their forward
outputs a renewable share and the wrappers are trivial. RAG and ICL are
"hard" (numpy retrieval bank / no config input / per-window loop), so they
are wrapped as :class:`FrozenConstant` — their predictions are computed once,
detached, and act as constants that still participate in fusion but do not
receive gradient.

The :class:`LearnedFusion` head replaces the global softmax fusion with a
**per-window** weight generator conditioned on the 5 predictions + config, so
the (differentiable) directions co-adapt through the fusion gradient. This is
"Plan A" from the design discussion (parallel + learned fusion), explicitly
*not* serial chaining.

This module is additive: the existing numpy ``predict_*_zs`` functions and the
frozen-prediction joint training pipeline are untouched.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

__all__ = [
    "TorchNativePredictor",
    "NativePhys",
    "NativeCausal",
    "NativeHier",
    "NativeRAG",
    "NativeICL",
    "FrozenConstant",
    "LearnedFusion",
    "pad_config_t",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pad_config_t(config: torch.Tensor, config_dim: int) -> torch.Tensor:
    """Right-pad a config tensor to ``config_dim`` (tensor version of
    ``transcif.physics.bounds.pad_config``).

    Accepts ``(D,)`` or ``(B, D)``. Missing entries are zero-filled, matching
    the numpy pad used by the direction trainers.
    """
    if config.dim() == 1:
        cur = config.shape[0]
        if cur >= config_dim:
            return config
        pad = torch.zeros(config_dim - cur, dtype=config.dtype, device=config.device)
        return torch.cat([config, pad])
    cur = config.shape[1]
    if cur >= config_dim:
        return config
    pad = torch.zeros(config.shape[0], config_dim - cur, dtype=config.dtype,
                      device=config.device)
    return torch.cat([config, pad], dim=1)


# ---------------------------------------------------------------------------
# Base contract
# ---------------------------------------------------------------------------

class TorchNativePredictor(nn.Module, ABC):
    """A zero-shot direction whose CIF predictions carry gradient.

    Subclasses implement :meth:`forward_cif` returning a ``(B, HORIZON)``
    tensor of CIF predictions. The output MUST keep ``requires_grad`` True if
    the underlying model has unfrozen parameters (so the joint trainer can
    backprop into the direction).
    """

    @abstractmethod
    def forward_cif(
        self,
        x: torch.Tensor,           # (B, SEQ_LEN) renewable-share windows
        config: torch.Tensor,      # (B, D) target config
        ef_r: float,
        ef_nr: float,
    ) -> torch.Tensor:             # (B, HORIZON) CIF predictions
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Live (differentiable) wrappers — the three "easy" directions
# ---------------------------------------------------------------------------

class NativePhys(TorchNativePredictor):
    """Wraps an ``AdaptivePersistDLinear`` (used by phys_irm).

    ``model.forward(x, config) -> share`` in [0, 1]; CIF = share·ef_r + (1-share)·ef_nr.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward_cif(self, x, config, ef_r, ef_nr):
        share = self.model(x, config)
        return share * ef_r + (1.0 - share) * ef_nr


class NativeHier(TorchNativePredictor):
    """Wraps a ``HierDLinear``.

    ``model.forward(x, config) -> (hourly, daily, weekly)``; ``hourly`` is the
    renewable-share forecast; CIF = hourly·ef_r + (1-hourly)·ef_nr.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward_cif(self, x, config, ef_r, ef_nr):
        hourly, _daily, _weekly = self.model(x, config)
        return hourly * ef_r + (1.0 - hourly) * ef_nr


class NativeCausal(TorchNativePredictor):
    """Wraps a ``CausalDomainVAE``.

    Mirrors ``predict_causal_zs``: ``encode`` → ``predict_share``, skipping the
    decoder (inference path only). CIF = share·ef_r + (1-share)·ef_nr, matching
    the inline torch physics already used at ``causal.py:334``.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward_cif(self, x, config, ef_r, ef_nr):
        z_inv, _z_spec, *_ = self.model.encode(x, config)
        share = self.model.predict_share(z_inv, config, x)
        return share * ef_r + (1.0 - share) * ef_nr


class NativeRAG(TorchNativePredictor):
    """Wraps a ``RagDLinear`` with a torch-native retrieval over a memory bank.

    The memory bank (built by ``train_rag_zero_shot`` from source windows) is
    stored as two buffers: ``X`` (N, SEQ_LEN) contexts and ``Y`` (N, HORIZON)
    renewable-share targets. Retrieval is a differentiable matmul-based kNN:
    distances ``||x||² + ||X||² − 2 x·Xᵀ`` → top-k → softmax(−d/temp) weights →
    weighted average of the neighbours' share targets. The discrete neighbour
    *selection* is non-differentiable (it is a discrete choice), but the
    weighting and the downstream ``RagDLinear.forward`` carry gradient to the
    model parameters (``rag_proj``, ``rag_gate``, DLinear heads).

    The model's ``forward`` outputs a renewable share; CIF = share·ef_r +
    (1−share)·ef_nr.
    """

    def __init__(self, model: nn.Module, bank_X: torch.Tensor,
                 bank_Y: torch.Tensor, k: int = 5, temp: float = 1.0):
        super().__init__()
        self.model = model
        self.register_buffer("bank_X", bank_X.as_tensor() if hasattr(bank_X, "as_tensor") else torch.as_tensor(bank_X, dtype=torch.float32))
        self.register_buffer("bank_Y", bank_Y.as_tensor() if hasattr(bank_Y, "as_tensor") else torch.as_tensor(bank_Y, dtype=torch.float32))
        self.k = int(min(k, self.bank_X.shape[0]))
        self.temp = float(temp)

    def _retrieve(self, x: torch.Tensor):
        """Differentiable kNN retrieval. Returns (rag_target (B,H), rag_dist (B,1))."""
        k = self.k
        # squared L2 distances: (B, N)
        x2 = (x * x).sum(dim=1, keepdim=True)            # (B, 1)
        X2 = (self.bank_X * self.bank_X).sum(dim=1)      # (N,)
        d = x2 + X2.unsqueeze(0) - 2.0 * (x @ self.bank_X.t())
        topk_d, topk_idx = torch.topk(d, k, dim=1, largest=False)  # (B, k)
        w = torch.softmax(-topk_d / self.temp, dim=1)    # (B, k)
        Yk = self.bank_Y[topk_idx]                       # (B, k, H) — gather
        rag_target = (w.unsqueeze(-1) * Yk).sum(dim=1)   # (B, H)
        rag_dist = topk_d.mean(dim=1, keepdim=True)      # (B, 1)
        return rag_target, rag_dist

    def forward_cif(self, x, config, ef_r, ef_nr):
        rag_target, rag_dist = self._retrieve(x)
        share = self.model(x, config, rag_target, rag_dist)
        return share * ef_r + (1.0 - share) * ef_nr


class NativeICL(TorchNativePredictor):
    """Wraps an ``ICTransformer`` with torch-native context assembly.

    ICL's non-differentiable parts are the per-query example *retrieval*
    (``select_examples``: config-distance + L2 ranking over source windows) and
    the context *assembly* (``build_context``). These are treated as
    no-gradient preprocessing — like RAG's neighbour selection, retrieval is a
    discrete choice that does not need to carry gradient. The transformer
    ``forward`` over the assembled context IS differentiable, so its parameters
    (input_proj, transformer, pred_head) receive gradient end-to-end.

    Source windows are precomputed once at construction (the per-query
    ``build_windows`` would otherwise dominate the training loop) and the L2
    ranking runs against the cached windows each call.

    The model has no config input — config only affects retrieval — so
    ``forward_cif`` ignores ``config`` for the forward pass (it was already used
    to build ``all_regions`` configs at construction time).
    """

    def __init__(self, model: nn.Module, all_regions: dict, target_name: str,
                 n_examples: int = 3, horizon: int = 24):
        super().__init__()
        self.model = model
        self.target_name = target_name
        self.n_examples = n_examples
        self.horizon = horizon
        self.target_cfg = np.asarray(all_regions[target_name]["config"],
                                     dtype=np.float32)
        # Precompute source windows once (the expensive part of select_examples).
        from transcif.data.windows import build_windows
        self._ctx_cache = {}  # query-content hash -> (values_np, roles_np)
        self._sources = []  # list of (config, x_win (N,SEQ_LEN), y_win (N,H))
        for name, data in all_regions.items():
            if name == target_name:
                continue
            try:
                xw, _, yw = build_windows(data["rs"], data["cif"])
            except Exception:
                continue
            if len(xw) == 0:
                continue
            self._sources.append((
                np.asarray(data["config"], dtype=np.float32),
                xw.astype(np.float32), yw.astype(np.float32),
            ))

    def _select_examples(self, target_full: np.ndarray):
        """Cached-source version of icl.select_examples. Returns (wins, outs)."""
        scores = []
        for cfg, xw, yw in self._sources:
            n_cmp = min(len(cfg), len(self.target_cfg))
            config_dist = float(np.linalg.norm(cfg[:n_cmp] - self.target_cfg[:n_cmp]))
            L = min(xw.shape[1], target_full.shape[0])
            sims = -np.linalg.norm(xw[:, :L] - target_full[:L][None, :], axis=1)
            best_idx = int(np.argmax(sims))
            score = float(sims[best_idx]) - 0.1 * config_dist
            scores.append((score, best_idx, xw, yw))
        scores.sort(key=lambda t: -t[0])
        wins, outs = [], []
        for i in range(min(self.n_examples, len(scores))):
            _, idx, xw, yw = scores[i]
            wins.append(xw[idx, -self.horizon:])
            outs.append(yw[idx])
        return wins, outs

    def forward_cif(self, x, config, ef_r, ef_nr):
        from transcif.models.zeroshot.icl import build_context
        # Retrieval + assembly: no-grad preprocessing (discrete neighbour choice).
        # The retrieved context depends ONLY on the (fixed) query window, not on
        # model parameters, so across the ~60 training steps that reuse the same
        # 12 origins it is identical — cache it by query content to avoid 720x
        # redundant L2 retrievals per pair.
        with torch.no_grad():
            target_full = x[0].detach().cpu().numpy().astype(np.float32)
            key = hash(target_full.tobytes())
            cached = self._ctx_cache.get(key)
            if cached is None:
                wins, outs = self._select_examples(target_full)
                if not wins:  # no sources — fall back to last-horizon persistence share
                    share = x[:, -self.horizon:]
                    return share * ef_r + (1.0 - share) * ef_nr
                target_h = target_full[-self.horizon:]
                cached = build_context(target_h, wins, outs, horizon=self.horizon)
                self._ctx_cache[key] = cached
            values_np, roles_np = cached
            values = torch.as_tensor(values_np, dtype=torch.float32, device=x.device)
            roles = torch.as_tensor(roles_np, dtype=torch.long, device=x.device)
        # Transformer forward: differentiable through model params.
        share = self.model(values, roles)
        return share * ef_r + (1.0 - share) * ef_nr


# ---------------------------------------------------------------------------
# Frozen constant — for RAG / ICL (gradient stops here)
# ---------------------------------------------------------------------------

class FrozenConstant(TorchNativePredictor):
    """Wraps a numpy ``predict_fn`` into a graph-detached constant predictor.

    Used for RAG (numpy KNN retrieval bank) and ICL (numpy example selection +
    per-window loop) whose internals are not yet torch-native. The prediction
    is computed under ``no_grad`` and returned as a tensor with
    ``requires_grad=False``, so gradient stops here but the value still
    participates in downstream fusion.

    ``predict_fn`` must follow the standard contract:
    ``predict_fn(x_rs (B,SEQ_LEN) ndarray, config (D,) ndarray, ef_r, ef_nr) ->
    (B, HORIZON) ndarray``. Callers close over any bank/regions the direction
    needs.
    """

    def __init__(self, predict_fn: Callable):
        super().__init__()
        self.predict_fn = predict_fn

    def forward_cif(self, x, config, ef_r, ef_nr):
        with torch.no_grad():
            x_np = x.detach().cpu().numpy().astype(np.float32)
            # config may be (B, D); take the first row for the numpy contract
            # (the standard predict contract takes a single (D,) config).
            cfg_np = config.detach().cpu().numpy()
            if cfg_np.ndim == 2:
                cfg_np = cfg_np[0]
            cif_np = np.asarray(
                self.predict_fn(x_np, cfg_np.astype(np.float32), ef_r, ef_nr),
                dtype=np.float32,
            )
        # Detached constant: no gradient to any upstream learnable parameter.
        return torch.as_tensor(cif_np, dtype=torch.float32, device=x.device)


# ---------------------------------------------------------------------------
# Learned per-window fusion (Plan A)
# ---------------------------------------------------------------------------

class LearnedFusion(nn.Module):
    """Per-window learned fusion over the 5 direction predictions.

    Unlike :class:`~transcif.models.zeroshot.fusion.BasisMixFusion` (a single
    global softmax over directions), this head generates **per-window** weights
    conditioned on the predictions themselves + the target config, so the
    fusion can favour different directions for different windows and the
    differentiable directions co-adapt through this gradient path.

    Forward:
        stack  : ``(B, n_directions, HORIZON)`` per-direction CIF predictions
        config : ``(B, config_dim)`` target config
        -> (B, HORIZON) fused CIF  [+ optional (B, n_directions) weights]

    The weight features are per-direction mean and std of the HORIZON
    predictions (rotation-invariant summaries) concatenated with the config.
    """

    def __init__(
        self,
        n_directions: int = 5,
        config_dim: int = 2,
        horizon: int = 24,
        hidden: int = 32,
    ):
        super().__init__()
        self.n_directions = n_directions
        self.horizon = horizon
        feat_dim = 2 * n_directions + config_dim
        self.weight_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_directions),
        )

    def forward(
        self,
        stack: torch.Tensor,
        config: torch.Tensor,
        return_weights: bool = False,
    ):
        if stack.shape[1] != self.n_directions:
            raise ValueError(
                f"stack axis 1 = {stack.shape[1]}, expected n_directions="
                f"{self.n_directions}"
            )
        means = stack.mean(dim=2)            # (B, n_dir)
        stds = stack.std(dim=2)              # (B, n_dir)
        feat = torch.cat([means, stds, config], dim=1)   # (B, 2*n_dir + cfg)
        logits = self.weight_net(feat)      # (B, n_dir)
        weights = torch.softmax(logits, dim=1)
        out = (stack * weights.unsqueeze(-1)).sum(dim=1)   # (B, HORIZON)
        if return_weights:
            return out, weights
        return out
