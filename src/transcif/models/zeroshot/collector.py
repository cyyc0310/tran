"""Source-region stack collector for fusion training (Task 1.3).

This module provides the core data collection logic for training the fusion head:
for each source region, run the 5 zero-shot direction methods and collect their
CIF predictions into stacks of shape ``(n_i, 5, HORIZON)`` where ``n_i`` is the
number of TEST windows in that source region.

Zero-shot validity (CRITICAL):
    For each source region, ONLY windows from that source region's TEST split
    are used. The split point is ``int(n_hours * TRAIN_FRACTION)`` (TRAIN_FRACTION
    is in ``transcif.config``). Windows are built from ``rs[split - SEQ_LEN:]``
    and ``cif[split - SEQ_LEN:]`` with stride ``TEST_STRIDE`` and horizon
    ``HORIZON``. This ensures no train/test leakage at the source-region level.

    Source regions are NOT the target region, so using their labels does NOT
    violate the zero-shot constraint on the target. The fusion head learns to
    combine 5 direction predictors using source-region errors, then we apply
    it zero-shot to the held-out target.

The collector is the bridge between:
    - Per-direction predictors (:mod:`transcif.models.zeroshot.{rag,phys_irm,causal,icl,hier}`)
    - Fusion training (:func:`transcif.models.zeroshot.fusion.train_fusion`)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Mapping, Sequence

import numpy as np

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.windows import build_windows

if TYPE_CHECKING:
    from transcif.models.zeroshot.fusion import PredictorFn


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def collect_source_stacks(
    all_regions: Mapping[str, np.ndarray],
    target_name: str,
    seed: int = 0,
    device: str | None = None,
    source_names: list[str] | None = None,
    progress: bool = False,
) -> tuple[list[np.ndarray], list[np.ndarray], list[str]]:
    """Run 5 directions on each source region's TEST window.

    For each source region (all regions except ``target_name``), this function:
        1. Loads the region's rs/cif data
        2. Determines the train/test split point: ``split = int(n_hours * TRAIN_FRACTION)``
        3. Builds TEST windows from ``rs[split - SEQ_LEN:]`` and ``cif[split - SEQ_LEN:]``
        4. Trains each of the 5 direction predictors on the source region
        5. Runs zero-shot prediction on the source's TEST windows
        6. Stacks the 5 CIF predictions into ``(n_i, 5, HORIZON)``

    Zero-shot guarantee:
        The TEST window index for each source region starts at ``split`` (the first
        hour after the training portion). Source regions are distinct from the target,
        so using their labels does not violate the zero-shot constraint. The source
        TEST window index does NOT overlap with the target's TEST window index
        (different region, but this is called out explicitly for clarity).

    Args:
        all_regions: Mapping ``region_name -> {rs, cif, config, ef_r, ef_nr, ...}``.
        target_name: Name of the target region (excluded from sources).
        seed: Random seed for direction predictor training.
        device: ``"cuda"`` or ``None`` (CPU).
        source_names: Explicit list of source region names. If ``None``, all regions
                      except ``target_name`` are used as sources.
        progress: If ``True``, print progress messages (useful for debugging).

    Returns:
        cif_stacks: List of ``(n_i, 5, HORIZON)`` arrays, one per source region.
                    Each array contains the 5 direction predictors' CIF outputs
                    stacked along axis 1, in the order defined by
                    ``DIRECTION_ORDER`` (``rag, phys, causal, icl, hier``).
        cif_true:   List of ``(n_i, HORIZON)`` arrays, one per source region.
                    The ground-truth CIF values for the source's TEST windows.
        names:       List of source region names, in the same order as ``cif_stacks``
                    and ``cif_true``. ``target_name`` is never included.

    Raises:
        ValueError: If a source region has insufficient data for window building.
        ImportError: If a direction predictor module cannot be imported.

    Example:
        >>> cif_stacks, cif_true, names = collect_source_stacks(
        ...     all_regions, target_name="AU1", seed=0, device=None
        ... )
        >>> len(names)  # Number of source regions used
        28
        >>> cif_stacks[0].shape  # First source's stack: (n_windows, 5, HORIZON)
        (42, 5, 24)
    """
    # Determine which regions to use as sources
    if source_names is None:
        source_names = [name for name in all_regions if name != target_name]
    else:
        # Validate that target is not in the explicit source list
        if target_name in source_names:
            raise ValueError(
                f"target_name '{target_name}' cannot be in source_names. "
                "Source regions must be distinct from the target region."
            )
        # Validate that all source names exist in all_regions
        missing = set(source_names) - set(all_regions.keys())
        if missing:
            raise ValueError(
                f"source_names contains regions not in all_regions: {sorted(missing)}"
            )

    if not source_names:
        return [], [], []

    # Import DIRECTION_ORDER from fusion to ensure consistent ordering
    from transcif.models.zeroshot.fusion import DIRECTION_ORDER

    # Collect stacks and true CIF arrays for each source region
    cif_stacks: list[np.ndarray] = []
    cif_true: list[np.ndarray] = []
    valid_names: list[str] = []

    for i, source_name in enumerate(source_names):
        if progress:
            print(
                f"  [collector {i+1}/{len(source_names)}] Processing {source_name}...",
                flush=True
            )

        # Load source region data
        if source_name not in all_regions:
            if progress:
                print(f"    [SKIP] {source_name} not in all_regions")
            continue

        source_data = all_regions[source_name]
        rs = source_data["rs"]
        cif = source_data["cif"]
        config = source_data["config"]
        ef_r = source_data["ef_r"]
        ef_nr = source_data["ef_nr"]

        n_hours = len(rs)
        split = int(n_hours * TRAIN_FRACTION)

        # Build TEST windows from the source region
        # CRITICAL: Windows are built from rs[split - SEQ_LEN:] and cif[split - SEQ_LEN:]
        # The first window's prediction target starts at 'split', ensuring we only
        # use TEST data (no leakage from the training portion)
        try:
            x_rs_test, _, y_cif_test = build_windows(
                rs[split - SEQ_LEN :],
                cif[split - SEQ_LEN :],
                seq_len=SEQ_LEN,
                horizon=HORIZON,
                stride=TEST_STRIDE,
            )
        except Exception as e:
            if progress:
                print(f"    [SKIP] {source_name}: window building failed: {e}")
            continue

        if len(x_rs_test) == 0:
            if progress:
                print(f"    [SKIP] {source_name}: no TEST windows")
            continue

        # Run 5 direction predictors on the source region
        # Import each direction module here so tests can monkeypatch at import time
        direction_predictions = {}

        # RAG
        try:
            from transcif.models.zeroshot.rag import predict_rag_zs, train_rag_zero_shot
            rag_model, rag_bank = train_rag_zero_shot(
                all_regions, source_name, seed=seed, device=device
            )
            rag_pred = predict_rag_zs(
                rag_model, rag_bank, x_rs_test.astype(np.float32),
                config.astype(np.float32), ef_r, ef_nr
            )
            direction_predictions["rag"] = rag_pred
        except Exception as e:
            if progress:
                print(f"    [WARN] {source_name} RAG failed: {e}")
            continue

        # Phys-IRM
        try:
            from transcif.models.zeroshot.phys_irm import predict_phys_irm, train_phys_irm
            phys_model, _ = train_phys_irm(
                all_regions, source_name, seed=seed,
                gamma_irm=0.1, lambda_cif=0.5, device=device
            )
            phys_pred = predict_phys_irm(
                phys_model, x_rs_test.astype(np.float32),
                config.astype(np.float32), ef_r, ef_nr
            )
            direction_predictions["phys"] = phys_pred
        except Exception as e:
            if progress:
                print(f"    [WARN] {source_name} Phys-IRM failed: {e}")
            continue

        # Causal
        try:
            from transcif.models.zeroshot.causal import predict_causal_zs, train_causal_zero_shot
            causal_model, _ = train_causal_zero_shot(
                all_regions, source_name, seed=seed, device=device
            )
            causal_pred = predict_causal_zs(
                causal_model, x_rs_test.astype(np.float32),
                config.astype(np.float32), ef_r, ef_nr
            )
            direction_predictions["causal"] = causal_pred
        except Exception as e:
            if progress:
                print(f"    [WARN] {source_name} Causal failed: {e}")
            continue

        # ICL
        try:
            from transcif.models.zeroshot.icl import predict_icl_zs, train_icl
            icl_model = train_icl(
                all_regions, source_name, seed=seed, device=device
            )
            icl_pred = predict_icl_zs(
                icl_model, all_regions, source_name,
                x_rs_test.astype(np.float32), ef_r, ef_nr
            )
            direction_predictions["icl"] = icl_pred
        except Exception as e:
            if progress:
                print(f"    [WARN] {source_name} ICL failed: {e}")
            continue

        # Hier
        try:
            from transcif.models.zeroshot.hier import predict_hier_zs, train_hier
            hier_model = train_hier(
                all_regions, source_name, seed=seed, device=device
            )
            hier_pred = predict_hier_zs(
                hier_model, x_rs_test.astype(np.float32),
                config.astype(np.float32), ef_r, ef_nr
            )
            direction_predictions["hier"] = hier_pred
        except Exception as e:
            if progress:
                print(f"    [WARN] {source_name} Hier failed: {e}")
            continue

        # Verify all 5 directions produced predictions
        if len(direction_predictions) != len(DIRECTION_ORDER):
            if progress:
                print(
                    f"    [SKIP] {source_name}: only {len(direction_predictions)}/{len(DIRECTION_ORDER)} "
                    f"directions succeeded"
                )
            continue

        # Stack predictions into (n_i, 5, HORIZON) array
        # Order: DIRECTION_ORDER = ("rag", "phys", "causal", "icl", "hier")
        n_i = len(x_rs_test)
        stack = np.empty((n_i, len(DIRECTION_ORDER), HORIZON), dtype=np.float32)

        for d, direction in enumerate(DIRECTION_ORDER):
            pred = direction_predictions[direction]
            if pred.shape != (n_i, HORIZON):
                raise ValueError(
                    f"Direction '{direction}' returned shape {pred.shape}, "
                    f"expected ({n_i}, {HORIZON})"
                )
            stack[:, d, :] = pred

        cif_stacks.append(stack)
        cif_true.append(y_cif_test)
        valid_names.append(source_name)

        if progress:
            print(f"    [OK] {source_name}: {n_i} TEST windows, stack shape {stack.shape}")

    return cif_stacks, cif_true, valid_names


__all__ = ["collect_source_stacks"]
