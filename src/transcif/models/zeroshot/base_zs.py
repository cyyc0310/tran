"""Baseline zero-shot training and single-target LORO evaluation.

Contains ``train_zero_shot`` (the flagship cross-region trainer) and
``evaluate_target`` (the full persistence / PatchTST / ZS / ZS+ / optional
direction-methods evaluation pipeline).  Direction method branches
(RAG, Phys-IRM, Causal, ICL, Hier) are imported lazily inside the
corresponding ``if`` blocks so that importing this module does not require
the optional direction submodules.
"""

import random

import numpy as np
import torch
import torch.nn as nn

from transcif.config import (
    SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE, TRAIN_FRACTION,
    EPOCHS_ZERO_SHOT, BATCH_SIZE, EPOCHS_SUPERVISED,
)
from transcif.data.loaders import load_region_data
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.models.base import AdaptivePersistDLinear
from transcif.models.patchtst import train_patchtst
from transcif.training.losses import ramp_aware_loss
from transcif.training.augment import MissingMaskAugmentor
from transcif.training.schedulers import get_cosine_warmup_scheduler
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration.zs_plus import zs_plus_predict


# When True (and CUDA is available), the optional direction methods are trained
# and inferred concurrently on separate cuda streams instead of strictly
# serially.  Set to False to force the original serial ordering (e.g. if you
# hit GPU memory pressure from multiple models being resident at once).
USE_CUDA_STREAMS = True


def train_zero_shot(all_regions, target_name, seed=42,
                    model_class=None, use_weighted=True,
                    use_ramp_loss=False, mask_augment_prob=0.0,
                    epochs=EPOCHS_ZERO_SHOT, lr=1e-3, device=None):
    """Train the zero-shot model on all source regions for one LORO target.

    Args:
        all_regions : dict  {name: {"rs":..., "cif":..., "config":...}}
        target_name : str   region to leave out
        seed        : int
        model_class : nn.Module class (default AdaptivePersistDLinear)
        use_weighted: bool  config-distance source weighting
        use_ramp_loss : bool  use ramp-weighted L1 loss instead of plain L1
        mask_augment_prob : float  probability of input masking (0=off)
        epochs, lr  : training hyperparameters
    """
    if model_class is None:
        model_class = AdaptivePersistDLinear
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    model = model_class(seq_len=SEQ_LEN, horizon=HORIZON)
    if device:
        model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = get_cosine_warmup_scheduler(optimizer, max(1, epochs // 10), epochs)
    mask_aug = MissingMaskAugmentor(prob=mask_augment_prob) if mask_augment_prob > 0 else None
    xs, ys, cfgs, weights = [], [], [], []
    for name, data in all_regions.items():
        if name == target_name:
            continue
        x_win, y_win, _ = build_windows(data["rs"], data["cif"])
        if len(x_win) == 0:
            continue
        xs.append(x_win)
        ys.append(y_win)
        cfgs.append(np.tile(data["config"], (len(x_win), 1)))
        if use_weighted:
            dist = abs(data["mean_rs"] - all_regions[target_name]["mean_rs"])
            w = 1.0 / (dist + 0.05)
        else:
            w = 1.0
        weights.append(np.full(len(x_win), w, dtype=np.float32))
    x_all = torch.tensor(np.concatenate(xs))
    y_all = torch.tensor(np.concatenate(ys))
    c_all = torch.tensor(np.concatenate(cfgs))
    w_all = torch.tensor(np.concatenate(weights))
    w_all = w_all / w_all.sum() * len(w_all)
    if device:
        x_all, y_all, c_all, w_all = (x_all.to(device), y_all.to(device),
                                      c_all.to(device), w_all.to(device))
    n_samples = len(x_all)
    batch_size = min(512, n_samples)
    model.train()
    for epoch in range(epochs):
        idx = torch.randperm(n_samples)[:batch_size]
        x_batch = x_all[idx]
        y_batch = y_all[idx]
        c_batch = c_all[idx]
        w_batch = w_all[idx]
        if mask_aug is not None:
            x_batch, _ = mask_aug(x_batch)
        pred = model(x_batch, c_batch)
        if use_ramp_loss:
            per_element = ramp_aware_loss(pred, y_batch, reduction='none')
            loss = (w_batch.unsqueeze(1) * per_element).mean()
        else:
            loss = (w_batch.unsqueeze(1) * torch.abs(pred - y_batch)).mean()
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    model.eval()
    return model


def evaluate_target(target_name, all_regions, seed=42,
                    model_class=None, use_ramp_loss=False,
                    use_rag=False, use_phys_irm=False,
                    use_causal=False, use_icl=False, use_hier=False):
    """Full evaluation on one target: persistence, PatchTST-sup, ZS, ZS+,
    [RAG], [Phys-IRM], [Causal], [ICL], [Hier].

    Optional direction methods are trained/evaluated only when their flag is
    True; their imports are resolved lazily from ``transcif.models.zeroshot``.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Use CUDA when available so every method (PatchTST, ZS, ZS+, and all
    # optional direction methods) trains/inferes on the GPU.  Each predict
    # routine follows the model's own device, so we only need to pass `device`
    # into the trainers and move the few tensors we build inline here.
    device = "cuda" if torch.cuda.is_available() else None

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    n_hours = len(rs)
    split_hour = int(n_hours * TRAIN_FRACTION)

    x_rs_test, _, y_cif_test = build_windows(
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE)
    cif_offset = cif[split_hour - SEQ_LEN:]
    x_cif_test = []
    for start in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
        x_cif_test.append(cif_offset[start:start + SEQ_LEN])
    if not x_cif_test:
        return None
    x_cif_test = np.stack(x_cif_test)

    x_cif_train, y_cif_train = [], []
    for start in range(0, split_hour - SEQ_LEN - HORIZON + 1, TRAIN_STRIDE):
        x_cif_train.append(cif[start:start + SEQ_LEN])
        y_cif_train.append(cif[start + SEQ_LEN:start + SEQ_LEN + HORIZON])
    x_cif_train = np.stack(x_cif_train)
    y_cif_train = np.stack(y_cif_train)

    results = {"target": target_name, "seed": seed, "mean_rs": data["mean_rs"],
               "ef_nr": data["ef_nr"], "n_test": len(x_rs_test)}

    # 1. Persistence
    persist_pred = x_cif_test[:, -HORIZON:]
    results["persistence"] = compute_metrics(persist_pred, y_cif_test)

    # 2. PatchTST supervised
    ptst = train_patchtst(x_cif_train, y_cif_train, epochs=EPOCHS_SUPERVISED,
                          device=device)
    with torch.no_grad():
        ptst_pred = ptst(
            torch.tensor(x_cif_test, dtype=torch.float32).to(device)).numpy()
    results["patchtst_sup"] = compute_metrics(ptst_pred, y_cif_test)

    # 3. TransCIF zero-shot
    zs_model = train_zero_shot(all_regions, target_name, seed=seed,
                               model_class=model_class,
                               use_ramp_loss=use_ramp_loss, device=device)
    target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(
        len(x_rs_test), -1).to(device)
    with torch.no_grad():
        zs_rs_pred = zs_model(
            torch.tensor(x_rs_test, dtype=torch.float32).to(device),
            target_cfg).numpy()
    zs_cif_pred = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
    results["transcif_zs"] = compute_metrics(zs_cif_pred, y_cif_test)

    # 4. TransCIF-ZS+
    origins = [split_hour + st
               for st in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)
    results["transcif_zs_plus"] = compute_metrics(zsp_pred, y_cif_test)

    # 5-9. Optional direction methods.
    #
    # Each direction method (RAG, Phys-IRM, Causal, ICL, Hier) trains an
    # independent model from its own seed and its own data, so the train+predict
    # of different directions are completely independent.  When CUDA is
    # available we launch each direction on its OWN cuda stream so their kernels
    # overlap on the GPU instead of running strictly serially; we then
    # `synchronize` and collect metrics.  Determinism is preserved: every method
    # keeps its own seed and its own model, and cuda-stream scheduling does not
    # change the result of any individual op.
    #
    # Disable concurrency (fall back to strict serial) by setting
    # ``USE_CUDA_STREAMS = False`` below or when CUDA is unavailable.
    use_streams = USE_CUDA_STREAMS and device is not None and "cuda" in str(device)

    # Each entry: (flag, label, task).  ``task`` returns the cif prediction
    # array (numpy) or raises; it is executed either inline (serial) or inside
    # a dedicated cuda stream (concurrent).
    direction_tasks = []

    if use_rag:
        def _task_rag():
            from transcif.models.zeroshot.rag import (RagMemoryBank, RagDLinear,
                                                      train_rag_zero_shot, predict_rag_zs)
            rag_model, bank = train_rag_zero_shot(
                all_regions, target_name, seed=seed, device=device)
            return predict_rag_zs(rag_model, bank, x_rs_test.astype(np.float32),
                                   data["config"].astype(np.float32), ef_r, ef_nr)
        direction_tasks.append(("rag", "RAG", _task_rag,
                                "transcif_rag", "ratio_rag_vs_zs"))

    if use_phys_irm:
        def _task_phys():
            from transcif.models.zeroshot.phys_irm import (train_phys_irm,
                                                           predict_phys_irm,
                                                           train_phys_weighted_only)
            phys_model, _ = train_phys_irm(
                all_regions, target_name, seed=seed, gamma_irm=0.1,
                lambda_cif=0.5, device=device)
            cif_phys = predict_phys_irm(phys_model, x_rs_test.astype(np.float32),
                                        data["config"].astype(np.float32), ef_r, ef_nr)
            pw_model, _ = train_phys_weighted_only(
                all_regions, target_name, seed=seed, lambda_cif=0.5, device=device)
            cif_pw = predict_phys_irm(pw_model, x_rs_test.astype(np.float32),
                                      data["config"].astype(np.float32), ef_r, ef_nr)
            return cif_phys, cif_pw
        direction_tasks.append(("phys-irm", "Phys-IRM", _task_phys,
                                "transcif_phys_irm", "ratio_phys_irm_vs_zs"))

    if use_causal:
        def _task_causal():
            from transcif.models.zeroshot.causal import (train_causal_zero_shot,
                                                         predict_causal_zs)
            causal_model, _ = train_causal_zero_shot(
                all_regions, target_name, seed=seed, device=device)
            return predict_causal_zs(
                causal_model, x_rs_test.astype(np.float32),
                data["config"].astype(np.float32), ef_r, ef_nr)
        direction_tasks.append(("causal", "Causal", _task_causal,
                                "transcif_causal", "ratio_causal_vs_zs"))

    if use_icl:
        def _task_icl():
            from transcif.models.zeroshot.icl import (ICTransformer, train_icl,
                                                      predict_icl_zs)
            icl_model = train_icl(all_regions, target_name, seed=seed, device=device)
            return predict_icl_zs(
                icl_model, all_regions, target_name,
                x_rs_test.astype(np.float32), ef_r, ef_nr)
        direction_tasks.append(("icl", "ICL", _task_icl,
                                "transcif_icl", "ratio_icl_vs_zs"))

    if use_hier:
        def _task_hier():
            from transcif.models.zeroshot.hier import train_hier, predict_hier_zs
            hier_model = train_hier(all_regions, target_name, seed=seed, device=device)
            return predict_hier_zs(
                hier_model, x_rs_test.astype(np.float32),
                data["config"].astype(np.float32), ef_r, ef_nr)
        direction_tasks.append(("hier", "Hier", _task_hier,
                                "transcif_hier", "ratio_hier_vs_zs"))

    # Launch concurrently on separate streams (or serially as a fallback).
    outcomes = {}  # flag -> ("ok", payload) | ("err", msg)
    if use_streams and direction_tasks:
        streams = {flag: torch.cuda.Stream() for flag, _, _, _, _ in direction_tasks}
        launched = {}
        for flag, label, task, _, _ in direction_tasks:
            print(f"    [{label}] training...", end="", flush=True)
            s = streams[flag]
            def _run(task=task, s=s):  # capture defaults
                try:
                    with torch.cuda.stream(s):
                        return "ok", task()
                except Exception as e:  # noqa: BLE001
                    return "err", e
            launched[flag] = _run()
        torch.cuda.synchronize()
        for flag, (status, payload) in launched.items():
            outcomes[flag] = (status, payload)
    else:
        for flag, label, task, _, _ in direction_tasks:
            print(f"    [{label}] training...", end="", flush=True)
            try:
                outcomes[flag] = ("ok", task())
                print(" done", flush=True)
            except Exception as e:  # noqa: BLE001
                outcomes[flag] = ("err", e)

    # Collect results (order-independent; ratios use the already-computed ZS MAE)
    zs_mae = results["transcif_zs"]["mae"]
    for flag, label, task, key, ratio_key in direction_tasks:
        status, payload = outcomes.get(flag, ("err", None))
        if status == "ok":
            if flag == "phys-irm":
                cif_phys, cif_pw = payload
                results["transcif_phys_irm"] = compute_metrics(cif_phys, y_cif_test)
                results["ratio_phys_irm_vs_zs"] = (
                    results["transcif_phys_irm"]["mae"] / max(zs_mae, 1e-6))
                results["transcif_phys_weighted"] = compute_metrics(cif_pw, y_cif_test)
                results["ratio_phys_weighted_vs_zs"] = (
                    results["transcif_phys_weighted"]["mae"] / max(zs_mae, 1e-6))
                results["irm_benefit"] = (
                    results["transcif_phys_irm"]["mae"]
                    / max(results["transcif_phys_weighted"]["mae"], 1e-6))
            else:
                results[key] = compute_metrics(payload, y_cif_test)
                results[ratio_key] = results[key]["mae"] / max(zs_mae, 1e-6)
            # In stream mode "training..." was printed at launch; finish the line.
            if use_streams and direction_tasks:
                print(" done", flush=True)
        else:
            e = payload
            results[key] = None
            results[ratio_key] = None
            print(f"  [WARN] {label} failed for {target_name}: {e}")

    # Ratios
    results["ratio_vs_patchtst"] = results["transcif_zs"]["mae"] / results["patchtst_sup"]["mae"]
    results["ratio_vs_persist"] = results["transcif_zs"]["mae"] / results["persistence"]["mae"]
    results["ratio_plus_vs_patchtst"] = (
        results["transcif_zs_plus"]["mae"] / results["patchtst_sup"]["mae"])
    results["ratio_plus_vs_persist"] = (
        results["transcif_zs_plus"]["mae"] / results["persistence"]["mae"])

    return results
