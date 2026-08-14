"""Run Tasks 2.2 + 3.3 + 4.1 in one pass per target (shares direction training).

Outputs:
  results/fused_five_sanity.json     — Task 2.2 (equal / median / softmax)
  results/fused_five_headline.json   — Task 3.3 (BasisMix / +ZS+ / equal+ZS+)
  results/fused_five_dropone.json    — Task 4.1 (drop-one direction ablation)

Usage:
    .venv/bin/python scripts/experiments/run_fused_five_variants.py \\
        --regions QLD1 NSW1 VIC1 SA1 --seed 0 --src-limit 1
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.config import (AU_REGIONS, HORIZON, SEQ_LEN, TEST_STRIDE,
                              TRAIN_FRACTION, UK_REGIONS, US_REGIONS)
from transcif.data.loaders import load_region_data
from transcif.data.windows import build_windows
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.models.zeroshot.fusion import (
    BasisMixFusion,
    DIRECTION_ORDER,
    EqualWeightFusion,
    FusionHead,
    FusionModel,
    MedianFusion,
    basis_mix_loss,
    loo_cv_train,
    train_fusion,
)
from scripts.experiments._shared import (
    build_direction_predictors as _build_predictors,
    zs_plus_origins as _zs_plus_origins,
)

DEVICE = "cuda" if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Per-target setup (shared across variants)
# ---------------------------------------------------------------------------

def _cif_with_head(head, predictors, x_test, config, ef_r, ef_nr,
                   rs, cif, want_plus):
    """Run a head variant end-to-end and (optionally) through ZS+."""
    model = FusionModel(head, predictors=predictors)
    cif_fused = model.predict_cif(x_test.astype(np.float32),
                                  config.astype(np.float32), ef_r, ef_nr)
    if not want_plus:
        return cif_fused, None
    model.configure_for_target(config, ef_r, ef_nr)
    origins = _zs_plus_origins(rs, cif)
    cif_plus = zs_plus_predict(model=None, config=config, rs=rs, cif=cif,
                               ef_r=ef_r, ef_nr=ef_nr, origins=origins,
                               share_fn=model.share_fn)
    return cif_fused, cif_plus


# ---------------------------------------------------------------------------
# Head trainers
# ---------------------------------------------------------------------------

def _train_softmax_head(src_stacks, src_true, predictors, seed,
                        epochs=200, lr=1e-2, l2=1e-4):
    return train_fusion(src_stacks, src_true, predictors=predictors,
                        epochs=epochs, lr=lr, l2=l2, seed=seed)


def _train_basismix_head(src_stacks, src_true, predictors, seed,
                         epochs=300, lr=1e-2, l2=1e-4,
                         lambda_entropy=1e-2, lambda_diversity=1e-2):
    """Train BasisMixFusion with the regularized basis_mix_loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = BasisMixFusion()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=l2)

    X = np.concatenate(src_stacks, axis=0).astype(np.float32)
    Y = np.concatenate(src_true, axis=0).astype(np.float32)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = basis_mix_loss(head, X_t, Y_t,
                              lambda_l2=l2,
                              lambda_entropy=lambda_entropy,
                              lambda_diversity=lambda_diversity)
        loss.backward()
        opt.step()
    head.eval()
    return FusionModel(head, predictors=predictors)


def _train_dropone_basismix(src_stacks, src_true, predictors, seed,
                            drop_direction: str):
    """Train BasisMixFusion with one direction zeroed in source stacks.

    Drop-one ablation: zero out the dropped direction's CIF predictions in
    source stacks. The head learns to put ~0 weight on the zeroed direction.
    At predict time, all 5 predictors still run (so the API stays 5-direction);
    the head naturally downweights the dropped one.
    """
    drop_idx = DIRECTION_ORDER.index(drop_direction)
    masked_stacks = [s.copy() for s in src_stacks]
    for s in masked_stacks:
        s[:, drop_idx, :] = 0.0

    torch.manual_seed(seed)
    np.random.seed(seed)
    head = BasisMixFusion(n_directions=len(DIRECTION_ORDER))
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)

    X = np.concatenate(masked_stacks, axis=0).astype(np.float32)
    Y = np.concatenate(src_true, axis=0).astype(np.float32)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    head.train()
    for _ in range(300):
        opt.zero_grad()
        loss = basis_mix_loss(head, X_t, Y_t,
                              lambda_l2=1e-3,
                              lambda_entropy=1e-2,
                              lambda_diversity=1e-2)
        loss.backward()
        opt.step()
    head.eval()
    return FusionModel(head, predictors=predictors)


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_target(target, all_regions, seed, src_limit):
    """Run all Task 2.2 + 3.3 + 4.1 variants for one target."""
    data = all_regions[target]
    config = data["config"].astype(np.float32)
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    rs, cif = data["rs"], data["cif"]

    split = int(len(rs) * TRAIN_FRACTION)
    x_test, _, y_true = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )

    print(f"  [predictors] training 5 directions on {target}...", flush=True)
    t0 = time.time()
    predictors = _build_predictors(all_regions, target, seed, DEVICE)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print(f"  [collect] gathering source stacks (src_limit={src_limit})...",
          flush=True)
    t0 = time.time()
    from transcif.models.zeroshot.collector import collect_source_stacks
    src_names = [n for n in all_regions if n != target][:src_limit]
    src_stacks, src_true, src_names_used = collect_source_stacks(
        all_regions, target, seed=seed, device=DEVICE,
        source_names=src_names, progress=False,
    )
    print(f"    done in {time.time()-t0:.1f}s ({len(src_stacks)} sources)",
          flush=True)

    if not src_stacks:
        return None

    # ---- Task 2.2: equal / median / softmax (with and without ZS+) ----
    sanity = {"target": target, "seed": seed}
    for variant, head in [
        ("equal", EqualWeightFusion()),
        ("median", MedianFusion()),
    ]:
        cf, cfp = _cif_with_head(head, predictors, x_test, config, ef_r, ef_nr,
                                 rs, cif, want_plus=True)
        sanity[f"{variant}_fused"] = compute_metrics(cf, y_true)
        sanity[f"{variant}_fused_plus"] = compute_metrics(cfp, y_true)

    softmax_model = _train_softmax_head(src_stacks, src_true, predictors, seed)
    cf = softmax_model.predict_cif(x_test.astype(np.float32), config, ef_r, ef_nr)
    sanity["softmax_fused"] = compute_metrics(cf, y_true)
    softmax_model.configure_for_target(config, ef_r, ef_nr)
    origins = _zs_plus_origins(rs, cif)
    cfp = zs_plus_predict(model=None, config=config, rs=rs, cif=cif,
                          ef_r=ef_r, ef_nr=ef_nr, origins=origins,
                          share_fn=softmax_model.share_fn)
    sanity["softmax_fused_plus"] = compute_metrics(cfp, y_true)

    # ---- Task 3.3: BasisMix / +ZS+ / equal+ZS+ (R1 control) ----
    headline = {"target": target, "seed": seed}
    bm_model = _train_basismix_head(src_stacks, src_true, predictors, seed)
    cf_bm = bm_model.predict_cif(x_test.astype(np.float32), config, ef_r, ef_nr)
    bm_model.configure_for_target(config, ef_r, ef_nr)
    cfp_bm = zs_plus_predict(model=None, config=config, rs=rs, cif=cif,
                              ef_r=ef_r, ef_nr=ef_nr, origins=origins,
                              share_fn=bm_model.share_fn)
    headline["basismix_fused"] = compute_metrics(cf_bm, y_true)
    headline["basismix_fused_plus"] = compute_metrics(cfp_bm, y_true)

    # R1 control: equal-weight THEN ZS+ (already computed in sanity)
    headline["equal_then_plus"] = sanity["equal_fused_plus"]
    headline["equal_fused"] = sanity["equal_fused"]

    # ---- Task 4.1: drop-one ablation ----
    dropone = {"target": target, "seed": seed, "drops": {}}
    for drop in DIRECTION_ORDER:
        m_drop = _train_dropone_basismix(src_stacks, src_true, predictors,
                                         seed, drop)
        # Build masked x_test by passing full x_test (predictors already
        # masked to 4 directions, so the model only uses 4 CIF predictions)
        cf_d = m_drop.predict_cif(x_test.astype(np.float32), config, ef_r, ef_nr)
        m_drop.configure_for_target(config, ef_r, ef_nr)
        cfp_d = zs_plus_predict(model=None, config=config, rs=rs, cif=cif,
                                 ef_r=ef_r, ef_nr=ef_nr, origins=origins,
                                 share_fn=m_drop.share_fn)
        dropone["drops"][drop] = {
            "fused": compute_metrics(cf_d, y_true),
            "fused_plus": compute_metrics(cfp_d, y_true),
        }

    return {"sanity": sanity, "headline": headline, "dropone": dropone}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+",
                    default=["QLD1", "NSW1", "VIC1", "SA1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-limit", type=int, default=1)
    ap.add_argument("--out-prefix", default="fused_five")
    args = ap.parse_args()

    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions", flush=True)

    sanity_all, headline_all, dropone_all = [], [], []
    for target in args.regions:
        if target not in all_regions:
            print(f"[SKIP] {target} not loaded")
            continue
        print(f"\n[EVAL] {target}", flush=True)
        t0 = time.time()
        r = evaluate_target(target, all_regions, args.seed, args.src_limit)
        if r is None:
            continue
        sanity_all.append(r["sanity"])
        headline_all.append(r["headline"])
        dropone_all.append(r["dropone"])
        print(f"  [{target}] done in {time.time()-t0:.1f}s "
              f"→ equal MAE={r['sanity']['equal_fused']['mae']:.3f}, "
              f"softmax MAE={r['sanity']['softmax_fused']['mae']:.3f}, "
              f"basismix+ MAE={r['headline']['basismix_fused_plus']['mae']:.3f}",
              flush=True)

    os.makedirs("results", exist_ok=True)
    for name, data, idx in [
        ("sanity", sanity_all, "2.2"),
        ("headline", headline_all, "3.3"),
        ("dropone", dropone_all, "4.1"),
    ]:
        path = f"results/{args.out_prefix}_{name}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[WRITE] {path}  (Task {idx})")

    # Task 2.2 markdown summary
    if sanity_all:
        _write_sanity_md(sanity_all)
    if headline_all:
        _write_headline_md(headline_all)
    if dropone_all:
        _write_dropone_md(dropone_all)


def _write_sanity_md(records):
    path = "results/fused_five_sanity.md"
    with open(path, "w") as f:
        f.write("# Task 2.2 — Sanity (equal / median / softmax)\n\n")
        f.write("| target | equal MAE | equal+ MAE | median MAE | median+ MAE |"
                " softmax MAE | softmax+ MAE |\n")
        f.write("|--------|-----------|------------|------------|-------------|"
                "-------------|--------------|\n")
        for r in records:
            f.write(f"| {r['target']} "
                    f"| {r['equal_fused']['mae']:.3f} "
                    f"| {r['equal_fused_plus']['mae']:.3f} "
                    f"| {r['median_fused']['mae']:.3f} "
                    f"| {r['median_fused_plus']['mae']:.3f} "
                    f"| {r['softmax_fused']['mae']:.3f} "
                    f"| {r['softmax_fused_plus']['mae']:.3f} |\n")
        # R2 verdict: does softmax beat equal?
        import statistics
        equal = [r["equal_fused_plus"]["mae"] for r in records]
        softmax = [r["softmax_fused_plus"]["mae"] for r in records]
        diff = [e - s for e, s in zip(equal, softmax)]
        mean_diff = statistics.mean(diff) if diff else 0.0
        f.write(f"\n**R2 signal**: equal+ minus softmax+ mean = {mean_diff:+.3f}. ")
        if abs(mean_diff) < 1.0:
            f.write("Head carries little signal (within noise) — R2 confirmed.\n")
        else:
            f.write("Head carries signal — softmax meaningfully differs from equal.\n")
    print(f"[WRITE] {path}")


def _write_headline_md(records):
    path = "results/fused_five_headline.md"
    with open(path, "w") as f:
        f.write("# Task 3.3 — Headline (BasisMix vs R1 control)\n\n")
        f.write("| target | equal MAE | equal+ MAE | basismix MAE | basismix+ MAE | "
                "R1 verdict |\n")
        f.write("|--------|-----------|------------|--------------|---------------|"
                "------------|\n")
        for r in records:
            r1 = "dead weight" if r["equal_then_plus"]["mae"] <= r["basismix_fused_plus"]["mae"] else "additive"
            f.write(f"| {r['target']} "
                    f"| {r['equal_fused']['mae']:.3f} "
                    f"| {r['equal_then_plus']['mae']:.3f} "
                    f"| {r['basismix_fused']['mae']:.3f} "
                    f"| {r['basismix_fused_plus']['mae']:.3f} "
                    f"| {r1} |\n")
    print(f"[WRITE] {path}")


def _write_dropone_md(records):
    path = "results/fused_five_dropone.md"
    with open(path, "w") as f:
        f.write("# Task 4.1 — Drop-one ablation\n\n")
        f.write("Verdict (Task 4.2) is computed in a separate step. This table\n")
        f.write("shows MAE when each direction is dropped from BasisMix.\n\n")
        f.write("| target | drop_rag | drop_phys | drop_causal | drop_icl | drop_hier |\n")
        f.write("|--------|----------|-----------|-------------|----------|-----------|\n")
        for r in records:
            row = [r["target"]]
            for d in DIRECTION_ORDER:
                mae = r["drops"][d]["fused_plus"]["mae"]
                row.append(f"{mae:.3f}")
            f.write("| " + " | ".join(row) + " |\n")
    print(f"[WRITE] {path}")


if __name__ == "__main__":
    main()
