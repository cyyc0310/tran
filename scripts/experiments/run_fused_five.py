"""Fuse RAG/Phys-IRM/Causal/ICL/Hier into a single zero-shot algorithm with ZS+ calibration.

Usage:
    .venv/bin/python scripts/experiments/run_fused_five.py --regions AU1 AU2 AU3 AU4 --seed 0

Task 1.3 will extract the source-region stack collector into a reusable module.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.config import SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, AU_REGIONS, UK_REGIONS, US_REGIONS
from transcif.data.loaders import load_region_data
from transcif.data.windows import build_windows
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.models.zeroshot.fusion import DIRECTION_ORDER, FusionModel, FusionHead, train_fusion

DEVICE = "cuda" if torch.cuda.is_available() else None


def _train_and_predict_direction(direction, region_name, all_regions, seed, x_test, config, ef_r, ef_nr):
    """Train and predict a single direction method."""
    if direction == "rag":
        from transcif.models.zeroshot.rag import train_rag_zero_shot, predict_rag_zs
        model, bank = train_rag_zero_shot(all_regions, region_name, seed=seed, device=DEVICE)
        return predict_rag_zs(model, bank, x_test.astype(np.float32), config.astype(np.float32), ef_r, ef_nr)
    elif direction == "phys":
        from transcif.models.zeroshot.phys_irm import train_phys_irm, predict_phys_irm
        model, _ = train_phys_irm(all_regions, region_name, seed=seed, gamma_irm=0.1, lambda_cif=0.5, device=DEVICE)
        return predict_phys_irm(model, x_test.astype(np.float32), config.astype(np.float32), ef_r, ef_nr)
    elif direction == "causal":
        from transcif.models.zeroshot.causal import train_causal_zero_shot, predict_causal_zs
        model, _ = train_causal_zero_shot(all_regions, region_name, seed=seed, device=DEVICE)
        return predict_causal_zs(model, x_test.astype(np.float32), config.astype(np.float32), ef_r, ef_nr)
    elif direction == "icl":
        from transcif.models.zeroshot.icl import train_icl, predict_icl_zs
        model = train_icl(all_regions, region_name, seed=seed, device=DEVICE)
        return predict_icl_zs(model, all_regions, region_name, x_test.astype(np.float32), ef_r, ef_nr)
    elif direction == "hier":
        from transcif.models.zeroshot.hier import train_hier, predict_hier_zs
        model = train_hier(all_regions, region_name, seed=seed, device=DEVICE)
        return predict_hier_zs(model, x_test.astype(np.float32), config.astype(np.float32), ef_r, ef_nr)


def _build_predictor_dict(region_name, all_regions, seed):
    """Build predictor dict and get test data."""
    data = all_regions[region_name]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_test, _, y_test = build_windows(data["rs"][split - SEQ_LEN:], data["cif"][split - SEQ_LEN:],
                                      seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE)

    predictors = {}
    for d in DIRECTION_ORDER:
        if d == "rag":
            from transcif.models.zeroshot.rag import train_rag_zero_shot, predict_rag_zs
            m, bank = train_rag_zero_shot(all_regions, region_name, seed=seed, device=DEVICE)
            predictors[d] = lambda x, cfg, ef_r, ef_nr, m=m, b=bank: predict_rag_zs(m, b, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
        elif d == "phys":
            from transcif.models.zeroshot.phys_irm import train_phys_irm, predict_phys_irm
            m, _ = train_phys_irm(all_regions, region_name, seed=seed, gamma_irm=0.1, lambda_cif=0.5, device=DEVICE)
            predictors[d] = lambda x, cfg, ef_r, ef_nr, m=m: predict_phys_irm(m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
        elif d == "causal":
            from transcif.models.zeroshot.causal import train_causal_zero_shot, predict_causal_zs
            m, _ = train_causal_zero_shot(all_regions, region_name, seed=seed, device=DEVICE)
            predictors[d] = lambda x, cfg, ef_r, ef_nr, m=m: predict_causal_zs(m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
        elif d == "icl":
            from transcif.models.zeroshot.icl import train_icl, predict_icl_zs
            m = train_icl(all_regions, region_name, seed=seed, device=DEVICE)
            predictors[d] = lambda x, cfg, ef_r, ef_nr, m=m: predict_icl_zs(m, all_regions, region_name, x.astype(np.float32), ef_r, ef_nr)
        elif d == "hier":
            from transcif.models.zeroshot.hier import train_hier, predict_hier_zs
            m = train_hier(all_regions, region_name, seed=seed, device=DEVICE)
            predictors[d] = lambda x, cfg, ef_r, ef_nr, m=m: predict_hier_zs(m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)

    return predictors, data, x_test, y_test


def _collect_source_stacks(source_names, all_regions, seed, src_limit):
    """Collect source region CIF stacks for fusion head training."""
    names = source_names[:src_limit] if src_limit > 0 else source_names
    src_stacks, src_true = [], []

    for i, name in enumerate(names):
        try:
            data = all_regions[name]
            split = int(len(data["rs"]) * TRAIN_FRACTION)
            x_test, _, y_test = build_windows(data["rs"][split - SEQ_LEN:], data["cif"][split - SEQ_LEN:],
                                             seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE)

            stack = np.stack([_train_and_predict_direction(d, name, all_regions, seed, x_test,
                                                            data["config"].astype(np.float32),
                                                            data["ef_r"], data["ef_nr"])
                              for d in DIRECTION_ORDER], axis=1).astype(np.float32)

            src_stacks.append(stack)
            src_true.append(y_test)
            print(f"    [src {i+1}/{len(names)}] {name} ok", flush=True)
        except Exception as e:
            print(f"    [WARN] skip {name}: {e}", flush=True)

    return src_stacks, src_true


def evaluate_fused(target_name, all_regions, seed, src_limit, use_head):
    """Evaluate fused model with optional ZS+ calibration."""
    predictors, data, x_test, y_true = _build_predictor_dict(target_name, all_regions, seed)

    src_stacks, src_true = _collect_source_stacks([n for n in all_regions if n != target_name],
                                                   all_regions, seed, src_limit)

    if use_head and src_stacks:
        fusion_model = train_fusion(src_stacks, src_true, predictors=predictors, epochs=200, lr=1e-2, l2=1e-4, seed=seed)
    else:
        equal_head = FusionHead()
        with torch.no_grad():
            equal_head.logit.fill_(0.0)
        fusion_model = FusionModel(equal_head, predictors=predictors)

    cif_fused = fusion_model.predict_cif(x_test.astype(np.float32),
                                         data["config"].astype(np.float32),
                                         data["ef_r"], data["ef_nr"])

    result = {"target": target_name, "seed": seed, "transcif_fused5": compute_metrics(cif_fused, y_true)}

    fusion_model.configure_for_target(data["config"], data["ef_r"], data["ef_nr"])
    rs, cif = data["rs"], data["cif"]
    split = int(len(rs) * TRAIN_FRACTION)
    origins = [split + st for st in range(0, len(cif) - split - HORIZON + 1, TEST_STRIDE)]

    cif_fused_plus = zs_plus_predict(model=None, config=data["config"], rs=rs, cif=cif, ef_r=data["ef_r"],
                                      ef_nr=data["ef_nr"], origins=origins, share_fn=fusion_model.share_fn)
    result["transcif_fused5_plus"] = compute_metrics(cif_fused_plus, y_true)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["QLD1", "NSW1", "VIC1", "SA1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-head", action="store_true", help="use equal-weight fusion")
    ap.add_argument("--src-limit", type=int, default=1, help="max source regions for head training")
    ap.add_argument("--out", default=None, help="output filename (default: fused_five_smoke.json)")
    args = ap.parse_args()

    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {name: load_region_data(name, all_configs) for name in all_configs if name in all_configs}

    out = []
    for target in args.regions:
        if target not in all_regions:
            print(f"[SKIP] {target} not loaded")
            continue
        print(f"[EVAL] {target} ...", flush=True)
        r = evaluate_fused(target, all_regions, args.seed, args.src_limit, use_head=not args.no_head)
        out.append(r)
        print(f"  fused5 MAE={r['transcif_fused5']['mae']:.3f}  fused5+ MAE={r['transcif_fused5_plus']['mae']:.3f}", flush=True)

    os.makedirs("results", exist_ok=True)
    fn = args.out or ("fused_five_smoke.json" if not args.no_head
                      else "fused_five_equalweight_smoke.json")
    with open(os.path.join("results", fn), "w") as f:
        json.dump(out, f, indent=2)
    print(f"[DONE] wrote results/{fn}")


if __name__ == "__main__":
    main()
