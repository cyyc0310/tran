"""Task 8.4 driver: QLD1 sanity run for joint training.

Trains on QLD1 target with NSW1+VIC1+SA1 sources, seed 0.
Writes:
  results/joint_train_sanity/stage1_metrics.json
  results/joint_train_sanity/stage2_metrics.json
  results/joint_train_sanity/summary.json
  results/joint_train_sanity.md  (loss curves + MAE summary)

Budget: ≤ 2 GPU-hr. On CPU/MPS this is faster but bounded by data load.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/cyyc0310/Downloads/transcif")

from transcif.data.loaders import load_region_data, all_region_configs
from scripts.experiments.run_joint_train import run_joint_train


def main():
    target = "QLD1"
    out_dir = Path("results/joint_train_sanity")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] loading regions...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}

    src_names = [n for n in all_regions if n != target][:3]
    small_regions = {target: all_regions[target]}
    for n in src_names:
        small_regions[n] = all_regions[n]

    print(f"[LOAD] target={target} sources={src_names}", flush=True)

    # Stage budget: keep total under 2 GPU-hr. On CPU this is fast.
    # 30 origins × 30 steps × 2 stages is enough to observe convergence.
    t0 = time.time()
    summary = run_joint_train(
        small_regions, target,
        stages=("stage1", "stage2"),
        n_origins=12,
        out_dir=str(out_dir),
        seed=0,
        n_steps_stage1=30,
        n_steps_stage2=30,
        lr_stage1=5e-2,
        lr_stage2=1e-2,
        margin=0.10,
    )
    elapsed = time.time() - t0
    summary["elapsed_seconds"] = elapsed
    summary["target"] = target
    summary["sources"] = src_names
    summary["n_origins"] = 12
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Markdown report
    m1 = json.loads((out_dir / "stage1_metrics.json").read_text())
    m2 = json.loads((out_dir / "stage2_metrics.json").read_text())

    md = [
        f"# Joint Train Sanity: {target} seed 0",
        f"",
        f"- Target: `{target}`",
        f"- Sources: `{', '.join(src_names)}`",
        f"- Elapsed: `{elapsed:.1f}s` ({elapsed / 60:.2f} min)",
        f"- Origins: 12",
        f"",
        f"## Stage 1 (ZS+ attention + BasisMix head)",
        f"",
        f"- Initial train loss: `{m1['train_loss'][0]:.4f}`",
        f"- Final train loss:   `{m1['train_loss'][-1]:.4f}`",
        f"- Final val MAE:      `{m1['val_mae'][-1]:.4f}`",
        f"- Steps: {len(m1['train_loss'])}",
        f"",
        f"## Stage 2 (+ per-direction correction)",
        f"",
        f"- Initial train loss: `{m2['train_loss'][0]:.4f}`",
        f"- Final train loss:   `{m2['train_loss'][-1]:.4f}`",
        f"- Final val MAE:      `{m2['val_mae'][-1]:.4f}`",
        f"- Steps: {len(m2['train_loss'])}",
        f"",
        f"## Verdict",
        f"",
    ]
    if summary.get("stage2_final_mae") is not None:
        final_mae = summary["stage2_final_mae"]
        if final_mae < 41:
            md.append(f"- **Stage 2 val MAE {final_mae:.2f} < 41 target**: GO")
        elif final_mae < 46:
            md.append(f"- **Stage 2 val MAE {final_mae:.2f} in [41, 46)**: tune & retry")
        else:
            md.append(f"- **Stage 2 val MAE {final_mae:.2f} ≥ 46**: underperforms BasisMix+ baseline (46.89)")

    (Path("results/joint_train_sanity.md")).write_text("\n".join(md))
    print(f"[DONE] {elapsed:.1f}s — summary: {summary}", flush=True)


if __name__ == "__main__":
    main()
