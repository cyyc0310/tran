"""Stage E: diagnose per-region difficulty and produce a stratified report.

Reads the leak-fixed fused_five + joint_train results and stratifies regions by:
  - persistence MAE (how hard the raw signal is)
  - ef_nr comparability (whether donor pools contain similar regions)
  - fuel regime (cluster from the fuel-share vectors)

Outputs:
  - results/regime_split_report.json  (machine-readable)
  - docs/REGIME_ANALYSIS.md           (human-readable report)

Usage:
    PYTHONPATH=src python scripts/experiments/diagnose_regime_split.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from transcif.config import RESULTS_DIR, DATA_DIR

AU = {"QLD1": 841.59, "NSW1": 875.23, "VIC1": 1160.12, "SA1": 490.43}
US = {"US_CISO": 342.8, "US_PJM": 347.6, "US_MISO": 534.1, "US_ERCO": 470.3,
      "US_ISNE": 299.1, "US_NYIS": 287.3, "US_FPL": 340.9, "US_BPAT": 207.5}


def load_region_stats():
    """Per-region persistence MAE + ef_nr from raw data (train-split aware)."""
    rows = []
    for f in sorted(Path(DATA_DIR).glob("*_2023_hourly.csv")):
        name = f.stem.replace("_2023_hourly", "")
        df = pd.read_csv(f, parse_dates=["hour"])
        cif = df["cif_real_gco2_per_kwh"].values.astype(float)
        valid = np.isfinite(cif) & (cif >= 0)
        cif = cif[valid]
        n24 = (len(cif) // 24) * 24
        persist_mae = np.abs(
            cif[:n24].reshape(-1, 24)[1:] - cif[:n24].reshape(-1, 24)[:-1]
        ).mean()
        if name in AU:
            ef_nr = AU[name]
        elif name in US:
            ef_nr = US[name]
        else:
            rs = df["renew_share"].values.astype(float)
            rs_tr = rs[: int(len(rs) * 0.8)]
            cif_tr = cif[: int(len(cif) * 0.8)]
            mask = (rs_tr < 0.5) & (rs_tr > 0.1) & (cif_tr > 0)
            ef_nr = float(np.median(cif_tr[mask] / (1 - rs_tr[mask]))) if mask.sum() > 100 else np.nan
        rows.append({"region": name, "persist_mae": persist_mae, "ef_nr": ef_nr})
    return pd.DataFrame(rows)


def load_joint_results():
    """Load per-region median MAE from the joint_train full run."""
    path = RESULTS_DIR / "joint_train_full.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    by_region = {}
    for r in rows:
        t = r["target"]
        by_region.setdefault(t, []).append(r["held_out_mae"])
    return {t: float(np.median(v)) for t, v in by_region.items()}


def load_fuel_clusters():
    """Cluster regions by fuel mix (Stage A data)."""
    regions = {}
    for fname in ("fuel_shares_us.json", "fuel_shares_uk.json"):
        path = DATA_DIR / "fuel" / fname
        if path.exists():
            doc = json.loads(path.read_text())
            regions.update(doc.get("regions", {}))
    if not regions:
        return {}
    order = ["coal", "gas", "nuclear", "hydro", "solar", "wind"]
    names = list(regions)
    vecs = np.array([[regions[n].get(k, 0) for k in order] for n in names])
    Z = linkage(vecs, method="ward")
    clusters = fcluster(Z, t=4, criterion="maxclust")
    return dict(zip(names, clusters))


def main():
    print("=" * 80)
    print("Stage E: Regime/difficulty stratification report")
    print("=" * 80)

    stats = load_region_stats()
    joint = load_joint_results()
    clusters = load_fuel_clusters()

    stats["joint_mae"] = stats["region"].map(joint)
    stats["cluster"] = stats["region"].map(clusters)

    # Stratify.
    stats["difficulty"] = pd.cut(
        stats["persist_mae"], bins=[0, 30, 60, 1e4],
        labels=["easy(<30)", "medium(30-60)", "pathological(>60)"])
    stats["ef_comparable"] = stats["ef_nr"].apply(
        lambda x: "comparable(<600)" if pd.notna(x) and x < 600
        else ("incomparable(>=600)" if pd.notna(x) else "unknown"))

    print("\n--- By difficulty ---")
    for tier in ["easy(<30)", "medium(30-60)", "pathological(>60)"]:
        sub = stats[stats.difficulty == tier]
        jm = sub.joint_mae.dropna()
        print(f"  {tier:18s}: {len(sub):2d} regions  "
              f"joint MAE median={jm.median():.1f}" if len(jm) else
              f"  {tier:18s}: {len(sub):2d} regions  (no joint data)")

    print("\n--- By ef_nr comparability ---")
    for tier in ["comparable(<600)", "incomparable(>=600)", "unknown"]:
        sub = stats[stats.ef_comparable == tier]
        jm = sub.joint_mae.dropna()
        print(f"  {tier:22s}: {len(sub):2d} regions  "
              f"joint MAE median={jm.median():.1f}" if len(jm) else
              f"  {tier:22s}: {len(sub):2d} regions  (no joint data)")

    # Comparable subset benchmark.
    comparable = stats[stats.ef_comparable == "comparable(<600)"]
    comp_joint = comparable.joint_mae.dropna()
    all_joint = stats.joint_mae.dropna()
    print(f"\n--- Comparable-subset benchmark ---")
    print(f"  All 29 regions:       joint median MAE = {all_joint.median():.2f}  (mean {all_joint.mean():.2f})")
    print(f"  Comparable subset:    joint median MAE = {comp_joint.median():.2f}  (mean {comp_joint.mean():.2f})")
    print(f"  Pathological removed: {set(stats[stats.ef_comparable=='incomparable(>=600)'].region)}")

    # Per-region table.
    print(f"\n--- Per-region detail (sorted by joint MAE) ---")
    print(f"{'region':<32} {'persist':>8} {'ef_nr':>7} {'joint':>7} {'cluster':>7} {'tier':>18}")
    print("-" * 90)
    for _, r in stats.sort_values("joint_mae", na_position="last").iterrows():
        ef = f"{r.ef_nr:.0f}" if pd.notna(r.ef_nr) else "nan"
        jm = f"{r.joint_mae:.1f}" if pd.notna(r.joint_mae) else "nan"
        cl = f"{int(r.cluster)}" if pd.notna(r.cluster) else "?"
        print(f"{r.region:<32} {r.persist_mae:>8.1f} {ef:>7} {jm:>7} {cl:>7} {str(r.ef_comparable):>18}")

    # Write JSON.
    out = {
        "per_region": stats.to_dict(orient="records"),
        "aggregate": {
            "all_median": float(all_joint.median()),
            "all_mean": float(all_joint.mean()),
            "comparable_median": float(comp_joint.median()),
            "comparable_mean": float(comp_joint.mean()),
        },
    }
    out_path = RESULTS_DIR / "regime_split_report.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
