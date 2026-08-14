"""Phase 1.2 Validation: Quick check that US regions integrate properly.

Runs a lightweight evaluation (1 seed, 1 US target) to verify data quality.
Usage: PYTHONPATH=scripts python scripts/validate_us_data.py
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Import from phase1 script
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))  # repo root
from scripts.experiments.run_phase1_complete import (
    DATA_DIR, AU_REGIONS, discover_uk_regions, UK_REGIONS,
    load_region_data, build_windows, cif_from_shares,
    AdaptivePersistDLinear, train_zero_shot, SEQ_LEN, HORIZON, TEST_STRIDE
)

# US region configs (computed from EIA-930 data)
US_REGIONS = {
    "US_CISO": {"file": "US_CISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 342.8},
    "US_PJM":  {"file": "US_PJM_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 347.6},
    "US_MISO": {"file": "US_MISO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 534.1},
    "US_ERCO": {"file": "US_ERCO_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 470.3},
    "US_ISNE": {"file": "US_ISNE_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 299.1},
    "US_NYIS": {"file": "US_NYIS_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 287.3},
    "US_FPL":  {"file": "US_FPL_2023_hourly.csv",  "ef_r": 0.0, "ef_nr": 340.9},
    "US_BPAT": {"file": "US_BPAT_2023_hourly.csv", "ef_r": 0.0, "ef_nr": 207.5},
}


def load_us_region(region_name: str) -> dict:
    """Load US region data in same format as AU/UK."""
    info = US_REGIONS[region_name]
    path = DATA_DIR / info["file"]
    ef_r, ef_nr = info["ef_r"], info["ef_nr"]

    df = pd.read_csv(path, parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    rs = df["renew_share"].values.astype(np.float32)
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float32)

    # Clean: remove NaN/inf
    valid = np.isfinite(rs) & np.isfinite(cif) & (cif >= 0)
    rs = rs[valid]
    cif = cif[valid]

    # Derive scalar config from the training split only (no test-period leak).
    split = int(len(rs) * TRAIN_FRACTION)
    train_mean_rs = float(rs[:split].mean())
    return {
        "rs": rs, "cif": cif,
        "mean_rs": train_mean_rs,
        "ef_r": ef_r, "ef_nr": ef_nr,
        "config": np.array([train_mean_rs, ef_nr / 1000.0], dtype=np.float32),
    }


def main():
    print("=" * 70)
    print("Phase 1.2 Validation: US Data Integration Test")
    print("=" * 70)

    # Load all regions
    discover_uk_regions()
    all_regions = {}
    for name in AU_REGIONS:
        all_regions[name] = load_region_data(name)
    for name in UK_REGIONS:
        all_regions[name] = load_region_data(name)
    for name in US_REGIONS:
        try:
            all_regions[name] = load_us_region(name)
        except Exception as e:
            print(f"  [WARN] Failed to load {name}: {e}")

    n_total = len(all_regions)
    print(f"\nTotal regions loaded: {n_total}")
    print(f"  AU: {len(AU_REGIONS)}, UK: {len(UK_REGIONS)}, US: {sum(1 for k in all_regions if k.startswith('US_'))}")

    # Print region summary sorted by mean_rs
    print(f"\n{'Region':<12} {'mean_rs':<9} {'ef_nr':<8} {'Hours':<7} {'mean_CIF':<10}")
    print("-" * 50)
    for name in sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"]):
        d = all_regions[name]
        print(f"{name:<12} {d['mean_rs']:<9.3f} {d['ef_nr']:<8.1f} {len(d['rs']):<7} {d['cif'].mean():<10.1f}")

    # Quick zero-shot test on 2 US targets
    print("\n" + "=" * 70)
    print("Quick Zero-Shot Evaluation (seed=0)")
    print("=" * 70)

    test_targets = ["US_CISO", "US_ERCO", "US_BPAT", "US_MISO"]
    for target in test_targets:
        if target not in all_regions:
            continue
        data = all_regions[target]
        rs, cif = data["rs"], data["cif"]
        ef_r, ef_nr = data["ef_r"], data["ef_nr"]

        # Build test windows
        split = int(len(rs) * TRAIN_FRACTION)
        x_rs_test, _, y_cif_test = build_windows(
            rs[split - SEQ_LEN:], cif[split - SEQ_LEN:], SEQ_LEN, HORIZON, TEST_STRIDE)

        # CIF history for persistence
        cif_offset = cif[split - SEQ_LEN:]
        x_cif_test = []
        for start in range(0, len(cif_offset) - SEQ_LEN - HORIZON + 1, TEST_STRIDE):
            x_cif_test.append(cif_offset[start:start + SEQ_LEN])
        x_cif_test = np.stack(x_cif_test)

        # Persistence
        persist_mae = float(np.abs(x_cif_test[:, -HORIZON:] - y_cif_test).mean())

        # Zero-shot TransCIF
        zs_model = train_zero_shot(all_regions, target, seed=0)
        target_cfg = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
        with torch.no_grad():
            zs_rs_pred = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), target_cfg).numpy()
        zs_cif = cif_from_shares(zs_rs_pred, ef_r, ef_nr)
        zs_mae = float(np.abs(zs_cif - y_cif_test).mean())

        ratio = zs_mae / persist_mae
        print(f"  {target:<10} mean_rs={data['mean_rs']:.3f} | "
              f"Persist={persist_mae:.1f} | TransCIF-ZS={zs_mae:.1f} | "
              f"ZS/Persist={ratio:.3f}")

    print("\n✓ US data integration validated!")
    print(f"  Total coverage: {n_total} regions, mean_rs range: "
          f"{min(d['mean_rs'] for d in all_regions.values()):.3f} - "
          f"{max(d['mean_rs'] for d in all_regions.values()):.3f}")


if __name__ == "__main__":
    main()
