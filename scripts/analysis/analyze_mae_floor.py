"""MAE floor analysis: what bounds CIF forecasting error per region, and how
low can MAE plausibly go?

This is the data spine for the "reduce MAE to 10" investigation. For each of
the 29 grids it computes:

  - Descriptive stats: CIF mean/std/CV, renewable penetration, ef_nr.
  - **Persistence floor**: the 24h-ahead MAE of lag-24 persistence on the test
    split (the best "do nothing sophisticated" reference). A model that cannot
    beat this is not learning the diurnal cycle.
  - **Noise floor**: 0.8 x std(CIF - hour-of-week climatology). This is an
    OPTIMISTIC theoretical lower bound — the irreducible error of any model
    that predicts only the deterministic seasonal pattern and treats the rest
    as noise. It is *not* a hard limit (recent-observation models can partly
    predict weather-driven deviations), but regions whose current MAE is near
    this floor have little headroom.
  - Diurnal amplitude, autocorrelation (lag 24 / 168), day-to-day volatility.
  - Joint-trained median MAE per region (from results/joint_train_full.json),
    the gap-to-floor, and an easy/medium/hard tier.

Correlates features with per-region MAE to identify what makes a grid hard,
and quantifies whether a system-wide median MAE of 10 is physically plausible.

Output: results/mae_floor_analysis.json
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION, DATA_DIR
from transcif.data.loaders import all_region_configs

HERE = Path(__file__).resolve().parent.parent.parent
RESULTS = HERE / "results"


def load_region_csv(name, cfg):
    """Load raw hourly CSV for a region; return dataframe sorted by hour."""
    df = pd.read_csv(DATA_DIR / cfg["file"], parse_dates=["hour"])
    df = df.sort_values("hour").reset_index(drop=True)
    return df


def persistence_floor(cif, rs, ef_r, ef_nr):
    """24h-ahead lag-24 persistence MAE on the test split (two variants).

    Returns (cif_lag_mae, rs_reconstruct_mae). The CIF-lag variant is the best
    simple reference; the rs-reconstruct variant matches fused_five_full's
    'persistence' column.
    """
    n = len(cif)
    split = int(n * TRAIN_FRACTION)
    # test origins: stride TEST_STRIDE, each forecasts [t0, t0+HORIZON)
    origins = [split + st for st in range(0, n - split - HORIZON + 1, TEST_STRIDE)]
    cif_errs, rs_errs = [], []
    for t0 in origins:
        true_c = cif[t0:t0 + HORIZON]
        # CIF lag: yesterday's actual CIF
        lag_c = cif[t0 - HORIZON:t0]
        cif_errs.append(np.abs(lag_c - true_c).mean())
        # rs reconstruct (fused_five_full protocol)
        lag_rs = rs[t0 - HORIZON:t0]
        rec = lag_rs * ef_r + (1 - lag_rs) * ef_nr
        rs_errs.append(np.abs(rec - true_c).mean())
    return float(np.mean(cif_errs)), float(np.mean(rs_errs))


def noise_floor(cif, hour_of_week):
    """Optimistic theoretical MAE floor: 0.8 * std(detrended CIF).

    Detrending = subtract the per-(day-of-week, hour) climatology mean.
    0.8 ≈ sqrt(2/pi) maps Gaussian std to mean absolute deviation.
    """
    clim = np.zeros(168)
    for hw in range(168):
        m = cif[hour_of_week == hw]
        if len(m) > 0:
            clim[hw] = m.mean()
    detrended = cif - clim[hour_of_week]
    std = float(np.std(detrended))
    return 0.8 * std, std


def autocorr(x, lag):
    """Lagged autocorrelation of a 1-D series."""
    x = x - x.mean()
    denom = np.dot(x, x)
    if denom == 0:
        return 0.0
    return float(np.dot(x[:-lag], x[lag:]) / denom)


def analyze_region(name, cfg, df, joint_mae):
    cif = df["cif_real_gco2_per_kwh"].values.astype(np.float64)
    rs = df["renew_share"].values.astype(np.float64)
    ef_r = float(cfg["ef_r"])
    ef_nr = float(cfg["ef_nr"])
    n = len(cif)

    # hour-of-week index (0..167) for climatology
    dow = df["hour"].dt.dayofweek.values
    hod = df["hour"].dt.hour.values
    hour_of_week = (dow * 24 + hod).astype(np.int64)

    pers_cif, pers_rs = persistence_floor(cif, rs, ef_r, ef_nr)
    noise_mae, noise_std = noise_floor(cif, hour_of_week)

    hour_means = np.array([cif[hod == h].mean() if (hod == h).any() else 0.0
                           for h in range(24)])
    diurnal_amp = float(hour_means.max() - hour_means.min())

    delta24 = cif[HORIZON:] - cif[:-HORIZON]
    return {
        "region": name,
        "n_hours": int(n),
        "ef_nr": ef_nr,
        "ef_r": ef_r,
        "cif_mean": float(cif.mean()),
        "cif_std": float(cif.std()),
        "cif_cv": float(cif.std() / (abs(cif.mean()) + 1e-8)),
        "cif_median": float(np.median(cif)),
        "cif_p10": float(np.percentile(cif, 10)),
        "cif_p90": float(np.percentile(cif, 90)),
        "renewable_penetration": float(rs.mean()),
        "rs_std": float(rs.std()),
        "persistence_floor_cif_mae": pers_cif,
        "persistence_floor_rs_mae": pers_rs,
        "noise_floor_mae": float(noise_mae),
        "noise_std": float(noise_std),
        "diurnal_amplitude": diurnal_amp,
        "acf_lag24": autocorr(cif, 24),
        "acf_lag168": autocorr(cif, 168),
        "day_to_day_volatility": float(np.std(delta24)),
        "joint_median_mae": float(joint_mae) if joint_mae is not None else None,
        "gap_to_persistence": (float(joint_mae) - pers_cif) if joint_mae is not None else None,
        "gap_to_noise_floor": (float(joint_mae) - noise_mae) if joint_mae is not None else None,
    }


def load_joint_per_region():
    """Median joint-trained MAE per region (over seeds)."""
    path = RESULTS / "joint_train_full.json"
    if not path.exists():
        return {}
    rows = json.loads(path.read_text())
    by_region = {}
    for r in rows:
        if "held_out_mae" in r and r["held_out_mae"] is not None:
            by_region.setdefault(r["target"], []).append(r["held_out_mae"])
    return {k: float(np.median(v)) for k, v in by_region.items()}


def correlate(regions, target_key, feature_keys):
    """Pearson + Spearman correlation of each feature with the target."""
    from scipy.stats import pearsonr, spearmanr
    y = np.array([r[target_key] for r in regions], dtype=np.float64)
    out = {}
    for fk in feature_keys:
        x = np.array([r[fk] for r in regions], dtype=np.float64)
        if np.std(x) == 0 or np.std(y) == 0:
            out[fk] = {"pearson": None, "spearman": None}
            continue
        p = pearsonr(x, y)
        s = spearmanr(x, y)
        out[fk] = {
            "pearson": float(p[0]), "pearson_p": float(p[1]),
            "spearman": float(s.correlation), "spearman_p": float(s.pvalue),
        }
    return out


def tier_of(r):
    """Easy / medium / hard by the persistence floor (best simple reference)."""
    pf = r["persistence_floor_cif_mae"]
    if pf < 25:
        return "easy"
    if pf < 50:
        return "medium"
    return "hard"


def main():
    configs = all_region_configs()
    joint_per_region = load_joint_per_region()
    print(f"[LOAD] {len(configs)} region configs; "
          f"{len(joint_per_region)} with joint MAE", flush=True)

    regions = []
    for name, cfg in configs.items():
        df = load_region_csv(name, cfg)
        jm = joint_per_region.get(name)
        rec = analyze_region(name, cfg, df, jm)
        rec["tier"] = tier_of(rec)
        regions.append(rec)
        print(f"  {name:32s} CIF_mean={rec['cif_mean']:6.1f} "
              f"pers_floor={rec['persistence_floor_cif_mae']:5.1f} "
              f"noise_floor={rec['noise_floor_mae']:5.1f} "
              f"joint={rec['joint_median_mae']}" if jm is not None else
              f"  {name:32s} CIF_mean={rec['cif_mean']:6.1f} "
              f"pers_floor={rec['persistence_floor_cif_mae']:5.1f} "
              f"noise_floor={rec['noise_floor_mae']:5.1f} joint=None",
              flush=True)

    # correlations with joint MAE (only regions with joint MAE)
    with_mae = [r for r in regions if r["joint_median_mae"] is not None]
    feature_keys = [
        "cif_mean", "cif_std", "cif_cv", "renewable_penetration", "rs_std",
        "ef_nr", "persistence_floor_cif_mae", "noise_floor_mae", "noise_std",
        "diurnal_amplitude", "acf_lag24", "acf_lag168", "day_to_day_volatility",
    ]
    corr = correlate(with_mae, "joint_median_mae", feature_keys)

    # tier aggregates
    tier_summary = {}
    for tier in ["easy", "medium", "hard"]:
        members = [r for r in with_mae if r["tier"] == tier]
        if not members:
            continue
        tier_summary[tier] = {
            "n_regions": len(members),
            "members": [m["region"] for m in members],
            "joint_mae_median": float(np.median([m["joint_median_mae"] for m in members])),
            "joint_mae_min": float(np.min([m["joint_median_mae"] for m in members])),
            "joint_mae_max": float(np.max([m["joint_median_mae"] for m in members])),
            "persistence_floor_median": float(np.median([m["persistence_floor_cif_mae"] for m in members])),
            "noise_floor_median": float(np.median([m["noise_floor_mae"] for m in members])),
            "mean_gap_to_noise": float(np.mean([m["gap_to_noise_floor"] for m in members])),
        }

    # Can system median reach 10? Estimate the floor for system median.
    all_noise = np.array([r["noise_floor_mae"] for r in regions])
    all_pers = np.array([r["persistence_floor_cif_mae"] for r in regions])
    feasibility = {
        "median_noise_floor_all_regions": float(np.median(all_noise)),
        "median_persistence_floor_all_regions": float(np.median(all_pers)),
        "n_regions_noise_floor_below_10": int((all_noise < 10).sum()),
        "n_regions_persistence_floor_below_10": int((all_pers < 10).sum()),
        "interpretation": (
            "A system-wide median MAE of 10 requires at least half the regions "
            "to have an MAE near 10. The noise floor is an OPTIMISTIC bound "
            "(perfect seasonal model + Gaussian noise); the persistence floor "
            "is a PRACTICAL bound (best lag reference). If the median "
            "persistence floor exceeds 10, reaching median MAE 10 is not "
            "physically plausible without fundamentally richer inputs "
            "(e.g. weather/price forecasts)."
        ),
    }

    out = {
        "regions": regions,
        "correlations_with_joint_mae": corr,
        "tier_summary": tier_summary,
        "feasibility_median_mae_10": feasibility,
    }
    out_path = RESULTS / "mae_floor_analysis.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\n[WRITE] {out_path}")

    # console summary
    print("\n=== Tier summary ===")
    for tier, s in tier_summary.items():
        print(f"  {tier:7s} n={s['n_regions']:2d}  joint_med={s['joint_mae_median']:5.1f}  "
              f"pers_floor={s['persistence_floor_median']:5.1f}  "
              f"noise_floor={s['noise_floor_median']:5.1f}  "
              f"mean_gap_to_noise={s['mean_gap_to_noise']:5.1f}")
    print(f"\n=== Feasibility of median MAE 10 ===")
    print(f"  median noise floor:        {feasibility['median_noise_floor_all_regions']:.1f}")
    print(f"  median persistence floor:  {feasibility['median_persistence_floor_all_regions']:.1f}")
    print(f"  regions with noise floor < 10: {feasibility['n_regions_noise_floor_below_10']}/{len(regions)}")
    print("\n=== Top feature correlations with joint MAE (Spearman) ===")
    ranked = sorted(corr.items(), key=lambda kv: abs(kv[1].get("spearman") or 0), reverse=True)
    for fk, v in ranked[:6]:
        sp = v.get("spearman")
        print(f"  {fk:32s} spearman={sp:+.2f} (p={v.get('spearman_p'):.2e})")


if __name__ == "__main__":
    main()
