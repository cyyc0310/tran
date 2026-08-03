"""Zero-shot config-only foundation check on the REAL SA1 sample.

Purpose
-------
Before rebuilding TransCIF around a pure zero-shot / config-only design, we must
verify the load-bearing assumption of that design:

    CIF(t) can be synthesized from a (domain-invariant) generation STRUCTURE plus a
    domain-specific CONFIG (emission factors ef_g), i.e. CIF = f(structure ; config).

This script does NOT train the full model. It runs three honest diagnostics on the
only real data present on this machine (tests/fixtures/real_aemo_sample_sa1.csv,
300 hourly rows, binary renew/non-renew aggregates for SA1 only):

  E1. Physics-synthesis ceiling: reconstruct CIF from the TRUE renew_share + SA1
      config factors. This isolates the error of the binary physics layer itself
      (structure assumed known perfectly). It tells us whether upgrading the
      binary split to a full per-source s_g(t) is necessary.

  E2. Config-swap sensitivity: apply OTHER regions' config factors to SA1's true
      structure and measure the CIF-level shift. This quantifies "change config =>
      change domain" — the core mechanism of config-only cross-region transfer.

  E3. Persistence bar: day-ahead (lag-24h) persistence MAE — the baseline the
      zero-shot pipeline must beat.

Data limitation (reported, not hidden): full multi-region NEMED data (/tmp/nemed_output)
and per-source generation are absent here, so a real multi-ISO LORO zero-shot run is
BLOCKED on data acquisition. This script validates the mechanism foundation only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from transcif.physics.cif import cif_from_shares, get_emission_factors

FIXTURE = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "real_aemo_sample_sa1.csv"


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def main() -> None:
    df = pd.read_csv(FIXTURE)
    real_cif = df["cif_real_gco2_per_kwh"].to_numpy(dtype=float)
    renew_share = df["renew_share"].to_numpy(dtype=float)
    n = len(df)
    print(f"Loaded REAL SA1 sample: {n} hourly rows "
          f"({df['hour'].iloc[0]} .. {df['hour'].iloc[-1]})")
    print(f"CIF range {real_cif.min():.1f}..{real_cif.max():.1f} gCO2/kWh, "
          f"mean {real_cif.mean():.1f}; renew_share mean {renew_share.mean():.3f}\n")

    # --- E1: physics-synthesis ceiling (true structure + SA1 config) ---
    r_sa, n_sa = get_emission_factors("AU_SA")
    cif_phys = cif_from_shares(renew_share, r_sa, n_sa)
    e1 = mae(cif_phys, real_cif)
    # data-implied effective non-renewable factor (per hour), for context
    eff_nonrenew = real_cif / np.clip(1 - renew_share, 1e-6, None)
    print("E1  Physics ceiling (TRUE renew_share + AU_SA config 0/490.43):")
    print(f"    binary-physics MAE = {e1:.3f} gCO2/kWh")
    print(f"    data-implied non-renew factor: mean {eff_nonrenew.mean():.1f} "
          f"(std {eff_nonrenew.std():.1f}) vs config 490.43 -> "
          f"binary constant is an approximation\n")

    # --- E3: day-ahead persistence bar (lag-24h) ---
    lag = 24
    persist_pred = real_cif[:-lag]
    persist_true = real_cif[lag:]
    e3 = mae(persist_pred, persist_true)
    print(f"E3  Persistence (lag-{lag}h) MAE = {e3:.3f} gCO2/kWh  (bar to beat)\n")

    # --- E2: config-swap sensitivity ---
    print("E2  Config-swap on SA1 TRUE structure (mean synthesized CIF):")
    for code in ("AU_SA", "AU_NSW", "AU_QLD", "AU_VIC"):
        r, nf = get_emission_factors(code)
        cif_swapped = cif_from_shares(renew_share, r, nf)
        print(f"    {code:7s} ef=({r:.2f},{nf:7.2f}) -> mean CIF {cif_swapped.mean():7.1f} gCO2/kWh")
    print("    (only the config changed; structure held fixed => config controls level)\n")

    # --- verdict ---
    print("VERDICT")
    verdict = ("physics ceiling BEATS persistence -> binary decomposition adequate; "
               "bottleneck is STRUCTURE prediction"
               if e1 < e3 else
               "physics ceiling LOSES to persistence even with TRUE structure -> "
               "binary split too coarse; full per-source s_g upgrade is necessary")
    print(f"    E1={e1:.2f} vs E3={e3:.2f} -> {verdict}")


if __name__ == "__main__":
    main()
