"""Cross-region persistence-baseline comparison, reusing the already-trained rotation results
cached by theorem1_domain_rotation.py (/tmp/transcif_result_rot_{region}_{slug}.json) instead of
retraining. Fills the one gap that experiment left open: Item1 only reported the Theorem 1
decomposition (L_T, term1_share_pct) per region, never persistence_mae, so there was no
region-by-region answer to "does D+E actually beat the naive persistence baseline outside SA1?"
This is exactly the evidence the "plug-and-play generalizes across regions" paper claim needs.

No model training happens here -- persistence_mae is computed purely from data (last observed
RenewShare repeated across the horizon), using the same load_source_and_target() + CALIB_FRACTION
split that decompose() used when producing the cached corrected_mae values, so the two numbers
are directly comparable.

Run with: PYTHONPATH=src python scripts/region_persistence_comparison.py
"""

import json
import re

import numpy as np

from theorem1_domain_rotation import AU_REGIONS, VARIANTS, load_source_and_target
from sa1_ablation import CALIB_FRACTION, HORIZON, REGION_TO_FACTOR_CODE
from sa1_domain_adaptation import INCLUDE_GENERATION, INCLUDE_TEMPERATURE
from transcif.evaluation.metrics import mae
from transcif.physics.cif import cif_from_shares, get_emission_factors


def result_path_for(target_region: str, name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_")
    return f"/tmp/transcif_result_rot_{target_region}_{slug}.json"


def persistence_mae_for_region(region: str) -> float:
    _, x_target, _, ci_true_target = load_source_and_target(region, INCLUDE_GENERATION, INCLUDE_TEMPERATURE)

    n = x_target.shape[0]
    split = int(n * CALIB_FRACTION)
    x_eval = x_target[split:]
    ci_true_eval = ci_true_target[split:]

    renew_factor, nonrenew_factor = get_emission_factors(REGION_TO_FACTOR_CODE[region])

    last_observed_share = x_eval[:, -1, 0].numpy()
    persistence_share_pred = np.repeat(last_observed_share[:, None], HORIZON, axis=1)
    ci_persistence_eval = cif_from_shares(persistence_share_pred, renew_factor, nonrenew_factor)

    return mae(ci_true_eval.reshape(-1), ci_persistence_eval.reshape(-1))


if __name__ == "__main__":
    rows = []
    for region in AU_REGIONS:
        persistence_mae = persistence_mae_for_region(region)
        print(f"{region}: persistence_mae={persistence_mae}", flush=True)

        cached = {}
        for variant in VARIANTS:
            path = result_path_for(region, variant["name"])
            with open(path) as f:
                result = json.load(f)
            cached[variant["name"]] = result

        row = {"region": region, "L_T": cached[VARIANTS[0]["name"]]["L_T"], "persistence_mae": persistence_mae}
        for variant in VARIANTS:
            corrected_mae = cached[variant["name"]]["mean_abs_total_error"]
            vs_persistence_pct = (corrected_mae - persistence_mae) / persistence_mae * 100
            row[f"corrected_mae[{variant['name']}]"] = corrected_mae
            row[f"vs_persistence_pct[{variant['name']}]"] = vs_persistence_pct
        rows.append(row)
        print(row, flush=True)

    print("\n=== cross-region persistence comparison summary ===")
    for row in rows:
        print(row, flush=True)

    with open("/tmp/transcif_region_persistence_comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
