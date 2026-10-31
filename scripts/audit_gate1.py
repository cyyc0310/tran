"""Post-run semantic/numerical audit; does not change the fixed candidate."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research.toy import ToyMechanism
from transcif.research.gates12 import _cdf_at
from transcif.research.weighted_metrics import PreparedDistribution, weighted_quantile


def main():
    destination = ROOT / "results/research/gates12"
    before = hashlib.sha256((destination / "summary.json").read_bytes()).hexdigest()
    checks = {}
    for case in ("harmful", "harmless", "resolved"):
        model = ToyMechanism(case)
        coarse, fine = model.reference(20001), model.reference(40001)
        quantiles = np.linspace(.025, .975, 39)
        thresholds = weighted_quantile(fine["predictions"], fine["weights"], quantiles)
        coarse_quantiles = weighted_quantile(coarse["predictions"], coarse["weights"], quantiles)
        interval_error = float(np.abs(weighted_quantile(coarse["predictions"], coarse["weights"], [.05, .95]) - weighted_quantile(fine["predictions"], fine["weights"], [.05, .95])).max())
        cdf_error = float(np.abs(_cdf_at(coarse["predictions"], coarse["weights"], thresholds) - _cdf_at(fine["predictions"], fine["weights"], thresholds)).max())
        z = np.column_stack([np.linspace(-4, 4, 101), np.zeros(101)])
        observation = model.historical_observations(z)
        reconstructed_likelihood = -.5 * ((observation["normalized_daily_total_minus_one"] - model.amplitude ** 2) / model.squared_observation_sd) ** 2
        if case == "resolved":
            reconstructed_likelihood -= .5 * ((observation["peak_contrast"] - model.amplitude) / model.sign_observation_sd) ** 2
        likelihood_error = float(np.abs(reconstructed_likelihood - model.evaluate(z)[0]).max())
        checks[case] = {"max_cdf_grid_refinement_error": cdf_error,
                        "max_quantile_grid_refinement_error_g_per_kwh": float(np.abs(coarse_quantiles - thresholds).max()),
                        "interval90_endpoint_refinement_error_g_per_kwh": interval_error,
                        "physical_observation_log_likelihood_error": likelihood_error}
    model = ToyMechanism("harmful")
    pair = np.array([[-1.6, 0], [1.6, 0]])
    aggregate = model.historical_observations(pair)
    pair_error = float(np.abs(aggregate["daily_fuel_mwh"][0] - aggregate["daily_fuel_mwh"][1]).max())
    predictions = model.evaluate(pair)[1]
    # Deliberately change the unobserved target sign prior. Its total-energy
    # observation distribution is unchanged, so conditioning cannot detect it.
    ref = model.reference(2001)
    shifted_weights = ref["weights"] * np.where(ref["z"][:, 0] > 0, 1.8, .2)
    shifted_weights /= shifted_weights.sum()
    base = PreparedDistribution(ref["predictions"], ref["weights"])
    oracle = PreparedDistribution(ref["predictions"], shifted_weights)
    rng = np.random.default_rng(250991)
    truths = ref["predictions"][rng.choice(len(shifted_weights), 512, p=shifted_weights)]
    stress = {"target_positive_mode_mass": float(shifted_weights[ref["z"][:, 0] > 0].sum()),
              "assumed_posterior_positive_mode_mass": float(ref["weights"][ref["z"][:, 0] > 0].sum()),
              "assumed_posterior_crps": float(np.mean([base.score(truth)["crps"] for truth in truths])),
              "known_shift_oracle_crps": float(np.mean([oracle.score(truth)["crps"] for truth in truths])),
              "note": "Oracle knows the hidden target prior shift; not a deployable baseline. 512 common truth curves, fixed new audit seed."}
    audit = {"original_summary_sha256": before, "likelihood_changed": False, "candidate_tuned": False,
             "semantic_note": "The squared observation is exactly normalized aggregate generation; monthly fuel shares alone carry no amplitude information in this toy. Peak contrast is an additional historical statistic.",
             "checks": checks, "opposite_phase_fuel_energy_max_difference": pair_error,
             "opposite_phase_hourly_cif_mae": float(np.abs(predictions[0] - predictions[1]).mean()),
             "prior_shift_stress": stress,
             "cdf_and_interval_audit_passed": pair_error < 1e-10 and all(c["max_cdf_grid_refinement_error"] < .01 and c["interval90_endpoint_refinement_error_g_per_kwh"] < 1 and c["physical_observation_log_likelihood_error"] < 1e-8 for c in checks.values()),
             "all_quantiles_below_one_g_per_kwh": all(c["max_quantile_grid_refinement_error_g_per_kwh"] < 1 for c in checks.values()),
             "quantile_limitation": "The initial audit's all-quantile <1 criterion failed at the harmful median (1.36 g/kWh); the two-mode CDF is nearly flat around probability .5. Report this diagnostic separately; the CDF and actual 90% interval endpoints converge. Original initial-audit artifact is retained, not overwritten.",
             "audit_scope": "Added after initial gate1/2 results; numerical/semantic validation only. Original outcomes preserved; no hyperparameter selection."}
    (destination / "gate1_audit_completed.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
