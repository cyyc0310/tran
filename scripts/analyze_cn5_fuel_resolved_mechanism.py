"""Post-hoc mechanism diagnostic for the strict-CN5 fuel-share response head."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_cn5_fuel_resolved_response as runner  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sealed", type=Path,
                        default=ROOT / "results/cn5-fuel-resolved-response-r1")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sealed = args.sealed.resolve()
    seal_path = sealed / "prediction_seal.json"
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    for item in seal:
        if sha(sealed / item["path"]) != item["sha256"]:
            raise RuntimeError(f"sealed point prediction changed for {item['target']}")

    prepared = {code: nine.prepare_country(code)[:2] for code in runner.COUNTRIES}
    summary_rows, hourly_rows = [], []
    for target in runner.COUNTRIES:
        sources = [code for code in runner.COUNTRIES if code != target]
        train_parts = [prepared[code][1].loc[prepared[code][1].year.eq(2022)]
                       for code in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([
            runner.fuel_for(prepared[code][0], part)
            for code, part in zip(sources, train_parts)
        ])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[code][0].iloc[part.raw_row.to_numpy()])
            for code, part in zip(sources, train_parts)
        ])
        base = model.fit(train, train_fuel, train_coarse)
        head = runner.fit_fuelshare(train, train_fuel)

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        target_fuel = runner.fuel_for(raw, query)
        target_total = target_fuel[:, :4].sum(axis=1)
        active = target_total > 0
        truth_share = np.divide(target_fuel[:, :4], target_total[:, None],
                                out=np.full((len(target_fuel), 4), np.nan),
                                where=target_total[:, None] > 0.)
        learned_share = np.column_stack([
            learner.predict(query[list(model.HISTORY)]) for learner in head["heads"]
        ])
        learned_share = np.maximum(learned_share, 0.)
        learned_total = learned_share.sum(axis=1)
        learned_share = np.divide(
            learned_share, learned_total[:, None],
            out=np.tile(head["prior"], (len(query), 1)), where=learned_total[:, None] > 0.
        )
        prior_share = np.tile(head["prior"], (len(query), 1))

        coarse = query[list(model.GROUPS)].to_numpy(float)
        profile = base["profile"][query.clock.dt.hour.to_numpy()]
        coarse_hour = coarse * profile
        total_hour = coarse_hour.sum(axis=1)
        thermal_weight = np.divide(coarse_hour[:, 0], total_hour,
                                   out=np.zeros(len(query)), where=total_hour > 0.)
        learned_y = (learned_share @ runner.FUEL_FACTORS) * thermal_weight
        prior_y = (prior_share @ runner.FUEL_FACTORS) * thermal_weight
        truth = runner.truth_for(raw, query)
        sealed_prediction = pd.read_csv(sealed / f"{target.upper()}_heldout_predictions.csv.gz")
        expected_time = raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        if not np.array_equal(sealed_prediction.time.astype(str).to_numpy(), expected_time):
            raise RuntimeError(f"sealed timestamp mismatch for {target}")
        if len(sealed_prediction) != len(query):
            raise RuntimeError(f"sealed row count mismatch for {target}")

        for name, predicted in (("learned", learned_share), ("source_mean", prior_share)):
            per_fuel = np.nanmean(np.abs(truth_share[active] - predicted[active]), axis=0)
            summary_rows.append({
                "target_country": target.upper(), "composition_model": name,
                "coal_share_mae": float(per_fuel[0]), "gas_share_mae": float(per_fuel[1]),
                "oil_share_mae": float(per_fuel[2]), "other_thermal_share_mae": float(per_fuel[3]),
                "mean_fuel_share_mae": float(per_fuel.mean()),
                "effective_factor_mae": float(np.mean(np.abs(
                    (predicted @ runner.FUEL_FACTORS)[active]
                    - (truth_share @ runner.FUEL_FACTORS)[active]))),
                "hourly_proxy_mae": float(np.mean(np.abs(truth[active] -
                    (learned_y if name == "learned" else prior_y)[active]))),
            })

        hourly_rows.extend({
            "target_country": target.upper(), "time": time,
            "truth_thermal_share_coal": t[0], "truth_thermal_share_gas": t[1],
            "truth_thermal_share_oil": t[2], "truth_thermal_share_other": t[3],
            "pred_thermal_share_coal": p[0], "pred_thermal_share_gas": p[1],
            "pred_thermal_share_oil": p[2], "pred_thermal_share_other": p[3],
            "prior_thermal_share_coal": q[0], "prior_thermal_share_gas": q[1],
            "prior_thermal_share_oil": q[2], "prior_thermal_share_other": q[3],
            "truth_proxy": y, "fuelshare_prediction": fy, "source_mean_prediction": py,
            "sealed_fuelshare_prediction": sy,
            "sealed_ratio_prediction": ry,
        } for time, t, p, q, y, fy, py, sy, ry in zip(
            expected_time, truth_share, learned_share, prior_share, truth, learned_y,
            prior_y, sealed_prediction.fuelshare_history.to_numpy(),
            sealed_prediction.ratio_recomposition.to_numpy()) if np.all(np.isfinite(t)))

    pd.DataFrame(summary_rows).to_csv(out / "country_mechanism_metrics.csv", index=False)
    pd.DataFrame(hourly_rows).to_csv(out / "hourly_mechanism.csv.gz", index=False,
                                     compression="gzip")
    (out / "status.json").write_text(json.dumps({
        "status": "POSTHOC_MECHANISM_DIAGNOSTIC",
        "target_year": 2024,
        "target_labels_previously_viewed": True,
        "blind_confirmation": False,
        "diagnostic": "Compare learned source-trained thermal fuel shares with fixed source mean shares",
        "sealed_prediction_manifest_sha256": sha(seal_path),
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.DataFrame(summary_rows).groupby("composition_model").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
