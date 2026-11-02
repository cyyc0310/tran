"""Strict five-group pseudo-target conditional stacking, nine-country LOCO."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

import run_energy_charts_nine_country_cn5_loco as nine
from transcif.research import multicountry_cn5 as model
from transcif.research.monthly_adaptation import weighted_quantile

ROOT = nine.ROOT
COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/PSEUDOTARGET_COARSE_CONDITIONAL_STACK_CN5_PROTOCOL_20261101.md"
GATE_FEATURES = (*model.GROUPS, "thermal_mean", "thermal_std", "thermal_change",
                 "total_cv", "observations")
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gate_weights(x: np.ndarray, beta: np.ndarray, mean: np.ndarray,
                 scale: np.ndarray, conditional: bool) -> np.ndarray:
    design = (np.column_stack([np.ones(len(x)), (x - mean) / scale])
              if conditional else np.ones((len(x), 1)))
    logits = design @ beta
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_gate(x: np.ndarray, expert: np.ndarray, y: np.ndarray, groups: np.ndarray,
             penalty: float, conditional: bool) -> dict:
    x, expert, y = np.asarray(x, float), np.asarray(expert, float), np.asarray(y, float) / 1000.
    if len(x) != len(expert) or len(y) != len(x) or not np.isfinite(expert).all():
        raise ValueError("invalid pseudo-target gate training arrays")
    mean = np.zeros(x.shape[1]); scale = np.ones(x.shape[1])
    if conditional:
        country_weights = model.weights(pd.DataFrame({"country": groups, "region": groups}))
        mean = np.sum(x * country_weights[:, None], axis=0)
        scale = np.sqrt(np.sum((x - mean) ** 2 * country_weights[:, None], axis=0))
        scale[scale < 1e-8] = 1.
        design = np.column_stack([np.ones(len(x)), (x - mean) / scale])
    else:
        design = np.ones((len(x), 1))
    row_weights = model.weights(pd.DataFrame({"country": groups, "region": groups}))
    _, n_experts = expert.shape
    n_features = design.shape[1]

    def objective(flat):
        beta = flat.reshape(n_features, n_experts)
        logits = design @ beta
        logits -= logits.max(axis=1, keepdims=True)
        gate = np.exp(logits)
        gate /= gate.sum(axis=1, keepdims=True)
        point = np.sum(gate * expert / 1000., axis=1)
        residual = point - y
        value = float(np.sum(row_weights * residual ** 2))
        if conditional and n_features > 1:
            value += .5 * penalty * float(np.sum(beta[1:] ** 2))
        dlogit = (2 * row_weights * residual)[:, None] * gate * (expert / 1000. - point[:, None])
        grad = design.T @ dlogit
        if conditional and n_features > 1:
            grad[1:] += penalty * beta[1:]
        return value, grad.ravel()

    result = minimize(objective, np.zeros(n_features * n_experts), method="L-BFGS-B", jac=True,
                      options={"maxiter": 250, "ftol": 1e-10, "maxls": 30})
    if not result.success and result.status not in (1,):
        raise RuntimeError(f"gate optimizer failed: {result.message}")
    return {"beta": result.x.reshape(n_features, n_experts), "mean": mean, "scale": scale,
            "conditional": conditional, "penalty": float(penalty),
            "optimizer_message": str(result.message), "objective": float(result.fun)}


def predict_gate(gate: dict, x: np.ndarray, expert: np.ndarray) -> np.ndarray:
    alpha = gate_weights(x, gate["beta"], gate["mean"], gate["scale"], gate["conditional"])
    return np.sum(alpha * expert, axis=1)


def target_rows(prepared: dict, country: str, year: int, half: int | None):
    raw, features = prepared[country]
    query = features.loc[features.year.eq(year)].copy()
    if half == 1:
        query = query.loc[query.clock.dt.month.le(6)].copy()
    elif half == 2:
        query = query.loc[query.clock.dt.month.gt(6)].copy()
    if query.empty:
        raise ValueError(f"empty query: {country} {year} half={half}")
    fuel = nine.fuel_for(raw, query)
    labels = fuel @ model.FACTORS / fuel.sum(axis=1)
    return query, labels


def source_fit(prepared: dict, sources: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
    train = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([nine.fuel_for(prepared[c][0], part) for c, part in zip(sources, parts)])
    coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
                             for c, part in zip(sources, parts)])
    if train.empty or len(train) != len(fuel) or len(train) != len(coarse):
        raise ValueError(f"misaligned source fit for {sources}")
    return model.fit(train, fuel, coarse)


def oof_pseudo_predictions(prepared: dict, sources: list[str]) -> dict:
    result = {}
    for pseudo in sources:
        models = source_fit(prepared, [c for c in sources if c != pseudo])
        rows = {}
        for half in (1, 2):
            query, y = target_rows(prepared, pseudo, 2023, half)
            rows[half] = {"query": query, "y": y, "point": model.predict(models, query),
                          "x": query[list(GATE_FEATURES)].to_numpy(float)}
        result[pseudo] = rows
        print("pseudo target crossfit complete", pseudo.upper(), flush=True)
    return result


def tune_penalty(oof: dict) -> tuple[float, pd.DataFrame]:
    rows = []
    for penalty in PENALTIES:
        fold_mae = []
        for heldout in oof:
            train_countries = [c for c in oof if c != heldout]
            train_parts = [oof[c][1] for c in train_countries]
            x = np.concatenate([p["x"] for p in train_parts])
            y = np.concatenate([p["y"] for p in train_parts])
            experts = np.concatenate([np.column_stack([p["point"][m] for m in model.METHODS])
                                      for p in train_parts])
            groups = np.concatenate([np.repeat(c, len(p["y"]))
                                     for c, p in zip(train_countries, train_parts)])
            gate = fit_gate(x, experts, y, groups, penalty, True)
            test = oof[heldout][1]
            pred = predict_gate(gate, test["x"],
                                np.column_stack([test["point"][m] for m in model.METHODS]))
            fold_mae.append(float(np.abs(pred - test["y"]).mean()))
        rows.append({"penalty": penalty, "inner_country_mae": float(np.mean(fold_mae)),
                     "fold_scores": json.dumps(fold_mae)})
    table = pd.DataFrame(rows).sort_values(["inner_country_mae", "penalty"])
    return float(table.iloc[0].penalty), table


def daily_shape(query: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    daily = pd.DataFrame({"truth": y, "prediction": p,
                          "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["truth", "prediction"]] - daily.groupby("day")[["truth", "prediction"]].transform("mean")
    return float(np.abs(centered.truth - centered.prediction).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_paths = [Path(__file__).resolve(), Path(nine.__file__).resolve(), PROTOCOL,
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n", encoding="utf-8")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n",
                                         encoding="utf-8")

    seals, prediction_sets, selections = [], {}, []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        oof = oof_pseudo_predictions(prepared, sources)
        penalty, penalty_table = tune_penalty(oof)
        penalty_table.to_csv(out / f"{target.upper()}_inner_penalty_cv.csv", index=False)
        h1 = [oof[c][1] for c in sources]
        h2 = [oof[c][2] for c in sources]
        x1, y1 = np.concatenate([p["x"] for p in h1]), np.concatenate([p["y"] for p in h1])
        expert1 = np.concatenate([np.column_stack([p["point"][m] for m in model.METHODS]) for p in h1])
        groups1 = np.concatenate([np.repeat(c, len(p["y"])) for c, p in zip(sources, h1)])
        gate = fit_gate(x1, expert1, y1, groups1, penalty, True)
        static_gate = fit_gate(x1, expert1, y1, groups1, 0., False)
        model_mae = {m: float(np.mean([np.abs(oof[c][1]["point"][m] - oof[c][1]["y"]).mean()
                                       for c in sources])) for m in model.METHODS}
        selected = min(model_mae, key=model_mae.get)
        selections.append({"target": target.upper(), "selected_method": selected,
                           "selected_inner_mae": model_mae[selected], "penalty": penalty,
                           "gate_objective": gate["objective"]})
        h2_y = np.concatenate([p["y"] for p in h2])
        h2_x = np.concatenate([p["x"] for p in h2])
        h2_expert = np.concatenate([np.column_stack([p["point"][m] for m in model.METHODS]) for p in h2])
        h2_groups = np.concatenate([np.repeat(c, len(p["y"])) for c, p in zip(sources, h2)])
        conditional_alpha = gate_weights(h2_x, gate["beta"], gate["mean"], gate["scale"], True)
        static_alpha = gate_weights(h2_x, static_gate["beta"], static_gate["mean"],
                                    static_gate["scale"], False)
        h2_points = {m: h2_expert[:, i] for i, m in enumerate(model.METHODS)}
        h2_points["uniform_ensemble"] = h2_expert.mean(axis=1)
        h2_points["static_stack"] = np.sum(static_alpha * h2_expert, axis=1)
        h2_points["conditional_stack"] = np.sum(conditional_alpha * h2_expert, axis=1)
        h2_points["pseudo_selected"] = h2_expert[:, model.METHODS.index(selected)]
        cal_weights = model.weights(pd.DataFrame({"country": h2_groups, "region": h2_groups}))
        radii = {m: float(weighted_quantile(np.abs(h2_y - p), cal_weights, .9))
                 for m, p in h2_points.items()}

        base_models = source_fit(prepared, sources)
        raw, query = prepared[target]
        query = query.loc[query.year.eq(2024)].copy()
        candidates = model.predict(base_models, query)
        expert = np.column_stack([candidates[m] for m in model.METHODS])
        xq = query[list(GATE_FEATURES)].to_numpy(float)
        points = dict(candidates)
        points["uniform_ensemble"] = expert.mean(axis=1)
        points["static_stack"] = predict_gate(static_gate, xq, expert)
        points["conditional_stack"] = predict_gate(gate, xq, expert)
        points["pseudo_selected"] = candidates[selected]
        payload = {"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(), **points}
        for method, prediction in points.items():
            payload[f"{method}_lo"] = prediction - radii[method]
            payload[f"{method}_hi"] = prediction + radii[method]
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame(payload).to_csv(path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "path": path.name, "sha256": sha(path),
                      "hours": len(query), "source_countries": [c.upper() for c in sources]})
        prediction_sets[target] = (query, path)
        print(f"{target.upper()} sealed {len(query)} hours; pseudo-selected {selected}; penalty {penalty}",
              flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    (out / "pseudo_target_selection.json").write_text(json.dumps(selections, indent=2) + "\n",
                                                     encoding="utf-8")
    if any(sha(Path(path)) != expected for path, expected in input_hashes.items()):
        raise RuntimeError("frozen research input changed during execution")
    if any(sha(Path(path)) != expected for path, expected in code_hashes.items()):
        raise RuntimeError("frozen code/protocol changed during execution")

    rows = []
    for target, (query, path) in prediction_sets.items():
        raw = prepared[target][0]
        fuel = nine.fuel_for(raw, query)
        truth = fuel @ model.FACTORS / fuel.sum(axis=1)
        scored = pd.read_csv(path)
        methods = [*model.METHODS, "uniform_ensemble", "static_stack", "conditional_stack", "pseudo_selected"]
        for method in methods:
            pred = scored[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - pred[mask]
                lo = scored[f"{method}_lo"].to_numpy(float)[mask]
                hi = scored[f"{method}_hi"].to_numpy(float)[mask]
                rows.append({"target_country": target.upper(), "method": method, "month": int(month),
                             "hours": int(mask.sum()), "mae": float(np.abs(error).mean()),
                             "shape_mae": daily_shape(query.loc[mask], truth[mask], pred[mask]),
                             "coverage90": float(((truth[mask] >= lo) & (truth[mask] <= hi)).mean()),
                             "width90": float(np.mean(hi - lo)),
                             "interval_score90": float(np.mean(hi - lo + 20 * np.maximum(lo - truth[mask], 0)
                                                                 + 20 * np.maximum(truth[mask] - hi, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    fields = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[fields].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    country.groupby("method")[fields].mean().reset_index().to_csv(out / "macro_metrics.csv", index=False)
    print(pd.read_csv(out / "macro_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
