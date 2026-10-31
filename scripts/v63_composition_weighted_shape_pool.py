"""Nested source-region tuning of composition-weighted donor fuel shapes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402

SOURCE = data.SOURCE_REGIONS
TARGETS = data.ALL_REGIONS
FUELS = data.FUELS
EF = data.EF
TAUS = (0.05, 0.10, 0.20, 0.40, 0.80, float("inf"))
OUTPUT = ROOT / "results/v63-composition-weighted-shape-pool-r1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def latest_row(monthly: pd.DataFrame, region: str, origin: pd.Timestamp) -> pd.Series | None:
    rows = monthly.loc[monthly.region.eq(region) & monthly.period_start.lt(origin)
                       & monthly.period_end.le(origin) & monthly.available_at.le(origin)]
    return rows.sort_values(["period_start", "available_at"]).iloc[-1] if len(rows) else None


def shares(row: pd.Series) -> np.ndarray:
    x = np.array([float(row[f"fuelshare_{fuel}"]) for fuel in FUELS], dtype=float)
    if not np.isfinite(x).all() or (x < 0).any() or x.sum() <= 0:
        raise ValueError("Bad monthly share row")
    return x / x.sum()


def bray_curtis(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.abs(left - right).sum() / np.maximum((left + right).sum(), 1e-15))


def weights(monthly: pd.DataFrame, target: str, donors: tuple[str, ...],
            origin: pd.Timestamp, tau: float) -> tuple[np.ndarray, np.ndarray]:
    target_row = latest_row(monthly, target, origin)
    if target_row is None:
        raise ValueError(f"No target monthly mix: {target}/{origin}")
    target_share = shares(target_row)
    distances, valid_idx = [], []
    for i, donor in enumerate(donors):
        donor_row = latest_row(monthly, donor, origin)
        if donor_row is not None:
            distances.append(bray_curtis(target_share, shares(donor_row)))
            valid_idx.append(i)
    if not valid_idx:
        raise ValueError(f"No donor monthly mixes: {target}/{origin}")
    d = np.asarray(distances)
    w = np.ones(len(d)) if np.isinf(tau) else np.exp(-d / tau)
    w /= w.sum()
    return target_share, np.column_stack((np.asarray(valid_idx, dtype=int), w))


def main() -> None:
    protocol = ROOT / "docs/V63_COMPOSITION_WEIGHTED_SHAPE_POOL_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, monthly, release_audits, input_hashes = data.prepare_regions()
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Protocol changed during data preparation")

    profile_cache: dict[tuple[str, int, int, str], np.ndarray | None] = {}

    def profile(region: str, month: int, year: int, fuel: str) -> np.ndarray | None:
        key = (region, month, year, fuel)
        if key in profile_cache:
            return profile_cache[key]
        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        selected = frame.loc[times.dt.year.eq(year) & times.dt.month.eq(month)
                             & frame.valid].copy()
        if selected.empty:
            profile_cache[key] = None
            return None
        selected["hour"] = pd.to_datetime(selected.time, utc=True).dt.hour
        x = selected.groupby("hour")[f"generation_{fuel}"].mean().reindex(range(24), fill_value=0).to_numpy(float)
        profile_cache[key] = x / x.sum() if x.sum() > 0 else None
        return profile_cache[key]

    def predict(region: str, donors: tuple[str, ...], origin: pd.Timestamp,
                forecast_year: int, tau: float) -> dict | None:
        target_row = latest_row(monthly, region, origin)
        if target_row is None:
            return None
        target_share, wt = weights(monthly, region, donors, origin, tau)
        used_donors = [donors[int(i)] for i in wt[:, 0]]
        donor_weights = wt[:, 1]
        profiles = np.zeros((24, len(FUELS)))
        fallbacks = []
        for j, fuel in enumerate(FUELS):
            pool, pool_weights = [], []
            for donor, weight in zip(used_donors, donor_weights):
                p = profile(donor, origin.month, forecast_year - 1, fuel)
                if p is not None:
                    pool.append(p)
                    pool_weights.append(float(weight))
            if not pool:
                profiles[:, j] = 1 / 24
                fallbacks.append(fuel)
            else:
                pool_weights = np.asarray(pool_weights, dtype=float)
                pool_weights /= pool_weights.sum()
                profiles[:, j] = np.average(np.stack(pool), axis=0, weights=pool_weights)
                profiles[:, j] /= profiles[:, j].sum()

        frame = hourly[region]
        times = pd.to_datetime(frame.time, utc=True)
        current = frame.loc[times.dt.year.eq(origin.year) & times.dt.month.eq(origin.month)
                            & frame.valid].copy()
        if current.empty:
            return None
        actual = np.stack([current[f"generation_{fuel}"].to_numpy(float) for fuel in FUELS], axis=1)
        valid = np.isfinite(actual).all(axis=1) & (actual >= 0).all(axis=1) & (actual.sum(axis=1) > 0)
        current, actual = current.loc[valid], actual[valid]
        if len(current) < 480:
            return None
        hours = pd.to_datetime(current.time, utc=True).dt.hour.to_numpy()
        parts = profiles[hours] * target_share[None, :]
        prediction = parts @ EF / np.maximum(parts.sum(axis=1), 1e-12)
        truth = actual @ EF / actual.sum(axis=1)
        return {"truth": truth, "prediction": prediction, "residual": truth - prediction,
                "fallbacks": fallbacks}

    def months(year: int) -> list[pd.Timestamp]:
        return [pd.Timestamp(f"{year}-{m:02d}-01", tz="UTC") for m in range(1, 7)]

    # Cache 2024 tuning scores for each outer fold, validation source and bandwidth.
    selected_tau: dict[str, float] = {}
    tuning_records = []
    for outer in TARGETS:
        validation = [r for r in SOURCE if r != outer]
        losses = {}
        for tau in TAUS:
            by_region = []
            for val in validation:
                donors = tuple(r for r in SOURCE if r not in (outer, val))
                preds = [predict(val, donors, m, 2024, tau) for m in months(2024)]
                preds = [p for p in preds if p is not None]
                if preds:
                    by_region.append(float(np.mean(np.concatenate(
                        [np.abs(p["residual"]) for p in preds]))))
            losses[tau] = float(np.mean(by_region)) if by_region else float("inf")
        tau_star = min(TAUS, key=lambda x: (losses[x], -x))
        selected_tau[outer] = tau_star
        for tau, loss in losses.items():
            tuning_records.append({"outer_target": outer, "tau": tau if np.isfinite(tau) else "equal",
                                   "source_macro_2024_h1_mae": loss,
                                   "selected": bool(tau == tau_star)})

    scores = []
    for outer in TARGETS:
        tau = selected_tau[outer]
        cal_regions = [r for r in SOURCE if r != outer]
        cal_residuals = {}
        for cal in cal_regions:
            donors = tuple(r for r in SOURCE if r not in (outer, cal))
            preds = [predict(cal, donors, m, 2025, tau) for m in months(2025)]
            preds = [p for p in preds if p is not None]
            if preds:
                cal_residuals[cal] = np.concatenate([p["residual"] for p in preds])
        if len(cal_residuals) < 9:
            raise ValueError(f"Too few calibration regions for {outer}: {len(cal_residuals)}")
        values, weights_q, source_q = [], [], []
        for residual in cal_residuals.values():
            absolute = np.abs(residual)
            values.extend(absolute.tolist())
            weights_q.extend(np.full(len(absolute), 1 / (len(cal_residuals) * len(absolute))).tolist())
            source_q.append(float(np.quantile(absolute, .90)))
        order = np.argsort(values)
        arr, wts = np.asarray(values)[order], np.asarray(weights_q)[order]
        pooled = float(arr[min(np.searchsorted(np.cumsum(wts), .9 * wts.sum()), len(arr) - 1)])
        max_q = max(source_q)
        donors = tuple(r for r in SOURCE if r != outer)
        for month in months(2026):
            target = predict(outer, donors, month, 2026, tau)
            if target is None:
                continue
            for calibration, radius in (("source_equal_pooled_q90", pooled), ("max_source_q90", max_q)):
                y, pred = target["truth"], target["prediction"]
                lo, hi = pred - radius, pred + radius
                winkler = 2 * radius + 20 * np.maximum(lo - y, 0) + 20 * np.maximum(y - hi, 0)
                scores.append({"target_region": outer, "month": str(month.date()), "tau": tau if np.isfinite(tau) else "equal",
                               "calibration": calibration, "n_hours": len(y),
                               "point_mae": float(np.mean(np.abs(y - pred))),
                               "bias": float(np.mean(y - pred)),
                               "coverage": float(np.mean((y >= lo) & (y <= hi))),
                               "width": 2 * radius, "winkler": float(np.mean(winkler)),
                               "fallback_fuels": ";".join(target["fallbacks"]),
                               "calibration_regions": len(cal_residuals)})

    frame = pd.DataFrame(scores)
    tuning = pd.DataFrame(tuning_records)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    frame.to_csv(OUTPUT / "region_month_scores.csv", index=False)
    tuning.to_csv(OUTPUT / "tuning_scores.csv", index=False)
    region = (frame.groupby(["calibration", "target_region"])
              .apply(lambda part: pd.Series({
                  "coverage": np.average(part.coverage, weights=part.n_hours),
                  "point_mae": np.average(part.point_mae, weights=part.n_hours),
                  "bias": np.average(part.bias, weights=part.n_hours),
                  "width": np.average(part.width, weights=part.n_hours),
                  "winkler": np.average(part.winkler, weights=part.n_hours),
                  "months": part.month.nunique()}), include_groups=False).reset_index())
    region.to_csv(OUTPUT / "region_scores.csv", index=False)
    macro = (region.groupby("calibration")
             .agg(regions=("target_region", "nunique"), macro_coverage=("coverage", "mean"),
                  worst_coverage=("coverage", "min"), point_mae=("point_mae", "mean"),
                  bias=("bias", "mean"), mean_width=("width", "mean"),
                  winkler=("winkler", "mean"), mean_months=("months", "mean")).reset_index())
    macro.to_csv(OUTPUT / "macro.csv", index=False)
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Protocol changed during the run")
    tau_json = {region: (value if np.isfinite(value) else "equal")
                for region, value in selected_tau.items()}
    summary = {"status": "DEVELOPMENT_ONLY_ALL_REGIONS_PREVIOUSLY_OPENED",
               "target_regions": list(TARGETS), "source_regions": list(SOURCE),
               "tuning_period": "2024 H1", "calibration_period": "2025 H1", "target_period": "2026 H1",
               "selected_tau": tau_json, "candidate_tau_grid": [x if np.isfinite(x) else "equal" for x in TAUS],
               "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__).resolve()),
               "input_sha256": input_hashes, "release_audit": release_audits,
               "macro": macro.to_dict("records")}
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print("selected_tau:", json.dumps(tau_json, sort_keys=True))


if __name__ == "__main__":
    main()
