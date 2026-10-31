"""Source-held-out, one-month-ahead tail-frequency probe using legal monthly mix."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as source_data  # noqa: E402

PROTOCOL = ROOT / "docs/V68_COARSE_MIX_TAIL_FREQUENCY_FORECAST_PROTOCOL.md"
OUT = ROOT / "results/v68-coarse-mix-tail-frequency-forecast-r2"
TARGET_MONTHS = tuple(f"2026-{m:02d}" for m in range(1, 7))
TAUS = (100.0, 300.0, 600.0, 900.0)
EF = source_data.EF.astype(float)
FUELS = tuple(source_data.FUELS)
SEED = 68027


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def month_features(monthly: pd.DataFrame, region: str, origin: pd.Timestamp) -> np.ndarray | None:
    rows = monthly.loc[monthly.region.eq(region) & monthly.period_end.le(origin)
                       & monthly.available_at.le(origin)].sort_values("period_start")
    if len(rows) < 3:
        return None
    latest = rows.iloc[-1]
    recent = rows.iloc[-3:]
    shares_latest = np.array([latest[f"fuelshare_{f}"] for f in FUELS], dtype=float)
    weights = recent.total_mwh.to_numpy(float)
    shares_3m = np.average(recent[[f"fuelshare_{f}" for f in FUELS]].to_numpy(float),
                           axis=0, weights=weights)
    origin_month = int(origin.month)
    angle = 2 * np.pi * origin_month / 12
    latest_ef = float(shares_latest @ EF)
    return np.concatenate((shares_latest, shares_3m, [latest_ef / 1000.0,
                           np.log1p(float(latest.total_mwh)), np.sin(angle), np.cos(angle)]))


def monthly_labels(hourly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    fuel_cols = [f"generation_{f}" for f in FUELS]
    for region, frame in hourly.items():
        x = frame[fuel_cols].to_numpy(float)
        totals = x.sum(axis=1)
        times = pd.to_datetime(frame.time, utc=True)
        valid = frame.valid.to_numpy(bool) & np.isfinite(x).all(axis=1) & (x >= 0).all(axis=1) & (totals > 0)
        intensities = np.full(len(frame), np.nan)
        intensities[valid] = x[valid] @ EF / totals[valid]
        month_ids = times.dt.strftime("%Y-%m")
        index = pd.DataFrame({"month": month_ids, "intensity": intensities, "valid": valid})
        for month, rows in index.groupby("month", sort=True):
            start = pd.Timestamp(f"{month}-01", tz="UTC")
            stop = start + pd.offsets.MonthBegin(1)
            expected = int((stop - start) / pd.Timedelta(hours=1))
            vals = rows.loc[rows.valid, "intensity"].to_numpy(float)
            if len(vals) / expected < 0.8:
                continue
            record = {"region": region, "month": month, "period_start": start,
                      "period_end": stop, "label_available_at": stop + pd.Timedelta(days=30),
                      "valid_hours": len(vals), "expected_hours": expected}
            for tau in TAUS:
                record[f"rate_tau_{int(tau)}"] = float(np.mean(vals > tau))
                record[f"any_tau_{int(tau)}"] = int(np.any(vals > tau))
            records.append(record)
    return pd.DataFrame(records)


def equal_region_weights(regions: pd.Series) -> np.ndarray:
    counts = regions.value_counts()
    w = regions.map(lambda r: 1.0 / counts[r]).to_numpy(float)
    return w / w.mean()


def fit_predict(train: pd.DataFrame, test_x: np.ndarray, target: str) -> dict[str, float]:
    x = np.stack(train.features.to_list())
    y = train[target].to_numpy(float)
    w = equal_region_weights(train.region)
    month_number = train.period_start.dt.month.to_numpy(int)
    calendar = test_x[-2:]
    query_month = int(round((np.arctan2(calendar[0], calendar[1]) % (2 * np.pi)) * 12 / (2 * np.pi)))
    query_month = 12 if query_month == 0 else query_month
    seasonal = float(np.mean(y[month_number == query_month])) if np.any(month_number == query_month) else float(np.average(y, weights=w))

    # Simple level baseline: latest fixed-EF monthly intensity and calendar.
    one_columns = np.array([16, 17, 18, 19])
    one = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    one.fit(x[:, one_columns], y, ridge__sample_weight=w)
    pred_one = float(one.predict(test_x[one_columns].reshape(1, -1))[0])

    # Full composition + three-month context and calendar, linear and nonlinear.
    full = make_pipeline(StandardScaler(), Ridge(alpha=20.0))
    full.fit(x, y, ridge__sample_weight=w)
    pred_full = float(full.predict(test_x.reshape(1, -1))[0])
    tree = ExtraTreesRegressor(n_estimators=100, min_samples_leaf=12, max_features=0.8,
                               random_state=SEED, n_jobs=1)
    tree.fit(x, y, sample_weight=w)
    pred_tree = float(tree.predict(test_x.reshape(1, -1))[0])
    return {"seasonal_source_mean": float(np.clip(seasonal, 0, 1)),
            "latest_ef_ridge": float(np.clip(pred_one, 0, 1)),
            "full_mix_ridge": float(np.clip(pred_full, 0, 1)),
            "full_mix_extra_trees": float(np.clip(pred_tree, 0, 1))}


def cluster_bootstrap(differences: pd.DataFrame) -> tuple[float, float]:
    regions = sorted(differences.region.unique())
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(5000):
        picked = rng.choice(regions, size=len(regions), replace=True)
        pieces = [differences.loc[differences.region.eq(region)] for region in picked]
        sample = pd.concat(pieces, ignore_index=True)
        draws.append(float(sample["diff"].mean()))
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> None:
    protocol_hash = sha256(PROTOCOL)
    hourly, monthly, release_audits, input_hashes = source_data.prepare_regions()
    labels = monthly_labels(hourly)
    feature_rows = []
    for _, label in labels.iterrows():
        feature = month_features(monthly, str(label.region), label.period_start)
        if feature is None:
            continue
        feature_rows.append({"region": label.region, "month": label.month,
                             "period_start": label.period_start, "period_end": label.period_end,
                             "label_available_at": label.label_available_at,
                             "features": feature, **{k: label[k] for k in label.index if k.startswith(("rate_tau_", "any_tau_"))}})
    panel = pd.DataFrame(feature_rows)
    test_rows = []
    for month in TARGET_MONTHS:
        origin = pd.Timestamp(f"{month}-01", tz="UTC")
        for region in sorted(hourly):
            target = panel.loc[panel.region.eq(region) & panel.month.eq(month)]
            qx = month_features(monthly, region, origin)
            if target.empty or qx is None:
                continue
            target_row = target.iloc[0]
            train = panel.loc[panel.region.ne(region) & panel.period_end.le(origin)
                              & panel.label_available_at.le(origin)]
            if train.region.nunique() < 8 or len(train) < 100:
                raise ValueError(f"Insufficient source training rows for {region}/{month}: {len(train)}")
            for tau in TAUS:
                target_col = f"rate_tau_{int(tau)}"
                actual = float(target_row[target_col])
                predictions = fit_predict(train, qx, target_col)
                for method, prediction in predictions.items():
                    test_rows.append({"region": region, "month": month, "origin": origin.isoformat(),
                                      "tau": int(tau), "method": method, "actual_event_rate": actual,
                                      "actual_any_event": int(target_row[f"any_tau_{int(tau)}"]),
                                      "prediction": prediction, "abs_error": abs(prediction - actual),
                                      "squared_error": (prediction - actual) ** 2,
                                      "training_regions": int(train.region.nunique()),
                                      "training_region_months": int(len(train))})
    detail = pd.DataFrame(test_rows)
    if detail.empty:
        raise ValueError("No eligible source-held-out 2026H1 region-months")
    OUT.mkdir(parents=True, exist_ok=False)
    detail.to_csv(OUT / "region_month_predictions.csv", index=False)
    summary = (detail.groupby(["tau", "method"]).agg(region_months=("month", "count"),
               regions=("region", "nunique"), mean_actual_rate=("actual_event_rate", "mean"),
               actual_positive_region_months=("actual_any_event", "sum"),
               mae=("abs_error", "mean"), rmse=("squared_error", lambda x: float(np.sqrt(x.mean()))))
               .reset_index())
    summary.to_csv(OUT / "summary_by_tau_method.csv", index=False)
    diffs = []
    for tau in TAUS:
        pivot = detail.loc[detail.tau.eq(tau)].pivot(index=["region", "month"], columns="method", values="abs_error").reset_index()
        for method in ("latest_ef_ridge", "full_mix_ridge", "full_mix_extra_trees"):
            frame = pivot[["region", "month", method, "seasonal_source_mean"]].copy()
            frame["diff"] = frame[method] - frame.seasonal_source_mean
            lo, hi = cluster_bootstrap(frame[["region", "diff"]])
            diffs.append({"tau": tau, "method": method, "mae_delta_vs_seasonal": float(frame["diff"].mean()),
                          "cluster_bootstrap_95_low": lo, "cluster_bootstrap_95_high": hi,
                          "improvement_fraction": float((frame[method] < frame.seasonal_source_mean).mean())})
    comparison = pd.DataFrame(diffs)
    comparison.to_csv(OUT / "paired_comparison_vs_seasonal.csv", index=False)
    payload = {"status": "DEVELOPMENT_SOURCE_HELD_OUT_TAIL_FREQUENCY_FORECAST_PROXY_NOT_INDEPENDENT_CONFIRMATION",
               "regions": sorted(detail.region.unique().tolist()), "target_months": list(TARGET_MONTHS),
               "rows": int(len(detail)), "tau_values": list(TAUS),
               "hourly_label_release_lag_assumption_days": 30,
               "monthly_feature_release_policy": "V54 full-month release at period_end+30d; must be <= target month origin",
               "release_audits": release_audits, "input_sha256": input_hashes,
               "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
               "summary": summary.to_dict("records"), "paired_comparison_vs_seasonal": comparison.to_dict("records"),
               "limitations": ["Hourly EIA labels are snapshot-revised and have no verified first-release archive; 30-day label lag is an explicit proxy assumption.",
                              "All 17 regions were opened in earlier work; this is not a new-region confirmation.",
                              "Targets use a fixed-factor EIA generation-side positive-gross-generation proxy, not full-grid consumption CIF.",
                              "Region-month errors are clustered by region; repeated months are not independent regions."]}
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(summary.to_string(index=False))
    print(comparison.to_string(index=False))
    print(f"rows={len(detail)} output={OUT}")


if __name__ == "__main__":
    main()
