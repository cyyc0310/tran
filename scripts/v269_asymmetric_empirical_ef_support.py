"""V269: apply asymmetric 90%-generation fuel-factor support to target mixes."""
from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402

PROTOCOL = ROOT / "docs/V269_ASYMMETRIC_EMPIRICAL_EF_SUPPORT_PROTOCOL.md"
ARCHIVE = ROOT / "data/external-cems/epacems-2025.zip"
INPUT_CERT = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1/threshold_side_certificate.csv"
RADIUS_REF = ROOT / "results/v267-cems-calibrated-threshold-gate-r2/cems_state_quarter_r90.csv"
OUT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2"
THRESHOLD = 500.0001
TON_TO_G = 907.18474
FUEL_GROUP = {"Coal": "coal", "Petroleum Coke": "coal",
              "Pipeline Natural Gas": "gas", "Natural Gas": "gas",
              "Residual Oil": "oil", "Diesel Oil": "oil", "Other Oil": "oil"}
USECOLS = ["State", "Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)",
           "CO2 Mass Measure Indicator", "Primary Fuel Type"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def shortest_weighted_interval(values: np.ndarray, weights: np.ndarray, mass: float = .90) -> tuple[float, float, float]:
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    target = mass * weights.sum()
    prefix = np.concatenate(([0.0], np.cumsum(weights)))
    best = None
    right = 0
    for left in range(len(values)):
        right = max(right, left)
        while right < len(values) and prefix[right + 1] - prefix[left] < target:
            right += 1
        if right == len(values):
            break
        width = values[right] - values[left]
        candidate = (width, values[left], values[right], left, right)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise RuntimeError("could not construct weighted support interval")
    _, lo, hi, left, right = best
    covered = float(weights[left:right + 1].sum() / weights.sum())
    return float(lo), float(hi), covered


def load_state_factors() -> pd.DataFrame:
    totals: dict[tuple[str, int, str], list[float]] = {}
    with zipfile.ZipFile(ARCHIVE) as archive:
        members = sorted(name for name in archive.namelist() if name.endswith(".csv"))
        for member in members:
            match = re.search(r"q([1-4])", member, flags=re.IGNORECASE)
            if not match:
                continue
            quarter = int(match.group(1))
            with archive.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=USECOLS, chunksize=500_000, low_memory=False):
                    indicator = chunk["CO2 Mass Measure Indicator"].astype("string").str.upper().str.strip()
                    chunk = chunk[indicator.isin({"MEASURED", "MEASURE"})].copy()
                    if chunk.empty:
                        continue
                    for col in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)"):
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                    chunk["fuel_group"] = chunk["Primary Fuel Type"].map(FUEL_GROUP)
                    valid = (chunk["Operating Time"].gt(0) & chunk["Gross Load (MW)"].gt(0)
                             & chunk["CO2 Mass (short tons)"].gt(0) & chunk["State"].notna()
                             & chunk["fuel_group"].notna())
                    chunk = chunk.loc[valid, ["State", "Operating Time", "Gross Load (MW)",
                                              "CO2 Mass (short tons)", "fuel_group"]]
                    if chunk.empty:
                        continue
                    chunk["mwh"] = chunk["Operating Time"] * chunk["Gross Load (MW)"]
                    chunk["tons"] = chunk["Operating Time"] * chunk["CO2 Mass (short tons)"]
                    grouped = chunk.groupby(["State", "fuel_group"], observed=True)[["mwh", "tons"]].sum()
                    for (state, group), row in grouped.iterrows():
                        key = (str(state), quarter, str(group))
                        item = totals.setdefault(key, [0.0, 0.0])
                        item[0] += float(row.mwh); item[1] += float(row.tons)
    frame = pd.DataFrame([{"state": state, "quarter": q, "fuel_group": group,
                           "mwh": mwh, "ef_g_kwh": tons / mwh * TON_TO_G}
                          for (state, q, group), (mwh, tons) in totals.items() if mwh > 0])
    return frame


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    state = load_state_factors()
    interval_rows = []
    for (quarter, group), subset in state.groupby(["quarter", "fuel_group"]):
        lo, hi, covered = shortest_weighted_interval(subset.ef_g_kwh.to_numpy(), subset.mwh.to_numpy())
        interval_rows.append({"quarter": int(quarter), "fuel_group": group,
                              "states": int(len(subset)), "generation_mwh": float(subset.mwh.sum()),
                              "support_low_g_kwh": lo, "support_high_g_kwh": hi,
                              "support_width_g_kwh": hi - lo, "generation_share_covered": covered})
    intervals = pd.DataFrame(interval_rows)
    interval_map = {(int(r.quarter), r.fuel_group): (float(r.support_low_g_kwh), float(r.support_high_g_kwh))
                    for r in intervals.itertuples(index=False)}
    r90ref = pd.read_csv(RADIUS_REF)
    r90map = {(int(r.quarter), r.fuel_group): (float(r.pooled_ef_g_kwh), float(r.r90_state_quarter))
              for r in r90ref.itertuples(index=False)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    certs = pd.read_csv(INPUT_CERT)
    if len(certs) != 16 or certs.duplicated(["region", "year", "month"]).any():
        raise RuntimeError("expected frozen 16 BA-month cohort")
    rows = []
    for cert in certs.itertuples(index=False):
        entry = series[(cert.region, int(cert.year), int(cert.month))]
        amounts = np.asarray(entry["fuel_totals"], float)
        labels = np.asarray(partial.EF, float)
        fuel_amounts = {1000.0: float(amounts[labels == 1000.0].sum()),
                        800.0: float(amounts[labels == 800.0].sum()),
                        400.0: float(amounts[labels == 400.0].sum())}
        total = float(amounts.sum())  # include non-fossil generation in the CIF denominator
        quarter = (int(cert.month) - 1) // 3 + 1
        group_for_ef = {1000.0: "coal", 800.0: "oil", 400.0: "gas"}
        low_sum = high_sum = sym_low_sum = sym_high_sum = 0.0
        for ef, mwh in fuel_amounts.items():
            group = group_for_ef[ef]
            lo, hi = interval_map[(quarter, group)]
            low_sum += mwh * lo; high_sum += mwh * hi
            center, radius = r90map[(quarter, group)]
            sym_low_sum += mwh * center * max(0.0, 1.0 - radius)
            sym_high_sum += mwh * center * (1.0 + radius)
        low, high = low_sum / total, high_sum / total
        sym_low, sym_high = sym_low_sum / total, sym_high_sum / total
        rows.append({"region": cert.region, "year": int(cert.year), "month": int(cert.month),
                     "quarter": quarter, "empirical_low_mean": low, "empirical_high_mean": high,
                     "symmetric_cems_low_mean": sym_low, "symmetric_cems_high_mean": sym_high,
                     "empirical_interval_crosses": low < THRESHOLD <= high,
                     "symmetric_cems_interval_crosses": sym_low < THRESHOLD <= sym_high,
                     "uniform_project_10pct_crosses": float(cert.critical_relative_ef_tolerance_to_reach_threshold) <= .10})
    cases = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    intervals.to_csv(OUT / "cems_asymmetric_support_intervals.csv", index=False)
    cases.to_csv(OUT / "ba_month_gate_comparison.csv", index=False)
    summary = pd.DataFrame([
        {"contract": "CEMS weighted-shortest 90% interval", "ba_months": len(cases),
         "threshold_crossings": int(cases.empirical_interval_crosses.sum()),
         "mean_low": cases.empirical_low_mean.mean(), "mean_high": cases.empirical_high_mean.mean()},
        {"contract": "CEMS symmetric r90", "ba_months": len(cases),
         "threshold_crossings": int(cases.symmetric_cems_interval_crosses.sum()),
         "mean_low": cases.symmetric_cems_low_mean.mean(), "mean_high": cases.symmetric_cems_high_mean.mean()},
        {"contract": "project nominal uniform ±10%", "ba_months": len(cases),
         "threshold_crossings": int(cases.uniform_project_10pct_crosses.sum()),
         "mean_low": np.nan, "mean_high": np.nan},
    ])
    summary.to_csv(OUT / "summary.csv", index=False)
    meta = {"status": "asymmetric_cems_empirical_factor_support_transport_stress_not_china_ef_estimate",
            "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
            "cems_archive_sha256": sha256(ARCHIVE), "radius_reference_sha256": sha256(RADIUS_REF),
            "certificate_sha256": sha256(INPUT_CERT), "eia_input_hashes": input_hashes,
            "threshold_g_kwh": THRESHOLD, "support_mass": .90, "ba_months": len(cases),
            "empirical_interval_crossings": int(cases.empirical_interval_crosses.sum()),
            "symmetric_r90_crossings": int(cases.symmetric_cems_interval_crosses.sum()),
            "uniform_10pct_crossings": int(cases.uniform_project_10pct_crosses.sum()),
            "warning": "US CEMS state-quarter factors transferred to already-open EIA BA mixes; not China factor uncertainty or prediction coverage"}
    (OUT / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(intervals.to_string(index=False))
    print(summary.to_string(index=False))
    print(cases[["region", "month", "empirical_low_mean", "empirical_high_mean",
                 "empirical_interval_crosses", "uniform_project_10pct_crosses"]].to_string(index=False))


if __name__ == "__main__":
    main()
