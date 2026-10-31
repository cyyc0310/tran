"""V267: calibrate the threshold-side gate with measured CEMS heterogeneity."""
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

PROTOCOL = ROOT / "docs/V267_CEMS_CALIBRATED_THRESHOLD_GATE_PROTOCOL.md"
ARCHIVE = ROOT / "data/external-cems/epacems-2025.zip"
OUT = ROOT / "results/v267-cems-calibrated-threshold-gate-r2"
INPUT_CERT = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1/threshold_side_certificate.csv"
NOMINAL_EF = {"coal": 1000.0, "oil": 800.0, "gas": 400.0}
THRESHOLD = 500.0001
TON_TO_G = 907.18474
USECOLS = ["State", "Date", "Operating Time", "Gross Load (MW)",
           "CO2 Mass (short tons)", "CO2 Mass Measure Indicator", "Primary Fuel Type"]
FUEL_GROUP = {
    "Coal": "coal", "Petroleum Coke": "coal",
    "Pipeline Natural Gas": "gas", "Natural Gas": "gas",
    "Residual Oil": "oil", "Diesel Oil": "oil", "Other Oil": "oil",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cems_state_quarter_factors() -> tuple[pd.DataFrame, dict]:
    totals: dict[tuple[str, int, str], list[float]] = {}
    included_rows = 0
    with zipfile.ZipFile(ARCHIVE) as archive:
        for member in sorted(name for name in archive.namelist() if name.endswith(".csv")):
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
                    for column in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)"):
                        chunk[column] = pd.to_numeric(chunk[column], errors="coerce")
                    chunk["fuel_group"] = chunk["Primary Fuel Type"].map(FUEL_GROUP)
                    valid = (chunk["Operating Time"].gt(0) & chunk["Gross Load (MW)"].gt(0)
                             & chunk["CO2 Mass (short tons)"].gt(0) & chunk["State"].notna()
                             & chunk["fuel_group"].notna())
                    chunk = chunk.loc[valid, ["State", "Operating Time", "Gross Load (MW)",
                                              "CO2 Mass (short tons)", "fuel_group"]]
                    if chunk.empty:
                        continue
                    included_rows += len(chunk)
                    chunk["mwh"] = chunk["Operating Time"] * chunk["Gross Load (MW)"]
                    chunk["tons"] = chunk["Operating Time"] * chunk["CO2 Mass (short tons)"]
                    grouped = chunk.groupby(["State", "fuel_group"], observed=True)[["mwh", "tons"]].sum()
                    for (state, group), row in grouped.iterrows():
                        key = (str(state), quarter, str(group))
                        acc = totals.setdefault(key, [0.0, 0.0])
                        acc[0] += float(row["mwh"])
                        acc[1] += float(row["tons"])
    rows = [{"state": state, "quarter": quarter, "fuel_group": group,
             "mwh": mwh, "tons": tons, "ef_g_kwh": tons / mwh * TON_TO_G}
            for (state, quarter, group), (mwh, tons) in totals.items() if mwh > 0]
    state = pd.DataFrame(rows)
    pooled = state.groupby(["quarter", "fuel_group"], as_index=False).agg(
        pooled_mwh=("mwh", "sum"), pooled_tons=("tons", "sum"))
    pooled["pooled_ef_g_kwh"] = pooled.pooled_tons / pooled.pooled_mwh * TON_TO_G
    state = state.merge(pooled[["quarter", "fuel_group", "pooled_ef_g_kwh"]],
                        on=["quarter", "fuel_group"], validate="many_to_one")
    state["relative_deviation"] = (state.ef_g_kwh / state.pooled_ef_g_kwh - 1.0).abs()
    radii = []
    for (quarter, group), frame in state.groupby(["quarter", "fuel_group"]):
        frame = frame.sort_values("relative_deviation")
        threshold_mwh = 0.90 * frame.mwh.sum()
        selected = frame.loc[frame.mwh.cumsum().ge(threshold_mwh)].iloc[0]
        radii.append({"quarter": int(quarter), "fuel_group": group,
                      "pooled_ef_g_kwh": float(frame.pooled_ef_g_kwh.iloc[0]),
                      "states": int(len(frame)), "generation_mwh": float(frame.mwh.sum()),
                      "r90_state_quarter": float(selected.relative_deviation),
                      "generation_share_inside_r90": float(frame.loc[
                          frame.relative_deviation.le(selected.relative_deviation), "mwh"].sum() / frame.mwh.sum())})
    return pd.DataFrame(radii), {"included_measured_rows": included_rows, "state_quarter_groups": len(state)}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    radii, cems_meta = cems_state_quarter_factors()
    radius_map = {(int(row.quarter), str(row.fuel_group)): float(row.r90_state_quarter)
                  for row in radii.itertuples(index=False)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    certificates = pd.read_csv(INPUT_CERT)
    if len(certificates) != 16 or certificates.duplicated(["region", "year", "month"]).any():
        raise RuntimeError("expected the frozen V240 16 BA-month cohort")
    rows = []
    for cert in certificates.itertuples(index=False):
        entry = series[(cert.region, int(cert.year), int(cert.month))]
        quarter = (int(cert.month) - 1) // 3 + 1
        amounts = np.asarray(entry["fuel_totals"], float)
        labels = np.asarray(partial.EF, float)
        grouped = {ef: float(amounts[labels == ef].sum()) for ef in (1000.0, 800.0, 400.0)}
        total = float(amounts.sum())  # include non-fossil generation in the CIF denominator
        if total <= 0:
            raise RuntimeError(f"empty fuel margin: {cert.region}/{cert.year}/{cert.month}")
        group_lookup = {1000.0: "coal", 800.0: "oil", 400.0: "gas"}
        lower_emissions = upper_emissions = nominal_emissions = 0.0
        for ef, amount in grouped.items():
            radius = radius_map[(quarter, group_lookup[ef])]
            nominal_emissions += amount * ef
            lower_emissions += amount * ef * max(0.0, 1.0 - radius)
            upper_emissions += amount * ef * (1.0 + radius)
        low_mean, high_mean = lower_emissions / total, upper_emissions / total
        r90_gate = low_mean < THRESHOLD <= high_mean
        common10_gate = float(cert.critical_relative_ef_tolerance_to_reach_threshold) <= 0.10
        rows.append({"region": cert.region, "year": int(cert.year), "month": int(cert.month),
                     "quarter": quarter, "n_hours": int(cert.n_hours),
                     "nominal_mean_g_kwh": nominal_emissions / total,
                     "r90_low_mean_g_kwh": low_mean, "r90_high_mean_g_kwh": high_mean,
                     "coal_r90": radius_map[(quarter, "coal")],
                     "oil_r90": radius_map[(quarter, "oil")],
                     "gas_r90": radius_map[(quarter, "gas")],
                     "crosses_threshold_at_500": bool(r90_gate),
                     "crosses_under_uniform_10pct": bool(common10_gate),
                     "critical_uniform_tolerance": float(cert.critical_relative_ef_tolerance_to_reach_threshold),
                     "hourly_total_can_shrink_robust_K_under_gate": not bool(r90_gate)})
    cases = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    radii.to_csv(OUT / "cems_state_quarter_r90.csv", index=False)
    cases.to_csv(OUT / "ba_month_gate_transfer_stress.csv", index=False)
    summary = cases.groupby("quarter", as_index=False).agg(
        ba_months=("region", "size"), r90_gate_crossed=("crosses_threshold_at_500", "sum"),
        uniform_10pct_crossed=("crosses_under_uniform_10pct", "sum"),
        mean_coal_r90=("coal_r90", "mean"), mean_oil_r90=("oil_r90", "mean"),
        mean_gas_r90=("gas_r90", "mean"))
    overall = pd.DataFrame([{"quarter": "ALL", "ba_months": len(cases),
                             "r90_gate_crossed": int(cases.crosses_threshold_at_500.sum()),
                             "uniform_10pct_crossed": int(cases.crosses_under_uniform_10pct.sum()),
                             "mean_coal_r90": cases.coal_r90.mean(), "mean_oil_r90": cases.oil_r90.mean(),
                             "mean_gas_r90": cases.gas_r90.mean()}])
    pd.concat([summary, overall], ignore_index=True).to_csv(OUT / "summary.csv", index=False)
    metadata = {"status": "empirical_cems_heterogeneity_threshold_gate_transport_stress_not_china_ef_estimate",
                "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
                "cems_archive_sha256": sha256(ARCHIVE), "input_certificate_sha256": sha256(INPUT_CERT),
                "eia_input_hashes": input_hashes, "cems": cems_meta,
                "fuel_group_mapping": FUEL_GROUP, "nominal_ef_g_kwh": NOMINAL_EF,
                "threshold_g_kwh": THRESHOLD, "state_quarter_generation_support": "symmetric radius containing >=90% generation",
                "ba_months": int(len(cases)), "r90_gate_crossed": int(cases.crosses_threshold_at_500.sum()),
                "uniform_10pct_gate_crossed": int(cases.crosses_under_uniform_10pct.sum()),
                "warning": "US CEMS state-quarter heterogeneity is transferred as stress multipliers to already-open EIA BA monthly fuel shares; not a China EF distribution or coverage statement"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(radii.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
