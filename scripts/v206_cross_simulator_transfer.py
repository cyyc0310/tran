"""Cross-simulator transfer stress test for V197 hourly CIF intervals."""
from __future__ import annotations

import base64
import calendar
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "results/china-projected-cif-v194-shape-transfer-r1"
PROTOCOL = ROOT / "docs/V206_CROSS_SIMULATOR_TRANSFER_PROTOCOL.md"
OUT = ROOT / "results/v206-cross-simulator-transfer-r1"
RAW_CACHE = OUT / "target-json-cache"
BASE = "https://planetaryenergy.ac.cn"
RELEASE = "20260813-import-adjusted-v3"
INDEX_URL = f"{BASE}/data/future-emissions/index.json?v={RELEASE}"
SOURCE_SCENARIOS = ("S1", "S2", "S3", "S4", "S5")
SOURCE_YEARS = (2030, 2035, 2040, 2045, 2050, 2055, 2060)
TARGET_YEARS = SOURCE_YEARS
TARGET_REGIONS = (
    "Beijing", "Tianjin", "Hebei", "Shanxi", "Liaoning", "Jilin", "Heilongjiang",
    "Shanghai", "Jiangsu", "Zhejiang", "Anhui", "Fujian", "Jiangxi", "Shandong",
    "Henan", "Hubei", "Hunan", "Guangdong", "Guangxi", "Hainan", "Chongqing",
    "Sichuan", "Guizhou", "Yunnan", "Tibet", "Shaanxi", "Gansu", "Qinghai",
    "Ningxia", "Xinjiang",
)
TARGET_NAME = {"Tibet": "Xizang"}
K = 15
FLOOR = 0.1
METHODS = ("constant_anchor", "k15_mean_shape", "k15_pooled90",
           "k15_dispersion_scaled90", "k15_particle90")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def digest(array: np.ndarray) -> str:
    return sha_bytes(np.asarray(array, dtype="<f8", order="C").tobytes())


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "TransCIF research evaluation/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def fetch_many(urls: list[str], workers: int = 8) -> dict[str, bytes]:
    """Download version-pinned JSON with resumable local caching and retries."""
    RAW_CACHE.mkdir(parents=True, exist_ok=True)

    def cached_fetch(url: str) -> bytes:
        cache_path = RAW_CACHE / f"{sha_bytes(url.encode())}.json"
        if cache_path.exists():
            return cache_path.read_bytes()
        last_error = None
        for attempt in range(5):
            try:
                payload = fetch(url)
                # Reject malformed/truncated JSON before it enters the cache.
                json.loads(payload)
                temp_path = cache_path.with_suffix(f".json.{attempt}.tmp")
                temp_path.write_bytes(payload)
                temp_path.replace(cache_path)
                return payload
            except Exception as exc:  # transient disconnects occur on this host
                last_error = exc
                time.sleep(min(2 ** attempt, 12))
        raise RuntimeError(f"failed after retries: {url}: {last_error}") from last_error

    data: dict[str, bytes] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(cached_fetch, url): url for url in urls}
        for completed, future in enumerate(concurrent.futures.as_completed(futures), 1):
            url = futures[future]
            data[url] = future.result()
            if completed % 100 == 0 or completed == len(urls):
                print(f"downloaded {completed}/{len(urls)} JSON files", flush=True)
    return data


def month_bounds(year: int) -> list[tuple[int, int]]:
    """Return standard 365-day month slices used by both projection models."""
    result, offset = [], 0
    for month in range(1, 13):
        end = offset + calendar.monthrange(2001, month)[1] * 24
        result.append((offset, end))
        offset = end
    if offset != 8760:
        raise AssertionError(f"standard-year length mismatch for {year}: {offset}")
    return result


def read_source() -> tuple[dict, dict[str, str]]:
    provinces = (
        "Beijing", "Tianjin", "Hebei", "Shanxi", "Inner Mongolia", "Liaoning",
        "Jilin", "Heilongjiang", "Shanghai", "Jiangsu", "Zhejiang", "Anhui",
        "Fujian", "Jiangxi", "Shandong", "Henan", "Hubei", "Hunan", "Guangdong",
        "Guangxi", "Hainan", "Chongqing", "Sichuan", "Guizhou", "Yunnan",
        "Tibet", "Shaanxi", "Gansu", "Qinghai", "Ningxia", "Xinjiang",
    )
    data = {}
    hashes = {}
    for scenario in SOURCE_SCENARIOS:
        path = SOURCE_DIR / f"CEF data for Scenario {scenario}.xlsx"
        hashes[path.name] = sha_file(path)
        workbook = load_workbook(path, read_only=True, data_only=True)
        expected_sheets = ["2025", "2030", "2035", "2040", "2045", "2050", "2055", "2060"]
        if workbook.sheetnames != expected_sheets:
            raise ValueError(f"unexpected source sheets {scenario}: {workbook.sheetnames}")
        for year in SOURCE_YEARS:
            worksheet = workbook[str(year)]
            rows = worksheet.iter_rows(values_only=True)
            header = next(rows)
            expected_header = ("Time", "Mainland China", *provinces)
            if tuple(header) != expected_header:
                raise ValueError(f"unexpected source header {scenario}/{year}: {header}")
            values = np.asarray(list(rows), dtype=np.float64)
            expected_hours = 8784 if calendar.isleap(year) else 8760
            if values.shape != (expected_hours, len(expected_header)):
                raise ValueError(f"source shape mismatch {scenario}/{year}: {values.shape}")
            if not np.array_equal(values[:, 0], np.arange(1, expected_hours + 1)):
                raise ValueError(f"source hourly index mismatch {scenario}/{year}")
            if calendar.isleap(year):
                # Target CISPO files encode every planning year as 8,760 hours.
                # Remove source Feb 29 to put both model worlds on the same
                # standard-year calendar before extracting monthly profiles.
                values = np.concatenate((values[: 24 * 59], values[24 * 60 :]), axis=0)
            if values.shape[0] != 8760:
                raise ValueError(f"standardized source year is not 8,760 hours: {scenario}/{year}")
            # Source workbook is tCO2/MWh; multiply by 1000 to match gCO2/kWh.
            data[(scenario, year)] = (values[:, 2:] * 1000.0, provinces)
        workbook.close()
    return data, hashes


def decode_target(payload: bytes, expected_hours: int, encoding: dict) -> np.ndarray:
    item = json.loads(payload)
    if not isinstance(item.get("series"), str):
        raise ValueError("target hourly JSON is missing encoded series")
    raw = base64.b64decode(item["series"], validate=True)
    values = np.frombuffer(raw, dtype="<i2").astype(np.float64)
    if values.size != expected_hours:
        raise ValueError(f"target series length {values.size} != {expected_hours}")
    missing = int(encoding["missing"])
    scale = float(encoding["scale"])
    values[values == missing] = np.nan
    values[~np.isnan(values)] *= scale
    return values


def read_target() -> tuple[dict, dict[str, str], dict]:
    index_bytes = fetch(INDEX_URL)
    index = json.loads(index_bytes)
    if index.get("release") != "2026-08-13" or index.get("scope") != "provincial-average-with-imports":
        raise ValueError("unexpected target dataset release/scope")
    if index.get("unit") != "gCO2/kWh":
        raise ValueError(f"unexpected target unit: {index.get('unit')}")
    scenario_ids = tuple(index["scenarios"])
    if len(scenario_ids) != 6:
        raise ValueError(f"expected six target scenarios, found {len(scenario_ids)}")
    summary_urls = []
    for scenario in scenario_ids:
        for year in TARGET_YEARS:
            path = index["scenarios"][scenario]["files"][str(year)]
            summary_urls.append(BASE + path + f"?v={RELEASE}")
    summary_bytes = fetch_many(summary_urls)
    summaries = {url: json.loads(payload) for url, payload in summary_bytes.items()}
    region_urls: dict[tuple[str, int, str], str] = {}
    for scenario in scenario_ids:
        for year in TARGET_YEARS:
            summary_url = BASE + index["scenarios"][scenario]["files"][str(year)] + f"?v={RELEASE}"
            summary = summaries[summary_url]
            if summary.get("scenario") != scenario or int(summary.get("year", -1)) != year:
                raise ValueError(f"target scenario/year mismatch: {scenario}/{year}")
            if summary.get("unit") != "gCO2/kWh" or summary.get("scope") != "provincial-average-with-imports":
                raise ValueError(f"target summary contract mismatch: {scenario}/{year}")
            for region in TARGET_REGIONS:
                raw_name = TARGET_NAME.get(region, region)
                if raw_name not in summary["provinces"]:
                    raise ValueError(f"missing target region {raw_name} in {scenario}/{year}")
                region_urls[(scenario, year, region)] = BASE + summary["provinces"][raw_name]["file"] + f"?v={RELEASE}"
    region_payloads = fetch_many(list(region_urls.values()))
    target = {}
    for key, url in region_urls.items():
        scenario, year, region = key
        item = json.loads(region_payloads[url])
        encoding = summaries[
            BASE + index["scenarios"][scenario]["files"][str(year)] + f"?v={RELEASE}"
        ]["encoding"]
        # CISPO publishes all planning years on a fixed 8,760-hour calendar,
        # including nominally leap years such as 2040 and 2060.
        hours = 8760
        values = decode_target(region_payloads[url], hours, encoding)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite target series: {scenario}/{year}/{region}")
        target[key] = values
    receipt = {
        "index_url": INDEX_URL,
        "index_sha256": sha_bytes(index_bytes),
        "summary_file_sha256": {url: sha_bytes(payload) for url, payload in summary_bytes.items()},
        "region_file_sha256": {url: sha_bytes(payload) for url, payload in region_payloads.items()},
        "release": index["release"],
        "scenario_ids": list(scenario_ids),
        "years": list(TARGET_YEARS),
        "regions": list(TARGET_REGIONS),
        "scope": index["scope"],
        "unit": index["unit"],
        "source_description": index.get("source"),
    }
    return target, receipt, index


def source_month_candidates(source: dict) -> dict:
    by_region_month = {region: {month: [] for month in range(1, 13)} for region in source[("S1", SOURCE_YEARS[0])][1]}
    for scenario in SOURCE_SCENARIOS:
        for year in SOURCE_YEARS:
            matrix, regions = source[(scenario, year)]
            for month, (lo, hi) in enumerate(month_bounds(year), 1):
                for index, region in enumerate(regions):
                    profile = matrix[lo:hi, index]
                    if not np.isfinite(profile).all():
                        continue
                    mean = float(profile.mean())
                    by_region_month[region][month].append(
                        (scenario, year, mean, profile - mean)
                    )
    return by_region_month


def choose_donors(candidates: dict, target_region: str, month: int, target_mean: float) -> list[tuple]:
    donors = []
    for region in candidates:
        if region == target_region:
            continue
        choices = candidates[region][month]
        if not choices:
            continue
        selected = min(
            choices,
            key=lambda item: (abs(item[2] - target_mean), SOURCE_SCENARIOS.index(item[0]), item[1]),
        )
        donors.append((region, selected[0], selected[1], selected[2], selected[3]))
    if len(donors) != 30:
        return []
    return donors


def quantile_higher(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q, method="higher"))


def predict(target_mean: float, donors: list[tuple]) -> tuple[dict, dict]:
    means = np.asarray([d[3] for d in donors], dtype=np.float64)
    shapes = np.stack([d[4] for d in donors])
    if shapes.shape[0] != 30 or not np.isfinite(shapes).all():
        raise ValueError("invalid source donor bank")
    residuals = np.empty_like(shapes)
    scales = np.empty_like(shapes)
    all_index = np.arange(shapes.shape[0])
    for i in range(shapes.shape[0]):
        keep = all_index != i
        donor_means = means[keep]
        donor_shapes = shapes[keep]
        nearest = np.argsort(np.abs(donor_means - means[i]), kind="stable")[:K]
        neighbor_shapes = donor_shapes[nearest]
        residuals[i] = shapes[i] - neighbor_shapes.mean(axis=0)
        scales[i] = np.maximum(neighbor_shapes.std(axis=0, ddof=1), FLOOR)
    pooled_q = quantile_higher(np.abs(residuals).ravel(), 0.90)
    standardized_q = quantile_higher((np.abs(residuals) / scales).ravel(), 0.90)
    target_order = np.argsort(np.abs(means - target_mean), kind="stable")[:K]
    target_shapes = shapes[target_order]
    center = target_mean + target_shapes.mean(axis=0)
    target_scale = np.maximum(target_shapes.std(axis=0, ddof=1), FLOOR)
    particle_low, particle_high = np.quantile(
        target_mean + target_shapes, [0.05, 0.95], axis=0, method="higher"
    )
    predictions = {
        # Same source-calibrated residual budget as the shape methods, but no
        # transferred hourly shape in the center. A zero-width interval here
        # would violate the frozen information-budget comparison.
        "constant_anchor": (np.full(center.shape, target_mean - pooled_q),
                            np.full(center.shape, target_mean + pooled_q),
                            np.full(center.shape, target_mean)),
        "k15_mean_shape": (center, center, center),
        "k15_pooled90": (center - pooled_q, center + pooled_q, center),
        "k15_dispersion_scaled90": (center - standardized_q * target_scale,
                                    center + standardized_q * target_scale, center),
        "k15_particle90": (particle_low, particle_high, center),
    }
    details = {
        "pooled_q90": pooled_q, "standardized_q90": standardized_q,
        "target_scale_mean": float(target_scale.mean()),
        "donor_shape_sha256": digest(shapes),
        "source_residual_sha256": digest(residuals),
        "source_scale_sha256": digest(scales),
        "target_center_sha256": digest(center),
        "target_scale_sha256": digest(target_scale),
        "target_neighbors": [donors[int(i)][0] for i in target_order],
    }
    return predictions, details


def score(lower: np.ndarray, upper: np.ndarray, center: np.ndarray, truth: np.ndarray,
          target_mean: float) -> dict:
    shape_truth = truth - target_mean
    shape_prediction = center - target_mean
    width = upper - lower
    miss = np.maximum(lower - truth, 0) + np.maximum(truth - upper, 0)
    return {
        "point_mae": float(np.mean(np.abs(center - truth))),
        "point_rmse": float(np.sqrt(np.mean((center - truth) ** 2))),
        "shape_mae": float(np.mean(np.abs(shape_prediction - shape_truth))),
        "shape_rmse": float(np.sqrt(np.mean((shape_prediction - shape_truth) ** 2))),
        "coverage90": float(np.mean((truth >= lower) & (truth <= upper))),
        "width90": float(np.mean(width)),
        "winkler90": float(np.mean(width + 20 * miss)),
        "level_mae": float(abs(float(center.mean()) - target_mean)),
    }


def main() -> None:
    if OUT.exists() and any(path.name != RAW_CACHE.name for path in OUT.iterdir()):
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    source, source_hashes = read_source()
    candidates = source_month_candidates(source)
    target, target_receipt, target_index = read_target()
    source_year_sheets_hash = {
        str(scenario): [year for year in SOURCE_YEARS] for scenario in SOURCE_SCENARIOS
    }
    query_specs = []
    for scenario in target_receipt["scenario_ids"]:
        for year in TARGET_YEARS:
            hours = 8760
            bounds = month_bounds(year)
            for region in TARGET_REGIONS:
                truth = target[(scenario, year, region)]
                if truth.size != hours:
                    raise ValueError(f"invalid target hour length {scenario}/{year}/{region}")
                for month, (lo, hi) in enumerate(bounds, 1):
                    y = truth[lo:hi]
                    if not np.isfinite(y).all():
                        continue
                    target_mean = float(y.mean())
                    donors = choose_donors(candidates, region, month, target_mean)
                    if len(donors) != 30:
                        continue
                    predictions, details = predict(target_mean, donors)
                    row = {
                        "scenario": scenario, "year": year, "province": region, "month": month,
                        "hours": hi - lo, "target_mean": target_mean,
                        "donor_regions": [d[0] for d in donors],
                        "donor_scenarios": [d[1] for d in donors],
                        "donor_years": [d[2] for d in donors],
                        "donor_means": [d[3] for d in donors],
                        "donor_bank_sha256": details["donor_shape_sha256"],
                        "calibration": {k: v for k, v in details.items() if k != "target_neighbors"},
                        "target_neighbors": details["target_neighbors"],
                        "prediction_sha256": {
                            method: {name: digest(values) for name, values in
                                     zip(("lower", "upper", "center"), arrays)}
                            for method, arrays in predictions.items()
                        },
                    }
                    query_specs.append(row)
    manifest_path = OUT / "prediction_manifest.jsonl"
    manifest_hash = hashlib.sha256()
    with manifest_path.open("w") as stream:
        for row in query_specs:
            line = json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            stream.write(line)
            manifest_hash.update(line.encode())
    source_query_hash = sha_file(PROTOCOL)
    seal = {
        "status": "PREDICTIONS_SEALED_BEFORE_SCORING",
        "protocol_sha256": source_query_hash,
        "runner_sha256": sha_file(Path(__file__)),
        "source_world_workbook_sha256": source_hashes,
        "target_data_receipt": target_receipt,
        "source_world": "Li et al. Figshare v3; projections",
        "target_world": "PlanetaryEnergy CISPO-China hourly model emissions release",
        "target_scenarios": target_receipt["scenario_ids"],
        "target_years": list(TARGET_YEARS),
        "target_regions": list(TARGET_REGIONS),
        "source_years": list(SOURCE_YEARS),
        "source_scenarios": list(SOURCE_SCENARIOS),
        "k": K, "scale_floor_g_kwh": FLOOR, "methods": list(METHODS),
        "query_count": len(query_specs),
        "prediction_manifest_sha256": manifest_hash.hexdigest(),
        "scope": "cross-simulator model-projection stress test; target monthly means are retrospective oracle anchors",
        "source_sheet_manifest": source_year_sheets_hash,
    }
    (OUT / "prediction_seal.json").write_text(json.dumps(seal, ensure_ascii=False, indent=2) + "\n")

    # Scoring begins only after the manifest and seal exist.
    scores = []
    manifest_rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    if len(manifest_rows) != len(query_specs):
        raise AssertionError("manifest row count changed")
    for row in manifest_rows:
        key = (row["scenario"], int(row["year"]), row["province"])
        truth_series = target[key]
        lo, hi = month_bounds(key[1])[int(row["month"]) - 1]
        y = truth_series[lo:hi]
        target_mean = float(row["target_mean"])
        donors = choose_donors(candidates, row["province"], int(row["month"]), target_mean)
        predictions, _ = predict(target_mean, donors)
        for method in METHODS:
            arrays = predictions[method]
            hashes = {name: digest(values) for name, values in
                      zip(("lower", "upper", "center"), arrays)}
            if hashes != row["prediction_sha256"][method]:
                raise AssertionError(f"sealed prediction mismatch {key}/{row['month']}/{method}")
            metrics = score(*arrays, y, target_mean)
            scores.append({
                "scenario": key[0], "year": key[1], "province": key[2],
                "month": int(row["month"]), "method": method, **metrics,
            })
    score_frame = pd.DataFrame(scores)
    score_frame.to_csv(OUT / "monthly_scores.csv", index=False)
    metrics = ["point_mae", "point_rmse", "shape_mae", "shape_rmse",
               "coverage90", "width90", "winkler90", "level_mae"]
    province = score_frame.groupby(
        ["province", "method"], as_index=False
    )[metrics].mean()
    province.to_csv(OUT / "province_summary.csv", index=False)
    macro = province.groupby("method", as_index=False)[metrics].mean()
    macro.to_csv(OUT / "summary_macro.csv", index=False)
    by_scenario = score_frame.groupby(["scenario", "method"], as_index=False)[metrics].mean()
    by_scenario.to_csv(OUT / "summary_scenario.csv", index=False)
    by_year = score_frame.groupby(["year", "method"], as_index=False)[metrics].mean()
    by_year.to_csv(OUT / "summary_year.csv", index=False)
    payload = {
        "status": "COMPLETED_CROSS_SIMULATOR_PROJECTION_STRESS_TEST",
        "queries": int(score_frame[["scenario", "year", "province", "month"]].drop_duplicates().shape[0]),
        "scored_rows": len(score_frame),
        "provinces": score_frame.province.nunique(),
        "scenarios": sorted(score_frame.scenario.unique().tolist()),
        "years": sorted(score_frame.year.unique().tolist()),
        "methods": list(METHODS),
        "macro": macro.to_dict("records"),
        "seal_sha256": sha_file(OUT / "prediction_seal.json"),
        "monthly_scores_sha256": sha_file(OUT / "monthly_scores.csv"),
        "limitations": [
            "Both source and target are published model-generated projection worlds, not observed Chinese hourly grid labels.",
            "The target monthly mean is an oracle derived from target monthly hourly values; it is stronger than the user's monthly generation table.",
            "The target model's data redistribution license was not located; this run stores only hashes and derived metrics, not raw target series.",
            "An initial internal pilot or this full run does not establish statistical coverage guarantees or real-world transfer.",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(macro.to_string(index=False))
    print(f"queries={payload['queries']} rows={payload['scored_rows']} "
          f"provinces={payload['provinces']} output={OUT}")


if __name__ == "__main__":
    main()
