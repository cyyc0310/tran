"""Retrospective source-only coarse-schema diagnosis, not a forecast experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/heldout-2026h1-data/monthly.csv"
SOURCE_SHA = "77cb3b47d079094f13d3437f19b0315ba09ac50305981d69fd4f9991a7f07ee5"
SOURCES = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
SHARES = [f"fuelshare_{f}" for f in FUELS]
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
SCHEMAS = {
    "eight": ((0,), (1,), (2,), (3,), (4,), (5,), (6,), (7,)),
    "merged_CGO": ((0, 1, 2), (3,), (4,), (5,), (6,), (7,)),
    "merged_positive_proxy": ((0, 1, 2, 7), (3,), (4,), (5,), (6,)),
    "CGO_plus_coal": ((0,), (1, 2), (3,), (4,), (5,), (6,), (7,)),
    "CGO_plus_gas": ((1,), (0, 2), (3,), (4,), (5,), (6,), (7,)),
    "CGO_plus_oil": ((2,), (0, 1), (3,), (4,), (5,), (6,), (7,)),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bounds(shares: np.ndarray, schema: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return sharp level bounds and attaining endpoint fuel allocations."""
    x = np.asarray(shares, float)
    if (x.ndim != 2 or x.shape[1] != 8 or not np.isfinite(x).all()
            or (x < 0).any() or not np.allclose(x.sum(1), 1., rtol=0, atol=1e-12)):
        raise ValueError("Finite nonnegative eight-fuel simplex rows required")
    groups = SCHEMAS[schema]
    low, high = np.zeros_like(x), np.zeros_like(x)
    for group in groups:
        ids = np.array(group)
        mass = x[:, ids].sum(1)
        low[:, ids[np.argmin(EF[ids])]] = mass
        high[:, ids[np.argmax(EF[ids])]] = mass
    return low @ EF, high @ EF, low, high


def validate(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate selected historical rows; do not silently repair duplicates."""
    x = frame.copy()
    if x.empty or not set(x.region) <= set(SOURCES):
        raise ValueError("Only nonempty registered source rows allowed")
    for col in ("period_start", "period_end", "available_at"):
        x[col] = pd.to_datetime(x[col], utc=True, errors="raise")
        if x[col].isna().any():
            raise ValueError("Finite timestamps required")
    if (not x.period_start.eq(x.period_start.dt.normalize()).all()
            or not x.period_start.dt.day.eq(1).all()
            or not x.period_end.eq(x.period_start + pd.offsets.MonthBegin(1)).all()
            or not x.available_at.ge(x.period_end).all()
            or x.duplicated(["region", "period_start"]).any()):
        raise ValueError("Unique one-month rows with valid release timing required")
    numeric = [*SHARES, "total_mwh", "observed_hours", "expected_hours", "coverage"]
    x[numeric] = x[numeric].apply(pd.to_numeric, errors="raise")
    if (not np.isfinite(x[numeric].to_numpy()).all()
            or not x.total_mwh.gt(0).all()
            or not x.observed_hours.gt(0).all()
            or not x.observed_hours.le(x.expected_hours).all()
            or not x.observed_hours.eq(np.floor(x.observed_hours)).all()
            or not np.array_equal(x.expected_hours.to_numpy(),
                                  ((x.period_end-x.period_start)/pd.Timedelta(hours=1)).to_numpy())
            or not np.allclose(x.coverage, x.observed_hours/x.expected_hours, rtol=0, atol=1e-12)):
        raise ValueError("Invalid energy totals or observed-hour coverage")
    bounds(x[SHARES].to_numpy(float), "eight")
    return x.sort_values(["region", "period_start"]).reset_index(drop=True)


def annualize(monthly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate only complete twelve-record proxy years, by observed energy."""
    x = validate(monthly)
    x["year"] = x.period_start.dt.year
    annual, support = [], []
    for (region, year), part in x.groupby(["region", "year"], sort=True):
        months = set(part.period_start.dt.month)
        complete = months == set(range(1, 13)) and len(part) == 12
        support.append(dict(region=region, year=year, months=len(part), complete=complete,
                            missing_months=",".join(map(str, sorted(set(range(1, 13))-months)))))
        if not complete:
            continue
        energy = part.total_mwh.to_numpy(float)
        mix = energy @ part[SHARES].to_numpy(float) / energy.sum()
        level = float(mix @ EF)
        monthly_levels = part[SHARES].to_numpy(float) @ EF
        end = pd.Timestamp(year=int(year)+1, month=1, day=1, tz="UTC")
        annual.append(dict(region=region, year=year, period_start=part.period_start.min(),
            period_end=end, available_at=max(end, part.available_at.max()),
            total_mwh=float(energy.sum()), observed_hours=int(part.observed_hours.sum()),
            expected_hours=int(part.expected_hours.sum()),
            coverage=float(part.observed_hours.sum()/part.expected_hours.sum()),
            month_level_min=float(monthly_levels.min()), month_level_max=float(monthly_levels.max()),
            month_level_range=float(np.ptp(monthly_levels)),
            month_abs_gap_from_annual=float(np.abs(monthly_levels-level).mean()),
            **dict(zip(SHARES, mix.tolist()))))
    return pd.DataFrame(annual), pd.DataFrame(support)


def diagnose(frame: pd.DataFrame, unit: str) -> pd.DataFrame:
    """Schema endpoint values; containment is algebra, not empirical coverage."""
    x = frame[SHARES].to_numpy(float)
    values = x @ EF
    outputs = []
    for schema in SCHEMAS:
        lo, hi, wl, wh = bounds(x, schema)
        if (np.any(values < lo-1e-9) or np.any(values > hi+1e-9)
                or np.any(hi < lo-1e-9)):
            raise AssertionError("Original proxy must lie in its schema level set")
        out = frame.drop(columns=SHARES).copy()
        out["unit"], out["schema"] = unit, schema
        out["proxy_level"], out["lower"], out["upper"] = values, lo, hi
        out["identification_width"] = hi-lo
        for i, fuel in enumerate(FUELS):
            out[f"lower_mass_{fuel}"] = wl[:, i]
            out[f"upper_mass_{fuel}"] = wh[:, i]
        outputs.append(out)
    return pd.concat(outputs, ignore_index=True)


def summaries(monthly: pd.DataFrame, annual: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    complete_keys = set(zip(annual.region, annual.year))
    matched = np.array([(r, t.year) in complete_keys
                        for r, t in zip(monthly.region, monthly.period_start)])
    regions = []
    datasets = [("monthly_2023_common", monthly.loc[monthly.period_start.dt.year.eq(2023)]),
                ("annual_2023_common", annual.loc[annual.year.eq(2023)]),
                ("monthly_all_2022_2024", monthly),
                ("monthly_complete_proxy_years", monthly.loc[matched]),
                ("annual_complete_proxy", annual)]
    for cohort, table in datasets:
        for (region, schema), part in table.groupby(["region", "schema"], sort=True):
            regions.append(dict(cohort=cohort, region=region, schema=schema, units=len(part),
                mean_identification_width=float(part.identification_width.mean()),
                mean_proxy_level=float(part.proxy_level.mean())))
    region = pd.DataFrame(regions)
    macro = region.groupby(["cohort", "schema"], sort=True).agg(
        regions=("region", "size"), units=("units", "sum"),
        mean_identification_width=("mean_identification_width", "mean"),
        mean_proxy_level=("mean_proxy_level", "mean")).reset_index()
    return region, macro


def verify_frozen(frozen: dict, output: Path, freeze_sha: str) -> None:
    if sha(output/"freeze.json") != freeze_sha:
        raise ValueError("Frozen run manifest changed during run")
    for path, expected in {**frozen["input_hashes"], **frozen["code_hashes"]}.items():
        if sha(Path(path)) != expected:
            raise ValueError("Frozen input or code changed during run")


def run(output: Path) -> None:
    """Freeze three code/doc hashes and one input before reading numeric data."""
    if sha(SOURCE) != SOURCE_SHA:
        raise ValueError("Pinned monthly source changed")
    code = [Path(__file__).resolve(), ROOT/"tests/test_coarse_fields_v24.py",
            ROOT/"docs/COARSE_FIELDS_V24_PROTOCOL.md"]
    frozen = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        purpose="retrospective identification, not hourly prediction or China validation",
        input_hashes={str(SOURCE): SOURCE_SHA}, code_hashes={str(p): sha(p) for p in code},
        sources=list(SOURCES), years=[2022, 2023, 2024], schemas=SCHEMAS,
        ef=EF.tolist(), no_hourly_files=True, no_target_files=True,
        real_publication_vintage_verified=False)
    output.mkdir(parents=True, exist_ok=False)
    (output/"freeze.json").write_text(json.dumps(frozen, indent=2)+"\n")
    freeze_sha = sha(output/"freeze.json")
    raw = pd.read_csv(SOURCE, dtype=str)
    if set(raw.region) != set(SOURCES):
        raise ValueError("Unexpected regions in the source-only monthly file")
    selected = raw.loc[raw.period_start.str[:4].isin(["2022", "2023", "2024"])].copy()
    monthly = validate(selected)
    common = monthly.loc[monthly.period_start.dt.year.eq(2023)]
    if common.groupby("region").size().to_dict() != {r: 12 for r in SOURCES}:
        raise ValueError("Prespecified common 2023 cohort is incomplete")
    annual, support = annualize(monthly)
    if annual.empty:
        raise ValueError("No complete proxy annual cohort")
    mb, ab = diagnose(monthly, "month"), diagnose(annual, "proxy_year")
    region, macro = summaries(mb, ab)
    tables = dict(monthly_bounds=mb, annual_bounds=ab, annual_support=support,
                  region_summary=region, macro_summary=macro)
    for name, table in tables.items():
        table.to_csv(output/f"{name}.csv", index=False)
    verify_frozen(frozen, output, freeze_sha)
    receipt = dict(status="COMPLETED_PENDING_INDEPENDENT_AUDIT", freeze_sha256=freeze_sha,
        monthly_rows=len(monthly),
        common_2023_rows=len(common), complete_proxy_years=len(annual),
        incomplete_proxy_years=int((~support.complete).sum()),
        outputs={f"{name}.csv": sha(output/f"{name}.csv") for name in tables},
        hourly_prediction_evaluated=False, new_holdout_opened=False, novelty_established=False)
    (output/"completed.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
