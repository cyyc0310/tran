"""V292: show what weak-set intersection coverage says about hourly CIF."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v226_frozen_four_field_confirmation as confirm
import v225_partial_snapshot_sharp_risk_bounds as partial
import v224_eia_sparse_daily_maxrisk as dayrisk

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V292_WEAK_COVERAGE_COUNTEREXAMPLE_PROTOCOL.md"
OUT = ROOT / "results/v292-weak-coverage-counterexample-r2"
HALF_WIDTHS = (0., 25., 50., 100., 200., 400., 800.)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    hourly_rows: list[dict] = []
    month_rows: list[dict] = []

    for key, d in sorted(series.items()):
        region, year, month = key
        if year != 2026 or month not in (7, 8, 9) or region in dev:
            continue
        times = pd.DatetimeIndex(d["time"])
        if len(np.unique(dayrisk.day_ids(times))) < 24:
            continue
        margins = np.asarray(d["fuel_totals"], float)
        active = margins > 1e-10
        ef = np.asarray(partial.EF, float)
        if not np.any(active) or margins.sum() <= 0:
            raise RuntimeError(f"invalid fuel margins: {key}")
        weak_lo = float(np.min(ef[active]))
        weak_hi = float(np.max(ef[active]))
        anchor = float(margins @ ef / margins.sum())
        y = np.asarray(d["cif"], float)
        if not (weak_lo - 1e-8 <= anchor <= weak_hi + 1e-8):
            raise RuntimeError(f"monthly anchor outside weak set: {key}")
        if np.any(y < weak_lo - 1e-7) or np.any(y > weak_hi + 1e-7):
            raise RuntimeError(f"hourly proxy outside weak set: {key}")

        month_cover = []
        for half_width in HALF_WIDTHS:
            lo, hi = anchor - half_width, anchor + half_width
            # Weak coverage is the JMLR weak-label event C(x) ∩ W != empty.
            weak_cover = float(lo <= weak_hi and hi >= weak_lo)
            strong_cover = float(np.mean((y >= lo) & (y <= hi)))
            month_cover.append(strong_cover)
            hourly_rows.append({
                "region": region, "year": year, "month": month,
                "half_width": half_width, "interval_width": 2 * half_width,
                "weak_set_lower": weak_lo, "weak_set_upper": weak_hi,
                "weak_set_width": weak_hi - weak_lo,
                "monthly_anchor": anchor, "weak_coverage": weak_cover,
                "proxy_strong_coverage": strong_cover, "hours": len(y),
            })
        for half_width, strong_cover in zip(HALF_WIDTHS, month_cover):
            month_rows.append({"region": region, "year": year, "month": month,
                               "half_width": half_width,
                               "proxy_strong_coverage": strong_cover})

    detail = pd.DataFrame(hourly_rows)
    months = pd.DataFrame(month_rows)
    n_bam = detail[["region", "month"]].drop_duplicates().shape[0]
    n_regions = detail.region.nunique()
    if n_bam < 30 or n_regions < 10:
        raise RuntimeError(f"cohort gate failed: {n_bam} BA-months, {n_regions} BAs")
    summary = detail.groupby("half_width", as_index=False).agg(
        interval_width=("interval_width", "first"),
        weak_coverage=("weak_coverage", "mean"),
        proxy_strong_coverage=("proxy_strong_coverage", "mean"),
        mean_weak_set_width=("weak_set_width", "mean"),
        n_hours=("hours", "sum"),
    )
    month_summary = months.groupby("half_width", as_index=False).agg(
        mean_ba_month_proxy_coverage=("proxy_strong_coverage", "mean"),
        min_ba_month_proxy_coverage=("proxy_strong_coverage", "min"),
        n_ba_months=("month", "count"),
    )
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "ba_month_width_metrics.csv", index=False)
    summary.merge(month_summary, on="half_width").to_csv(OUT / "summary.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash,
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "n_ba_months": int(n_bam), "n_regions": int(n_regions),
        "half_widths": list(HALF_WIDTHS),
        "weak_coverage_definition": "prediction interval intersects the monthly-margin-compatible pointwise CIF set",
        "strong_label_caveat": "EIA fixed-factor generation-derived hourly CIF proxy; not measured full-grid emissions",
        "cohort_caveat": "previously opened V231 development cohort; illustrative diagnostic, not untouched confirmation",
        "weak_coverage_is_not_strong_coverage": True,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.merge(month_summary, on="half_width").to_string(index=False))
    print(f"cohort={n_bam} BA-months/{n_regions} BAs; output={OUT}")


if __name__ == "__main__":
    main()
