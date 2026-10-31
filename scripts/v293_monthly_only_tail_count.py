"""V293: sharp outer bounds for monthly-only hourly CIF tail counts."""
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
PROTOCOL = ROOT / "docs/V293_MONTHLY_ONLY_TAIL_COUNT_PROTOCOL.md"
OUT = ROOT / "results/v293-monthly-only-tail-count-r1"
THRESHOLDS = (400., 500., 600., 800., 900.)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sharp_monthly_only_count_bounds(
    margins: np.ndarray, ef: np.ndarray, threshold: float, hours: int
) -> tuple[int, int, float]:
    """Return exact min/max tail-count envelope and monthly energy-weighted CIF."""
    active = margins > 1e-10
    if not np.any(active) or margins.sum() <= 0 or hours < 1:
        raise ValueError("positive monthly energy and at least one hour required")
    mu = float(margins @ ef / margins.sum())
    high = ef[active] > threshold
    if not np.any(high):
        return 0, 0, mu
    if np.all(high):
        return hours, hours, mu
    if mu > threshold:
        return 1, hours, mu
    return 0, hours - 1, mu


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    ef = np.asarray(partial.EF, float)
    rows: list[dict] = []
    for key, d in sorted(series.items()):
        region, year, month = key
        if year != 2026 or month not in (7, 8, 9) or region in dev:
            continue
        if len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"])))) < 24:
            continue
        margins = np.asarray(d["fuel_totals"], float)
        active = margins > 1e-10
        y = np.asarray(d["cif"], float)
        t = len(y)
        support_lo = float(np.min(ef[active]))
        support_hi = float(np.max(ef[active]))
        for threshold in THRESHOLDS:
            lower, upper, mu = sharp_monthly_only_count_bounds(margins, ef, threshold, t)
            observed = int(np.sum(y > threshold))
            if not lower <= observed <= upper:
                raise RuntimeError(
                    f"proxy count outside sharp set {key}, tau={threshold}: "
                    f"{observed} not in [{lower}, {upper}]"
                )
            rows.append({
                "region": region, "year": year, "month": month,
                "threshold": threshold, "hours": t,
                "monthly_anchor": mu, "anchor_side": "above" if mu > threshold else "at_or_below",
                "active_ef_support_lower": support_lo,
                "active_ef_support_upper": support_hi,
                "ef_support_straddles_threshold": support_lo <= threshold < support_hi,
                "sharp_count_lower": lower, "sharp_count_upper": upper,
                "sharp_count_width": upper - lower,
                "proxy_tail_count": observed,
                "proxy_tail_fraction": observed / t,
            })
    detail = pd.DataFrame(rows)
    n_bam = detail[["region", "month"]].drop_duplicates().shape[0]
    n_regions = detail.region.nunique()
    if n_bam != 74 or n_regions != 26 or len(detail) != 74 * len(THRESHOLDS):
        raise RuntimeError(f"V231 cohort mismatch: {n_bam} BA-months, {n_regions} BAs, {len(detail)} rows")
    summary = detail.groupby("threshold", as_index=False).agg(
        mean_hours=("hours", "mean"), mean_sharp_width=("sharp_count_width", "mean"),
        median_sharp_width=("sharp_count_width", "median"),
        mean_proxy_tail_count=("proxy_tail_count", "mean"),
        months_anchor_above=("anchor_side", lambda x: int((x == "above").sum())),
        months_support_straddles=("ef_support_straddles_threshold", "sum"),
        n_ba_months=("month", "count"),
    )
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "ba_month_threshold_metrics.csv", index=False)
    summary.to_csv(OUT / "threshold_summary.csv", index=False)
    metadata = {
        "protocol_sha256": protocol_hash,
        "script_sha256": sha256(Path(__file__).resolve()),
        "input_hashes": input_hashes,
        "n_ba_months": int(n_bam), "n_regions": int(n_regions),
        "thresholds_g_per_kwh": list(THRESHOLDS),
        "fixed_ef": ef.tolist(),
        "claim": "exact sharp min/max count envelope with monthly fuel margins and arbitrary positive hourly totals",
        "label_caveat": "EIA fixed-factor generation-derived CIF proxy; previously opened development cohort",
        "verification": "proxy tail count contained in analytic sharp envelope in every BA-month-threshold row",
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"all {len(detail)} proxy counts inside analytic bounds; output={OUT}")


if __name__ == "__main__":
    main()
