"""V231: pointwise hourly CIF identifiability from monthly fuel margins."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import v226_frozen_four_field_confirmation as confirm
import v228_factor_aware_minimum_channels as factor_span
import v225_partial_snapshot_sharp_risk_bounds as partial
import v224_eia_sparse_daily_maxrisk as dayrisk

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/V231_MONTHLY_ONLY_HOURLY_IDENTIFIABILITY_PROTOCOL.md"
OUT = ROOT / "results/v231-monthly-only-hourly-identifiability-r1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def greedy_bounds(fuel: np.ndarray, margins: np.ndarray, ef: np.ndarray) -> tuple[float, float]:
    """Sharp one-hour bounds under per-fuel monthly caps and known total G."""
    total = float(fuel.sum())
    monthly_total = float(margins.sum())
    lower = np.maximum(0., margins - (monthly_total - total))
    upper = np.minimum(margins, total)
    remaining = total - float(lower.sum())

    def extremum(order: np.ndarray) -> float:
        x = lower.copy()
        need = remaining
        for j in order:
            add = min(float(upper[j] - lower[j]), need)
            x[j] += add
            need -= add
            if need <= 1e-8:
                break
        if need > 1e-6:
            raise RuntimeError(f"infeasible fuel allocation: remaining={need}")
        return float(x @ ef / total)

    return extremum(np.argsort(ef)), extremum(np.argsort(-ef))


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    native, schemas, input_hashes, helpers = confirm.load_native()
    series = confirm.build_series(native, schemas, helpers)
    dev = set(confirm.DEV_REGIONS)
    rows = []
    selected = []
    zero_error = 0.
    for key, d in sorted(series.items()):
        region, year, month = key
        if year != 2026 or month not in (7, 8, 9) or region in dev:
            continue
        if len(np.unique(dayrisk.day_ids(pd.DatetimeIndex(d["time"])))) < 24:
            continue
        fuel = np.asarray(d["fuel_profile"], float) * np.asarray(d["fuel_totals"], float)[None, :]
        margins = np.asarray(d["fuel_totals"], float)
        ef = np.asarray(partial.EF, float)
        channels, coef, residual = factor_span.factor_span_fields(d)
        lo = np.empty(len(fuel)); hi = np.empty(len(fuel))
        for i, hour in enumerate(fuel):
            lo[i], hi[i] = greedy_bounds(hour, margins, ef)
        nominal = np.asarray(d["cif"], float)
        span_cif = factor_span.reconstruct_cif(d, channels, coef)
        full8_width = np.zeros(len(fuel), dtype=float)
        span_width = np.zeros(len(fuel), dtype=float)
        if np.any(lo - 1e-8 > nominal) or np.any(hi + 1e-8 < nominal):
            raise RuntimeError(f"nominal witness outside feasible set: {key}")
        err = float(np.max(np.abs(span_cif - nominal)))
        zero_error = max(zero_error, err)
        if err > 1e-8:
            raise RuntimeError(f"factor-span identity failed: {key}, {err}")
        monthly_only_width = hi - lo
        for method, width, contains in [
            ("monthly_margins_plus_hourly_total", monthly_only_width,
             (nominal >= lo - 1e-8) & (nominal <= hi + 1e-8)),
            ("minimum_factor_span_hourly_fields", span_width,
             np.abs(span_cif - nominal) <= 1e-8),
            ("full8_hourly_fuel_fields", full8_width, np.ones(len(fuel), dtype=bool)),
        ]:
            rows.append({"region": region, "year": year, "month": month, "method": method,
                "mean_width": float(np.mean(width)), "median_width": float(np.median(width)),
                "p90_width": float(np.quantile(width, .90)), "max_width": float(np.max(width)),
                "proxy_label_containment": float(np.mean(contains)), "hours": len(width),
                "n_local_hourly_group_fields": 0 if method == "monthly_margins_plus_hourly_total" else len(channels)})
        selected.append({"region": region, "month": month, "channels": channels,
            "numeric_local_hourly_fields_including_total": len(channels) + 1,
            "active_fuels": int(np.sum(margins > 1e-10)), "span_residual": residual})
    detail = pd.DataFrame(rows)
    if detail[["region", "month"]].drop_duplicates().shape[0] < 30 or detail.region.nunique() < 10:
        raise RuntimeError("frozen cohort gate failed")
    summary = detail.groupby("method", as_index=False).agg(
        mean_width=("mean_width", "mean"), median_width=("median_width", "median"),
        p90_width=("p90_width", "mean"), max_width=("max_width", "max"),
        proxy_label_containment=("proxy_label_containment", "mean"),
        n_ba_months=("month", "count"), n_regions=("region", "nunique"))
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "ba_month_metrics.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(selected).to_json(OUT / "selected_channels.json", orient="records", indent=2)
    meta = {"protocol_sha256": protocol_hash, "input_hashes": input_hashes,
        "n_ba_months": int(detail[["region", "month"]].drop_duplicates().shape[0]),
        "n_regions": int(detail.region.nunique()),
        "max_span_reconstruction_error": zero_error,
        "regime_caveat": "hourly total generation is granted to the monthly-only regime for an optimistic bound; user's supplied China data have monthly fields only",
        "bound_scope": "pointwise single-hour feasible sets; no joint across-hour optimization",
        "label_caveat": "EIA generation-derived fixed-factor proxy, not measured emissions"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"max_span_error={zero_error:.3g} output={OUT}")


if __name__ == "__main__":
    main()
