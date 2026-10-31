"""V219: transfer hourly fuel shapes, then reconcile to target monthly margins.

Retrospective, post-month reconstruction on the native EIA proxy. Hourly CIF is
mechanically derived from hourly fuel generation and fixed emission factors.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/v215_eia_hourly_series.pkl")
ENRICHED_CACHE = Path("/tmp/v219_eia_hourly_series.pkl")
OUT = ROOT / "results/v219-monthly-margin-shape-transfer-r2"
REGION_ROSTER = (
    "AVA", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "FMPP", "FPL", "GCPD",
    "GRID", "GVL", "IID", "JEA", "LDWP", "MISO", "NWMT", "NYIS", "PSEI",
    "SC", "SCEG", "SCL", "SEC", "TEC", "TIDC", "TPWR",
)
FUELS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
EF = np.array([1000.0, 400.0, 800.0, 400.0, 0.0, 0.0, 0.0, 0.0])
ALPHA = 0.10
MIN_MONTH_HOURS = 24 * 28


def load_series():
    if not CACHE.exists():
        raise FileNotFoundError(f"V215 source cache missing: {CACHE}")
    with CACHE.open("rb") as stream:
        series, hashes = pickle.load(stream)
    return series, hashes


def eligible_regions(series):
    return tuple(
        r for r in REGION_ROSTER
        if sum((r, 2025, m) in series for m in range(1, 13)) >= 9
        and sum((r, 2026, m) in series for m in range(1, 7)) >= 3
    )


def bray_curtis(a, b):
    return float(np.abs(a - b).sum() / max((a + b).sum(), 1e-12))


def predict_month_level(train_regions, target_mix, month, series):
    rows = []
    labels = []
    for region in train_regions:
        for source_month in range(1, 13):
            key = (region, 2025, source_month)
            if key not in series:
                continue
            d = series[key]
            rows.append((source_month, float(np.asarray(d["mix"]) @ EF) / 1000.0))
            labels.append(float(d["monthly_y"]))
    if not rows:
        raise ValueError("no source monthly rows for level model")
    row_months = np.asarray([row[0] for row in rows], dtype=int)
    design = np.column_stack([
        np.ones(len(rows)), np.eye(12)[row_months - 1], np.asarray([row[1] for row in rows])
    ])
    coef = np.linalg.lstsq(design, np.asarray(labels), rcond=None)[0]
    target = np.r_[1.0, np.eye(12)[month - 1], float(np.asarray(target_mix) @ EF) / 1000.0]
    return float(target @ coef)


def target_groups(fine: bool):
    # Coarse categories mirror common public reporting: thermal, nuclear,
    # hydro, wind, solar. Thermal composition is borrowed from donors.
    if fine:
        return np.eye(8)
    return np.array([
        [1, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
    ], dtype=float)


def profile_for(target_key, allowed_donors, series, fine, k=10, groups_override=None):
    """Build fuel profiles on target timestamps from donor month x weekday x hour."""
    target = series[target_key]
    month = target_key[2]
    target_times = pd.DatetimeIndex(target["time"])
    target_mix = np.asarray(target["mix"], dtype=float)
    groups = target_groups(fine) if groups_override is None else np.asarray(groups_override, dtype=float)
    target_group_mix = (groups @ target_mix)
    target_group_mix /= target_group_mix.sum()
    candidates = []
    for donor in allowed_donors:
        d = series.get((donor, 2025, month))
        if d is None:
            continue
        mix = np.asarray(d["mix"], dtype=float)
        gm = groups @ mix
        gm /= gm.sum()
        candidates.append((bray_curtis(target_group_mix, gm), donor, d))
    candidates.sort(key=lambda row: (row[0], row[1]))
    selected = candidates[:k]
    required = min(k, 5)
    if len(selected) < required:
        raise ValueError(f"fewer than {required} donor profiles for {target_key}")

    # Allocate each donor's fuel generation into weekday/hour cells. Convert
    # each cell to a per-fuel share of its monthly total, then map those shares
    # to target timestamps and average across donors.
    weekkey = target_times.dayofweek.to_numpy() * 24 + target_times.hour.to_numpy()
    donor_shares = []
    for _, _, d in selected:
        times = pd.DatetimeIndex(d["time"])
        values = np.asarray(d["mix"], dtype=float)  # overwritten below; cache stores shares
        # Recover monthly generation by fuel from mix and use each donor's
        # weekday/hour CIF only for shape is impossible from the compact cache.
        # V215 cache retains aggregate month shares and CIF, so profile fuel
        # shape is reconstructed from time-of-day CIF? Require enriched cache.
        if "fuel_profile" not in d:
            raise RuntimeError("V219 requires fuel_profile in cache; create via v219 cache builder")
        prof = np.asarray(d["fuel_profile"], dtype=float)
        donor_keys = times.dayofweek.to_numpy() * 24 + times.hour.to_numpy()
        cell = np.zeros((168, prof.shape[1]), dtype=float)
        counts = np.zeros(168, dtype=float)
        np.add.at(cell, donor_keys, prof)
        np.add.at(counts, donor_keys, 1.0)
        cell /= np.maximum(counts[:, None], 1.0)
        # A per-cell profile is compositional: normalize each fuel's donor
        # profile within month so later scaling enforces target margins.
        cell = np.maximum(cell, 0.0)
        sums = cell.sum(axis=0)
        cell = np.divide(cell, sums[None, :], out=np.zeros_like(cell), where=sums[None, :] > 0)
        donor_shares.append(cell[weekkey])
    avg_share = np.mean(np.stack(donor_shares), axis=0)
    share_sums = avg_share.sum(axis=0)
    avg_share = np.divide(
        avg_share, share_sums[None, :], out=np.zeros_like(avg_share), where=share_sums[None, :] > 0
    )

    target_total = np.asarray(target["fuel_totals"], dtype=float)
    if fine:
        allocated = target_total
    else:
        donor_mix = np.mean(np.stack([np.asarray(d["mix"], float) for _, _, d in selected]), axis=0)
        allocated = np.zeros(8, dtype=float)
        for gi, group in enumerate(groups):
            inds = np.flatnonzero(group)
            group_total = target_total[inds].sum()
            if group_total <= 0:
                continue
            weights = donor_mix[inds]
            if weights.sum() <= 0:
                weights = np.ones(len(inds), dtype=float)
            allocated[inds] = group_total * weights / weights.sum()
    zero_support = avg_share.sum(axis=0) <= 0
    if np.any(zero_support):
        avg_share[:, zero_support] = 1.0 / len(avg_share)
    fuel_hat = avg_share * allocated[None, :]
    # Exact reconciliation of every available target fuel/category margin.
    # For coarse thermal reporting, internal thermal weights are source-derived.
    if fine:
        check = fuel_hat.sum(axis=0)
        if not np.allclose(check, target_total, rtol=1e-10, atol=1e-8):
            raise AssertionError("fine monthly margins were not exactly reconciled")
    else:
        check = groups @ fuel_hat.sum(axis=0)
        expected = groups @ target_total
        if not np.allclose(check, expected, rtol=1e-8, atol=1e-5):
            raise AssertionError(f"coarse monthly margins were not exactly reconciled: {check} != {expected}")
    return fuel_hat @ EF / np.maximum(fuel_hat.sum(axis=1), 1e-12), selected


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    series, input_hashes = load_series()
    enriched_loaded = False
    if ENRICHED_CACHE.exists():
        with ENRICHED_CACHE.open("rb") as stream:
            enriched, enriched_hashes = pickle.load(stream)
        if enriched_hashes == input_hashes:
            series = enriched
            enriched_loaded = True
    if not enriched_loaded:
        # The compact V215 cache is expanded from raw input once, without
        # altering the original cache. Keep the step explicit for auditability.
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        import v54_unseen_local_ef_confirmation as source_data
        source_data.ALL_REGIONS = REGION_ROSTER
        hourly, _, _, raw_hashes = source_data.prepare_regions()
        if raw_hashes != input_hashes:
            raise RuntimeError("native EIA source hashes differ from V215 cache")
        for r in REGION_ROSTER:
            for year, months in ((2025, range(1, 13)), (2026, range(1, 7))):
                for month in months:
                    key = (r, year, month)
                    if key not in series:
                        continue
                    raw = hourly[r].copy()
                    raw["time"] = pd.to_datetime(raw.time, utc=True)
                    raw = raw.loc[raw.valid & raw.time.ge(f"{year}-{month:02d}-01T00:00:00Z")]
                    if month == 12:
                        next_date = f"{year+1}-01-01T00:00:00Z"
                    else:
                        next_date = f"{year}-{month+1:02d}-01T00:00:00Z"
                    raw = raw.loc[raw.time.lt(next_date)].sort_values("time")
                    cols = [f"generation_{f}" for f in FUELS]
                    z = raw[cols].to_numpy(float)
                    good = np.isfinite(z).all(axis=1) & (z >= 0).all(axis=1) & (z.sum(axis=1) > 0)
                    raw = raw.loc[good].copy(); z = z[good]
                    if len(z) < MIN_MONTH_HOURS or not np.array_equal(pd.DatetimeIndex(raw.time).asi8, pd.DatetimeIndex(series[key]["time"]).asi8):
                        raise RuntimeError(f"hourly timestamp mismatch for {key}")
                    series[key]["fuel_profile"] = z / z.sum(axis=0, keepdims=True).clip(min=1e-12)
                    series[key]["fuel_totals"] = z.sum(axis=0)
        with ENRICHED_CACHE.open("wb") as stream:
            pickle.dump((series, input_hashes), stream)

    regions = eligible_regions(series)
    targets = {}
    methods = ("source_shape", "monthly_constant", "coarse_margin_shape", "fine_margin_shape")
    for p in regions:
        donors = [r for r in regions if r != p]
        for month in range(1, 7):
            key = (p, 2026, month)
            if key not in series:
                continue
            d = series[key]
            # V215 source level + k10 composition-neighbor hourly CIF shape.
            donor_rows = sorted(
                [(bray_curtis(d["mix"], series[(q, 2025, month)]["mix"]), q)
                 for q in donors if (q, 2025, month) in series]
            )[:10]
            shape_cif = np.mean(np.stack([
                series[(q, 2025, month)]["hod"] - series[(q, 2025, month)]["monthly_y"]
                for _, q in donor_rows
            ]), axis=0)
            level = predict_month_level(donors, d["mix"], month, series)
            pred_source = level + shape_cif[d["hours"]]
            pred_coarse, _ = profile_for(key, donors, series, fine=False)
            pred_fine, _ = profile_for(key, donors, series, fine=True)
            # In this EIA proxy, monthly generation-weighted mean exactly
            # equals emissions / generation. This is an oracle post-month
            # anchor, not a claim about the time-average of a different label.
            monthly_anchor = float(d["mix"] @ EF)
            targets[(p, month)] = {
                "actual": np.asarray(d["cif"], float),
                "pred_source": pred_source,
                "pred_constant": np.full(len(d["cif"]), monthly_anchor),
                "pred_coarse": pred_coarse,
                "pred_fine": pred_fine,
                "n": len(d["cif"]),
            }

    # Nested leave-two-region-out source calibration. Each calibration BA is
    # excluded from the donor/model fit; target BA is excluded as well.
    source_scores = {name: [] for name in methods}
    for excluded in regions:
        pool = [r for r in regions if r != excluded]
        for q in pool:
            qdonors = [r for r in pool if r != q]
            for month in range(1, 13):
                key = (q, 2025, month)
                if key not in series:
                    continue
                d = series[key]
                donor_rows = sorted(
                    [(bray_curtis(d["mix"], series[(x, 2025, month)]["mix"]), x)
                     for x in qdonors if (x, 2025, month) in series]
                )[:10]
                if len(donor_rows) < 5:
                    continue
                shape = np.mean(np.stack([
                    series[(x, 2025, month)]["hod"] - series[(x, 2025, month)]["monthly_y"]
                    for _, x in donor_rows
                ]), axis=0)
                level = predict_month_level(qdonors, d["mix"], month, series)
                pred = level + shape[d["hours"]]
                source_scores["source_shape"].extend(np.abs(d["cif"] - pred))
                anchor = float(d["mix"] @ EF)
                source_scores["monthly_constant"].extend(np.abs(d["cif"] - anchor))
                for method, fine in (("coarse_margin_shape", False), ("fine_margin_shape", True)):
                    yhat, _ = profile_for(key, qdonors, series, fine=fine)
                    source_scores[method].extend(np.abs(d["cif"] - yhat))

    radii = {
        name: float(np.quantile(np.asarray(values), 1 - ALPHA, method="higher"))
        for name, values in source_scores.items()
    }
    rows = []
    region_rows = []
    by_method_region = {name: {} for name in methods}
    for (p, month), d in sorted(targets.items()):
        for name, pred_key in (("source_shape", "pred_source"), ("monthly_constant", "pred_constant"),
                               ("coarse_margin_shape", "pred_coarse"), ("fine_margin_shape", "pred_fine")):
            y = d["actual"]; pred = d[pred_key]; ae = np.abs(y - pred); rad = radii[name]
            metrics = {
                "coverage": float(np.mean(ae <= rad)),
                "width": float(2 * rad),
                "winkler90": float(np.mean(2 * rad + 20 * np.maximum(ae - rad, 0))),
                "point_mae": float(ae.mean()),
            }
            rows.append({"region": p, "month": month, "n_hours": d["n"], "method": name, **metrics})
            by_method_region[name].setdefault(p, []).append(metrics)
    for name in methods:
        for p, vals in by_method_region[name].items():
            x = {k: float(np.mean([v[k] for v in vals])) for k in ("coverage", "width", "winkler90", "point_mae")}
            region_rows.append({"method": name, "region": p, "n_months": len(vals), **x})
    region_df = pd.DataFrame(region_rows)
    macro = {}
    for name in methods:
        z = region_df.loc[region_df.method.eq(name)]
        macro[name] = {k: float(z[k].mean()) for k in ("coverage", "width", "winkler90", "point_mae")}
        macro[name]["worst_region_coverage"] = float(z.coverage.min())

    # Paired bootstrap across independent regions.
    rng = np.random.default_rng(219)
    nboot = 20000
    boot_rows = []
    bases = ("source_shape", "monthly_constant", "coarse_margin_shape")
    for method in ("coarse_margin_shape", "fine_margin_shape"):
        for base in bases:
            if method == base:
                continue
            a = region_df[region_df.method.eq(method)].set_index("region").sort_index()
            b = region_df[region_df.method.eq(base)].set_index("region").sort_index()
            delta = a.loc[:, ["coverage", "width", "winkler90", "point_mae"]].to_numpy() - b.loc[:, ["coverage", "width", "winkler90", "point_mae"]].to_numpy()
            draws = delta[rng.integers(0, len(delta), size=(nboot, len(delta)))].mean(axis=1)
            for j, metric in enumerate(("coverage", "width", "winkler90", "point_mae")):
                lo, hi = np.quantile(draws[:, j], [.025, .975])
                boot_rows.append({"candidate": method, "baseline": base, "metric": metric,
                                  "delta": float(delta[:, j].mean()), "ci95_low": float(lo), "ci95_high": float(hi),
                                  "n_resamples": nboot, "cluster": "region"})

    OUT.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(OUT / "query_metrics.csv", index=False, float_format="%.10g")
    region_df.to_csv(OUT / "region_metrics.csv", index=False, float_format="%.10g")
    pd.DataFrame(boot_rows).to_csv(OUT / "paired_region_bootstrap.csv", index=False, float_format="%.10g")
    summary = {
        "status": "RETROSPECTIVE_EIA_MONTHLY_MARGIN_CONSTRAINED_SHAPE_TRANSFER",
        "regions": list(regions), "n_regions": len(regions), "n_target_region_months": len(targets),
        "training_window": "2025-01..2025-12", "target_window": "2026-01..2026-06",
        "fixed_ef_gco2_per_kwh": dict(zip(FUELS, EF.tolist())),
        "calibration": "nested leave-two-region-out source residual q90, method-specific",
        "monthly_information": "target month fuel-generation totals; post-month retrospective oracle, no hourly target inputs",
        "coarse_categories": ["thermal", "nuclear", "hydro", "wind", "solar"],
        "fine_categories": list(FUELS), "radius_q90": radii, "macro_metrics": macro,
        "input_hashes": input_hashes,
        "limitations": [
            "hourly CIF is mechanically derived from EIA hourly generation estimates and fixed EF, not independent measured CIF",
            "EIA source family overlaps earlier project experiments; retrospective 2026H1 is development, not blind confirmation",
            "target monthly margins are only known after month-end and this is a post-month reconstruction task",
            "coarse thermal-category internal coal/gas/oil/other split is borrowed from source donors",
            "EIA category boundaries and emission factors do not certify Chinese public-table boundaries or factors",
            "method is conventional donor profile transfer plus margin reconciliation; no novelty claim",
        ],
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["macro_metrics"], indent=2), flush=True)
    print("WROTE", OUT, flush=True)


if __name__ == "__main__":
    main()
