"""Retrospective aggregation audit restricted to the seven inspected source BAs.

No fitted model, forecast runner, or external-BA hourly file is imported or read.
The monthly comparison always distinguishes an identical-hour identity from the
separately prepared/released monthly observation and its quality mask.
"""
from __future__ import annotations

import argparse
import calendar
import csv
from datetime import datetime, timezone
import hashlib
from io import StringIO
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
YEARS = (2022, 2023, 2024)
FUELS = ("coal", "gas", "oil", "nuclear", "hydro", "wind", "solar", "other")
EF = np.array([1000., 400., 800., 0., 0., 0., 0., 400.])
MIN_COVERAGE = .8
IDENTITY_ATOL = 1e-10


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")


def read_filtered_csv(path, *, hourly_region=None):
    """Filter region/year CSV text before any numeric fuel or CIF parsing."""
    with Path(path).open(newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader)
        region_position = header.index("region")
        time_position = header.index("time" if hourly_region else "period_start")
        selected = [row for row in reader
                    if row[region_position] in REGIONS
                    and (hourly_region is None or row[region_position] == hourly_region)
                    and row[time_position][:4] in {str(year) for year in YEARS}]
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(header)
    writer.writerows(selected)
    buffer.seek(0)
    return pd.read_csv(buffer)


def same_hour_identity(generation):
    """Population Cov(G,C)/mean(G), with the exact same legal hours throughout."""
    generation = np.asarray(generation, dtype=float)
    if generation.ndim != 2 or generation.shape[1] != len(FUELS) or not len(generation):
        raise ValueError("Nonempty hours x eight fuel array required")
    if not np.isfinite(generation).all() or np.any(generation < 0):
        raise ValueError("Finite nonnegative generation required")
    total = generation.sum(axis=1)
    if np.any(total <= 0):
        raise ValueError("Every selected hour must have positive total generation")
    carbon = (generation @ EF) / total
    mean_generation = total.mean()
    mean_carbon = carbon.mean()
    weighted_carbon = np.sum(total * carbon) / total.sum()
    covariance = np.mean((total - mean_generation) * (carbon - mean_carbon))
    delta = weighted_carbon - mean_carbon
    fuel_energy = generation.sum(axis=0)
    return {"samehours_total_mwh": float(total.sum()),
            "mean_generation_mw": float(mean_generation),
            "generation_std_mw": float(total.std(ddof=0)),
            "time_mean_cif": float(mean_carbon), "hourly_cif_std": float(carbon.std(ddof=0)),
            "z_samehours": float(weighted_carbon), "delta": float(delta),
            "population_cov_generation_cif": float(covariance),
            "covariance_over_mean_generation": float(covariance / mean_generation),
            "identity_residual": float(delta - covariance / mean_generation),
            "fuel_aggregation_identity_residual": float(weighted_carbon - (fuel_energy @ EF) / total.sum()),
            **{f"samehours_fuelshare_{fuel}": float(value / total.sum()) for fuel, value in zip(FUELS, fuel_energy)}}


def self_test():
    """Same hourly G and monthly fuel totals can imply three distinct mean CIFs."""
    paths = []
    for gas, hydro in (([0., 3.], [1., 0.]), ([.75, 2.25], [.25, .75]), ([1., 2.], [0., 1.])):
        generation = np.zeros((2, len(FUELS)))
        generation[:, FUELS.index("gas")] = gas
        generation[:, FUELS.index("hydro")] = hydro
        paths.append(generation)
    metrics = [same_hour_identity(path) for path in paths]
    for path in paths[1:]:
        np.testing.assert_allclose(path.sum(axis=0), paths[0].sum(axis=0), rtol=0, atol=1e-14)
        np.testing.assert_allclose(path.sum(axis=1), paths[0].sum(axis=1), rtol=0, atol=1e-14)
    np.testing.assert_allclose([m["z_samehours"] for m in metrics], [300.] * 3, rtol=0, atol=1e-12)
    np.testing.assert_allclose([m["time_mean_cif"] for m in metrics], [200., 300., 1000 / 3], rtol=0, atol=1e-12)
    np.testing.assert_allclose([m["delta"] for m in metrics], [100., 0., -100 / 3], rtol=0, atol=1e-12)
    assert max(abs(m["identity_residual"]) for m in metrics) < IDENTITY_ATOL
    constant = np.ones((24, len(FUELS)))
    assert same_hour_identity(constant)["delta"] == 0
    rng = np.random.default_rng(731)
    for n in (24, 672, 744):
        assert abs(same_hour_identity(rng.lognormal(5., 1., size=(n, len(FUELS))))["identity_residual"]) < IDENTITY_ATOL
    try:
        same_hour_identity(np.full((2, len(FUELS)), -1.))
    except ValueError:
        pass
    else:
        raise AssertionError("Negative generation must not be silently clipped")
    return {"passed": True, "same_generation_mw": [1., 3.],
            "same_monthly_gas_mwh": 3., "same_monthly_hydro_mwh": 1.,
            "same_monthly_energy_weighted_cif": 300.,
            "possible_time_mean_cif": [m["time_mean_cif"] for m in metrics],
            "possible_delta": [m["delta"] for m in metrics],
            "interpretation": "Even fixed hourly G, aggregate fuel mix, support and release age do not identify time-mean CIF."}


def calculate(data):
    monthly = read_filtered_csv(data / "monthly.csv")
    for column in ("period_start", "period_end", "available_at"):
        monthly[column] = pd.to_datetime(monthly[column], utc=True)
    if monthly.duplicated(["region", "period_start"]).any():
        raise ValueError("Duplicate monthly release support")
    output, excluded = [], []
    for region in REGIONS:
        frame = read_filtered_csv(data / f"{region}_hourly.csv", hourly_region=region)
        times = pd.DatetimeIndex(pd.to_datetime(frame.time, utc=True))
        if times.has_duplicates or not times.equals(times.floor("h")):
            raise ValueError("Source hourly inputs require unique whole UTC hours")
        generation = frame[[f"generation_{fuel}" for fuel in FUELS]].to_numpy(float)
        flag = frame.valid.astype(str).str.lower().isin(("true", "1")).to_numpy()
        finite = np.isfinite(generation).all(axis=1)
        nonnegative = (generation >= 0).all(axis=1)
        positive = generation.sum(axis=1) > 0
        legal = flag & finite & nonnegative & positive
        for year in YEARS:
            for month in range(1, 13):
                expected = calendar.monthrange(year, month)[1] * 24
                selection = (times.year == year) & (times.month == month)
                use = selection & legal
                n_hours = int(use.sum())
                start = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
                end = start + pd.offsets.MonthBegin(1)
                released = monthly.loc[(monthly.region == region) & (monthly.period_start == start)]
                metadata = {"region": region, "year": year, "month": month,
                    "expected_hours": expected, "raw_observed_hours": int(selection.sum()),
                    "valid_flag_hours": int((selection & flag).sum()), "legal_hours": n_hours,
                    "coverage": n_hours / expected,
                    "valid_but_illegal_fuel_hours": int((selection & flag & ~legal).sum()),
                    "released_month_present": bool(len(released))}
                if n_hours / expected < MIN_COVERAGE:
                    excluded.append({**metadata, "exclusion": "same-hour legal coverage below 0.8"})
                    continue
                row = {**metadata, **same_hour_identity(generation[use])}
                row.update(released_comparable=False, released_support_mismatch=False)
                if len(released):
                    release = released.iloc[0]
                    row["released_support_mismatch"] = bool(release.period_end != end)
                    if release.period_end != end:
                        row["released_comparison_status"] = "noncalendar_support"
                    else:
                        shares = release[[f"fuelshare_{fuel}" for fuel in FUELS]].to_numpy(float)
                        if not np.isfinite(shares).all() or np.any(shares < 0) or not np.isclose(shares.sum(), 1, rtol=0, atol=1e-8):
                            raise ValueError("Released monthly shares must be a finite simplex")
                        released_z = float(shares @ EF)
                        hourly_shares = np.array([row[f"samehours_fuelshare_{fuel}"] for fuel in FUELS])
                        hour_gap = float(release.observed_hours - n_hours)
                        energy_gap = float(release.total_mwh - row["samehours_total_mwh"])
                        share_gap = float(np.abs(shares - hourly_shares).sum())
                        same_counts = bool(hour_gap == 0 and float(release.expected_hours) == expected)
                        same_energy = bool(np.isclose(release.total_mwh, row["samehours_total_mwh"], rtol=1e-10, atol=1e-6))
                        same_mix = bool(np.allclose(shares, hourly_shares, rtol=0, atol=1e-10))
                        row.update(released_comparable=True, released_comparison_status="compared",
                            released_available_at=str(release.available_at), released_coverage=float(release.coverage),
                            released_observed_hours=float(release.observed_hours), released_expected_hours=float(release.expected_hours),
                            released_total_mwh=float(release.total_mwh), released_z=released_z,
                            released_z_minus_samehours_z=released_z - row["z_samehours"],
                            released_z_minus_time_mean=released_z - row["time_mean_cif"],
                            released_observed_hours_minus_legal_hours=hour_gap,
                            released_total_mwh_minus_samehours_total_mwh=energy_gap,
                            released_fuelshare_l1_difference=share_gap,
                            released_hour_count_mismatch=not same_counts,
                            released_energy_mismatch=not same_energy,
                            released_fuelmix_mismatch=not same_mix,
                            released_aggregation_mismatch=not (same_counts and same_energy and same_mix))
                else:
                    row["released_comparison_status"] = "missing_release"
                output.append(row)
    return pd.DataFrame(output), pd.DataFrame(excluded)


def summaries(per_month):
    records = []
    for (region, year), group in per_month.groupby(["region", "year"], sort=True):
        comparable = group.loc[group.released_comparable]
        record = {"region": region, "year": int(year), "eligible_months": len(group),
            "eligible_legal_hours": int(group.legal_hours.sum()), "delta_mean": float(group.delta.mean()),
            "delta_std_months_ddof1": float(group.delta.std(ddof=1)),
            "delta_abs_mean": float(group.delta.abs().mean()), "delta_abs_p95": float(group.delta.abs().quantile(.95)),
            "delta_min": float(group.delta.min()), "delta_max": float(group.delta.max()),
            "delta_hours_weighted_mean": float(np.average(group.delta, weights=group.legal_hours)),
            "positive_delta_months": int((group.delta > 1e-10).sum()),
            "negative_delta_months": int((group.delta < -1e-10).sum()),
            "identity_max_abs_residual": float(group.identity_residual.abs().max()),
            "released_comparable_months": len(comparable),
            "missing_released_months": int((~group.released_month_present).sum()),
            "released_support_mismatches": int(group.released_support_mismatch.sum()),
            "released_aggregation_mismatches": int(comparable.released_aggregation_mismatch.sum()),
            "released_hour_count_mismatches": int(comparable.released_hour_count_mismatch.sum()),
            "released_z_vs_samehours_abs_max": float(comparable.released_z_minus_samehours_z.abs().max()) if len(comparable) else None,
            "released_z_vs_samehours_abs_mean": float(comparable.released_z_minus_samehours_z.abs().mean()) if len(comparable) else None}
        records.append(record)
    annual = pd.DataFrame(records)
    cross_year, paired = [], []
    for region in REGIONS:
        group = per_month.loc[per_month.region == region]
        yearly = annual.loc[annual.region == region]
        record = {"region": region, "eligible_months": len(group),
            "cross_year_mean_delta_equal_years": float(yearly.delta_mean.mean()),
            "cross_year_std_annual_mean_ddof1": float(yearly.delta_mean.std(ddof=1)),
            "all_months_delta_std_ddof1": float(group.delta.std(ddof=1)),
            "all_months_delta_abs_mean": float(group.delta.abs().mean())}
        for year in YEARS:
            item = yearly.loc[yearly.year == year].iloc[0]
            record[f"year{year}_delta_mean"] = float(item.delta_mean)
            record[f"year{year}_delta_std_months_ddof1"] = float(item.delta_std_months_ddof1)
        cross_year.append(record)
        pivot = group.pivot(index="month", columns="year", values="delta")
        for earlier, later in ((2022, 2023), (2023, 2024), (2022, 2024)):
            aligned = pivot[[earlier, later]].dropna()
            change = aligned[later] - aligned[earlier]
            correlation = aligned[earlier].corr(aligned[later]) if len(aligned) > 1 else np.nan
            paired.append({"region": region, "earlier_year": earlier, "later_year": later,
                "same_calendar_month_pairs": len(aligned), "delta_change_mean": float(change.mean()),
                "delta_change_std_ddof1": float(change.std(ddof=1)),
                "delta_change_abs_mean": float(change.abs().mean()),
                "delta_same_month_correlation": float(correlation) if np.isfinite(correlation) else None,
                "delta_sign_change_months": int(((aligned[earlier] * aligned[later]) < 0).sum())})
    return annual, pd.DataFrame(cross_year), pd.DataFrame(paired)


def brief(per_month, excluded, annual, cross_year, paired, toy):
    comparable = per_month.loc[per_month.released_comparable]
    mean_by_year = annual.groupby("year").delta_mean.mean()
    max_identity = per_month.identity_residual.abs().max()
    lines = [
        "# 源地区粗聚合偏差诊断", "",
        "本诊断仅使用已查看的 BPAT、CISO、ERCO、FPL、ISNE、MISO、NYIS 在 2022—2024 年的数据；未打开新增六个地区的小时文件。没有训练、修改或选择任何预测模型。", "",
        "固定 EF、八燃料、非负发电的发电侧代理，单位为 gCO₂/kWh。每月只保留 hourly.valid 为真、八燃料有限且非负、总发电 G > 0 的同一组小时；小时覆盖至少 80%。固定 EF 并非完整电网排放真值。", "",
        "对同一小时集合，Cₜ = Σf Gₜf EFf / Gₜ，z = ΣₜGₜCₜ / ΣₜGₜ，δ = z − mean(C) = Covₚₒₚ(G,C) / mean(G)。协方差分母为 n，不是 n−1。下面的恒等式不涉及未来、预测或传输假设。", "",
        f"共检查 252 个地区年月，{len(per_month)} 个月达到小时覆盖阈值，{len(excluded)} 个月剔除。恒等式最大绝对残差为 {max_identity:.3e}；预设容差为 {IDENTITY_ATOL:.1e}。", "",
        f"合格月中，δ 的等月均值为 {per_month.delta.mean():.3f}，月间样本标准差为 {per_month.delta.std(ddof=1):.3f}，平均绝对值为 {per_month.delta.abs().mean():.3f}，绝对值 95% 分位为 {per_month.delta.abs().quantile(.95):.3f}，最大绝对值为 {per_month.delta.abs().max():.3f}。这些都是已查看源数据的描述性统计。", "",
        "各年先对每个地区的合格月份求均值，再对地区等权平均：" + "；".join(f"{year} 年 {value:.3f}" for year, value in mean_by_year.items()) + "。", "",
        "下表每格为该地区当年的月 δ 均值（月间样本标准差）；不将月份、小时或地区视为独立同分布的重复试验。", "",
        "| 地区 | 2022 | 2023 | 2024 | 年均 δ 的跨年标准差 |", "|---|---:|---:|---:|---:|"
    ]
    for _, row in cross_year.iterrows():
        values = [f"{row[f'year{year}_delta_mean']:.3f} ({row[f'year{year}_delta_std_months_ddof1']:.3f})" for year in YEARS]
        lines.append(f"| {row.region} | " + " | ".join(values) + f" | {row.cross_year_std_annual_mean_ddof1:.3f} |")
    lines.extend(["",
        f"独立 released monthly 记录中可比月为 {len(comparable)}，缺失记录 {int((~per_month.released_month_present).sum())}，非日历月支持失配 {int(per_month.released_support_mismatch.sum())}。可比月的小时数、总能量或燃料份额任一失配共 {int(comparable.released_aggregation_mismatch.sum())}；其中小时数失配 {int(comparable.released_hour_count_mismatch.sum())}。released z 与同小时 z 的平均／最大绝对差为 {comparable.released_z_minus_samehours_z.abs().mean():.3e}／{comparable.released_z_minus_samehours_z.abs().max():.3e}。", "",
        "per_month.csv 分开列出 released_z_minus_samehours_z 与 released_z_minus_time_mean，及小时数、总能量、燃料份额差。前者可反映聚合集合或质量 mask 的不同；不能把它归因于 Cov(G,C)/mean(G)。小时数、总能量和份额一致只说明这些汇总量相容，并不能证明两个小时集合完全相同。", "",
        f"同一日历月配对的跨年 δ 变化绝对值平均为 {paired.delta_change_abs_mean.mean():.3f}（地区和年份对等权）。该量只刻画源地区跨年的变化，不验证给定燃料份额、统计年龄后的跨地区稳定性。配对明细见 year_pair_delta_shift.csv。", "",
        "数学反例保持两小时的 G = (1,3) 和月总燃料能量不变：燃气 3、零排放燃料 1，燃气 EF 为 400。三种合法小时燃料分配都有 z = 300，但时间均值分别为 200、300、333.333，对应 δ 为 100、0、−33.333。它们可以具有相同统计支持、发布时间和年龄。脚本 self_test 已对该反例、恒定序列和随机非负序列验证恒等式。", "",
        "因此，月燃料份额与年龄本身不能识别时间均值，也不能识别未来日水平、日内形状或水平—形状残差的联合分布。即使补充逐时总发电 G，反例仍成立。把已发布粗 z 当作未来小时时间均值的无偏观测，需要额外的协方差、时间演化与跨域稳定性假设；在源地区学习到的条件关系属于可检验的迁移假设，不是由聚合恒等式推出的保证。", "",
        "本诊断没有对 fuel mix × age 拟合传输回归，也没有据此调整任何冻结模型。源数据的跨年统计、残差校准或组件采样不能自动证明新地区的条件可迁移性。这里没有提出新的理论方法。", "",
        "所有输出属于源假设诊断，不是外部验证成绩。region_year_summary.csv 给出等月与按合法小时加权的两种均值；cross_year_delta.csv 区分年内月间变化与年均值跨年变化。", ""
    ])
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("results/heldout-2026h1-data"))
    parser.add_argument("--output", type=Path, default=Path("results/source-aggregation-bias-v4"))
    parser.add_argument("--self-test-only", action="store_true")
    args = parser.parse_args()
    toy = self_test()
    if args.self_test_only:
        print(json.dumps(toy, indent=2))
        return
    data, output = args.data.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Immutable audit output: choose a fresh directory")
    # Enumerate only the seven inspected sources; never glob or open external BA data.
    paths = [data / f"{region}_hourly.csv" for region in REGIONS] + [data / "monthly.csv"]
    if any(not path.is_file() for path in paths):
        raise FileNotFoundError("All seven inspected source hourly files and monthly.csv are required")
    frozen = {str(path): sha(path) for path in paths}
    code_digest = sha(Path(__file__))
    output.mkdir(parents=True)
    write_json(output / "prefit_inputs.json", {"created_at_utc": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True, "new_external_hourly_read": False, "regions": REGIONS, "years": YEARS,
        "min_month_hour_coverage": MIN_COVERAGE, "identity_tolerance": IDENTITY_ATOL,
        "fuel_order": FUELS, "fixed_ef": EF.tolist(), "input_hashes": frozen,
        "script_sha256": code_digest, "environment": {"python": sys.version, "executable": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__}})
    shutil.copyfile(Path(__file__), output / Path(__file__).name)
    write_json(output / "self_test.json", toy)
    per_month, excluded = calculate(data)
    if per_month.empty or per_month.identity_residual.abs().max() > IDENTITY_ATOL:
        raise AssertionError("No eligible months or aggregation identity exceeded tolerance")
    annual, cross_year, paired = summaries(per_month)
    per_month.to_csv(output / "per_month.csv", index=False)
    excluded.to_csv(output / "excluded_months.csv", index=False)
    annual.to_csv(output / "region_year_summary.csv", index=False)
    cross_year.to_csv(output / "cross_year_delta.csv", index=False)
    paired.to_csv(output / "year_pair_delta_shift.csv", index=False)
    (output / "diagnostic_brief.md").write_text(brief(per_month, excluded, annual, cross_year, paired, toy))
    if any(sha(path) != digest for path, digest in frozen.items()) or sha(Path(__file__)) != code_digest:
        raise ValueError("Source input or diagnostic code changed during analysis")
    write_json(output / "postrun_verification.json", {"all_input_hashes_match": True,
        "script_hash_matches": True, "new_external_hourly_read": False,
        "eligible_months": len(per_month), "excluded_months": len(excluded),
        "identity_max_abs_residual": float(per_month.identity_residual.abs().max()),
        "output_hashes": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}})
    print(annual.to_string(index=False))
    print("Output:", output)


if __name__ == "__main__":
    main()
