"""Post-score regime diagnostics; never used to select heldout predictions."""

import numpy as np
import pandas as pd


def paired_month_interval(scores, candidate, baseline="latest", repeats=5000, seed=10313):
    """Resample the same full calendar months in every region.

    Preserve within-region day counts before equal-region averaging. The
    interval describes the observed year/regions, not independent global grids.
    """
    frame = scores[scores.method.isin([candidate, baseline])].copy()
    if frame.duplicated(["region", "origin", "method"]).any():
        raise ValueError("Select exactly one latency/year before pairing")
    paired = frame.pivot(index=["region", "origin"], columns="method", values="crps")
    if paired.isna().any().any() or not {candidate, baseline}.issubset(paired.columns):
        raise ValueError("Methods require identical scored populations")
    delta = (paired[candidate] - paired[baseline]).rename("delta").reset_index()
    delta["month"] = pd.to_datetime(delta.origin, utc=True).dt.month
    regions = sorted(delta.region.unique())
    months = sorted(delta.month.unique())
    grouped = delta.groupby(["region", "month"]).delta.agg(["sum", "count"])
    index = pd.MultiIndex.from_product([regions, months], names=["region", "month"])
    grouped = grouped.reindex(index, fill_value=0)
    sums = grouped["sum"].to_numpy().reshape(len(regions), len(months))
    counts = grouped["count"].to_numpy().reshape(len(regions), len(months))
    estimate = float(np.mean(sums.sum(axis=1) / counts.sum(axis=1)))
    draws = np.random.default_rng(seed).integers(len(months), size=(repeats, len(months)))
    numerator, denominator = sums[:, draws].sum(axis=2), counts[:, draws].sum(axis=2)
    valid = (denominator > 0).all(axis=0)
    samples = (numerator[:, valid] / denominator[:, valid]).mean(axis=0)
    if len(samples) == 0:
        raise ValueError("No bootstrap draw contains observations for every region")
    low, high = np.quantile(samples, [.025, .975])
    return {"difference": estimate, "descriptive_ci_low": float(low), "descriptive_ci_high": float(high),
            "regions": len(regions), "region_days": len(delta), "bootstrap_valid": len(samples)}


def input_regimes(monthly, origins, region, extra_delay):
    """Describe lawful inputs independently of outcomes or fitted models."""
    origins = pd.DatetimeIndex(pd.to_datetime(origins, utc=True))
    history = monthly.loc[monthly.region.eq(region)].copy()
    for col in ("period_start", "period_end", "available_at"):
        history[col] = pd.to_datetime(history[col], utc=True)
    history.available_at += pd.Timedelta(days=int(extra_delay))
    history = history.loc[history.period_start < history.period_end].sort_values(["period_end", "available_at"])
    end = history.period_end.to_numpy()
    released = history.available_at.to_numpy()
    eligible = (end[None, :] <= origins.to_numpy()[:, None]) & (released[None, :] <= origins.to_numpy()[:, None])
    selected = np.max(np.where(eligible, np.arange(len(history)), -1), axis=1) if len(history) else np.full(len(origins), -1)
    ages = np.full(len(origins), np.nan)
    coal = np.full(len(origins), np.nan)
    for index in np.unique(selected[selected >= 0]):
        take = selected == index
        row = history.iloc[index]
        ages[take] = (origins[take] - row.period_end).total_seconds() / 86400
        coal[take] = float(row.fuelshare_coal)
    same_month = (history.period_start.dt.month.to_numpy()[None, :] == origins.month.to_numpy()[:, None])
    seasonal_available = (eligible & same_month).any(axis=1)
    return pd.DataFrame({"origin": origins.astype(str), "region": region, "delay": extra_delay,
                         "age_days": ages, "history_months": eligible.sum(axis=1),
                         "seasonal_history": np.where(seasonal_available, "same-month history", "source seasonal fallback"),
                         "age_regime": pd.cut(ages, [-np.inf, 60, 120, 240, np.inf],
                                               labels=["<=60d", "61-120d", "121-240d", ">240d"]).astype(str),
                         "coal_regime": pd.cut(coal, [-np.inf, .1, .4, np.inf],
                                                labels=["<=10% coal", "10-40% coal", ">40% coal"]).astype(str)})
