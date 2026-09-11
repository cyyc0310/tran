"""FD-49 step 1: merit-order dispatch-prior mechanism audit (telemetry-only,
zero training).

Hypothesis (Deetjen & Azevedo ES&T 2019): the thermal mix filling the
dispatch gap shifts systematically with gap size — cheap units (coal)
first, expensive units (gas) last.  If this mapping is a stable function
of (gap, config) across source grids, a target region can inherit it as
a physics prior for its thermal-split head, replacing the static
config-anchored split.

Audit design (pre-registered):
  * per region, per hour: gap = 1 - (wind+share + solar share) from fuel
    telemetry (30-min UK / hourly AU+US data; the same stream the model
    would see in deployment).
  * fit the thermal-structure regression  coal_share/(coal+gas+petrol)
    ~ f(gap)  per region on the *source* side of the LORO split (the
    10-12 month training window), evaluate transfer on the held-out
    target region: predicted thermal mix vs actual.
  * transfer metric: MAE of the predicted coal/gas fractions, and the
    resulting CIF error  |sum_f pred_s_f * ef_f - cif|  using canonical
    EFs, compared against (a) the static config prior (current model
    behaviour at step zero) and (b) the monthly-mean thermal mix.
  * decision rule (pre-registered): the dispatch prior transfers if it
    beats the static config prior on >= 20/29 regions by a pooled margin
    >= 0.5 gCO2/kWh, without any region degrading > +3 gCO2/kWh.

Output: results/fd49_merit_audit.json + fd49_merit_audit.md
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_2023" / "fuel"
RESULTS = ROOT / "results"

# Canonical EFs (gCO2/kWh) — the same vector the physics layer uses.
EF = {"coal": 820.0, "gas": 490.0, "petroleum": 650.0,
      "nuclear": 12.0, "hydro": 24.0, "biomass": 230.0,
      "imports": 250.0, "other": 480.0, "solar": 48.0, "wind": 11.0}
THERMAL = ["coal", "gas", "petroleum"]

# LORO source/target split mirrors the training protocol: the last
# 10 months are the test window for the target; for the audit the
# *source* fit window = Jan-Oct, transfer evaluation = the target
# region's full year (deployment view).
FIT_END = "2023-11-01"


def load_region(region: str) -> pd.DataFrame:
    """Load hourly fuel telemetry, canonicalised to share columns."""
    f = DATA / f"{region}_fuel_2023_hourly.csv"
    df = pd.read_csv(f, parse_dates=["hour"])
    if "perc_coal" in df.columns:                       # UK schema
        out = pd.DataFrame({"hour": df["hour"]})
        for fu in ["coal", "gas", "petroleum", "nuclear", "hydro",
                   "biomass", "imports", "other", "solar", "wind"]:
            col = f"perc_{fu}"
            out[fu] = (df[col].astype(float) / 100.0
                       if col in df.columns else 0.0)
        out["cif"] = df["cif_gco2_per_kwh"].astype(float)
    else:                                               # AU/US schema
        out = pd.DataFrame({"hour": df["hour"]})
        for fu in ["coal", "gas", "petroleum", "nuclear", "hydro", "solar",
                   "wind", "biomass", "imports", "other"]:
            col = f"gen_{fu}"
            out[fu] = (df[col].astype(float) if col in df.columns else 0.0)
        tot = df["total_gen"].astype(float).clip(lower=1e-6)
        for fu in ["coal", "gas", "petroleum", "nuclear", "hydro", "solar",
                   "wind", "biomass", "imports", "other"]:
            out[fu] = out[fu] / tot
        # intensity-weighted CIF from the mix itself (telemetry-grounded)
        out["cif"] = sum(out[fu] * EF[fu] for fu in EF)
    return out


def thermal_mix(shares: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Vectorised coal/(coal+gas+petrol) and thermal total (Series in/out)."""
    t = shares["coal"] + shares["gas"] + shares["petroleum"]
    coal_frac = shares["coal"] / t.where(t > 1e-4)
    return coal_frac, t


def main() -> None:
    regions = sorted(
        p.name.split("_fuel_")[0]
        for p in DATA.glob("*_fuel_2023_hourly.csv"))
    rows = []
    for tgt in regions:
        try:
            df_t = load_region(tgt)
        except Exception as e:                           # noqa: BLE001
            rows.append({"target": tgt, "error": str(e)})
            continue
        # source fit: pooled fit-window data from all OTHER regions
        src = []
        for r in regions:
            if r == tgt:
                continue
            try:
                d = load_region(r)
            except Exception:                            # noqa: BLE001
                continue
            m = d["hour"] < FIT_END
            src.append(d[m])
        src = pd.concat(src, ignore_index=True)

        # target's own gap for evaluation: DEPLOYMENT-VISIBLE definition —
        # config annual shares scaled by weather CF (the model's own
        # information set), NOT telemetry shares.  Telemetry enters only
        # on the source side as the fitted signal.
        try:
            wx = pd.read_csv(ROOT / "data_2023" / "weather"
                             / f"{tgt}_weather_2023_hourly.csv")
            wcf = wx["wind_speed_100m"].astype(float).values / 3.6
            wcf = np.clip(wcf / 11.0, 0.0, 1.0)          # IEC-style CF proxy
            swr = wx["shortwave_radiation"].astype(float).values
            swr = np.clip(swr / 700.0, 0.0, 1.0)
            ann_wind = float(df_t["wind"].mean())
            ann_solar = float(df_t["solar"].mean())
            n = min(len(wcf), len(swr), len(df_t))
            gap_t = 1.0 - (ann_wind * wcf[:n] + ann_solar * swr[:n])
            df_t = df_t.iloc[:n].reset_index(drop=True)
            gap_t = np.clip(np.asarray(gap_t, dtype=float), 0.0, 1.0)
        except Exception:
            gap_t = 1.0 - (df_t["wind"] + df_t["solar"])
            gap_t = np.clip(np.asarray(gap_t, dtype=float), 0.0, 1.0)
        coal_frac_t, thermal_t = thermal_mix(df_t)

        # --- three priors on the target's thermal mix -----------------
        # (a) dispatch prior: bin source gap->coal-frac mapping
        src_gap = 1.0 - (src["wind"] + src["solar"])
        src_gap = np.clip(np.asarray(src_gap, dtype=float), 0.0, 1.0)
        src_coal_frac, _ = thermal_mix(src)
        bins = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 1.0])
        prior_coal = np.full(len(gap_t), np.nan)
        for i in range(len(bins) - 1):
            m = (gap_t >= bins[i]) & (gap_t < bins[i + 1])
            sm = (src_gap >= bins[i]) & (src_gap < bins[i + 1])
            if m.sum() == 0 or sm.sum() < 50:
                continue
            prior_coal[m] = src_coal_frac[sm].mean()
        # fallback bin for gap>=0.85
        m = (gap_t >= 0.85) & np.isnan(prior_coal)
        if m.sum() > 0:
            sm = src_gap >= 0.85
            if sm.sum() >= 50:
                prior_coal[m] = src_coal_frac[sm].mean()
        # gaps with no coverage: leave NaN, fall back to monthly mean
        nan_mask = np.isnan(prior_coal)
        mm = df_t.groupby(df_t["hour"].dt.month)[
            ["coal", "gas", "petroleum"]].transform("mean")
        coal_mm = (mm["coal"] /
                   mm[["coal", "gas", "petroleum"]].sum(axis=1)).values

        # Ablation arm (diagnostic only, violates zero-telemetry): the
        # target's OWN telemetry gap — if even this carries no signal,
        # the mechanism itself is absent at hourly granularity, not
        # merely non-transferable.
        own_gap = 1.0 - (df_t["wind"] + df_t["solar"])
        own_coal_prior = np.full(len(own_gap), np.nan)
        for i in range(len(bins) - 1):
            m = (own_gap >= bins[i]) & (own_gap < bins[i + 1])
            tm = (src_gap >= bins[i]) & (src_gap < bins[i + 1])
            if m.sum() == 0 or tm.sum() < 50:
                continue
            own_coal_prior[m] = src_coal_frac[tm].mean()
        m85 = (own_gap >= 0.85) & np.isnan(own_coal_prior)
        if m85.sum() > 0:
            tm = src_gap >= 0.85
            if tm.sum() >= 50:
                own_coal_prior[m85] = src_coal_frac[tm].mean()

        # (b) static config prior = target's own annual mean split
        cfg_coal = np.nanmean(coal_frac_t)

        # --- evaluate CIF error of each prior on the target ------------
        ev = ~nan_mask
        gap_ev = gap_t[ev]
        prior_c = prior_coal[ev]
        coal_ev = coal_frac_t[ev]
        thermal_ev = thermal_t[ev]

        def cif_err(coal_pred: np.ndarray) -> float:
            coal = coal_pred * thermal_ev
            gas = (1.0 - coal_pred) * thermal_ev
            # pet share assumed small/absorbed: split it as actual mean
            pet_ev = (df_t["petroleum"][ev] /
                      thermal_ev.clip(1e-6)).mean()
            coal = coal * (1 - pet_ev)
            gas = gas * (1 - pet_ev)
            cif_p = (coal * EF["coal"] + gas * EF["gas"]
                     + pet_ev * thermal_ev * EF["petroleum"]
                     + df_t["nuclear"][ev] * EF["nuclear"]
                     + df_t["hydro"][ev] * EF["hydro"]
                     + df_t["biomass"][ev] * EF["biomass"]
                     + df_t["imports"][ev] * EF["imports"]
                     + df_t["other"][ev] * EF["other"]
                     + df_t["solar"][ev] * EF["solar"]
                     + df_t["wind"][ev] * EF["wind"])
            return float(np.mean(np.abs(cif_p - df_t["cif"][ev])))

        e_dispatch = cif_err(prior_c)
        e_config = cif_err(np.full_like(prior_c, cfg_coal))
        e_monthly = cif_err(coal_mm[ev])
        # own-telemetry ablation (diagnostic)
        own_c = own_coal_prior[ev]
        own_ok = ~np.isnan(own_c)
        e_own = (cif_err(own_c[own_ok]) if own_ok.sum() > 100
                 else float("nan"))

        rows.append({
            "target": tgt,
            "n_eval": int(ev.sum()),
            "mae_dispatch": round(e_dispatch, 2),
            "mae_own": None if np.isnan(e_own) else round(e_own, 2),
            "mae_config": round(e_config, 2),
            "mae_monthly": round(e_monthly, 2),
            "delta_config": round(e_dispatch - e_config, 2),
        })
        print(f"{tgt:28s} dispatch {e_dispatch:6.2f} "
              f"own {e_own:6.2f} config {e_config:6.2f} "
              f"monthly {e_monthly:6.2f}", flush=True)

    pooled_d = np.mean([r["delta_config"] for r in rows
                        if "delta_config" in r])
    n_win = sum(1 for r in rows if r.get("delta_config", 0) < 0)
    n_tot = sum(1 for r in rows if "delta_config" in r)
    summary = {
        "date": "2026-09-11",
        "design": "source-pooled gap->coal-frac prior, transfer to target",
        "pooled_delta_vs_config": round(float(pooled_d), 2),
        "regions_won": n_win,
        "regions_total": n_tot,
        "rows": rows,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "fd49_merit_audit.json").write_text(json.dumps(summary, indent=1))
    print(f"\npooled delta vs config: {pooled_d:+.2f}  "
          f"({n_win}/{n_tot} regions won)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
