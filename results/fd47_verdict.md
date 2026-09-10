# FD-47 Verdict: Deployment-Real `fut_weather` (GFS/ICON vs ERA5 proxy)

**Status: POSITIVE — adopted as the deployment-default weather protocol.**
Date: 2026-09-09 (follow-up to FD-45/FD-46).

> **[SUPERSEDED 2026-09-10, FD-48 full-asset rerun]**  Every number
> below was computed on the DEGRADED data asset (post the 2026-09-06
> rebuild: farmblend files lost, 9 centroid files lost, AU+UK fuel
> telemetry lost).  The four arms were re-run on the fully restored
> asset on 2026-09-10 and the headline REVERSES:
> - pooled `era5` 56.05 (degraded) -> 43.98 (restored) — now matches
>   the FD-41 official baseline (42.89, repro pooled 42.86).
> - pooled `ensemble` 51.37 -> 50.41: the "zero deployment penalty"
>   finding (-1.44) is an artifact of the degraded asset; on the
>   restored asset the operational-forecast ensemble costs **+6.4
>   pooled MAE vs the ERA5 proxy**.
> - The infrastructure-gain story survives (gfs/icon give weather to
>   previously blind regions), but the fallback-family pooled -11.87
>   must be re-read as partially "gains from having weather at all"
>   vs the weather-blind state rather than forecast skill.
> - Deployment-default recommendation REVERSED: keep the ERA5-proxy
>   pipeline for the paper's headline; treat GFS/ICON as a
>   sensitivity/servability ablation (report the +6.4 penalty
>   honestly).  See `fd47_nwp_fut_weather.json` (restored) and the
>   backup `fd47_nwp_fut_weather_degraded_asset.json`.

## Question (pre-registered)

The FD stack feeds the 24 h horizon with an ERA5 *reanalysis proxy* for
day-ahead weather.  A deployer receives an operational NWP forecast
instead.  Two questions:

1. **Skill loss**: in regions WITH ERA5 weather (21/29), how much MAE
   does the proxy optimism hide — i.e. what does the FD-41/45 headline
   look like under the weather a real deployment gets?
2. **Infrastructure gain**: in regions WITHOUT any weather file (8/29,
   `has_real_weather=0` — the model previously ran weather-blind with
   zero channels), what does a public operational forecast archive
   (Open-Meteo `gfs_seamless`/`icon_seamless`, free, no account)
   provide?

## Protocol (mirrors FD-41/45 official pipeline bit-exact)

Same training (28-source leave-one-region-out, seed 0, ep900, p_cold
0.3, monthly config lag 1, tau 1.1 + ROUTE_TABLE), same test windows;
ONLY the target's inference-time future weather differs.  Arms:
`era5` (baseline), `gfs`, `icon`, `ensemble` (GFS+ICON mean).  The 336 h
history stays ERA5 in every arm (a deployer's past is observed).  For
the no-ERA5 family the history is also replaced (all-NWP deployment).
Verified: the `era5` arm reproduces the FD-45 dump bit-exact (QLD1
`pred_era5_icfg == pred_i_cfg`).

Join convention: valid-time UTC grid → region timeline axis (AU with
the month-based DST correction identical to `attach_fuel_and_exog`);
positional alignment for window patching (no timestamp round-trips —
AU DST duplicated hours cannot misalign).

Weather-derived future channels recomputed per arm: `fut_weather`
[temp, swr, w100, wind_cf, csi, regime24, tend6]; `fut_exog` cols
[wind_cf(2), csi(3), hdh(10), cdh(11), regime24(15), tend6(16)].
Gust/pressure future channels keep the ERA5 proxy (archive lacks the
columns; secondary features; documented approximation).

## Headline numbers (I_cfg, zero-telemetry tier)

### Family 1: era5-real (21 regions) — forecast skill loss ≈ negligible

| arm | pooled ΔMAE | regions better | regions worse |
|---|---|---|---|
| gfs | -0.84 | 11 | 9 |
| icon | -1.03 | 8 | 12 |
| ensemble | **-1.44** | 11 | 9 |

**The operational-forecast penalty is ZERO or slightly positive.**
Under the ensemble arm the pooled I_cfg MAE actually improves by 1.44
compared with the ERA5 proxy — reanalysis optimism is not a crutch the
headline depends on.  The FD-41/45 numbers stand under deployment-real
weather; the zero-telemetry cold-start bound is robust to forecast
error.  Largest single-region degradations: VIC1 +2.8 (ensemble),
NSW1 +1.8 (gfs); largest gains: US_ERCO -14.9, US_MISO -9.4 (both
high-wind-share US regions — see below).

### Family 2: no-era5 fallback (8 regions) — public NWP as zero-telemetry weather source

| arm | pooled ΔMAE | better | worse |
|---|---|---|---|
| gfs | -11.43 | 7/8 better | 1 worse |
| icon | -11.87 | 7/8 better | 1 worse |
| ensemble | -11.87 | 1 worse (US_BPAT) |

The 8 regions that previously ran weather-blind (UK_01/02/03/05/06/07/08
+ US_FPL/BPAT) gain **~12 MAE pooled** from the free public forecast
archive.  UK_05_Yorkshire: 66.1 → 33.5 (ensemble) — a 2x improvement
that closes the gap to the telemetry-rich regions.  US_FPL: 20.5 →
17.9.  This is the paper's "no infrastructure" narrative: the public
day-ahead forecast archive is itself a zero-telemetry weather source
that the framework converts into skill.

## Interpretation

1. **Deployment realism is now the default, not a caveat.**  The FD
   protocol can report the headline under operational NWP forecasts
   with no penalty — strengthening the claims rather than weakening
   them: the ERA5-proxy protocol was not implicitly reaping reanalysis
   skill.
2. **The GFS/ICON ensemble is the deployment-default arm.**  It wins
   or ties in both families (era5-real -1.44, fallback -11.87), 11 vs 9
   region wins, and is free/public.
3. **Where forecasts beat reanalysis (ERCO +14.9, MISO +9.4)**: both
   are high-wind-share regions where the ERA5 proxy was likely
   single-point mis-sampled (centroid vs fleet footprint); an
   operational forecast grid is smoother in the relevant band and the
   wind-CF channel noise reduces.
4. **Where forecasts lose (VIC1 +2.8, NSW1 +1.8, BPAT +3.4-4.1)**:
   AU summer DST seam and BPAT hour-offset — both are join/timeline
   artifacts to fix in FD-48 rather than forecast-skill issues.
   **[ERRATUM 2026-09-09, FD-48]**: this attribution is FALSIFIED by
   FD-48 join forensics (lag scans r=0.80-0.90 at best_lag=0, diurnal
   phase within 1 h, zero test-zone holes, seams train-zone-only and
   arm-symmetric).  The VIC1/NSW1 degradations are genuine day-ahead
   forecast error on high-wind-share regions; BPAT is an OOD
   composition effect (hydro region never trained with weather).  See
   `results/fd48_join_forensics.md` and `results/fd48_verdict.md`.
5. **No coverage gaps**: all 29 regions join at ≥ 99.87% coverage;
   GFS/ICON columns full-year non-null.

## Paper mapping (C1 information hierarchy)

FD-47 directly upgrades claim C1: the I_cfg tier is now demonstrated
with deployment-real weather in 29 regions — including 8 regions that
have NO reanalysis infrastructure at all and gain 12 MAE from the
public archive.  The "zero telemetry" claim extends from "no carbon
intensity history" to "no weather infrastructure required" — the
framework converts a free public forecast API into carbon-intensity
forecast skill.

## Follow-ups

- FD-48 (small): fix the AU DST seam in the AU timeline correction
  (VIC1/NSW1), re-examine BPAT hour offset; expected to recover the
  remaining degradations in the era5-real family.
  **[ERRATUM 2026-09-09, FD-48]**: the seam/offset attribution above
  was falsified; FD-48 ran as join forensics instead (see
  `results/fd48_verdict.md`).  Separately, FD-48 discovered the
  `data_2023/weather/` asset degradation (2026-09-06 rebuild lost all
  farmblend files + 9 centroid files).  FD-45/47 numbers in this
  verdict were computed on the degraded (centroid-only) asset and must
  be re-run after restoration before entering the paper.
- FD-49 (P1, pre-registered earlier): merit-order dispatch prior for
  event-day phase (FD-46 showed shallow statistical fixes have no
  space; FD-45's 6 significant event-inflation regions await).
- SPCI conformal upgrade (P2) uses the ensemble arm as the reference
  deployment configuration.
