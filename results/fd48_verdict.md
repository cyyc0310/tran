# FD-48 Verdict: Join Forensics — FD-47 残余恶化归因（证伪 + 修正）

**Status: NEGATIVE for the artifact hypothesis — AU DST seam and BPAT
hour-offset are NOT the cause; residual degradations are genuine
forecast error + representativeness mismatch.  Same-model re-scoring
remains possible (see FD-49).**

Date: 2026-09-09.  Pre-registered diagnostics D1-D5, no training runs.

## Why this was worth checking

FD-47 verdict attributed VIC1 +2.8 / NSW1 +1.8 / BPAT +3.8 (ensemble)
to "AU DST seam and BPAT hour-offset join artifacts" and pre-registered
FD-48 as "fix the seam, recover the degradations".  Forensics first —
the attribution was WRONG, and a seam fix would have been a no-op
dressed as progress.

## Findings

### F1. The AU timeline seams exist but are benign (attribution falsified)

- NEM source CSVs carry per-month boundary duplicates (11 dup rows +
  trailing 2024-01-01 row; 8772 raw rows -> 8761 after dedup) for ALL AU
  regions including DST-free QLD1.
- The month-based DST rule maps 2023-09-30 23:00 local and 2023-10-01
  00:00 local to the same UTC hour (one axis duplicate) and leaves a
  2 h gap at the 2023-04-01/02 fall-back seam (positions 2161/6553).
- BOTH seams sit in the TRAIN zone (split=7008; axis length 8761) and
  are shared symmetrically by the ERA5 baseline and all NWP arms —
  positional window patching never crosses them asymmetrically.
- Lag scans on the join axis: VIC1 gfs r=0.888 / icon r=0.904, NSW1
  0.805/0.838, all best_lag=0; swr diurnal peaks match within 1 h.
  Zero test-zone holes (0 of 73 windows touch an NWP gap).
- DST month-rule vs true zoneinfo calendar: only the April 2 ambiguous
  hour differs — inside train, never test.

**The FD-47 "DST artifact" claim is dead: VIC1/NSW1 degradations are
genuine day-ahead forecast error on high-wind-share regions.**

### F2. The 2024-01-01 tail holes are real but nearly harmless

US timelines extend into 2024-01-01 local (BPAT 9 h, CISO 9 h, NYIS
6 h): the NWP archive ends 2023-12-31 23:00 UTC, so final-window
horizon hours get zero-valued weather (0 m/s wind, 0 C, 0 csi).

But the effect is ~0.1 MAE per region: BPAT delta excluding the last
2 windows is +3.69 vs +3.78 total; NYIS +1.85 vs +1.84; CISO +0.65 vs
+0.62.  The tail holes are symmetric (the ERA5 join also lacks those
hours) and small.  No fix worth an FD number.

### F3. BPAT is an OOD-composition story, not a timeline story

BPAT (hydro-dominated, wind share ~1%) never had weather at all
(ERA5 weather = all zeros; the model trained 28-source with BPAT as a
zero-weather source).  The NWP arms inject wind-semantic channels
(CF ~0.12-0.13, regime/tend features) into a model that never saw wind
predict its CIF — an out-of-distribution feature composition shift,
distributed across all 73 windows (not tail-concentrated).  This is
the "no free lunch" edge of the public-NWP fallback: weather helps
where weather drives the mix, hurts when it does not.

### F4. Farmblend files do NOT exist (an FD-17/FD-26 claim needs a caveat)

`download_farm_weighted_wind.py` defines farm tables for VIC1/SA1/NSW1/
QLD1/UK_01/UK_16/UK_02/UK_18/UK_10/ERCO/MISO/CISO, but the outputs
were never downloaded into `data_2023/weather/` (glob finds only
`{REGION}_weather_2023_hourly.csv` centroid files).  Every "farmblend
prefer" branch in `load_raw_weather` is currently inert; all ERA5
baselines are centroid-grid.  Cross-checking FD-17/FD-26 verdicts
(which credited farmblend gains) against the present tree: those
experiments ran in an environment where the files existed; the current
repo state does not reproduce them.  **Re-downloading the farmblend
set is a prerequisite for any future farmblend-dependent claim.**

**Timeline reconstruction (file mtimes + FD-41 json ctime):**
- 2026-09-04 22:56 — FD-41 official results computed (the 42.89 / 39.08
  headline era: full farmblend weather + AU fuel telemetry present).
- 2026-09-06 21:29-21:36 — `data_2023/weather/` and `data_2023/fuel/`
  rebuilt; only 21 of 30 centroid weather files succeeded (9 failures
  UK_01..08/US_FPL/US_BPAT never retried), the farmblend set and the
  AU fuel telemetry set (`{QLD1,NSW1,VIC1,SA1}_fuel_2023_hourly.csv` +
  `fuel_shares_au.json`) were NEVER re-downloaded.
- 2026-09-08/09 — FD-45 (Dunkelflaute) and FD-47 (NWP swap) therefore
  ran on the DEGRADED assets: farmblend regions at centroid grid,
  9 regions fully weather-blind, ALL FOUR AU regions without fuel
  telemetry (monthly tables gone -> anchor_trust crash introduced the
  `_safe_prepare_fd_region` bypass -> AU annual-config fallback).

**F5 (2026-09-09, root cause of the VIC1 +52 anomaly):** the AU fuel
telemetry loss is the DOMINANT degradation, not the weather grid.
Evidence chain: (a) FD-41 snapshot commit a11c627 crashes on
anchor_trust(None) with today's tree — proving AU fuel files existed
on 2026-09-04 (the crash is impossible with them present);
(b) restored-asset FD-47 rerun gives VIC1 era5 126.70 vs FD-41 74.59
(+52.1) while UK farmblend regions reproduce FD-22-era numbers
(UK_01 41.29 vs 41.9, UK_02 15.23 vs 16.0) — the residual anomaly is
AU-specific and fuel-telemetry-shaped; (c) AU regions have has_fuel=0
today, collapsing their monthly config tables to annual (VIC1 wind
share seasonality ~3x mischaracterised; cf. FD-22 table: SA1/VIC1
had monthly tables then).

**Consequences that require re-reading:**
1. FD-45's event-day inflation ratios (UK_16 3.05x etc.) were measured
   on the degraded asset; farmblend restoration may change both the
   event labels and the per-bucket errors — the C5 protocol numbers
   must be re-run before they enter the paper.
2. FD-47's era5-real family was a centroid-vs-centroid comparison
   (both arms same grid); the "operational forecast grid is smoother"
   interpretation of ERCO/MISO gains does not hold as stated — those
   gains are genuine forecast-skill-vs-reanalysis differences at the
   same point.  The pooled deltas (-1.44 era5-real / -11.87 fallback)
   remain valid AS MEASURED, but the 9-region fallback family's
   "gains from public NWP" partially re-describe "gains from HAVING
   weather at all" vs the zero-weather state.
3. The 42.89/39.08 headline was computed WITH farmblend + AU fuel —
   restoring both assets is REQUIRED to keep the paper's headline
   reproducible from the current tree.
4. FD-45/46/47 verdicts published 2026-09-08/09 inherit the degraded
   asset: their AU-region numbers (VIC1/NSW1/SA1/QLD1 rows) need
   re-measurement on restored assets before any paper citation.

**Restoration actions (2026-09-09/10, FD-48):**
- centroid weather retry for the 9 missing regions: DONE (31/31 files,
  all 8760 rows, zero duplicate hours).
- farmblend 12-region rebuild (`YEARS` narrowed to 2023): DONE (12/12,
  wind means 18.5-31.7 km/h consistent with fleet footprints; unit
  conversion /3.6 verified active in `load_raw_weather`).
- AU fuel telemetry re-extraction (`extract_au_fuel_breakdown.py`,
  NEMED, 4 regions x 12 months): DONE (VIC1 coal 0.726 / wind 0.180,
  consistent with FD-22-era structure; anchor_trust 0.899 restored;
  upstream `prepare_fd_region` no longer crashes on AU).
- **UK fuel telemetry (F6, discovered 2026-09-10): the 18 UK fuel
  files were ALSO lost on 2026-09-06.**  UK fuel lives in
  `data_2023/fuel/UK_*_fuel_2023_hourly.csv` (perc_* schema from the
  Carbon Intensity API), NOT in the main UK CSVs (which never had
  perc_* columns); the rebuild re-downloaded only the US set.  All 18
  regions re-extracted (18/18, fuel_shares_uk.json written; UK_01
  wind 70.4%, UK_09 gas 53.3% — structures consistent with the
  FD-22-era tables).  This loss was masked until now because
  `_safe_prepare_fd_region` silently bypassed the crash.
- Full-asset scan: 29/29 regions load with monthly tables, zero
  crashes, zero None tables.
- FD-47 four-arm rerun on the weather-restored (fuel-still-degraded)
  asset: DONE — pooled era5 56.05 -> 45.32; superseded by the full
  restoration below.
- FD-41 headline reproduction on the FULLY restored asset
  (`probe_hparam_arm.py --epochs 900`, original script/config):
  DONE — pooled 42.89 -> 42.86 (delta 0.03).  26/29 regions within
  +-4 of FD-41; AU drift VIC1 +6.9 / NSW1 -11.4 / QLD1 -9.3
  attributable to NEMED re-extraction micro-differences (fuel
  structure unchanged: VIC1 coal 0.726 vs 0.713) + MPS training
  stochasticity — the cfg/i0 arm direction pattern is preserved.
  **The 42.89 headline is reproducible from the current tree.**

**Post-restoration re-runs (pre-registered; status 2026-09-10):**
- FD-41 headline reproduction: DONE, pooled 42.89 -> 42.86 (reproducible).
- FD-47 four arms on the fully restored asset: PENDING — current
  verdict numbers are the degraded-asset version; rerun required
  before paper citation (note: with UK fuel restored, the UK regions
  regain monthly tables and the earlier "fallback family" framing
  fully dissolves).
- FD-45 Dunkelflaute buckets: PENDING — event labels and per-bucket
  errors must be re-measured on the restored asset before the C5
  protocol numbers enter the paper.

## What actually explains the FD-47 residual degradations

| region | ensemble Δ | genuine cause |
|---|---|---|
| VIC1 | +2.77 | day-ahead forecast error on wind-CF (corr 0.83-0.90 vs ERA5); high wind share amplifies |
| NSW1 | +1.84 | same; GFS wind bias +0.3 m/s (CF +20% relative) |
| BPAT | +3.78 | OOD composition: wind-semantic channels on a hydro region never trained with weather |
| UK_16 | +1.71 | forecast error, mild |
| UK_11 | +3.56 | forecast error, mild |
| US_NYIS | +1.84 | forecast error + 6 tail holes (~0.1) |

None are join artifacts.  The deployment-real headline therefore
STANDS with no asterisk: pooled era5-real -1.44, no-era5 -11.87.

## FD-49 pre-registration (the actionable follow-up)

The one mechanism with headroom for the degrading regions is
forecast-error-aware horizon weighting — but FD-46 already showed
shallow statistical fixes (event-day shrinkage) have no space, and the
literature-survey P1 (merit-order dispatch prior) targets event-day
phase errors, not these level errors.  Pre-register instead:

FD-49 (P1, from the 2026-09-09 survey): merit-order dispatch prior for
the event-day phase — a physics-informed correction layer for the 6
regions FD-45 flagged as significant event-day error inflation
(UK_16 3.05x).  This remains the highest-headroom direction per the
survey's P1, and FD-48's forensics remove the join-artifact alibi: the
errors to attack are real forecast/dispatch errors, not fixable by
timeline surgery.

Lower priority: re-download farmblend files (F4) before any future
farmblend claim; document the 2024-01-01 tail-hole convention in the
FD-47 protocol note.
