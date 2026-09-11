# FD-49 Verdict: Merit-order dispatch prior — NEGATIVE (mechanism absent)

**Date**: 2026-09-11 ｜ **Protocol**: telemetry-only mechanism audit,
zero training, pre-registered decision rule (transfer if >= 20/29
regions won by pooled >= 0.5 gCO2/kWh with no region degrading > +3).

**Design**: three priors on the target's thermal split
(coal/(coal+gas+petrol)), evaluated as full-mix CIF MAE against the
telemetry-grounded CIF:

1. `dispatch` — source-pooled gap→coal-frac binned mapping, transferred
   to the target with a DEPLOYMENT-VISIBLE gap (config annual shares ×
   weather CF from ERA5 proxy; the model's own information set).
2. `own` (ablation, violates zero-telemetry, diagnostic only) — same
   binned mapping keyed on the target's OWN telemetry gap.
3. `config` — the target's annual-mean thermal split (the current
   model's step-zero behaviour via the config-anchored logits).
4. `monthly` — the target's monthly-mean split (upper reference).

**Result**: `dispatch` loses on **0/30 regions won, pooled +38.65**
vs `config`.  The `own`-telemetry ablation is the decisive evidence:
it approximately equals `dispatch` everywhere (e.g. VIC1 204.5 vs
205.4, UK_16 27.9 vs 29.2) and still loses to `config` on all regions.
**If even the target's own hourly gap carried no exploitable mapping
to its thermal structure, the mechanism itself is absent at hourly
granularity — this is not a transferability failure.**

**Interpretation**: the thermal mix filling the dispatch gap is set
by each grid's unit-level cost stack, interconnector scheduling and
internal balancing — idiosyncratic per grid, not a stable function of
(gap, config) that source grids could teach a target.  The Deetjen &
Azevedo (2019) result relied on curated unit registries (NERC); at
hourly aggregate-share granularity with zero telemetry, the
merit-order prior has no signal.  The P1 direction is CLOSED.

**Relation to prior negatives**: this closes the physical-prior branch
adjacent to FD-46 (event-day conservative shrinkage, negative: phase
missing).  Together they bracket the C5 improvement space: the
event-day error is not recoverable by either statistical shrinkage
(FD-46) or dispatch physics (FD-49) at zero telemetry; the remaining
lever is the calibration side (P2 SPCI) or foundation-model fusion
(P3).

**Artifacts**: `scripts/experiments/fd49_merit_audit.py`,
`results/fd49_merit_audit.json` (all-30-region table above; UK
petroleum columns absent by schema and zero-filled; US_BPAT/US_FPL
config MAE 0.0 = no thermal generation in the annual mix).

**Verdict**: NEGATIVE — do not invest training runs in merit-order
thermal priors.  Keep the audit as a paper-side note: the
"physics-first" narrative should cite FD-49 as the honest boundary of
physical priors at zero telemetry, complementing FD-45/46/47's
behavioural evidence.
