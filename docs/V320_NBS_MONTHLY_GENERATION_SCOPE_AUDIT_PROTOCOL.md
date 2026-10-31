# V320 protocol: audit the usable scope of NBS-derived monthly generation fields

Registered 2026-10-30 before analysis. The CSV is the OREO `China Provincial Monthly Electricity Generation by Technology` release downloaded from its public dataset page; the page identifies NBS as the source and CC BY 4.0 as the license. The linked October 2026 methodology is the authoritative source for scope and period interpretation.

## Question

Can this public province × month × technology table serve as a strict monthly supply cap in V319's OD allocation model, and can it close the annual provincial generation/OD/demand balance?

## Frozen checks

1. Record the category schema, provinces, dates, duplicate cells, missing cells, and 2023 coverage.
2. Treat February's reported generation as a **January–February cumulative value**, not February-only. For annual comparisons, use that value once plus March–December. Do not manufacture separate January and February observations.
3. Compare the OREO monthly-derived 2023 province total with the 2023 Ember monthly generation estimates and V310 annual Zenodo generation totals. Do not choose one source as truth; report paired gaps and rankings.
4. Recompute 2023 annual net balance diagnostics using OREO monthly-derived generation, V310 annual OD, and the existing V64 modeled demand only as a diagnostic. Explicitly preserve the known statistical-scope mismatch: OREO monthly generation covers industrial enterprises above designated size, while V64 demand is modeled all-sector consumption and V310 OD is a separate annual transfer product.
5. Interpret an above-designated-size generation total as **not a hard cap on total provincial exports** unless scope is restricted to those same plants or a validated full-generation adjustment is provided.

## Execution amendment: same-source annual reference data

Before extending the audit, publicly download OREO's NBS-derived annual generation (all plants) and annual electricity consumption tables, both CC BY 4.0. Compare the 2023 monthly-above-size generation sum against the same-source annual all-plant total, and independently compute the annual balance residual `G + OD net imports - demand` using the NBS annual totals. Keep the prior V64 load-proxy comparison as a historical diagnostic only. This amendment uses an independently documented same-provider annual table to measure the monthly-to-annual scope gap; it does not claim the V310 OD matrix shares the NBS accounting boundary.

No CIF values will be calculated, no hourly target opened, and no scope mismatch will be “balanced” by an estimated correction. This audit can accept or reject the NBS-derived series as a drop-in strict cap; a rejection does not invalidate other user-held data whose scope may be broader.
