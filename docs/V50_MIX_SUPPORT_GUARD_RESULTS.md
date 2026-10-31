# V50 monthly-mix source-support guard result

Run date: 2026-10-27. Protocol: [V50_MIX_SUPPORT_GUARD_PROTOCOL.md](V50_MIX_SUPPORT_GUARD_PROTOCOL.md). This is a development diagnostic; λ=0.75 was chosen from the preceding development shrinkage path.

## Result

The training-feature min/max guard applied the λ=.75 correction only to held days inside the range observed among the other nine sources. It fell back to the raw anchor on 100% of SC and SCL days and on none of the other eight regions.

The guard reduced the source-macro particle clipping rate from 8.97% to 5.97%, close to the raw-anchor 5.66%. It preserved pooled hourly CRPS improvement versus raw anchor (92.78 vs 125.14), and interval score (577.53 vs 712.16), with width 508.56 vs 527.98 and source-macro coverage 94.39% vs 88.50%. However it broke the required per-source coverage: SC fell from 99.29% under the unguarded λ=.75 correction to 75.06% after fallback. SC was the source with the largest level-error correction and its feature values were above training support. NWMT and WACM remained the lowest non-fallback coverage regions at 86.64% and 89.10%.

The guard did what it was designed to do for SCL: clipping returned from 86.1% under unguarded λ=.75 to its 56.5% raw-anchor level, but SCL hourly CRPS and interval score also returned from 0.39/17.8 to 1.62/40.5. Thus the support rule trades away gains whenever the source's feature is outside training range; it does not know whether that extrapolation was harmful (SCL) or valuable (SC).

## Decision

The fixed support-range fallback is rejected as a transfer rule. It exposes a key identifiability problem: a hard OOD gate can protect one unsupported source while removing the major gain on another unsupported source. The current correction therefore lacks a safe basis for extrapolating to an unseen high-intensity region such as a potentially coal-heavy China province. No method freeze and no unseen-source test are justified yet.

## Artifacts

- Script: `scripts/probe_v50_mix_support_guard.py`
- Output: `results/disjoint-calibration-v50-mix-support-guard/`
- Summary includes frozen input, seal, protocol, script, and output hashes.
