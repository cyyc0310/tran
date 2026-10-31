# Dynamic V2: frozen exploratory protocol (2026-10-23)

This protocol is frozen before inspecting new 2023 validation scores. Earlier
2024 experiments motivated the research question, so 2023 is source validation,
not independent confirmation. Never use 2024 hourly labels in this run.

## Hypothesis and specific mechanism

A released historical fuel mix informs future daily CIF through source-learned
cross-fuel / daily-shape dependencies. The influence depends on the observation
support and age, with a source-estimated persistence for each output coefficient.
This is a prototype of support/age-aware multiscale transfer, not a novelty claim
for Ridge, DCT, Haar, decay models or residual calibration.

For each coefficient j, learn correction a_j + b_j exp(-age/tau_j) x,
where x includes fuel-share innovation relative to source seasonal mix,
support duration / 90 days and coverage. Fit on source daily profile minus its
source seasonal calendar template. All fuel features can affect every scale;
there is no assumption that coarse observations affect only low frequencies.
Tau derives from source 30-day residual coefficient normalized autocorrelation,
clipped to [14,365] days. It is a heuristic transition, not a physical state law.
Tau may shrink all scales similarly; a wavelet gain is not presumed.

The time, DCT and full orthonormal 24-hour Haar representations all retain 24
outputs and ten correction inputs. Age-dependent attenuation makes these
different from equivalent full orthogonal rotations under isotropic Ridge.
Haar on 24 points recursively separates 12,6,3 differences and uses DCT on the
remaining 3-dimensional coarse space. Per-coefficient persistence estimates
have equal count across the three complete bases. Global-age and static
controls have the same correction feature capacity with fewer persistence
parameters. Ridge regularization is fixed at 30 for every learned correction.

## Inputs, comparisons and fairness

Seven leave-region-out folds. Fit other six regions' 2022 daily hourly proxy
profiles. Evaluate held-out 2023 days only. Target inputs are calendar and
previously released monthly fuel-generation shares, coverage and support.
No target hourly CIF, generation, load or retrospective daily mean is an input.
Source hourly generation/CIF is training-only information.

Fixed EF = [1000,400,800,0,0,0,0,400]. Generation-weighted monthly fuel mix is
not observed arithmetic hourly CIF mean. Monthly history is conditioned as a
learned predictor, not a hard constraint on future daily means or generation.
Nonnegative generation preprocessing and active-field prefix handling remain
those of the existing release-prefix data adapter. Actual publication vintages
are missing; release is controlled month-end +30 days.

Methods: source calendar; lagged proxy level + centered source template;
source direct Ridge (mix, age, availability indicator, four season features);
static innovation correction; global-age correction; coefficient-specific
time/DCT/Haar persistence; quarterly Haar information ablation. All clip raw
points and predictive samples to [0,1000]. Missing released history falls back
to source calendar for correction methods and level-template.

Quarterly inputs require three contiguous complete calendar months. Shares
are MWh-weighted, coverage is the minimum constituent coverage and availability
is the maximum constituent release. This is derived from allowed historical
monthly observations, not from target hourly data. Annual ablation is deferred:
2022-only source training has no fully released prior annual observation.

Residual calibration is identical in procedure and selected source day indices
for all methods: fit source January–June 2022; retain July–December 2022 daily
residual curves with bias; draw the same 128 source day indices using seed13017.
Each method has its own residual values from those same days. Source seasonal
templates lack unseen month cells during calibration, using the training-period
overall profile; this temporal shift is explicit and can overinflate intervals.
Final means fit all source2022. Residual samples provide an empirical predictive
distribution, with no target coverage guarantee. Report raw point, shape and
level errors separately from calibrated distribution mean metrics.

## Frozen execution and acceptance

Fresh output directory only. Save code/protocol/data hashes before run; save
forecasts plus residual banks and a SHA seal before opening held-out hourly2023
labels. Score complete quality-valid UTC days; report CRPS, 90% coverage/width,
point/shape/level errors per region and equal-region macro. No tuning after
observing scores; any later revision needs a new run name and explicit reason.

This is feasible prototype screening: recommend further study only if the new
candidate improves CRPS versus both strong level-template and Ridge controls,
has broadly distributed regional gains, does not sacrifice hourly shape or
coverage, and improves over static/global-age and alternative-basis controls.
These descriptive checks are not a significance test or permission to claim
China performance. Negative results remain in the artifact.
