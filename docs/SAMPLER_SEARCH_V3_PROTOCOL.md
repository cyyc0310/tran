# V3 sampler mechanism search: frozen sequential experiment

The mistake of interest is marginal predictive CDF error under a strict count of
all evaluated forward rows. CRPS excess, failure frequency (CDF > .25), retained
ESS, setup cost and runtime are secondary. Runtime occurs under concurrent load
and is descriptive, never a speed claim. This synthetic inverse benchmark is not
real-grid validation.

## Falsifiable mechanism changes

1. Global antithetic scrambled Sobol tests symmetry-aware coverage without pilot cost.
2. Full central-difference posterior Hessian (2d² forward rows per center plus one
recenter row) tests whether rotation-aware curvature offsets local setup cost.
Pilot fractions 1/8 and 1/4 ablate discovery versus production allocation.
3. Prediction-separated multistart defensive Student-t mixtures test mode coverage,
nuisance-aware center selection, heavy-tail robustness and full-density correction.
4. A budget policy runs geometry only if it leaves at least half the total budget
for production. If pilot likelihood ESS exceeds 35%, it uses independent final
Sobol draws. Infeasible geometry is explicitly logged and uses full-budget Sobol.
This is a standard diagnostic gate, not new theory. No axes are read.
5. Adaptive ESS-temperature SMC tests path-based concentration. Eight resample/move
stages use fixed pCN kernels reversible for the standard Gaussian prior; the last
stage reaches beta=1. All initial and move rows count. Weight ESS does not capture
resampling genealogy. Temperatures estimated from particles are an approximation.

All adaptive importance pilots and stencils are discarded. Final weights use the
complete frozen defensive mixture q, including its prior and every component.
Gaussian and Student-t proposals use their normalized full multivariate densities.
No evaluator attributes, quadrature coordinates, reference metrics or case labels
are available to infer_v3.

## Round boundaries and stopping rule

Before every round, write configuration, source SHA256 and full source snapshots.
Round 1: five existing development problems, seeds 6000–6009, budgets 128/512,
eight configurations (800 runs). Selection risk is equal-case/budget mean CDF +
.25 times worst cell mean CDF + .05 times fraction CDF>.25. Select at most three
non-baselines. Round 2: eight previously viewed stress problems explicitly relabeled
development, seeds 7000–7009, budgets 128/512; Sobol/prior and top three (800 runs).
Freeze the best non-baseline by the same rule. Round 3: six new compound cases with
new rotations and dimensions 4/8/16, combined shifts, curvature, mode multiplicity,
noise and observation ages, seeds 8000–8019; Sobol/prior and frozen winner (720 runs).
No optimization follows heldout scores. Total 2320 runs, failures retained.

Independent coarse/fine quadrature references are from benchmark_v2. Save CDF
refinement mean/max. Small improvements comparable to numerical reference error
are inconclusive, even when a descriptive paired interval excludes zero. Intervals
are descriptive, unadjusted for multiple comparisons. A useful algorithm claim
requires improvements over strong Sobol in multiple independent heldout regimes,
without material failure-rate regressions. Otherwise report regime dependence or
no improvement. No novelty claim follows a tuned standard sampler.

## Audit rerun

The initial experiment omitted transitive source hashes (inference.py and scoring
helpers). Its original manifests remain unchanged. An audited deterministic rerun
uses identical methods, cases, seeds and selection; it changes no algorithm based
on heldout scores. The new manifest includes the entire project-local import
closure, package initializers and focused tests. Metric CSVs must match the
initial runs exactly except timing. Original outputs receive only a supplemental
posthoc dependency attestation, never a retroactive pre-run hash claim.
