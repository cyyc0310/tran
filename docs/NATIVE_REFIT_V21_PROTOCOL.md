# V21 B0: same-offset native-feature residual controls

This source-development experiment tests whether ordinary source-trained
residual regression can use fine native structure that V17's direct regression
discarded. It was designed after V17/V18 outcomes; it is not independent
confirmation, a novel method, a new uncertainty law or permission to open target
or unseen regions. No such entry point exists. V7–V20/R1 remain immutable.

## Completed, independently audited parent

The fixed entry is `docs/evidence/native-hindcast-v20r1-verification.json`.
Prepare requires root's explicit 64-lowercase-hex `--parent-sha256`, a final
manifest field `numerical_audit_status='PASS'`, and completed R1 receipts bound
by that manifest. The receipts themselves remain PENDING_INDEPENDENT_AUDIT;
they must not be edited to admit this child. Missing, incomplete or pending-only
parents fail before numerical source parsing. R1 retains the original V20
generation exit 1, verification-only recovery, and aliases to unchanged original
archives. V21 never fabricates an original completion or regenerates a parent.

Verify all R1 manifest code/input/artifact hashes, its unchanged 65-code and
141-input closure and completed 91-prediction/21-bank aliases; recheck complete
fixed V18 evidence and the bound V17 packet/seal metadata. The V21 closure has
69 code/test/protocol files and exactly 158 input paths: inherited R1 inputs,
final R1 manifest and receipts, V17 packet/seal metadata and packets, V18 scored
reference curves, and three independent prelaunch reports. The interpreter is
already inherited and all environment package versions are frozen. Check the
entire registry, hashes and environment before and after each formal phase.
Prepare refuses a partial prior attempt; prediction and score refuse existing
outputs. Root generic, independent Python/math and science approval are required
before formal prepare/predict. Tests do not execute formal stages.

## One shared training task and fixed models

Use the same complete valid H1 proper cohort as V17: Jan25–Jun30 2024, 1100
region-days before each outer exclusion. Counts in source order are
157/158/158/153/158/158/158. For outer source r, use only the other six teachers s.
Each teacher's native N is the R1 H1 retrospective archive with both r and s
excluded from the fixed five-source H1 donor bank. These are geographic OOF
offline teachers and permit later H1 donor dates; they are not chronological
hindcasts. Every fitted response is the same 24-vector `y-native_mean(N)`.
The held source's fine responses are excluded from every fit. No 2025H1 scored
errors or H2 fine responses train any model.

Compare nine fixed input choices on exactly the same response rows:

- `native_only`: native particle mean24, standard deviation24 and common six
  calendar features, 54 columns total.
- `packet_only_D2/D8/T8/G5`: the full original V17 lag7 PacketHistory feature
  vector, including calendar and missing-history indicators. Feature dimensions
  are 19/51/51/35.
- `joint_D2/D8/T8/G5`: native mean24 and standard deviation24 plus the respective
  complete packet vector, 67/99/99/83 columns. Calendar appears only once.

Packets retain their original day-end+7-day release gate, 28-calendar-day
history window relative to the latest available packet, raw-energy EWMA scale14,
latest/std/weekday summaries and missing-history fallback. Four-packet daily
anchors are checked against D2 within 1e-9 for every training and query row.
No history-based row deletion is allowed. Future packet values remain strings
until selected by the lawful availability gate. The previous packet aggregator
did read source fine data; real publication vintages are still uncertified.

Every input uses region-balanced StandardScaler and the same fixed
ExtraTreesRegressor: 150 trees, min_samples_leaf10, random_state17023, n_jobs1,
region-balanced fit weights. There is no tuning grid or post-outcome model
replacement. Fit predicts an unclipped residual delta24. Packet-only residual
regression isolates the changed native offset task, native-only isolates
ordinary postprocessing of N, and joint-minus-packet isolates the addition of
native mean/std as inputs under the same task/model/cohort. This does not
complete a matched probabilistic residual-law comparison.

## Native noise, projections and compact archives

Seven 2025H1 queries use the inherited exact 181-day,128-particle native arrays,
origins and donor IDs from R1's original V18 copies. No new 2025 fine read occurs
during prediction. There are 55 raw arms: one shared alpha0 native identity plus
nine inputs times three projections (equal-hour day constant, four fixed UTC6h
block constants, full24) times alpha(.5,1). For each arm,
`raw_new=N+alpha*P(delta)`; the alpha0 branch returns an exact copy and preserves
native IDs. All 128 particles receive the same shift. This retains native noise
before clipping and introduces no bootstrap, coarse-error fit, heteroscedastic
law or QRF calibration. Day/block projections are ordinary equal-hour center
projections, not H/G conservation claims.

Each of seven NPZs contains only the original native raw/IDs, nine deltas, all
training/query input matrices, teacher path/region/date provenance, native
training mean/std, true training response and residual, anchors, region weights,
scaler statistics, the strict registered arm grid and per-arm center/width/clip
diagnostics. Seven JSON fit/as-of audits are separately sealed. Repeated expanded
raw arms need not be stored. Before sealing, reconstruct every registered arm,
check all values finite and enforce alpha0 identity, particle-noise preservation,
requested versus actual preclip center and realized clipped center. Recompute
cohort/features/provenance and scaler statistics from parent inputs; no second
tree fitting is required to verify the compact formula. Independent actual
audit may refit the fixed models if needed to validate fitted deltas.

Diagnostics retain requested point, actual unclipped point, realized initially
clipped point, original native saturation, newly clipped fraction, preclip and
initially clipped widths, and final spread clipping. A clip-induced narrowing
does not count as a new uncertainty mechanism. Point and scoring conventions
are the unchanged V18 `scaled_paths`: initially clip raw, take particle mean as
point, scale deviations about that point by the five fixed factors
(.75,1,1.25,1.5,2), then clip expanded paths. Hence all alpha0 native scores must
match V18 for all eight metrics within the existing 1e-9 score-check tolerance;
the raw/native/ID identity comparisons themselves remain exact.

## Complete sealing, scoring and references

The seven compact NPZs and seven JSON audits must have the exact registered
inventory and immutable hashes. All 55 formulas and the shared native identity
must pass before writing a prediction seal. Score refuses to read 2025 source
fine truth until the full seal/grid/hash and compact reconstruction checks pass.
Only then evaluate the original 1145 valid 2025H1 source days. All 275 curves
must retain that same cohort: 314875 day rows, 1925 regional rows and 275 macro
rows. Retain CRPS,90% coverage/width/interval score/lower and upper misses,
point-level MAE and point-shape MAE. Average within source before equal-source
macro comparisons and retain worst-source coverage.

Keep every joint-minus-packet comparison at the same packet, projection, alpha
and factor, plus explicitly labeled same-input-form T8-minus-G5 descriptive
comparisons. Preserve the full completed V18 curves and its inherited V17
ET/global/hetero/QRF and V16 additive references. Their day-anchor or prequential
response offsets and information sets differ from V21 and are disclosed as
background, not falsely called same-offset controls.

For each input and packet, select minimum macro CRPS among settings with macro
coverage in[90%,95%] and worst-source coverage>=85%; if none passes, choose
smallest violation, then CRPS with fixed lexical ties. T8's best packet-only or
joint setting passes the source budget gate only if it passes coverage and
beats native, native-only refit and the best D2/D8/G5 packets each by at least5%
CRPS, nonincreasing width and strictly lower interval score. Matched comparisons
remain fully reported regardless of this gate. No H-timing causal claim is made.
All custom-method, breakthrough, independent-confirmation, target authorization
and full-probabilistic-control flags remain false. The source gate is only a
development criterion; conditional-risk controls and independent confirmation
would require later, separately reviewed work.
