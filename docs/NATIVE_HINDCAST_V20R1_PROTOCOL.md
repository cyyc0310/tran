# V20R1 verification-only recovery

V20's original formal generation attempt failed with exit 1 during its pre-seal
exact replay. The original four producer/test/protocol files, freeze, log and
112 NPZs remain unchanged. No original V20 prediction seal or completed marker is
created. V20R1 generates no new predictions, changes no numerical values and
copies no archives. It registers aliases pointing to the original 91 primary
prediction archives and 21 separate bank provenance files. A bank file is not a
prediction archive.

The first failure is documented in the pinned original-directory diagnostic
script, JSON and report. NPZ serialization preserved every q value but changed
the original mixed byte strides to C-contiguous storage. The resulting arithmetic
order caused raw and center differences of roughly 1e-13 in the original checker.
Restoring only the original q layout restored exact equality in the isolated
first case. This is a verifier correction, not a model or calibration change;
the diagnostic alone does not certify the remaining archives.

## Immutable entry and closure

Only `prepare`, `verify` and `check` entry points exist. There is no prediction,
score or target entry point. Formal prepare/verify requires prior approval by
root generic, independent Python/math and scientific reviewers. Their three
reports in `results/native-hindcast-v20r1` must exist before prepare and are
hashed as inputs. Prepare refuses existing R1 freeze or output markers; verify
refuses existing R1 output markers. Existing partial attempts are preserved.

The original V20 freeze and failed log, the three diagnosis artifacts, all
112 original NPZs, the original 21 input paths and the three review reports enter
an exact 141-input closure. Original input hashes must match the original V20
freeze; source and parent links are rechecked by the unchanged V20 checker.
The code closure is the original 62 files plus this revision's runner, tests
and protocol, exactly 65 files. The interpreter/environment identity is checked
before and after verification. The failed attempt must match its fixed freeze,
log and diagnostic hashes, have exactly the registered archive inventory, and
have neither original completion marker. Arbitrary incomplete parents are
rejected; this exception admits only the explicitly pinned diagnosed failure.

## Exact verification

For each of 21 pair-excluded banks, numerically parse only the registered source
H1 complete valid days, recreate the original native bank with the frozen
function, and compare every saved bank field including provenance using exact
dtype equality and `np.array_equal`. Keep all original schema, support,
normalization and own-mix reconstruction checks. These legacy numerical
validation tolerances do not replace the additional exact field comparison.

After bank equality passes, replay using the freshly fitted bank, retaining the
same storage layout as original generation. Do not use the C-contiguous q loaded
from NPZ for replay. Recompute the original 14 legal coarse laws and original
sampler IDs; every field in all 84 teacher archives must match exactly. Any value,
ID, dtype, schema or provenance mismatch fails. The seven query archives must
match the inherited V18 native raw arrays, origins and IDs exactly and their
declared row provenance. No `allclose` or relaxed tolerance substitutes for any
prediction comparison.

The unchanged V20 split meanings remain explicit: 42 H1 Jan25–Jun30 teachers
are retrospective geographic OOF with both outer and teacher regions excluded
from the five-source bank. They may use later dates within the same H1 proper
window and are not chronological hindcasts. Another 42 H2 Jul1–Dec31 teachers
use the fixed pre-Jul1 bank and legal coarse prefixes as chronological hindcasts.
Seven 2025H1 query native archives are inherited exact copies. No H2 or 2025 fine
response is numerically parsed by this revision; no new model is trained or
scored. Prior coarse extraction did read fine data, and real publication vintages
remain uncertified. This recovery does not alter those limitations.

## Receipts and downstream eligibility

After all 91/21 checks pass, recheck the entire code/input/environment closure
and the captured freeze hash. Only then write new R1 `verification_seal.json`
and `completed.json`. Both record `producer_attempt_status='failed_exit_1'`,
`generation_attempt_success=false`,
`recovery='verification_only_fresh_bank_layout'`, and
`new_predictions_generated=false`. The seal registers original paths, exact
counts, comparison method and saved/fresh q strides. Completion binds the seal
and freeze hashes. `check` repeats the closure and full exact replay.

These receipts retain `numerical_audit_status='PENDING_INDEPENDENT_AUDIT'`.
A separately owned independent actual-data audit must pass before root seals
`docs/evidence/native-hindcast-v20r1-verification.json` with
`numerical_audit_status='PASS'`. V21 must reject any parent lacking that completed,
audited manifest. This revision does not authorize V21 prediction, target or
unseen data access, nor claim the original generation attempt succeeded.

Synthetic tests reproduce the NPZ storage-layout failure, prove exact replay
with verified fresh banks, reject corrupted values and IDs, reject unpinned or
ineligible parents, register the exact split grid, enforce three-review input
presence, and prevent completion after a freeze mutation. No formal stage is
executed as part of the tests.
