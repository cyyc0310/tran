# Capacity-Contract Correction Discovered during V139

The exact grouped-capacity solver in V92 formerly evaluated K=0 and K=T before verifying global monthly energy feasibility. For a physically incompatible input, those shortcuts could return a nonempty singleton even though no hourly allocation exists. It now returns `None` for such inputs. Calibration or forecasting code must distinguish unavailable physical information from a zero-risk certificate.

V139 joins the V120 r2 source rows to the independently saved V100 monthly-energy audit. The denominator and hours match. Of 589 months, 76 are capacity-infeasible: CHPD/FMPP/GRID have 24 each, AVA/GCPD have one each, and NWMT has two. Forty-four archived threshold intervals fail robust-label containment. All failures lie in the input-infeasible subset. All 1,026 endpoint pairs from the 513 feasible months exactly match V103.

| Artifact | Correction status |
|---|---|
| V120 r2 source CV and its headline | Invalid as overall performance evidence: infeasible months were counted as physical singleton intervals. Use the V139 audited full-denominator comparison instead. |
| V121 and V122 frozen support-fallback results | Their target-side capacity routines check monthly feasibility separately. The saved correction audit now recalibrates source support using only input-feasible months: V121 endpoints do not change; V122 changes one PACE τ=900 row from [0,508] to [0,68], with saved proxy endpoint coverage retained. This is a posthoc correction on saved target rows, not a new blind confirmation or full raw-data rerun. |
| V126 and V127 source support calibration | Their source routines call the same solver without an explicit monthly-energy check and use `None` for availability. The missing solver guard can affect archived support calibration; re-audit before reuse. |
| V100 / V102 / V103 | They prefilter using the input-only monthly-energy condition. V139 independently joins and verifies every V103 endpoint used in its feasible source subset. |
| V129–V138 | Their monthly and block contracts explicitly check energy feasibility or use a no-capacity fallback. Saved cohort/block availability reports and caller checks do not expose this endpoint-shortcut defect. This is a scope assessment, not a new full raw-data rerun. |

The archived result files are retained for provenance. Do not silently replace them or quote their previous counts as current results. The V139 output saves the affected rows, input hashes, four solver boundary checks, full-denominator fallback and new paired calibration frontier.

The [support-boundary correction runner](../scripts/v139_support_boundary_correction.py) saves its 86 target threshold rows and source-cutoff changes in `results/v139-support-boundary-correction-r1/`. All capacity-available saved targets remain covered after the cutoff correction; source availability counts and thresholds differ from the archived calibration. V126/V127 are not re-audited by this comparison.
