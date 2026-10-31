# V22: minimum-generation information for strict tail counts

This is source exploration designed after observing the V20 outcomes. It is not
preregistered independent confirmation, a new forecasting method, a new theorem,
or evidence of improved future prediction. No prediction model or target entry
point exists. All analysis concerns exact contemporaneous oracle information in
the seven already authorized source regions during 2024H2.

The question is limited: given block carbon energy H=sum(G_h*C_h), total G,
positive G_h and support 0<=C_h<=B=1000, how does knowing the exact minimum hourly
G in each six-hour block constrain K=sum(1[C_h>tau])? Thresholds remain the four
V20 values 100, 300, 600 and 900. Equal-to-threshold hours are not events. This is
an HG/support information class, not the full eight-fuel or physical dispatch
identification problem. Exact refers to recorded decimal proxy observations,
not certified real measurement precision or publication vintages.

## Fixed information comparison

Use the exact same seven-source 1000 complete valid UTC region-day mask sealed
by V20. Four budgets are computed for every day and threshold:

- T8: four block H/G pairs, with within-block G allocation otherwise positive
  and arbitrary. Its sharp count set is inherited unchanged from V20.
- T8_G2: additionally each block sum(G_h^2), four new G-only scalars. The
  inherited Cauchy/bounded-variance count set remains a necessary outer set,
  not a proven sharp moment identification set.
- T8_Gmin: additionally each block exact min(G_h), also four new G-only scalars.
  Its count set below is sharp for this HG/support/minimum information class.
- T8_G24: all hourly G values, inherited fixed-G sharp HG/support count set.

The scalar count is a dimension comparison, not Shannon information, measurement
cost, or certification of a real data channel. Gmin and G2 are different
information maps; neither is claimed to contain the other or dominate all risk
functions. Inclusions required are G24 subset Gmin subset T8 and separately
G24 subset G2 subset T8. No Gmin/G2 inclusion is enforced.

## Exact fixed-G sorting reference

Let S_k be the sum of the k smallest G values and L_k the sum of the k largest.
For a nonempty event subset A, feasibility is
`tau*G_A < H <= tau*G_total+(B-tau)*G_A`; K=0 is feasible iff H<=tau*G_total.
When H<=tau*G_total, the upper inequality is automatic: feasible counts are
0 through the largest k with `tau*S_k < H` (only zero when no positive k works).
When H>tau*G_total, the strict lower inequality is automatic: feasible counts
run from the smallest k with `H<=tau*G_total+(B-tau)*L_k` through n.
Hence fixed-G feasible integer counts are contiguous. The sorted computation
is a standard subset/knapsack reference, not an algorithmic novelty claim.

## Exact-minimum sharp set and construction

For n>=2, the exact minimum must satisfy 0<m<=G_total/n. For n=1, m=G_total
is necessary. K=0 is feasible iff H<=tau*G_total; K=n iff H>tau*G_total.
For 0<k<n:

- If H<=tau*G_total, k is feasible iff `H>tau*k*m`.
- If H>tau*G_total, k is feasible iff
  `H<=tau*G_total+(B-tau)*(G_total-(n-k)*m)`.

For the first case, give each selected event hour G=m, and place any remaining
generation above n*m on one complement hour. This attains the least selected
mass k*m. For the second case, give every complement hour G=m and put the excess
on an event hour, attaining selected mass G_total-(n-k)*m. In both constructions
the exact minimum remains m. Positive strict carbon margins and continuous
within-support carbon allocations attain every admitted H; the pure module
returns exact Fraction witnesses for every admitted k. Uniform G, n=1, n=2,
H=0, H=B*G and strict equalities are explicitly tested.

Consequently K=0 is certified (the whole count set is {0}) iff H<=tau*m, exactly
the same certificate as having all fixed G. This statement concerns absence of
exceedance in this information class. It does not assert Gmin identifies other
counts as well as G24, dominates G2 generally, or provides a universal minimum
measurement budget. Missing support or certified-capacity assumptions would
change the identification problem.

## Input, freeze and output contract

The fixed parent path is `docs/evidence/tail-count-v20-verification.json`.
Prepare requires root's explicit 64-lowercase-hex `--parent-sha256`; an absent,
unsealed or different parent fails before source numeric parsing. That digest
is written into the new freeze and cannot be resupplied to run/check. Verify
every hash in the parent manifest's code/input/artifact closure, the unchanged
Tail producer contract, and the fixed V19 ancestor closure. Tail completion must
bind exactly its four CSV outputs, source-only flags and 1000-day cohort.
The new direct closure contains eight code/test/protocol files (four frozen
Tail files plus four new V22 files) and 18 inputs: Tail manifest, freeze,
completion, four CSVs, seven source files, interpreter and three independent
prelaunch review reports. Environment identity is frozen and rechecked.

Only after root generic, independent Python/math and scientific prelaunch review
may prepare/run execute. The isolated inherited source parser reads only
registered seven-source 2024H2 rows, filters 2025 timestamps before numeric
conversion, and converts fuel decimal strings directly to Fraction. This oracle
extraction does use historical fine source data to simulate added statistics;
it does not pretend those new scalars were actually published or available for
future forecasts. No target, unseen-region or 2025 fine values are read.

Recompute all four budgets on the full grid: 16000 day rows, 64000 block rows,
112 region summary rows and 16 macro rows. Each row retains the complete integer
count set, lower/upper, span, cardinality, identification and K0/forced-event
flags, and actual strict count as a containment diagnostic. Four blocks combine
by exact Minkowski sum. The three inherited budgets' day and block curves must
match the parent CSVs exactly, including sets and flags. Actual count does not
select thresholds, cohorts or budgets. Summaries average within region first,
then seven regions equally; differing valid-day counts remain explicit.

Refuse overwrite of any freeze or partial output. Recheck all frozen inputs,
code and environment and the initially captured freeze hash before writing
completion. Completion binds all four output hashes and preserves explicit
post-outcome/source-oracle/no-prediction/no-target flags. Independent actual
numerical audit and result interpretation follow; synthetic tests alone do not
approve the real output values or establish information utility outside this
restricted historical setting.
