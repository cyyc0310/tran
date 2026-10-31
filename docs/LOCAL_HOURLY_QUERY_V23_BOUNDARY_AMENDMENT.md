# V23 source-day key handling amendment

Registered after the first V23 run stopped before writing any score table or summary. The first frozen file recorded inherited source-day keys as full UTC-midnight strings such as `2024-07-01T00:00:00+00:00`, while the isolated source reader normalized each selected row to `YYYY-MM-DD`. After it had read the registered 14 source-days' numeric fuel rows, the final selected-key equality gate rejected this representation mismatch. No identification set, K-span, certificate metric, strategy comparison or summary was written or inspected.

The R1 implementation normalizes the already frozen day keys to their first ten ISO date characters for row selection and retains the original full UTC keys in its freeze. The 14 source-days, labels, query policies, support, thresholds, budgets, formulas and interpretation gates are unchanged. R1 writes to a new result directory and pins this amendment; it does not overwrite or reuse the failed V23 freeze.

This is a data-key normalization correction, not a response-driven method or cohort change. The first failed attempt is preserved at `results/local-hourly-query-v23/` with its original freeze and script digest. It is explicitly counted as source-row numeric access but produced no query policy or identification results.
