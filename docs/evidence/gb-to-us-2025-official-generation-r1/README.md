# Superseded first scoring run (feature duplication)

This run is retained for audit only. Review found `mix_ridge`/`mix_hgb` received duplicated month-mix-by-hour interaction columns because the calendar feature builder created same-named features before the target monthly values replaced them. Do not use its scores. Corrected feature construction and the valid rerun are in `gb-to-us-2025-official-generation-r2`.
