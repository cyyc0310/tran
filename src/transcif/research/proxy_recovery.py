"""Month-local reported-generation proxies; unreported fuels are not observed zeros."""
from __future__ import annotations

import pandas as pd

from transcif.research.eia_proxy import _proxy_rows


def month_local_proxy(rows: pd.DataFrame, region: str, schemas: dict):
    if rows.index.tz is None or not rows.index.is_unique:
        raise ValueError('unique timezone-aware source timestamps required')
    rows=rows.sort_index()
    periods=rows.index.tz_convert('UTC').tz_localize(None).to_period('M')
    frames,audits=[],{}
    for period in periods.unique():
        block=rows.loc[periods==period]
        hourly,audit=_proxy_rows(block,region,schemas)
        frames.append(hourly)
        audits[str(period)]=audit
    if not frames:raise ValueError('empty source window')
    return pd.concat(frames,ignore_index=True),audits
