"""Label-free target inference from timestamped public monthly generation tables."""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from transcif.research.multicountry import GROUPS,HISTORY,METHODS,predict,public_features


def verify_input_seal(seal: dict, panel_bytes: bytes, manifest_bytes: bytes) -> pd.Timestamp:
    if hashlib.sha256(panel_bytes).hexdigest()!=seal['panel_sha256']:
        raise ValueError('panel does not match sealed snapshot')
    if hashlib.sha256(manifest_bytes).hexdigest()!=seal['source_manifest_sha256']:
        raise ValueError('source manifest does not match sealed snapshot')
    stamp=pd.Timestamp(seal['input_available_at_utc'])
    if stamp.tzinfo is None:raise ValueError('sealed availability timestamp requires timezone')
    return stamp


def nbs_months(panel: pd.DataFrame, province: str, origin: pd.Timestamp,
               snapshot_at: pd.Timestamp) -> tuple[pd.DataFrame,dict]:
    """Only the recorded snapshot is available; never invent an older vintage."""
    if origin.tzinfo is None or snapshot_at.tzinfo is None:
        raise ValueError('origin and snapshot timestamps require timezones')
    if origin<snapshot_at:
        raise ValueError('snapshot was not available at origin')
    local=origin.tz_convert('Asia/Shanghai')
    cutoff=local.tz_localize(None).to_period('M')-2
    selected=panel.loc[panel.province.eq(province)].copy()
    if selected.empty:raise ValueError('province absent')
    selected['period']=pd.PeriodIndex(selected.year.astype(str)+'-'+selected.month.astype(str).str.zfill(2),freq='M')
    if selected.period.duplicated().any():raise ValueError('duplicate province months')
    selected=selected.loc[selected.period<=cutoff]
    fields=[f'{g}_100m_kwh' for g in GROUPS[:5]]
    rows,excluded=[],[]
    for _,row in selected.iterrows():
        values=row[fields].to_numpy(float)
        total=float(row.total_100m_kwh)
        reason=None
        if not np.isfinite(np.r_[values,total]).all():reason='missing_current_value'
        elif (values<0).any() or total<=0 or values.sum()<=0:reason='invalid_generation'
        elif abs(total-values.sum())>.30000001:reason='unresolved_scope_or_remainder'
        if reason:
            excluded.append({'month':str(row.period),'reason':reason});continue
        # Five reported technologies are renormalized; the small closure gap is
        # not treated as an observed renewable technology.
        rows.append({'period':row.period,**dict(zip(GROUPS[:5],values*100000.)),
                     'other_renewable':0.})
    if not rows:raise ValueError('no complete public months')
    months=pd.DataFrame(rows).set_index('period').sort_index()
    if cutoff not in months.index:raise ValueError('exact lag-2 month unavailable')
    audit={'province':province,'snapshot_available_at':snapshot_at.isoformat(),
           'latest_allowed_month':str(cutoff),'eligible_months':len(months),'excluded_months':excluded,
           'input_unit':'100 million kWh converted to MWh',
           'closure_policy':'five-share renormalization when abs(total-five)<=0.3 source units',
           'other_renewable':'zero model placeholder within rounding ambiguity; not an observed zero',
           'scope':'NBS above-designated-size industrial generation',
           'official_release_vintage_certified':False}
    return months,audit


def infer(package: dict, months: pd.DataFrame, origin: pd.Timestamp, hours: int = 24):
    if origin.tzinfo is None or not 1<=hours<=168:
        raise ValueError('timezone required and horizon must be 1..168 hours')
    if tuple(package['feature_columns'])!=tuple(HISTORY):raise ValueError('model feature contract differs')
    clock=pd.date_range(origin.tz_convert('Asia/Shanghai'),periods=hours,freq='h')
    needed=set(clock.tz_localize(None).to_period('M')-2)
    if not needed.issubset(set(months.index)):raise ValueError('forecast crosses unsupported month')
    f=public_features(months,pd.Series(clock.tz_localize(None)))
    if f[list(HISTORY)].isna().any().any():raise ValueError('incomplete public features')
    predictions=predict(package['models'],f)
    rows=[]
    for method in METHODS:
        radius=package['interval_radii'][method]
        for time,point in zip(clock,predictions[method]):
            rows.append({'time':time.isoformat(),'method':method,'point':float(point),
                         'source_reference_lower90':float(point-radius),
                         'source_reference_upper90':float(point+radius)})
    outside={}
    for column,bounds in package['feature_support'].items():
        values=f[column].to_numpy(float)
        mask=(values<bounds[0]-1e-10)|(values>bounds[1]+1e-10)
        if mask.any():outside[column]={'hours_outside':int(mask.sum()),'source_range':bounds,
                                      'target_range':[float(values.min()),float(values.max())]}
    metadata={'status':'LABEL_FREE_RESEARCH_INFERENCE_ONLY','origin':origin.isoformat(),'hours':hours,
              'target_hourly_labels_used':False,'target_model_fitting':False,
              'estimand':'generation-side fixed-factor proxy; not consumption CIF',
              'intervals':'source calibration references; China coverage has not been validated',
              'outside_source_coordinate_support':outside,
              'source_model':package['metadata']}
    if 'train_year' in package['metadata']:
        month=origin.tz_convert('Asia/Shanghai').tz_localize(None).to_period('M')
        metadata['model_training_age_months']=month.ordinal-pd.Period(f"{package['metadata']['train_year']}-12",freq='M').ordinal
    return pd.DataFrame(rows),f,metadata
