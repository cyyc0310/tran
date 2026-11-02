import numpy as np
import pandas as pd
import pytest

from transcif.research.multicountry import GROUPS,HISTORY,METHODS
from transcif.research.public_inference import infer,nbs_months,verify_input_seal


def panel():
    rows=[]
    for month in range(1,10):
        rows.append({'province':'Shanghai','year':2026,'month':month,'total_100m_kwh':100.1,
                     **dict(zip([f'{g}_100m_kwh' for g in GROUPS[:5]],[90.,0.,0.,5.,5.]))})
    return pd.DataFrame(rows)


def test_snapshot_cutoff_rounding_and_future_rows():
    p=panel();origin=pd.Timestamp('2026-11-01T00:00+08:00');snapshot=pd.Timestamp('2026-10-31T00:00Z')
    months,audit=nbs_months(p,'Shanghai',origin,snapshot)
    assert months.index.max()==pd.Period('2026-09')
    assert months.other_renewable.eq(0.).all()
    assert 'not an observed zero' in audit['other_renewable']
    np.testing.assert_allclose(months.thermal,9000000.)
    p=pd.concat([p,p.iloc[[-1]].assign(month=10,thermal_100m_kwh=9999.)],ignore_index=True)
    pd.testing.assert_frame_equal(months,nbs_months(p,'Shanghai',origin,snapshot)[0])
    with pytest.raises(ValueError,match='not available'):
        nbs_months(p,'Shanghai',origin,pd.Timestamp('2026-11-02T00:00Z'))


def test_missing_and_unresolved_latest_month_are_not_filled():
    p=panel();origin=pd.Timestamp('2026-11-01T00:00+08:00');snapshot=pd.Timestamp('2026-10-31T00:00Z')
    p.loc[p.month.eq(9),'nuclear_100m_kwh']=np.nan
    with pytest.raises(ValueError,match='lag-2'):
        nbs_months(p,'Shanghai',origin,snapshot)
    p=panel();p.loc[p.month.eq(9),'total_100m_kwh']=105.
    with pytest.raises(ValueError,match='lag-2'):
        nbs_months(p,'Shanghai',origin,snapshot)


def test_inference_no_labels_or_fitting_and_unsupported_month_rejected(monkeypatch):
    months,_=nbs_months(panel(),'Shanghai',pd.Timestamp('2026-11-01T00:00+08:00'),pd.Timestamp('2026-10-31T00:00Z'))
    monkeypatch.setattr('transcif.research.public_inference.predict',lambda m,f:{k:700*f.thermal.to_numpy() for k in METHODS})
    package={'models':{},'feature_columns':list(HISTORY),'metadata':{'source_countries':['US']},
             'interval_radii':{m:10. for m in METHODS},'feature_support':{c:[-1.,1.] for c in HISTORY}}
    prediction,features,meta=infer(package,months,pd.Timestamp('2026-11-01T00:00+08:00'))
    assert len(prediction)==24*len(METHODS) and len(features)==24
    assert not meta['target_hourly_labels_used'] and not meta['target_model_fitting']
    with pytest.raises(ValueError,match='unsupported month'):
        infer(package,months,pd.Timestamp('2026-12-01T00:00+08:00'))


def test_seal_rejects_mismatched_panel_and_manifest():
    import hashlib
    data=b'panel';manifest=b'manifest'
    seal={'panel_sha256':hashlib.sha256(data).hexdigest(),
          'source_manifest_sha256':hashlib.sha256(manifest).hexdigest(),
          'input_available_at_utc':'2026-10-31T00:00:00Z'}
    assert verify_input_seal(seal,data,manifest).tzinfo is not None
    with pytest.raises(ValueError,match='panel'):
        verify_input_seal(seal,b'newer panel',manifest)
    with pytest.raises(ValueError,match='manifest'):
        verify_input_seal(seal,data,b'wrong manifest')
