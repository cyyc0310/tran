from copy import deepcopy

import pytest
pytest.importorskip('eccodes')

from scripts.audit_gfs_pilot import validate_window


def fixture():
    points=[{'id':'P','latitude':40.,'longitude':-100.}]
    meta={'initialization_at':'2022-01-01T00:00Z','forecast_origin_at':'2022-01-02T00:00Z',
          'horizon':2,'native_leads':[24,25,26],'snapshots':[{'lead':h} for h in [24,25,26]]}
    weather={'initialization_at':meta['initialization_at'],'first_lead':24,'end_lead':26,
             'points':points,'rows':[{'point':'P','valid_at':f'2022-01-02T0{h}:00Z'} for h in [0,1]]}
    return meta,weather,points


@pytest.mark.parametrize('change',['short_window','run','origin','duplicate','outside','missing_snapshot','points'])
def test_auditor_rejects_partial_or_mislabelled_windows(change):
    meta,weather,points=deepcopy(fixture())
    validate_window(meta,weather,points)
    if change=='short_window':weather['end_lead']=25
    elif change=='run':weather['initialization_at']='2022-01-02T00:00Z'
    elif change=='origin':meta['forecast_origin_at']='2022-01-02T01:00Z'
    elif change=='duplicate':weather['rows'][1]=weather['rows'][0]
    elif change=='outside':weather['rows'][1]['valid_at']='2022-01-02T03:00Z'
    elif change=='missing_snapshot':meta['snapshots'].pop()
    elif change=='points':weather['points']=[{'id':'OTHER','latitude':40.,'longitude':-100.}]
    with pytest.raises(ValueError):validate_window(meta,weather,points)
