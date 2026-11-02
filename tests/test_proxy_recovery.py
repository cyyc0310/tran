import numpy as np
import pandas as pd

from transcif.research.proxy_recovery import month_local_proxy


def test_future_oil_reporting_does_not_remove_past_reported_proxy():
    jan=pd.date_range('2023-01-01',periods=72,freq='h',tz='UTC')
    feb=pd.date_range('2023-02-01',periods=72,freq='h',tz='UTC')
    rows=pd.DataFrame({'source_snapshot':0,'net_generation_mw':100.,'generation_coal_mw':60.,
                       'generation_gas_mw':40.,'generation_oil_mw':np.nan},index=jan.append(feb))
    rows.loc[feb[:24],'generation_oil_mw']=0.
    schema={'0':{'fuel_columns':['generation_coal_mw','generation_gas_mw','generation_oil_mw']}}
    result,audit=month_local_proxy(rows,'TEST',schema)
    first=month_local_proxy(rows.loc[jan],'TEST',schema)[0]
    pd.testing.assert_frame_equal(result.iloc[:72].reset_index(drop=True),first)
    assert result.iloc[:72].valid.all()
    assert result.iloc[72:96].valid.all() and not result.iloc[96:].valid.any()
    assert 'generation_oil_mw' in audit['2023-01']['snapshots']['0']['unobserved_columns_excluded_not_certified_zero']
