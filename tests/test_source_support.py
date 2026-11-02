import pandas as pd

from transcif.research.source_support import match_added


def test_month_matched_budget_keeps_metadata_and_no_duplicate_rows():
    clock=pd.date_range('2022-01-01',periods=24*60,freq='h')
    f=pd.DataFrame({'clock':clock,'raw_row':range(len(clock))})
    result=match_added(f,{1:50,2:30,3:0})
    assert result.clock.dt.month.value_counts().to_dict()=={1:50,2:30}
    assert result.raw_row.is_unique
