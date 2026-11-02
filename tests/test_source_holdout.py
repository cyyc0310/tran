import importlib.util
from pathlib import Path
import sys

import pandas as pd


def selector():
    scripts = Path(__file__).resolve().parents[1]/'scripts'
    sys.path.insert(0,str(scripts))
    try:
        spec=importlib.util.spec_from_file_location('source_holdout_test', scripts/'run_source_holdout.py')
        module=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.choose
    finally:
        sys.path.pop(0)


def test_macro_gain_without_four_region_wins_is_rejected():
    rows=[]
    for i in range(6):
        rows.extend([{'region':str(i),'head':'native','strength':0.,'mae':100.,'shape_mae':20.},
                     {'region':str(i),'head':'joint','strength':.5,'mae':50. if i<3 else 130.,'shape_mae':19.}])
    result=selector()(pd.DataFrame(rows))
    assert result['head']=='native' and not result['passed']


def test_geographic_gain_with_shape_damage_is_rejected():
    rows=[]
    for i in range(6):
        rows.extend([{'region':str(i),'head':'native','strength':0.,'mae':100.,'shape_mae':20.},
                     {'region':str(i),'head':'joint','strength':.5,'mae':80.,'shape_mae':21.}])
    assert not selector()(pd.DataFrame(rows))['passed']
