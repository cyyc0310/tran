import hashlib
import json

import pandas as pd
import pytest

from transcif.data.smard import CATEGORIES,load_native


def test_native_categories_preserved_and_hash_checked(tmp_path):
    records=[]
    stamp=int(pd.Timestamp('2022-01-01',tz='UTC').timestamp()*1000)
    for i,code in enumerate(CATEGORIES):
        content=json.dumps({'series':[[stamp,float(i+1)]]}).encode()
        path=tmp_path/f'{code}.json';path.write_bytes(content)
        records.append({'filter':code,'path':path.name,'sha256':hashlib.sha256(content).hexdigest()})
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({'year':2022,'records':records,'errors':[]}))
    frame,audit=load_native([manifest],tmp_path)
    assert frame.pumped_storage.iloc[0]>0 and frame.other_renewable.iloc[0]>0
    assert 'cif' not in frame.columns and 'thermal' not in frame.columns
    assert audit['years']['2022']['hours']==1
    second=tmp_path/'second.json'
    second.write_text(json.dumps({'year':2022,'records':[{**records[0],'sha256':'wrong'}],'errors':[]}))
    with pytest.raises(ValueError,match='hash mismatch'):
        load_native([manifest,second],tmp_path)
    (tmp_path/'1223.json').write_text('{}')
    with pytest.raises(ValueError,match='hash mismatch'):
        load_native([manifest],tmp_path)
