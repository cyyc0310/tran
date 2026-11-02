from pathlib import Path
import json
import hashlib

import pytest

from tests.test_gfs_window import mocked_window
from transcif.data import gfs_compact as compact, gfs_window as window


def test_compact_bundle_reuses_exact_points_and_preserves_raw_source(mocked_window,monkeypatch):
    path, points, _, _ = mocked_window
    monkeypatch.setattr(compact,'decode_gfs_points',window.decode_gfs_points)
    source = path/'window'
    window.collect_gfs_window('2023-01-01T00:00Z',23,2,points,path/'raw',source)
    before={str(p):p.read_bytes() for p in (path/'raw').rglob('*') if p.is_file()}
    report=compact.compact_window(source,path/'compact')
    assert not report['raw_grib_copied'] and not report['eligible_for_issued_public']
    assert not list((path/'compact').rglob('*.grib2'))
    assert compact.load_compact_window(path/'compact',points)['rows']
    assert all(Path(p).read_bytes()==value for p,value in before.items())
    with pytest.raises(ValueError,match='point plan'):
        compact.load_compact_window(path/'compact',[])
    (path/'compact/hourly_points.json').write_text('{}')
    with pytest.raises(ValueError,match='hash'):
        compact.load_compact_window(path/'compact')


@pytest.mark.parametrize('change',['missing_evidence','absolute_path','dot_alias','duplicate_path','weather_resealed_outer_only'])
def test_compact_loader_requires_complete_canonical_source_evidence(mocked_window,monkeypatch,change):
    path,points,_,_=mocked_window
    monkeypatch.setattr(compact,'decode_gfs_points',window.decode_gfs_points)
    source=path/'window';target=path/'compact'
    window.collect_gfs_window('2023-01-01T00:00Z',23,2,points,path/'raw',source)
    compact.compact_window(source,target)
    manifest=target/'manifest.json';report=json.loads(manifest.read_text())
    if change=='missing_evidence':report['files']=[r for r in report['files'] if not r['file'].startswith('evidence/')]
    elif change=='absolute_path':report['files'][0]['file']=str((target/report['files'][0]['file']).resolve())
    elif change=='dot_alias':report['files'][0]['file']='./'+report['files'][0]['file']
    elif change=='duplicate_path':report['files'].append(report['files'][0].copy())
    elif change=='weather_resealed_outer_only':
        p=target/'hourly_points.json';weather=json.loads(p.read_text());weather['eligible_for_issued_public']=True
        p.write_text(json.dumps(weather))
        for entry in report['files']:
            if entry['file']=='hourly_points.json':entry['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(report))
    with pytest.raises(ValueError):compact.load_compact_window(target)
