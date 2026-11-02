"""Bind completed local public files to a conservative verified availability time."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import pandas as pd


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    panel_bytes=args.panel.read_bytes();manifest_bytes=args.manifest.read_bytes()
    panel=pd.read_csv(args.panel);manifest=json.loads(manifest_bytes)
    if not {'province','year','month','total_100m_kwh'}.issubset(panel.columns):
        raise ValueError('panel schema is not an NBS monthly table')
    verified=0
    for request in manifest.get('requests',[]):
        raw=(args.manifest.parent/request['response_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=request['response_sha256']:
            raise ValueError('raw source hash differs from manifest')
        verified+=1
    if args.panel.read_bytes()!=panel_bytes or args.manifest.read_bytes()!=manifest_bytes:
        raise ValueError('snapshot changed during sealing')
    seal={'input_available_at_utc':datetime.now(timezone.utc).isoformat(),
          'panel_sha256':hashlib.sha256(panel_bytes).hexdigest(),
          'source_manifest_sha256':hashlib.sha256(manifest_bytes).hexdigest(),
          'raw_source_files_verified':verified,
          'original_collection_started_at':manifest.get('retrieved_at_utc'),
          'semantics':'completed local inputs verified now; no claim of earlier publication vintage'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(seal,indent=2));print(json.dumps(seal))


if __name__=='__main__':main()
