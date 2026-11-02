"""Run an exported foreign model using an audited NBS public-month snapshot only."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from transcif.research.public_inference import infer,nbs_months,verify_input_seal
from run_cross_country import ROOT,digest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--panel',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--province',default='Shanghai')
    parser.add_argument('--origin',required=True)
    parser.add_argument('--hours',type=int,default=24)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    panel_bytes=args.panel.read_bytes();manifest_bytes=args.manifest.read_bytes()
    seal=json.loads(args.seal.read_text())
    snapshot=verify_input_seal(seal,panel_bytes,manifest_bytes);origin=pd.Timestamp(args.origin)
    import io
    months,audit=nbs_months(pd.read_csv(io.BytesIO(panel_bytes)),args.province,origin,snapshot)
    package=joblib.load(args.model)
    prediction,features,meta=infer(package,months,origin,args.hours)
    prediction.to_csv(args.output/'predictions.csv',index=False)
    features.astype({'period':str}).to_csv(args.output/'public_features.csv',index=False)
    months.to_csv(args.output/'accepted_public_months.csv')
    (args.output/'inference.json').write_text(json.dumps({**meta,'input_audit':audit},indent=2))
    hashes={str(p):digest(p) for p in [args.model,args.panel,args.manifest,args.seal,Path(__file__),ROOT/'src/transcif/research/public_inference.py']}
    (args.output/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps({'rows':len(prediction),'latest_month':audit['latest_allowed_month'],
                      'outside_support_fields':list(meta['outside_source_coordinate_support'])}))


if __name__=='__main__':main()
