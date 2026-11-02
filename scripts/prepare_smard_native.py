"""Prepare native hourly generation without imposing an emissions mapping."""
import argparse
import json
from pathlib import Path

from transcif.data.smard import load_native
from run_cross_country import ROOT,digest


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--manifest',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    frame,audit=load_native(args.manifest,ROOT)
    frame.to_csv(args.output/'native_hourly.csv')
    (args.output/'audit.json').write_text(json.dumps(audit,indent=2))
    hashes={str(p):digest(p) for p in [*args.manifest,Path(__file__),ROOT/'src/transcif/data/smard.py']}
    (args.output/'provenance.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(audit))


if __name__=='__main__':
    main()
