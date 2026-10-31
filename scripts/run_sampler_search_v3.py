"""Run three sequential frozen sampler-search rounds."""
import argparse
from pathlib import Path
from transcif.research.sampler_search_v3 import run
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(run(Path(__file__).resolve().parents[1],args.output))
