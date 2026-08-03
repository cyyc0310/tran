#!/bin/zsh
# Sequential rerun of remaining ZS+-affected downstream experiments (low priority).
# conformal_prediction.py + temporal_ood.py already completed (JSONs saved).
cd /Users/cyyc0310/Downloads/transcif
export PYTHONPATH=src
python scripts/carboncast_analysis.py > results/carboncast_rerun.log 2>&1
python scripts/deployment_warmup.py > results/deployment_rerun.log 2>&1
python scripts/ablation_study.py > results/ablation_rerun.log 2>&1
echo DONE > results/downstream_chain.done
