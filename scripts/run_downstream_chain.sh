#!/bin/zsh
# Sequential rerun of downstream experiments (ablation, carboncast, deployment, conformal).
# All scripts internally add their own directory to sys.path; no PYTHONPATH needed.
cd "$(dirname "$0")/.."
python scripts/conformal_prediction.py > results/conformal_rerun.log 2>&1
python scripts/temporal_ood.py > results/temporal_ood_rerun.log 2>&1
python scripts/carboncast_analysis.py > results/carboncast_rerun.log 2>&1
python scripts/deployment_warmup.py > results/deployment_rerun.log 2>&1
python scripts/ablation_study.py > results/ablation_rerun.log 2>&1
echo DONE > results/downstream_chain.done
