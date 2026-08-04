#!/bin/zsh
# Sequential rerun of downstream experiments (conformal, temporal OOD, carboncast,
# deployment, ablation).
# All scripts internally add their own directory to sys.path; no PYTHONPATH needed.
cd "$(dirname "$0")/../.."

python scripts/benchmark/conformal_prediction.py > results/conformal_rerun.log 2>&1
python scripts/benchmark/temporal_ood.py > results/temporal_ood_rerun.log 2>&1
python scripts/figures/carboncast_analysis.py > results/carboncast_rerun.log 2>&1
python scripts/experiments/deployment_warmup.py > results/deployment_rerun.log 2>&1
python scripts/benchmark/ablation_study.py > results/ablation_rerun.log 2>&1
echo DONE > results/downstream_chain.done
