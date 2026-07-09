#!/bin/bash
#SBATCH --job-name=simclr_har
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/simclr_%j.out
#SBATCH --error=logs/simclr_%j.err

source /opt/apps/testapps/common/software/staging/Anaconda3/2024.02-1/etc/profile.d/conda.sh

# 환경 활성화
conda activate lejepa

echo "Python path: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"

# =======================================
cd /mnt/parscratch/users/acb20si/har-benchmark-pipeline/models/deep_learning

# ==========================================
echo "Starting SimCLR training at $(date)"

/users/acb20si/.conda/envs/lejepa/bin/python self_supervised.py

echo "Finished at $(date)"