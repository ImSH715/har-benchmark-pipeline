#!/bin/bash
#SBATCH --job-name=simclr_har
#SBATCH --partition=gpu          
#SBATCH --gres=gpu:1            
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00      
#SBATCH --output=logs/simclr_%j.out 
#SBATCH --error=logs/simclr_%j.err 



source ~/venv/bin/activate  
conda init
conda activate lejepa
python self_supervised.py

echo "Job finished"