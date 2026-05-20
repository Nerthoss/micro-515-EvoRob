#!/bin/bash
#SBATCH --job-name=ER_project_training
#SBATCH --output=logs/ER_project_training_%j.out
#SBATCH --error=logs/ER_project_training_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --partition=academic
#SBATCH --account=micro-515

source .venv/bin/activate
mkdir -p logs
python final_project_train.py