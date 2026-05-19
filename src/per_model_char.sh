#!/bin/bash
#SBATCH --job-name=embed_char
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs/embed_char_%j.out
#SBATCH --error=logs/embed_char_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00  

# activate venv
source .venv/bin/activate

# run pipeline
python -u src/per_model_char.py 