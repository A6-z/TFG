#!/bin/bash
#SBATCH --job-name=cross_embed_char
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs/cross_embed_char_%j.out
#SBATCH --error=logs/cross_embed_char_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00  

# activate venv
source .venv/bin/activate

# run pipeline
python -u src/cross_model_char.py 