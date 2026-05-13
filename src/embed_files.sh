#!/bin/bash
#SBATCH --job-name=embed_files
#SBATCH -D .
#SBATCH -A bsc14
#SBATCH --qos=acc_debug
#SBATCH --output=logs/embed_files_%j.out
#SBATCH --error=logs/embed_files_%j.err
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=02:00:00  

# activate venv
source .venv/bin/activate

# models to be run in this job
models=(
    "xlm-roberta-base" 
    "bert-base-multilingual-cased" 
    "bert-base-spanish-wwm-cased" 
    "bertin-roberta-base-spanish" 
    "bsc-bio-es" 
    "bsc-bio-ehr-es" 
    "roberta-base-biomedical-clinical-es" 
    "bert-base-italian-xxl-cased" 
    "umberto-commoncrawl-cased-v1" 
    "bioBIT" 
    "medBIT"
    )

# run pipeline with set arguments
for model in "${models[@]}" 
do
    echo "=== Running for: $model ==="
    python -u src/embed_files.py --model "$model"
done