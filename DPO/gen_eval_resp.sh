#!/bin/bash
#SBATCH --mail-type=BEGIN
#SBATCH -c 2  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 4:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH --constraint=[a100]
export PYTHONPATH="${PYTHONPATH}=$(pwd):$PYTHONPATH"

module load miniconda/22.11.1-1 cuda/11.3.1

conda activate NLP685

python LLM_Alignment/gpu_check.py


python LLM_Alignment/generate_responses_for_safe_datasets.py \
    --model_name_or_path ./saved-models/DPO_LLAMA-7B/merged_model \
    --save_name DPO_LLAMA-7B
    
    
python LLM_Alignment/LLAMAguard_evaluate.py

python LLM_Alignment/evaluate_helpfulness.py \
    --model_name_or_path ./saved-models/DPO_LLAMA-7B/merged_model
