#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 10:00:00  # Job time limit
#SBATCH -o ./jobs/%j.out  # %j = job ID
#SBATCH --constraint=[a100]

module load miniconda/22.11.1-1 cuda/11.3.1
# /modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
if [ ! -d "./jobs" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs
fi

conda activate self_llm_env

python "./scripts/helpfulness_reward.py" \
      --num "$1"