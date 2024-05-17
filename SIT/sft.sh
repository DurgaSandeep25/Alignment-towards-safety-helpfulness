#!/bin/bash
#SBATCH -c 4  # Number of Cores per Task
#SBATCH --mem=40G  # Requested Memory
#SBATCH -p gpu-preempt # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 30:00:00  # Job time limit
#SBATCH -o ./jobs/replicate_%j.out  # %j = job ID
#SBATCH --constraint=[a100]

module load miniconda/22.11.1-1 cuda/11.3.1
# /modules/apps/cuda/10.1.243/samples/bin/x86_64/linux/release/deviceQuery
if [ ! -d "./jobs" ]; then
    echo "Creating ./jobs directory..."
    mkdir -p ./jobs
fi

conda activate self_llm_env

python gpu_check.py

python "./scripts/training/00_sft.py" \
      --model_name_or_path yahma/llama-7b-hf \
      --train_data_path .././SIT/data/training/alpaca_small.json \
      --per_device_train_batch_size 32 \
      --per_device_eval_batch_size 32 \
      --gradient_accumulation_steps 4 \
      --learning_rate 1e-4 \
      --report_to tensorboard \
      --run_name mistral_it \
      --max_seq_length 512 \
      --num_train_epochs 4 \
      --evaluation_strategy steps \
      --eval_steps 100 \
      --logging_strategy steps \
      --log_steps 500 \
      --logging_first_step \
      --save_strategy epoch \
      --save_steps 1 \
      --lora_rank 4 \
      --lora_alpha 16 \
      --lora_dropout 0.05 \
      --output_dir ./models/save_model_here