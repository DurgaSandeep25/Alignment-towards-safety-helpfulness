module load miniconda/22.11.1-1 cuda/11.3.1

conda activate NLP685

python LLM_Alignment/gpu_check.py

python LLM_Alignment/sft_inference.py

python LLM_Alignment/DPO_inference.py

python LLM_Alignment/evaluate.py --path LLM_Alignment/output/sft_inference.csv

python LLM_Alignment/evaluate.py --path LLM_Alignment/output/DPO_inference.csv