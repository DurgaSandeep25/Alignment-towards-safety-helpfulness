# to curate the data responses for helpfulness evaluation

python helpfulness_model_resps.py \
    --model_name_or_path "dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca-no-quant" \
    --save_name "llama-it-base"
    
python helpfulness_model_resps.py \
    --model_name_or_path /home/jupyter/models/01_raft_500_llamaGaurd/merged_model \
    --save_name raft-human-1
    
python helpfulness_model_resps.py \
    --model_name_or_path /home/jupyter/models/4_raft_100_deberta/merged_model \
    --save_name raft-deberta

python helpfulness_model_resps.py \
    --model_name_or_path /home/jupyter/models/2_raft_1000_human/merged_model \
    --save_name raft-human-2