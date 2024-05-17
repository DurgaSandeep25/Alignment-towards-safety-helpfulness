This readme.md consists of instructions to replicate Safety Instruction Tuning results

Create Virtual Environment: Python 3.8, name - self_llm_env
- pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 -f https://download.pytorch.org/whl/torch_stable.html
- pip install transformers
- pip install scikit-learn
- pip install pandas
- pip install tensorboard
- pip install trl
- pip install peft bitsandbytes
- pip install vllm

Note: Current Directory is submission

1. Baseline Model training 
- training on 20k alpaca dataset - ./SIT/data/training/alpaca_small.json
- bash sft.sh
    - model_path - yahma/llama-7b-hf
    - data_path - .././SIT/data/training/alpaca_small.json (give the absolute path instead of relative)

- trained model saved here in huggingface hub- dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca


2. Training on 20k + n safety samples. n can be 100, 300, 500, 1000, 1500, 2000
- bash sft.sh
    - model_path - dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca
    - data_path - .././SIT/data/training/saferpaca_Instructions_100.json
(please modify the model/dataset paths if there are any errors)


For evaluation: 

Harmfulness: In the following notebook, we have computed for baseline and other models
- submission/SIT/scripts/evaluation/harmfulness/

Helpfulness: In the following notebook, we have computed for baseline 
- for QA tasks - submission/SIT/scripts/evaluation/helpfulness/QA
- for computing reward scores - submission/SIT/scripts/evaluation/helpfulness/helpfulness.sh
- autometrics are computed - submission/SIT/scripts/evaluation/helpfulness/autometrics/autometrics.py