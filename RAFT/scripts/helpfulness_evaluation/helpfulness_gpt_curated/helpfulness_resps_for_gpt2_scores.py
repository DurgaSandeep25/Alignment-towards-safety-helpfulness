import os
import gc
import torch
import tqdm as notebook_tqdm
from tqdm import tqdm
import argparse
import sys
sys.path.append("/home/jupyter/LLM_Alignment/updated_sft/scripts")
import json

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from peft import prepare_model_for_int8_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, load_from_disk
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers import GenerationConfig
from prompter import Prompter


from datasets import load_from_disk, Dataset
import random
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
from vllm import LLM, SamplingParams
import torch
import json


os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models


seed=42

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Helpfulness evaluation")
    parser.add_argument("--model_name_or_path", type=str, default="dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca-no-quant", help="Model name or path")
    parser.add_argument("--save_name", type=str, default="", help="Save Data Path")
    
    input_args = parser.parse_args()
    model_name_or_path = input_args.model_name_or_path
    save_name = input_args.save_name
    
    # generate responses
    test_data_file = "alpaca_test.json"
    with open(test_data_file, 'r') as f:
        data = json.load(f)
    
    # prompter
    pr = Prompter("alpaca")
    model = LLM(model=model_name_or_path,
                tokenizer=model_name_or_path, 
                tensor_parallel_size=torch.cuda.device_count(), 
                seed=seed,
                gpu_memory_utilization=0.95, 
                dtype=torch.float16,
                enforce_eager=True,
                max_model_len=512 
                    )
    sampling_params = SamplingParams(n=1, 
                      max_tokens=120, 
                      # top_k=self.input_args.top_k, 
                      top_p=0.95, 
                      temperature=0.0,
                      frequency_penalty=1
                        )
    # process 50 responses per batch
    n_batch = 50
    n_rows = len(data["instructions"])
    all_outputs = []
    for i in range(0, n_rows, n_batch):
        end_index = min(i+n_batch, n_rows)
        prompts = [pr.generate_prompt(data["instructions"][x]) for x in range(i,end_index)]
        curr_output = model.generate(prompts, sampling_params)

        for j in range(len(prompts)):
            for ele in curr_output[j].outputs:
                all_outputs.append(ele.text)
        torch.cuda.empty_cache()
        
    # saving the model generated responses
    save_file = f"{save_name}_alpaca_test.json"
    data["model_responses"] = all_outputs
    with open(save_file, 'w') as f:
        json.dump(data, f)