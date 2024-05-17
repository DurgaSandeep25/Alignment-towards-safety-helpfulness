from datasets import load_from_disk, Dataset
import random
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.append("/home/jupyter/LLM_Alignment/updated_sft/scripts")
from vllm import LLM, SamplingParams
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
# from transformers import AutoModelForCausalLM, AutoTokenizer
seed=42
from prompter import Prompter
import argparse
# Set the environment variable
import os
import json
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Arguments")
    dd = "/home/jupyter/LLM_Alignment/updated_sft/data/safety_only_data_Instructions.json"
    parser.add_argument("--model_name_or_path", type=str, default="dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca-no-quant", help="Model name or path")
    parser.add_argument("--train_data_path", type=str, default=dd, help="Training Data Path")
    parser.add_argument("--n_rows", type=int, default=1000, help="Total data points to consider")
    parser.add_argument("--iteration", type=int, default=1, help="which iteration of raft")
    parser.add_argument("--n_batch", type=int, default=20, help="batch size")
    
    parser.add_argument("--save_path", type=str, default="llama-7b-alpaca-1000.json", help="save path")
    input_args = parser.parse_args()
    
    # parameters
    n_rows = input_args.n_rows
    iteration = input_args.iteration
    n_batch = input_args.n_batch
    
    # step-1: sampling prompts dataset
    if input_args.train_data_path.ends_with(".json")
        train_dataset = Dataset.from_json(input_args.train_data_path)
        train_dataset = Dataset.from_dict(train_dataset[n_rows*(iteration-1): n_rows*iteration])
    else:
        train_dataset = 
    # to be removed
    print("save path")
    print(input_args.save_path)
    # print(train_dataset[0])
    # import sys
    # sys.exit(0)
    model = LLM(model=input_args.model_name_or_path,
                    tokenizer=input_args.model_name_or_path, 
                    tensor_parallel_size=torch.cuda.device_count(), 
                    seed=seed,
                    gpu_memory_utilization=0.95, 
                    dtype=torch.float16,
                    enforce_eager=True,
                    max_model_len=512 
                )
    pr = Prompter("alpaca")
    sampling_params = SamplingParams(n=8, 
                              max_tokens=120, 
                              # top_k=self.input_args.top_k, 
                              top_p=0.95, 
                              temperature=0.85,
                              # frequency_penalty=1
                                )


    all_outputs =[]
    for i in range(0, n_rows, n_batch):

        prompts = [
            pr.generate_prompt(train_dataset[i+x]['instruction']) for x in range(n_batch)
        ]
        curr_output = model.generate(prompts, sampling_params)
        torch.cuda.empty_cache()
        for j in range(n_batch):
            temp = []
            for ele in curr_output[j].outputs:
                temp.append(ele.text)
            all_outputs.append(temp)
    dataset = train_dataset[:len(all_outputs)]
    dataset["responses"] = all_outputs
    dataset = Dataset.from_dict(dataset)
    print(dataset)
    data_file = input_args.save_path
    dataset.to_json(data_file)
    # with open(data_file, "w") as f:
    #     json.dump(dataset, f)