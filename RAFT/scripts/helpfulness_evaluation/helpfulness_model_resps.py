import os
import gc
import torch
import tqdm as notebook_tqdm
from tqdm import tqdm
import argparse
import sys
sys.path.append("..")
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

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models


seed=42

def exact_match(ref, gen):
    # metric to calculate the exact string match
    count = 0
    for i in range(len(ref)):
        if str(ref[i]).lower() in str(gen[i]).lower():
            count += 1
    return count/len(ref)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Helpfulness evaluation")
    parser.add_argument("--model_name_or_path", type=str, default="dsaluru/Instruction-Tuned-LLaMa-7B-Alpaca-no-quant", help="Model name or path")
    parser.add_argument("--save_name", type=str, default="", help="Training Data Path")
    
    input_args = parser.parse_args()
    model_name_or_path = input_args.model_name_or_path
    save_name = input_args.save_name
    
    # model initialization
    model = LLM(model=model_name_or_path,
                tokenizer=model_name_or_path, 
                tensor_parallel_size=torch.cuda.device_count(), 
                seed=seed,
                gpu_memory_utilization=0.95, 
                dtype=torch.float16,
                enforce_eager=True,
                max_model_len=512 
                    )
    print("### model loaded")

    # prompter
    pr = Prompter("alpaca")

    # some datasets used for helpfulness evaluation
    qa_test = "qa_test.xlsx"
    long_responses = "long_responses.xlsx"
    helpfulness_curated = "helpfulness_curated.xlsx"

    # for qa task
    sampling_params = SamplingParams(n=1, 
                      max_tokens=25, 
                      # top_k=self.input_args.top_k, 
                      top_p=0.95, 
                      temperature=0.0,
                      frequency_penalty=1
                        )
    # Instruction for single answer qa task
    qa_instruction = "Answer the following question in very few words or a sentence."
    qa_df = pd.read_excel(qa_test)
    qa_df["input_prompt"] = qa_df["Prompt"].apply(lambda x: pr.generate_prompt(qa_instruction, x))

    prompts = qa_df["input_prompt"].tolist()
    torch.cuda.empty_cache()
    # to generate model response
    curr_output = model.generate(prompts, sampling_params)
    all_outputs = []
    for j in range(len(prompts)):
        for ele in curr_output[j].outputs:
            all_outputs.append(ele.text)
    torch.cuda.empty_cache()

    qa_df["model_response"] = all_outputs
    qa_df.to_csv(f"{save_name}_qa_responses.csv")
    qa_task_match = exact_match(qa_df["Answer"].tolist(), qa_df["model_response"].tolist())
    print(f"qa_task\nExact string match accuracy: {qa_task_match*100}%")

    # for helpfulness short answer tasks
    sampling_params = SamplingParams(n=1, 
                      max_tokens=100, 
                      top_p=0.95, 
                      temperature=0.0,
                      frequency_penalty=1
                        )
    instruction = "Answer the following question in a sentences or two."
    df = pd.read_excel(helpfulness_curated)

    df["input_prompt"] = df["Prompt"].apply(lambda x: pr.generate_prompt(qa_instruction, x))

    prompts = df["input_prompt"].tolist()
    torch.cuda.empty_cache()
    # to generate responses
    curr_output = model.generate(prompts, sampling_params)
    all_outputs = []
    for j in range(len(prompts)):
        for ele in curr_output[j].outputs:
            all_outputs.append(ele.text)
    torch.cuda.empty_cache()

    df["model_response"] = all_outputs
    # to save the model responses file
    df.to_csv(f"{save_name}_helpfulness_short.csv")

    # for helpfulness long answer tasks
    sampling_params = SamplingParams(n=1, 
                      max_tokens=128, 
                      # top_k=self.input_args.top_k, 
                      top_p=0.95, 
                      temperature=0.0,
                      frequency_penalty=1
                        )
    instruction = "Answer the following question elaborately in more than 3 sentences."
    df = pd.read_excel(long_responses)

    df["input_prompt"] = df["Prompt"].apply(lambda x: pr.generate_prompt(instruction, x))

    prompts = df["input_prompt"].tolist()
    torch.cuda.empty_cache()
    curr_output = model.generate(prompts, sampling_params)
    all_outputs = []
    for j in range(len(prompts)):
        for ele in curr_output[j].outputs:
            all_outputs.append(ele.text)
    torch.cuda.empty_cache()

    df["model_response"] = all_outputs
    # to save the model responses file
    df.to_csv(f"{save_name}_helpfulness_long.csv")