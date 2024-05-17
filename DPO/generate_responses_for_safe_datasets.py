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

from vllm import LLM, SamplingParams

    
seed=42


eval_datasets = [
    'I-Alpaca.json',
    'I-CoNa.json',
    'I-Controversial.json',
    'I-MaliciousInstructions.json',
    'I-PhysicalSafetySafe.json',
    'I-PhysicalSafetyUnsafe.json',
]
eval_datasets_root = "LLM_Alignment"
eval_resps_save_root = "LLM_Alignment/output"



if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SFT Training Arguments")

    parser.add_argument("--model_name_or_path", type=str, default="./saved-models/DPO_LLAMA-7B/merged_model", help="Model name or path")
    parser.add_argument("--save_name", type=str, default="llama-it", help="Training Data Path")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum Sequence length")

    input_args = parser.parse_args()

    model = LLM(model=input_args.model_name_or_path,
                    tokenizer=input_args.model_name_or_path, 
                    tensor_parallel_size=torch.cuda.device_count(), 
                    seed=seed,
                    gpu_memory_utilization=0.95, 
                    dtype=torch.float16,
                    enforce_eager=True,
                    max_model_len=512 
                )
    print("### model loaded")


    sampling_params = SamplingParams(n=1, 
                      max_tokens=120, 
                      # top_k=self.input_args.top_k, 
                      top_p=0.95, 
                      temperature=0.0,
                      frequency_penalty=1
                        )
    
    pr = Prompter("alpaca")
    
    for d in tqdm(eval_datasets):
        eval_data = f"{eval_datasets_root}/{d}"
        print(f"Eval data: {eval_data}")
        with open(eval_data, 'rb') as f:
            data = json.load(f)
        data = Dataset.from_dict({
            "instruction": data["instructions"]
        })
        responses = {
            "instruction": [],
            "response": []
        }
        n_rows = len(data)
        n_batch = 100
        for i in tqdm(range(0, n_rows, n_batch)):
            samples = data[i:i+n_batch]
            prompts = [pr.generate_prompt(instruction) for instruction in samples['instruction']]
            
            curr_outputs = model.generate(prompts, sampling_params)
            torch.cuda.empty_cache()

            for j in range(len(prompts)):
                for ele in curr_outputs[j].outputs:
                    responses["instruction"].append(data[j]["instruction"])
                    responses["response"].append(ele.text)

        save_path = f"{eval_resps_save_root}/{input_args.save_name}/{d}"
        with open(save_path, 'w') as f:
            json.dump(responses, f)