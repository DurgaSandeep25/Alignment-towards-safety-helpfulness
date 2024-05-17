import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig


# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # To write to HF account


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Pushing trained model to Hub")
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--model_id", type=str)
    
    input_args = parser.parse_args()
    
    model_name = input_args.model_name_or_path
    

    # Account under which we need to push to hub
    account = "AbhishekSureddy"
    model_id = f"{account}/{input_args.model_id}"

    # to load the model
    model = AutoModelForCausalLM.from_pretrained(input_args.model_name_or_path, 
                                                    device_map='auto',
                                                    use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(input_args.model_name_or_path)
    tokenizer.pad_token_id = (0)
    tokenizer.padding_side = "right" 

    # push model and tokenizer to hub
    model.push_to_hub(model_id)
    tokenizer.push_to_hub(model_id)
    
    