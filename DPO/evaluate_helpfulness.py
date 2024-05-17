import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse

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

import json
from prompter import Prompter

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models
class InputArgs:
    def __init__(self):
        self.model_name_or_path = "./saved-models/DPO_LLAMA-7B/merged_model"
        self.max_seq_length = 512
        # self.train_data_path = "/home/asureddy_umass_edu/llm-alignment/LLM_Alignment/updated_sft/alpaca_small.json"


class Inferencer:
    def __init__(self, input_args):
        self.input_args = input_args
        self.prompter = Prompter("alpaca")

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.input_args.model_name_or_path)
        
        # unk. we want this to be different from the eos token
        tokenizer.pad_token_id = (0)
        
        # Allow batched inference
        tokenizer.padding_side = "right"  
        
        self.tokenizer = tokenizer
        return
    
    def tokenize(self, prompt, add_eos_token=True):
        # there's probably a way to do this with the tokenizer settings
        # but again, gotta move fast
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.input_args.max_seq_length,
            padding=False,
            return_tensors=None,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.input_args.max_seq_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point):
        full_prompt = self.prompter.generate_prompt(
            data_point["instruction"],
            data_point["input"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        
            
        tokenized_full_prompt['text'] = full_prompt
        return tokenized_full_prompt

    def get_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.bnb_config = bnb_config
        return

    # def get_dataset(self, val_set_size=0):
        
    #     if self.input_args.train_data_path.endswith(".json") or self.input_args.train_data_path.endswith(".jsonl"):
    #         data = load_dataset("json", data_files=self.input_args.train_data_path)
    #     else:
    #         data = load_dataset(self.input_args.train_data_path)
        
    #     if 'yahma' in self.input_args.model_name_or_path:
    #         val_set_size=500
        
    #     if val_set_size > 0:
    #         train_val = data["train"].train_test_split(
    #             test_size=val_set_size, shuffle=True, seed=42
    #         )
    #         train_data = (
    #             train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
    #         )
    #         val_data = (
    #             train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
    #         )
    #     else:
    #         train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
    #         val_data = None
        
    #     self.train_dataset = train_data
    #     self.eval_dataset = val_data

    def get_inference_model(self):
        self.get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path, 
                                                    quantization_config=self.bnb_config,
                                                    device_map='auto',
                                                    use_cache=False)
        return model

def process_data(data_point):
    test_sample = {
        "instruction": "Please respond to the following question by selecting the most appropriate option, while responding only give A or B or C or D",
        "input": f"{data_point['question']} \nA. {data_point['choices']['text'][0]} \nB. {data_point['choices']['text'][1]} \nC. {data_point['choices']['text'][2]} \nD. {data_point['choices']['text'][3]}",
        "output": f"{data_point['answerKey']}"
    }
    return test_sample
    
def compute_acc(actual_op, model_op):
    count = 0
    for i in range(len(actual_op)):
        count += actual_op[i]==model_op[i]
    return count/len(actual_op)*100

def parse_op(op):
    for l in op:
        if l in ["A","B","C","D"]:
            return l
    return "E" 

gen_kwargs = {
                "max_new_tokens":2,
                "top_p":0.95
            }


def return_outputs(dataset, inferencer, model, n=100):
    actual_op = []
    model_op = []
    print(len(dataset["train"]))
    for i in range(n):
        processed_data = process_data(dataset["train"][i])
        if processed_data['output'] not in ["A", "B", "C", "D"]:
            continue
        processed = inferencer.generate_and_tokenize_prompt(processed_data)
        outputs = model.generate(torch.tensor(processed["input_ids"]).unsqueeze(0), **gen_kwargs)
        torch.cuda.empty_cache()
        predicted = inferencer.tokenizer.batch_decode(outputs[:, len(processed["input_ids"]):], skip_special_tokens=True)
        # print(f"actual: {processed_data['output']}, predicted: {predicted}")
        predicted = parse_op(predicted[0])
        actual_op.append(processed_data['output'])
        model_op.append(predicted)
    return actual_op, model_op




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Helpfulness task")
    parser.add_argument("--model_name_or_path", type=str, default="./saved-models/DPO_LLAMA-7B/merged_model", help="Model name or path")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum Sequence length")
    input_args = parser.parse_args()

    inferencer = Inferencer(input_args)
    model = inferencer.get_inference_model()

    inferencer.get_tokenizer()
    
    eval_resps_save_root = "LLM_Alignment/output"
    model_save = "DPO_LLAMA-7B"
    
    results = {}

    for data_name, split_name in [("allenai/ai2_arc", "ARC-Easy"), ("google/boolq", "default"), ("allenai/openbookqa", "main"), ("piqa", "plain_text")]:
        print(data_name)
        dataset = load_dataset(data_name, split_name)
        actual_op, model_op = return_outputs(dataset, inferencer, model)
        acc = compute_acc(actual_op, model_op)
        print(f"Accuracy: {acc}")
        
        results[data_name] = acc
        
    
    save_path = f"{eval_resps_save_root}/{model_save}/evaluate_helpfulness_results.json"
    with open(save_path, 'w') as f:
        json.dump(results, f)