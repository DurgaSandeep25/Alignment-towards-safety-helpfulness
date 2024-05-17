import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, load_from_disk
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import pandas as pd

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"

# replace following with your own huggingface account key
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models

device = "cpu" # or "cuda" if you have a GPU
if torch.cuda.is_available():
    device = 'cuda'


output_dir = "./saved-models/SFT/merged_model"  

tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
print("############ Tokenizer loaded")


bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
print("############### BNB config done")

model = AutoModelForCausalLM.from_pretrained(output_dir, 
                                             quantization_config=bnb_config, 
                                             device_map={"": 0},
                                            use_cache=False)
print("############## Model loaded successfully")

train_dataset = load_from_disk("LLM_Alignment/data/sft_safe_rlhf_train_data.hf")
eval_dataset = load_from_disk("LLM_Alignment/data/sft_safe_rlhf_eval_data.hf")
test_dataset = load_from_disk("LLM_Alignment/data/sft_safe_rlhf_test_data.hf")


# dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")

# train_dataset = dataset['train']
# test_dataset = dataset['test']

# train_dataset = train_dataset.to_pandas()
# test_dataset = test_dataset.to_pandas()

# # considering only 1000 for now in training, 100 in testing
# train_dataset = train_dataset[(train_dataset['is_response_0_safe'] == True) & (train_dataset['is_response_1_safe'] == False)][:1000]
# test_dataset = test_dataset[(test_dataset['is_response_0_safe'] == True) & (test_dataset['is_response_1_safe'] == False)][:100]

# train_dataset = Dataset.from_pandas(train_dataset)
# split = train_dataset.train_test_split(test_size=0.2, seed=42)
# train_dataset, eval_dataset = split['train'], split['test']

# test_dataset = Dataset.from_pandas(test_dataset)

print("Train Data")
print(train_dataset)

print("Eval Data")
print(eval_dataset)


print("Test Data")
print(test_dataset)

print("########### Dataset is loaded properly")

def get_prompt(question):
    return "### Instruction: Answer the following question, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " + question + "\n### Response: "

test_dataset = test_dataset.to_pandas()
test_dataset['sft_prompt'] = test_dataset['prompt'].apply(get_prompt)
test_dataset = Dataset.from_pandas(test_dataset)

text = test_dataset['sft_prompt'][:5]

inputs = tokenizer(text, add_special_tokens=True,return_tensors='pt', padding=True).to(device)
outputs = model.generate(inputs['input_ids'], pad_token_id=tokenizer.pad_token_id, max_new_tokens=100,num_beams=1, do_sample = True)
outputs = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
torch.cuda.empty_cache()

i = 0
print(text[i])
print(outputs[i])



# store in csv
df_dict = {"prompt": text, "desired_response": test_dataset['label'][:5] , "model_response": outputs }
df = pd.DataFrame.from_dict(df_dict)
df.to_csv("LLM_Alignment/output/sft_inference.csv")

# evaluation metrics
