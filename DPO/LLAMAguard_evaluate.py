import os
import prompter
import imp
imp.reload(prompter)
import pandas as pd
from datasets import Dataset


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"

# replace following with your own huggingface account key
# os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models
os.environ["HF_TOKEN"] = "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"


model_id = "meta-llama/LlamaGuard-7b"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)


pr = prompter.Prompter("alpaca")
def calc_safe_acc(safetys):
    return sum([val=="safe" for val in safetys])/len(safetys)

def run_end_to_end(data_file, save_back=False):
    with open(data_file, 'r') as f:
        data = json.load(f)
    llama_gaurd_resps = []
    # len(data['prompt'])
    for i in tqdm(range(len(data['instruction']))):
        prompt = pr.generate_prompt(data['instruction'][i], None, None)
        inp = [
            {"role":"user", "content": f"{prompt}"},
            {"role":"assistant", "content": f"{data['response'][i]}"}
        ]
        resp = moderate(inp)
        llama_gaurd_resps.append(resp)
        torch.cuda.empty_cache()
        
    acc = calc_safe_acc(llama_gaurd_resps)
    if save_back:
        data["safety"] = llama_gaurd_resps
        with open(data_file, 'w') as f:
            json.dump(data, f)
            
    return acc, llama_gaurd_resps
    



eval_datasets = [
    'I-Alpaca.json',
    'I-CoNa.json',
    'I-Controversial.json',
    'I-MaliciousInstructions.json',
    'I-PhysicalSafetySafe.json',
    'I-PhysicalSafetyUnsafe.json',
]

eval_resps_save_root = "LLM_Alignment/output"
model_save = "DPO_LLAMA-7B"

results = {
    
}
for dataset in eval_datasets:
    data_file = f"{eval_resps_save_root}/{model_save}/{dataset}"
    print(dataset)
    acc, _ = run_end_to_end(data_file)
    print(acc)
    if model_save in results:
        results[model_save][dataset]=acc
    else:
        results[model_save] = {
            dataset: acc
        }
    
    

save_path = f"{eval_resps_save_root}/{model_save}/LLAMAguard_results.json"
with open(save_path, 'w') as f:
    json.dump(results, f)