from datasets import load_dataset
import json
import pandas as pd
import sys
sys.path.append("/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/scripts")
from prompter import Prompter
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import argparse

import os
import json
import os.path as osp
from typing import Union

import os
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

# from tap import Tap

# Check if GPU is available
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    
# Check if MPS is available
try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass


class GenerationArguments:
    def __init__(self):
        
        # self.base_model = "yahma/llama-7b-hf"
        self.base_model = "meta-llama/Llama-2-7b-hf"
        self.lora_weights = "safep/lora-alpaca-small-100-yahma" # this is after training
        self.load_8bit = True

        # generation arguments
        self.max_new_tokens = 256
        self.num_beams = 4
        self.top_k = 40
        self.top_p = 0.75
        self.temperature = 0.1
            

        ## Input and output files
        self.prompt_template_path = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/data/templates/alpaca.json"
        # self.input_path = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/data/safety_tuned/I-Alpaca.json"
        self.input_path = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/data/safety_tuned/I-MaliciousInstructions.json"
        self.output_path = "sample.json"

# Evaluation function
def evaluate(
    model,
    tokenizer,
    prompter,
    instruction,
    input=None,
    temperature=0.1,
    top_p=0.75,
    top_k=40,
    num_beams=4,
    max_new_tokens=128,
    stream_output=False,
    **kwargs,
):
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=True,
        **kwargs,
    )

    generate_params = {
        "input_ids": input_ids,
        "generation_config": generation_config,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": max_new_tokens,
    }

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=max_new_tokens,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    return prompter.get_response(output)

# Main function
def main(args):
    # Load the input data (.json)
    input_path = args.input_path
    with open(input_path) as f:
        input_data = json.load(f)
    instructions = input_data["instructions"]
    inputs = input_data["inputs"]

    # Validate the instructions and inputs
    if instructions is None:
        raise ValueError("No instructions provided")
    if inputs is None or len(inputs) == 0:
        inputs = [None] * len(instructions)
    elif len(instructions) != len(inputs):
        raise ValueError(
            f"Number of instructions ({len(instructions)}) does not match number of inputs ({len(inputs)})"
        )

    # Load the prompt template
    prompter = Prompter("alpaca")

    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if device == "cuda":
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            load_in_8bit=args.load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        raise ValueError("No GPU available - resubmit the jobs")

    if not args.load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    # Generate the outputs
    outputs = []
    for instruction, input in tqdm(
        zip(instructions, inputs),
        total=len(instructions),
        desc=f"Evaluate {args.lora_weights}",
    ):
        output = evaluate(
            model=model,
            tokenizer=tokenizer,
            prompter=prompter,
            instruction=instruction,
        )
        outputs.append(output)
        
    return instructions, inputs, outputs

if __name__ == "__main__":
    
    default_args = GenerationArguments()
    dd = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/analysis/helpfulness/data/"
    default_args.input_path = dd + "alpaca_test.json"
    
    
    parser = argparse.ArgumentParser(description="Helpfulness reward")
    parser.add_argument("--num", type=str, default="0", help="Model number")
    input_args = parser.parse_args()
    
    print("Number of safety samples: ", input_args.num)
    if int(input_args.num) == 0:
        default_args.base_model = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/models/safety_llama_replicate/merged_model"
    else:
        default_args.base_model = f"/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/safety_llama_paper/models/safety_llama_replicate_{int(input_args.num)}_safe/merged_model"
    
    print("Using basemodel: ", default_args.base_model)
    
    insts, inps, responses = main(default_args)

    output_path = dd + f"responses_alpaca_test_{int(input_args.num)}_safe.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "parameters": {
                    "model": default_args.base_model,
                    "prompt_template": default_args.prompt_template_path,
                    "lora_weights": default_args.lora_weights,
                    "load_8bit": default_args.load_8bit,
                },
                "inputs": inps,
                "instructions": insts,
                "outputs": responses,
            },
            f,
            indent=4,
        )
    
    print("Output: ", output_path)
    

    rm_tokenizer = AutoTokenizer.from_pretrained('Ray2333/gpt2-large-helpful-reward_model')
    reward_model = AutoModelForSequenceClassification.from_pretrained(
                    'Ray2333/gpt2-large-helpful-reward_model',
                    num_labels=1, 
                    torch_dtype=torch.bfloat16,
                    device_map=0,
                    )

    output_path = dd + f"responses_alpaca_test_{int(input_args.num)}_safe.json"
    with open(output_path) as f:
        data = json.load(f)


    del data['parameters']


    data = pd.DataFrame(data)

    all_rewards = []
    for i in range(len(data)):
        q, a = f"\n\nHuman: {data.iloc[i]['instructions']} \n\nAssistant:", data.iloc[i]['outputs']
        inputs = rm_tokenizer(q, a, return_tensors='pt', truncation=True)
        with torch.no_grad():
            reward = reward_model(**(inputs.to(0))).logits[0].cpu().detach().item()
            torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        
        all_rewards.append(reward)

    data['gpt2_large_reward'] = all_rewards
    data.to_csv(dd + f'gpt2_large_{int(input_args.num)}_safety.csv', index=False)
    
    print(f"Saved at : {dd}gpt2_large_{int(input_args.num)}_safety.csv")



