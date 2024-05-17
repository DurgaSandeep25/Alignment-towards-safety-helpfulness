#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 22:53:44 2024

@author: avinashamballa
"""

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

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"

# replace following with your own huggingface account key
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models

class SavePeftModelCallback(TrainerCallback):
    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}")

        kwargs["model"].save_pretrained(checkpoint_folder)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        torch.save({}, pytorch_model_path)
        return control
    
    
def return_prompt_and_responses(samples):
    
        return {
            "prompt": 
                "Question: " + samples["prompt"] + "\n\nAnswer: "
            ,
            "chosen": samples["response_0"] ,   # rated better than 1
            "rejected": samples["response_1"] , # rated worse than 0
        }

   
    
def DPO(input_args):

    # base_model_id = input_args.model_name_or_path # f"MBZUAI/LaMini-GPT-124M"
    
    
    sft_model_dir= input_args.model_name_or_path  # "./saved-models/SFT/merged_model"
    
    print("########## SFT model dir : ", sft_model_dir)


    #################################### Tokenizer ##############################################
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    ####################################### Load Dataset #########################################
    
    train_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_train_data.hf")
    eval_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_eval_data.hf")
    test_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_test_data.hf")
    
    
    
    # dataset = load_dataset(input_args.dataset_name_or_path)




    # train_dataset = dataset['train']
    # test_dataset = dataset['test']
    
    # original_columns = train_dataset.column_names

    # train_dataset = train_dataset.to_pandas()
    # test_dataset = test_dataset.to_pandas()
    
    # trainsamples = 1000
    # testsamples = 100

    # # considering only 1000 for now in training, 100 in testing (preference ranking)
    # train_dataset = train_dataset[(train_dataset['is_response_0_safe'] == True) & (train_dataset['is_response_1_safe'] == False)][:trainsamples]
    # test_dataset = test_dataset[(test_dataset['is_response_0_safe'] == True) & (test_dataset['is_response_1_safe'] == False)][:testsamples]

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

    ####################################### Load Model #########################################
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )


    # model should be sft pretrained model
    
    
    model = AutoModelForCausalLM.from_pretrained(sft_model_dir, 
                                                quantization_config=bnb_config, 
                                                device_map={"": 0},
                                                use_cache=False)
       
    
    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    
    model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)

    print("Model Loaded")
    print()

    ####################################### QLoRA setting #########################################
    
    config = LoraConfig(
        r=input_args.lora_rank,
        lora_alpha=input_args.lora_alpha,
        target_modules=[
            # "attn_out",
            # "ff_out",
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=input_args.lora_dropout,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print("Model Loaded with LoRA config")

    # --gradient_checkpointing True, workers speed up processing,  grad accumulation - 8, 16, 22 (less memory)

    ####################################### Training Arguments #########################################
    args=transformers.TrainingArguments(
        output_dir=input_args.output_dir,
        warmup_steps=1,
        per_device_train_batch_size=input_args.per_device_train_batch_size,
        # per_device_eval_batch_size=input_args.per_device_eval_batch_size,
        gradient_accumulation_steps=input_args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        # group_by_length=True,
        num_train_epochs=input_args.num_train_epochs,
        learning_rate=input_args.learning_rate,
        optim="paged_adamw_32bit",
        
        logging_strategy=input_args.logging_strategy,
        logging_steps=input_args.log_steps,              # When to start reporting loss
        save_strategy=input_args.save_strategy,       # Save the model checkpoint every logging step
        save_steps=input_args.save_steps,                # Save checkpoints every 100 steps
        # evaluation_strategy=input_args.evaluation_strategy, # Evaluate the model every epoch
        # eval_steps=input_args.eval_steps,               # Evaluate and save checkpoints every 100 steps
        # do_eval=True,                # Perform evaluation at the end of training
        report_to=input_args.report_to,           # Comment this out if you don't want to use weights & baises
        dataloader_pin_memory=True,                           
        dataloader_num_workers=4,
        dataloader_prefetch_factor=1,
        logging_first_step=input_args.logging_first_step,
        lr_scheduler_type="cosine",
        seed=42,
        bf16=True,
        warmup_ratio=0.05,
        # fp16=False,
        # tf32=True,
        disable_tqdm=False
    )
    


    ####################################### Data Formatting #########################################
    
    # train_dataset = train_dataset.map(
    #     return_prompt_and_responses,
    #     remove_columns=original_columns
    # )
    
    # eval_dataset = eval_dataset.map(
    #     return_prompt_and_responses,
    #     remove_columns=original_columns
    # )
    
    print("sample data : ", train_dataset[0])
    
    print("######### Output Directory : ", input_args.output_dir)

    # Seq Length - Based on distribution - 496 is the max length interms of space separated tokens
    ####################################### SFT Training #########################################
    print("################# Training started")
    
    ### DPO Trainer
    trainer = DPOTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        beta = 0.1, 
        tokenizer=tokenizer,
        peft_config=config,
        args=args,
        max_length=1536,
        max_prompt_length=1024,
    )
    
    # try:
    #     trainer.train()
    
    #     print("################# Training is done")
        
    #     trainer.model.save_pretrained(f"{input_args.output_dir}/final_checkpoint")
    #     tokenizer.save_pretrained(f"{input_args.output_dir}/final_checkpoint")

        
        
    # except Exception as e:
    #     # Flush memory
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     print(e)
    #     return

    trainer.train()

    print("################# Training is done")
    
    trainer.model.save_pretrained(f"{input_args.output_dir}/final_checkpoint")
    tokenizer.save_pretrained(f"{input_args.output_dir}/final_checkpoint")
    

    # Flush memory
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    


    print("################# reloading and merging")
    ### Reload and Merge
    # Reload model in FP16 (instead of NF4)
    base_model = AutoModelForCausalLM.from_pretrained(
        sft_model_dir,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(sft_model_dir, trust_remote_code=True)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, model_id=f"{input_args.output_dir}/final_checkpoint")
    model = model.merge_and_unload()

    print("################# Merged")
    # Save model and tokenizer

    model.save_pretrained(input_args.output_dir+"/merged_model")
    tokenizer.save_pretrained(input_args.output_dir+"/merged_model")

    # Flush memory
    del model, base_model
    gc.collect()
    torch.cuda.empty_cache()

    print("####################### Model saved at : ", input_args.output_dir+"/merged_model")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Training Arguments")
    
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b", help="Model name or path")
    parser.add_argument("--dataset_name_or_path", type=str, default="PKU-Alignment/PKU-SafeRLHF", help="Dataset name or path")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting destination")
    parser.add_argument("--run_name", type=str, default="DPO-Training", help="Name of the run")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum Sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")

    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Evaluation steps")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--log_steps", type=int, default=5000, help="Logging steps")
    parser.add_argument("--logging_first_step", action="store_true", help="Log the first step")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save steps")
    
    parser.add_argument("--lora_rank", type=int, default=32, help='Rank in LoRA config')
    parser.add_argument("--lora_alpha", type=int, default=16, help='Alpha in LoRA config')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='Dropout in LoRA config')

    parser.add_argument("--output_dir", type=str, default="./saved-models/no_name-sft", help="Output directory")
    input_args = parser.parse_args()

    DPO(input_args)