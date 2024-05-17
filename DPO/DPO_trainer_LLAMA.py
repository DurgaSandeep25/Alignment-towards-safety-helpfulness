import os
import sys
sys.path.append("..")
import gc
import torch
import tqdm as notebook_tqdm
import argparse
import pandas as pd
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

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models



class Trainer:
    def __init__(self, input_args):
        self.input_args = input_args
        self.common_instruction = "Answer the following question, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content."

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.input_args.model_name_or_path)
        
        # unk. we want this to be different from the eos token
        tokenizer.pad_token_id = (0)
        
        # Allow batched inference
        tokenizer.padding_side = "right"  
        
        # tokenizer = AutoTokenizer.from_pretrained(
        #     self.input_args.model_name_or_path,
        #     padding_side="right",
        #     # add_eos_token=True,
        #     # add_bos_token=True,
        #     trust_remote_code=True,
        # )
        # tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token_id = tokenizer.eos_token_id

        self.tokenizer = tokenizer
        return


    def get_bnb_config(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.bnb_config = bnb_config
        return


    def get_trainable_model(self):
        self.get_bnb_config()
        # model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path,
        #                                              load_in_8bit=True,
        #                                              torch_dtype=torch.float16,
        #                                              device_map='auto',
        #                                              trust_remote_code=True
        #                                              )
        # model = prepare_model_for_int8_training(model)
        
        model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path, 
                                                    quantization_config=self.bnb_config, 
                                                    device_map='auto',
                                                    use_cache=False)
        
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        model.enable_input_require_grads()

        return model

    def get_qlora_config(self):
        config = LoraConfig(
            r=self.input_args.lora_rank,
            lora_alpha=self.input_args.lora_alpha,
            target_modules=[
                # "attn_out",
                # "ff_out",
                "q_proj",
                "k_proj",
                "v_proj",
                # "o_proj",
                # "gate_proj",
                # "up_proj",
                # "down_proj",
                # "lm_head",
            ],
            bias="none",
            lora_dropout=self.input_args.lora_dropout,  # Conventional
            task_type="CAUSAL_LM",
        )
        self.lora_config = config
        return
    
    def get_sft_training_args(self):
        # --gradient_checkpointing True, workers speed up processing,  grad accumulation - 8, 16, 22 (less memory)

        args=transformers.TrainingArguments(
            output_dir=self.input_args.output_dir,
            warmup_steps=1,
            per_device_train_batch_size=self.input_args.per_device_train_batch_size,
            # per_device_eval_batch_size=self.input_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.input_args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            # group_by_length=False,
            num_train_epochs=self.input_args.num_train_epochs,
            learning_rate=self.input_args.learning_rate,
            optim="adamw_torch",
            logging_strategy=self.input_args.logging_strategy,
            logging_steps=self.input_args.log_steps,              # When to start reporting loss
            save_strategy=self.input_args.save_strategy,       # Save the model checkpoint every logging step
            save_steps=self.input_args.save_steps,                # Save checkpoints every 100 steps
            save_total_limit=30,
            evaluation_strategy=self.input_args.evaluation_strategy, # Evaluate the model every epoch
            eval_steps=self.input_args.eval_steps,               # Evaluate and save checkpoints every 100 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to=self.input_args.report_to,           # Comment this out if you don't want to use weights & baises
            dataloader_pin_memory=True,                           
            dataloader_num_workers=4,
            dataloader_prefetch_factor=1,
            logging_first_step=self.input_args.logging_first_step,
            lr_scheduler_type="cosine",
            seed=42,
            # bf16=True,
            fp16=True,
            # tf32=True,
            disable_tqdm=False
        )
        self.training_args = args
        return
        
    
    def run(self):

        print("########## Model name : ", self.input_args.model_name_or_path)

        # Load Tokenizer
        self.get_tokenizer()

        # Load Dataset - refer to data-scripts - 05_ift_eft_data.ipynb - here we created final ift+eft data

        self.train_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_train_data.hf")
        self.eval_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_eval_data.hf")
        self.test_dataset = load_from_disk("LLM_Alignment/data/DPO_safe_rlhf_test_data.hf")
        
        
        train_dataset_pd = self.train_dataset.to_pandas()
        n_sample = int(self.input_args.train_size*len(train_dataset_pd))
        train_dataset_sample = train_dataset_pd.sample(n_sample)
        self.train_dataset = Dataset.from_pandas(train_dataset_sample)
        

        print("Train Dataset : ", self.train_dataset)
        print("Eval Dataset : ", self.eval_dataset)

        print("########### Dataset is loaded properly")

        # Load the Raw model and prepare for k-bit training
        model = self.get_trainable_model()
        print("########### Model Loaded")
        print()

        # QLoRA config
        self.get_qlora_config()

        print("LORA Config : ", self.lora_config)

        # Get the PEFT model i.e., adapters
        model = get_peft_model(model, self.lora_config)
        
        print("Trainable parameters: ")
        model.print_trainable_parameters()
        
        # get sft training arguments - this decides the GPU RAM consumption mostly
        self.get_sft_training_args()
        
        
        # compute loss only for the summary, not for the prompt (conversation)
        # response_template = "### Response:"
        # collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer, mlm=False)
        
        # data_collator=transformers.DataCollatorForSeq2Seq(
        #     self.tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        # )
        
        print("sample data : ", self.train_dataset[0])
                
        print("######### Output Directory : ", self.input_args.output_dir)
        
        # print(self.train_dataset)
        # self.train_dataset.save_to_disk("sample.hf")
        
        # SFT Training
        print("#################  DPO Training started")
        trainer = DPOTrainer(
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            beta = 0.1, 
            tokenizer=self.tokenizer,
            args=self.training_args,
            max_length=1536,
            max_prompt_length=1024,
        )
        
        model.config.use_cache = False
        
        trainer.train()

        print("################# Training is done")


        trainer.model.save_pretrained(f"{self.input_args.output_dir}/final_checkpoint")
        self.tokenizer.save_pretrained(f"{self.input_args.output_dir}/final_checkpoint")

        # Flush memory
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()


        print("################# reloading and merging")
        ### Reload and Merge
        # Reload model in FP16 (instead of NF4)
        base_model = AutoModelForCausalLM.from_pretrained(
            self.input_args.model_name_or_path,
            return_dict=True,
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.input_args.model_name_or_path)

        # Merge base model with the adapter
        model = PeftModel.from_pretrained(base_model, model_id=f"{self.input_args.output_dir}/final_checkpoint")
        model = model.merge_and_unload()

        print("################# Merged")
        # Save model and tokenizer

        model.save_pretrained(self.input_args.output_dir+"/merged_model")
        tokenizer.save_pretrained(self.input_args.output_dir+"/merged_model")

        # Flush memory
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache()

        print("####################### Model saved at : ", self.input_args.output_dir+"/merged_model")

        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SFT Training Arguments")

    dd = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/Self-Rewarding-LM/"
    dd += "data/sft_training_data_only_ift.hf"
    
    parser.add_argument("--model_name_or_path", type=str, default="allenai/OLMo-1B", help="Model name or path")
    parser.add_argument("--train_size", type=float, default=1, help="Train data size")
    parser.add_argument("--train_data_path", type=str, default=dd, help="Training Data Path")

    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--report_to", type=str, default="wandb", help="Reporting destination")
    parser.add_argument("--run_name", type=str, default="SFT-Training", help="Name of the run")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum Sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=2, help="Number of training epochs")

    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=500, help="Evaluation steps")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--log_steps", type=int, default=500, help="Logging steps")
    parser.add_argument("--logging_first_step", action="store_true", help="Log the first step")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=500, help="Save steps")
    
    parser.add_argument("--lora_rank", type=int, default=32, help='Rank in LoRA config')
    parser.add_argument("--lora_alpha", type=int, default=16, help='Alpha in LoRA config')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='Dropout in LoRA config')

    parser.add_argument("--output_dir", type=str, default="./saved-models/no_name-sft", help="Output directory")
    input_args = parser.parse_args()
    
    print("SFT training")
    print("########### Learning rate : ", input_args.learning_rate)
    print("########### LORA rank : ", input_args.lora_rank)
    print("########### Output dir : ", input_args.output_dir)
    
    trainer = Trainer(input_args)
    trainer.run()
    