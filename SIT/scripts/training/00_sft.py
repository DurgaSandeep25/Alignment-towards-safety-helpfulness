# Some snippets in the following code is referred from https://github.com/vinid/safety-tuned-llamas

import os
import gc
import torch
import tqdm as notebook_tqdm
import argparse

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from peft import prepare_model_for_int8_training
from trl import DPOTrainer, SFTTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset, load_from_disk
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from prompter import Prompter

# Set the environment variable
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models



class Trainer:
    def __init__(self, input_args):
        self.input_args = input_args

    def get_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.input_args.model_name_or_path)
        
        # unk. we want this to be different from the eos token
        tokenizer.pad_token_id = (0)
        
        # Allow batched inference
        tokenizer.padding_side = "right"  
        
        self.tokenizer = tokenizer
        return
    
    def tokenize(self, prompt, add_eos_token=True):
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
            data_point["output"],
        )
        tokenized_full_prompt = self.tokenize(full_prompt)
        
        # True means train on inputs also (not just response)
        if not True:
            user_prompt = self.prompter.generate_prompt(
                data_point["instruction"], data_point["input"]
            )
            tokenized_user_prompt = tokenize(
                user_prompt, add_eos_token=add_eos_token
            )
            user_prompt_len = len(tokenized_user_prompt["input_ids"])

            if add_eos_token:
                user_prompt_len -= 1

            tokenized_full_prompt["labels"] = [
                -100
            ] * user_prompt_len + tokenized_full_prompt["labels"][
                user_prompt_len:
            ]  # could be sped up, probably
            
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

    def get_dataset(self, val_set_size=0):
        self.prompter = Prompter("alpaca")
        if self.input_args.train_data_path.endswith(".json") or self.input_args.train_data_path.endswith(".jsonl"):
            data = load_dataset("json", data_files=self.input_args.train_data_path)
        else:
            data = load_dataset(self.input_args.train_data_path)
        
        if 'yahma' in self.input_args.model_name_or_path:
            val_set_size=500
        
        if val_set_size > 0:
            train_val = data["train"].train_test_split(
                test_size=val_set_size, shuffle=True, seed=42
            )
            train_data = (
                train_val["train"].shuffle().map(self.generate_and_tokenize_prompt)
            )
            val_data = (
                train_val["test"].shuffle().map(self.generate_and_tokenize_prompt)
            )
        else:
            train_data = data["train"].shuffle().map(self.generate_and_tokenize_prompt)
            val_data = None
        
        self.train_dataset = train_data
        self.eval_dataset = val_data
        
        # # dd + "data/sft_training_data_only_eft.hf"
        # train = load_from_disk(self.input_args.train_data_path)
        
        # dd = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/"
        # eval = load_from_disk(dd + "LLM_Alignment/data/alpaca_val_35k.hf")

        # print("train data path : ", self.input_args.train_data_path)

        # self.train_dataset = train
        # self.eval_dataset = eval
        return

    def get_trainable_model(self):
        self.get_bnb_config()
        model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path,
                                                     load_in_8bit=True,
                                                     torch_dtype=torch.float16,
                                                     device_map='auto',
                                                     trust_remote_code=True
                                                     )
        model = prepare_model_for_int8_training(model)
        
        ## Another way of training model
        # model = AutoModelForCausalLM.from_pretrained(self.input_args.model_name_or_path, 
        #                                             quantization_config=self.bnb_config, 
        #                                             device_map='auto',
        #                                             use_cache=False)
        
        # model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        # model.enable_input_require_grads()

        return model

    def get_qlora_config(self):
        config = LoraConfig(
            r=self.input_args.lora_rank,
            lora_alpha=self.input_args.lora_alpha,
            target_modules=[
                # "attn_out",
                # "ff_out",
                "q_proj",
                # "k_proj",
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
            warmup_steps=10,
            per_device_train_batch_size=self.input_args.per_device_train_batch_size,
            # per_device_eval_batch_size=self.input_args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.input_args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            group_by_length=False,
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
            # do_eval=True,                # Perform evaluation at the end of training
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

        self.get_dataset()

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
        
        print("######### Output Directory : ", self.input_args.output_dir)
        
        print(self.train_dataset)
        self.train_dataset.save_to_disk("sample.hf")
        
        # SFT Training
        print("################# Training started")
        ### SFT Trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            # peft_config=self.lora_config,
            dataset_text_field="text",
            max_seq_length=self.input_args.max_seq_length,  # You can specify the maximum sequence length here
            tokenizer=self.tokenizer,
            args=self.training_args,
            packing=False,
            # data_collator=data_collator,
            # formatting_func=formatting_prompts_func,
            # callbacks=[SavePeftModelCallback()],
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
    