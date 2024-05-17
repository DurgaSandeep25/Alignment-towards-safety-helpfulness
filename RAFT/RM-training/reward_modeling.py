import os
import gc
import torch
from tqdm import tqdm
import argparse
import time
import transformers
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from trl import DPOTrainer, SFTTrainer, RewardTrainer
import bitsandbytes as bnb
from trl import DataCollatorForCompletionOnlyLM

from datasets import load_dataset, Dataset
from transformers import TrainerCallback, TrainerState, TrainerControl
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

# model_name = "facebook/opt-350m"
model_name = "google/gemma-2b"
base_model_id = model_name
# output_dir = "./reward_model"
output_dir = "/work/pi_dhruveshpate_umass_edu/project_21/llm_alignment/reward_models/gemma2b"

bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
print("############### BNB config done")

# model = AutoModelForSequenceClassification.from_pretrained(output_dir+"/merged_model", 
#                                              quantization_config=bnb_config, 
#                                              device_map={"": 0},
#                                              use_cache=False)
# print("############## Model loaded successfully")

# tokenizer = AutoTokenizer.from_pretrained(output_dir+"/merged_model", trust_remote_code=True)

# function to evaluate the reward model
# testing if the function put in reward_modeling.py is working fine...
# Tested OK
# no memory leaks
def evaluate_reward_model_acc_(model, tokenizer, dataset, batch_size=1):
  print("from reward_modeling.py")
  acc = 0
  try:
    for i in tqdm(range(0, len(dataset), batch_size)):
        subset = dataset[i:i+batch_size]
        inputs_chosen = tokenizer(subset['chosen'], return_tensors='pt', padding=True)
        inputs_rejected = tokenizer(subset['rejected'], return_tensors='pt', padding=True)
        # print(inputs_chosen['input_ids'].shape, inputs_rejected['input_ids'].shape)
        pos_scores = model(input_ids=inputs_chosen['input_ids'], attention_mask=inputs_chosen['attention_mask'])
        neg_scores = model(input_ids=inputs_rejected['input_ids'], attention_mask=inputs_rejected['attention_mask'])
        pos_scores_cpu = pos_scores.logits.cpu().detach().squeeze(-1)
        neg_scores_cpu = neg_scores.logits.cpu().detach().squeeze(-1)

        del pos_scores, neg_scores
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        
        # print("cleared mem!!")
        # print(pos_scores, neg_scores)
        acc += ((pos_scores_cpu-neg_scores_cpu)>=0).sum()
    return acc/len(dataset)
  except Exception as e:
    print(e)
    del pos_scores
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    print("cleared pos_scores from GPU")
    del neg_pos
    gc.collect()
    with torch.no_grad():
        torch.cuda.empty_cache()
    print("cleared neg_scores from GPU")


class RewardModel:
    def __init__(self, model, tokenizer, batch_size = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    @staticmethod
    def init_trainer(model, tokenizer, train_dataset, eval_dataset, input_args):
        def preprocess_function(examples):
            new_examples = {
                "input_ids_chosen": [],
                "attention_mask_chosen": [],
                "input_ids_rejected": [],
                "attention_mask_rejected": [],
            }
            for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
                tokenized_j = tokenizer(chosen, truncation=True)
                tokenized_k = tokenizer(rejected, truncation=True)

                new_examples["input_ids_chosen"].append(tokenized_j["input_ids"])
                new_examples["attention_mask_chosen"].append(tokenized_j["attention_mask"])
                new_examples["input_ids_rejected"].append(tokenized_k["input_ids"])
                new_examples["attention_mask_rejected"].append(tokenized_k["attention_mask"])

            return new_examples

        def _process(dataset_):
            dataset_ = dataset_.map(
                preprocess_function,
                batched=True,
                num_proc=4,
            )
            print(dataset_)
            dataset_ = dataset_.filter(
                lambda x: len(x["input_ids_chosen"]) <= input_args.max_seq_length
                and len(x["input_ids_rejected"]) <= input_args.max_seq_length
            )
            return dataset_
        train_dataset = _process(train_dataset)
        eval_dataset = _process(eval_dataset)

        print(train_dataset)
        
        model.config.end_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        model = prepare_model_for_kbit_training(model,use_gradient_checkpointing=False)
        config = LoraConfig(
                        r=input_args.lora_rank,
                        lora_alpha=input_args.lora_alpha,
                        bias="none",
                        task_type="SEQ_CLS",
                        modules_to_save=["scores"]
                    )
        model = get_peft_model(model, config)
        training_args=transformers.TrainingArguments(
            output_dir=input_args.output_dir,
            warmup_steps=1,
            per_device_train_batch_size=input_args.per_device_train_batch_size,
            per_device_eval_batch_size=input_args.per_device_eval_batch_size,
            gradient_accumulation_steps=input_args.gradient_accumulation_steps,
            gradient_checkpointing=True,
            num_train_epochs=input_args.num_train_epochs,
            learning_rate=input_args.learning_rate,
            optim="paged_adamw_32bit",
            logging_strategy=input_args.logging_strategy,
            logging_steps=input_args.log_steps,              # When to start reporting loss
            save_strategy=input_args.save_strategy,       # Save the model checkpoint every logging step
            save_steps=input_args.save_steps,                # Save checkpoints every 100 steps
            evaluation_strategy=input_args.evaluation_strategy, # Evaluate the model every epoch
            eval_steps=input_args.eval_steps,               # Evaluate and save checkpoints every 100 steps
            do_eval=True,                # Perform evaluation at the end of training
            report_to=input_args.report_to,           # Comment this out if you don't want to use weights & baises
            dataloader_pin_memory=True,                           
            dataloader_num_workers=4,
            dataloader_prefetch_factor=1,
            logging_first_step=input_args.logging_first_step,
            lr_scheduler_type="cosine",
            seed=42,
            # bf16=True,
            # fp16=False,
            # tf32=True,
            disable_tqdm=False
        )
        # trainer = RewardTrainer(
        #     model=model,
        #     args=training_args,
        #     tokenizer=tokenizer,
        #     train_dataset=dataset,
        #     peft_config=config,
        # )
        trainer = RewardTrainer(
                model=model,
                tokenizer=tokenizer,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset = eval_dataset,
                peft_config=config
                # max_length=input_args.max_seq_length,
            )
        return trainer
    
    @staticmethod
    def save_model_and_flush(model_path, model, tokenizer, trainer, save_dir):

        trainer.model.save_pretrained(f"{save_dir}/final_checkpoint")
        tokenizer.save_pretrained(f"{save_dir}/final_checkpoint")

        # Flush memory
        del trainer, model
        gc.collect()
        torch.cuda.empty_cache()

        # reload and merge
        base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            return_dict=True,
            torch_dtype=torch.float16,
            num_labels=1,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # merge base model with adapter
        model = PeftModel.from_pretrained(base_model, model_id=f"{save_dir}/final_checkpoint")
        model = model.merge_and_unload()

        model.save_pretrained(save_dir+"/merged_model")
        tokenizer.save_pretrained(save_dir+"/merged_model")

        # Flush memory
        del model, base_model
        gc.collect()
        torch.cuda.empty_cache()


    def reward_scores(self, response_texts, gc_collect=True):
        resps_tok = self.tokenizer(response_texts, return_tensors='pt', padding=True)
        all_reward_scores = []
        for i in tqdm(range(0, len(response_texts), self.batch_size)):
          end_idx = min(len(response_texts), i+self.batch_size)
          input_ids = resps_tok['input_ids'][i:end_idx]
          attention_mask = resps_tok['attention_mask'][i:end_idx]
          scores_ = self.model(input_ids=input_ids, attention_mask=attention_mask)
          scores = scores_.logits.squeeze(-1).cpu().detach().tolist()
          print("***"*30)
          print(scores)
          print("***"*30)
          all_reward_scores.extend(scores)
          if gc_collect:
              del scores_
              gc.collect()
              with torch.no_grad():
                  torch.cuda.empty_cache()
        for i in range(len(response_texts)):
           print(f"{response_texts[i]} \nscore:{all_reward_scores[i]}")
        print("***"*30)
        return all_reward_scores

# Tested OK
# working fine
def load_reward_model(model_path, config=bnb_config):
    if model_path=="OpenAssistant/reward-model-deberta-v3-large-v2":
        model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                quantization_config=config, 
                                                device_map={"": 0},
                                                trust_remote_code=True)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path, 
                                                quantization_config=config, 
                                                device_map={"": 0},
                                                trust_remote_code=True,
                                                num_labels=1,
                                                use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    rm = RewardModel(model, tokenizer)
    return rm

# Tested Ok
# working fine
def unload_reward_model(rm):
   del rm.model
   del rm
   gc.collect()
   with torch.no_grad():
      torch.cuda.empty_cache()

