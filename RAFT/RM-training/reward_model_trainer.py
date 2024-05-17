from dataset_processor import DATASET_PREPROCESSOR
from reward_modeling import load_reward_model, unload_reward_model, RewardModel

import argparse

def do_rm_training(input_args, dataset_preprocessor):
    rm_model = load_reward_model(input_args.model_name_or_path)
    trainer = RewardModel.init_trainer(rm_model.model, rm_model.tokenizer, dataset_preprocessor.train_dataset, dataset_preprocessor.eval_dataset, input_args)
    trainer.train()
    RewardModel.save_model_and_flush(input_args.model_name_or_path, rm_model.model, rm_model.tokenizer, trainer, input_args.output_dir)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="SFT Training Arguments")
    
    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-2b", help="Model name or path")
    parser.add_argument("--dataset_name_or_path", type=str, default="PKU-Alignment/PKU-SafeRLHF", help="Dataset name or path")
    
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Training batch size per device")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4, help="Evaluation batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps")
    
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Reporting destination")
    parser.add_argument("--run_name", type=str, default="SFT-Training", help="Name of the run")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum Sequence length")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")

    parser.add_argument("--evaluation_strategy", type=str, default="steps", help="Evaluation strategy")
    parser.add_argument("--eval_steps", type=int, default=5000, help="Evaluation steps")
    parser.add_argument("--logging_strategy", type=str, default="steps", help="Logging strategy")
    parser.add_argument("--log_steps", type=int, default=5000, help="Logging steps")
    parser.add_argument("--logging_first_step", action="store_true", help="Log the first step")
    parser.add_argument("--save_strategy", type=str, default="steps", help="Save strategy")
    parser.add_argument("--save_steps", type=int, default=5000, help="Save steps")
    
    parser.add_argument("--lora_rank", type=int, default=8, help='Rank in LoRA config')
    parser.add_argument("--lora_alpha", type=int, default=16, help='Alpha in LoRA config')
    parser.add_argument("--lora_dropout", type=float, default=0.05, help='Dropout in LoRA config')

    parser.add_argument("--output_dir", type=str, default="/scratch/workspace/asureddy_umass_edu-llm_alignment/reward_models/test_rm_model_no_train", help="Output directory")
    input_args = parser.parse_args()
    dataset_preprocessor = DATASET_PREPROCESSOR[input_args.dataset_name_or_path](train_type="rm")
    do_rm_training(input_args, dataset_preprocessor)
