
import argparse
import sys
import os
import json

# Load vLLM for infernce

from datasets import load_from_disk, Dataset, load_dataset
import random
import pandas as pd
import argparse
import warnings
warnings.filterwarnings("ignore")
from vllm import LLM, SamplingParams
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
import gc
import torch
seed=42

# to access gated models
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # to avoid warning "Tokenizer deadlocks"
os.environ["HF_TOKEN"] = "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" # my huggingface key to access llama models

# automated metrics
from sacrebleu import corpus_bleu
import nltk
from nltk.translate.bleu_score import sentence_bleu
from bert_score import BERTScorer
from bert_score import score as bert_score
from bleurt import score
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
from bert_score import BERTScorer

# BART Score
import sys
sys.path.append("/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/Evaluation_Metrics/MedBART/BARTScore")
from bart_score import BARTScorer
dd = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/Evaluation_Metrics/MedBART/BARTScore/"
# bart_score.pth
bart_scorer = BARTScorer(device='cuda:0', checkpoint="facebook/bart-large-cnn")
bart_scorer.load(path= dd + 'bart_score.pth')

class Evaluate:
    def __init__(self):
        self.scorer = None
        
class ScoreBleu(Evaluate):
    def __init__(self):
        super().__init__()
        self.scorer = sentence_bleu
        
    def score(self, ref, output):
        score = self.scorer([ref.split()], output.split())
        return score
    
class ScoreRouge(Evaluate):
    def __init__(self):
        super().__init__()
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def score(self, ref, output):
        rouge_scores = self.scorer.score(ref, output)
        rouge1 = rouge_scores['rouge1'].fmeasure
        rouge2 = rouge_scores['rouge2'].fmeasure
        rougeL = rouge_scores['rougeL'].fmeasure
        return rouge1, rouge2, rougeL
    
class AutoMetricsPredictor:
    
    def __init__(self, args):
        self.model_name = args.model_name_or_path
        self.output_dir = args.output_dir
    
    def load_model(self):
        model = LLM(model=self.model_name,
                    tokenizer=self.model_name, 
                    tensor_parallel_size=torch.cuda.device_count(), 
                    seed=seed,
                    gpu_memory_utilization=0.7, 
                    dtype=torch.float16,
                    enforce_eager=True,
                    max_model_len=1024 # 512 is small for the BoolQ dataset, so changing it to 1024
        )
        self.model = model
        print("Model Loaded Successfully: ", self.model_name)
    
    def load_datasets(self):
        base_path = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/data/"
        self.alpaca_test = load_from_disk(base_path + "alpaca_test.hf").to_pandas()
        
    def gen_responses(self):
        
        sampling = SamplingParams(n=1,
                          max_tokens=300,
                          top_k=50,
                          top_p=0.9,
                          temperature=0.0,
                          # # frequency_penalty=1.0
                        )
        
        n = 100
        all_results = []
        for i in range(0, len(self.alpaca_test), n):
            output = self.model.generate(self.alpaca_test['instruction_prompt'].tolist()[i:i+n], sampling)
            torch.cuda.empty_cache()
            
            for ele in output:
                all_results.append(ele.outputs[0].text)
        self.alpaca_test['generated_response'] = all_results
        
        Dataset.from_pandas(self.alpaca_test).save_to_disk(self.output_dir + "alpaca_test.hf")
        
    def get_automated_metrics(self):
        prompts = self.alpaca_test['instruction_prompt'].tolist()
        refs = self.alpaca_test['output'].to_list()
        outputs = self.alpaca_test['generated_response'].to_list()

        res = {}

        e = ScoreBleu()
        scores = [e.score(x,y) for x,y in zip(refs,outputs)]
        bleu  = sum(scores)/len(scores)

        res['bleu'] = bleu
        
        print("Bleu done")
        bert_score = BERTScorer(model_type="microsoft/deberta-xlarge-mnli", lang="en", rescale_with_baseline=True, device='cuda')
        p,r,f = bert_score.score(refs, outputs)
        torch.cuda.empty_cache()
        res['bert_score'] = f.mean().item()
        print("Bert score done")
        
        e = ScoreRouge()
        scores = [e.score(x,y) for x,y in zip(refs,outputs)]
        r1 = [x[0] for x in scores]
        r2 = [x[1] for x in scores]
        rl = [x[2] for x in scores]

        res['rouge-1'] = sum(r1)/len(r1)
        res['rouge-2'] = sum(r2)/len(r2)
        res['rouge-l'] = sum(rl)/len(rl)

        print("Rouge Done")
        
        # save it
        # Open the file in write mode
        with open(self.output_dir + 'autometrics_eval.json', "w") as json_file:
            # Write the data to the file
            json.dump(res, json_file)
        print("Saved at before BART ", self.output_dir + 'autometrics_eval.json')
        
        # bart score
        bart = bart_scorer.score(refs, outputs, batch_size=4)
        torch.cuda.empty_cache()
        res['bart'] = sum(bart)/len(bart)
        
        # save it
        # Open the file in write mode
        with open(self.output_dir + 'autometrics_eval.json', "w") as json_file:
            # Write the data to the file
            json.dump(res, json_file)
        print("Saved at ", self.output_dir + 'autometrics_eval.json')
        
        
        
    def run(self):
        self.load_model()
        self.load_datasets()
        self.gen_responses()
        
        # delete the model
        destroy_model_parallel()
        del self.model.llm_engine.driver_worker
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        torch.distributed.destroy_process_group()
        self.get_automated_metrics()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated Metrics Evaluation")
    
    dd = "/project/pi_hongyu_umass_edu/zonghai/clinical-llm-alignment/durga_sandeep/self_rewarding/LLM_Alignment/evals/eval_results/"
    parser.add_argument("--model_name_or_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.1", help="Model name or path")
    parser.add_argument("--output_dir", type=str, default=dd + "default_model", help="Output Directory for generated results")
    
    input_args = parser.parse_args()
    
    print("Model : ", input_args.model_name_or_path)
    print("output dir : ", input_args.output_dir)
    
    evaluator = AutoMetricsPredictor(input_args)
    evaluator.run()