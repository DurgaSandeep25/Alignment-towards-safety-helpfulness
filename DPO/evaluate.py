# !pip install sacrebleu bert-score
# !pip install git+https://github.com/google-research/bleurt.git
# !wget https://storage.googleapis.com/bleurt-oss-21/BLEURT-20-D12.zip
# !unzip BLEURT-20-D12.zip
# !pip install nltk
# nltk.download('punkt')
# !pip install tensorflow
# !pip install rouge-score
# !pip install unbabel-comet


from sacrebleu import corpus_bleu
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


from bert_score import BERTScorer
from bert_score import score as bert_score
from bleurt import score
from rouge_score import rouge_scorer
from comet import download_model, load_from_checkpoint
from bert_score import BERTScorer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--metric', type=str, default="bleu", help='evaluation metric')
parser.add_argument('--path', type=str, default='safe_gemma_10epochs_5000samples_train.csv', help='model responses csv path')
opt = parser.parse_args()
print (opt)

class Evaluate:
    def __init__(self):
        self.scorer = None

class ScoreBert(Evaluate):
    def __init__(self):
        super().__init__()
        self.scorer = BERTScorer(lang="en", model_type="microsoft/deberta-xlarge-mnli")
        
    def score(self, ref, output):
        P, R, F1 = self.scorer.score([output], [ref])
        bertscore = F1.mean().item()
        return bertscore

class ScoreBleu(Evaluate):
    def __init__(self):
        super().__init__()
        self.scorer = sentence_bleu
        
    def score(self, ref, output):
        score = self.scorer([ref.split()], output.split(), smoothing_function=SmoothingFunction().method4)
        return score

class ScoreBleurt(Evaluate):
    def __init__(self):
        super().__init__()
        bleurt_checkpoint = "./BLEURT-20-D12"
        self.scorer = score.BleurtScorer(bleurt_checkpoint)
        
    def score(self, ref, output):
        bleurt_score = self.scorer.score(references=[ref], candidates=[output])
        return bleurt_score[0]
    
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
    
class ScoreComet(Evaluate):
    def __init__(self):
        super().__init__()
        model_path = download_model("wmt20-comet-da")
        self.scorer = load_from_checkpoint(model_path)
        self.data = []
    def appenddata(self, ref, output, prompt):
        self.data.append({"src": prompt, "mt": output, "ref": ref})
    def score(self):
        comet_score = self.scorer.predict(self.data)
        return comet_score
    
import pandas as pd
df = pd.read_csv(opt.path)
prompts = df['prompt'].to_list()
refs = df['desired_response'].to_list()
outputs = df['model_response'].to_list()
processed_outputs = [x[2:-2] for x in outputs]

if opt.metric=="bleu":
    e = ScoreBleu()
    scores = [e.score(x,y) for x,y in zip(refs,outputs)]
    print("bleu:",sum(scores)/len(scores))
elif opt.metric=="bleurt":
    e = ScoreBleurt()
    scores = [e.score(x,y) for x,y in zip(refs,outputs)]
    print("bleurt:",sum(scores)/len(scores))
elif opt.metric=="bert":
    e = ScoreBert()
    scores = [e.score(x,y) for x,y in zip(refs,outputs)]
    print("bert:",sum(scores)/len(scores))
elif opt.metric=="rouge":
    e = ScoreRouge()
    scores = [e.score(x,y) for x,y in zip(refs,outputs)]
    r1 = [x[0] for x in scores]
    r2 = [x[1] for x in scores]
    rl = [x[2] for x in scores]
    print("r1:",sum(r1)/len(r1))
    print("r2:",sum(r2)/len(r2))
    print("rl:",sum(rl)/len(rl))
elif opt.metric=="comet":
    e = ScoreComet()
    for x,y,z in zip(refs,outputs,prompts):
        e.appenddata(x,y,z) 
    scores = e.score()
    # print(scores)
    print("comet:",scores.system_score)
    
