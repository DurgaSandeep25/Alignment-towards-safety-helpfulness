from transformers import BartForSequenceClassification, BartTokenizer
import pandas as pd
import argparse
import copy
import torch
import gc

parser = argparse.ArgumentParser()
parser.add_argument('--claims_golden', type=str, default="../helpfulness_recent_nli/nli/golden_claims.csv", help='path for golden response claims')
parser.add_argument('--claims_resps', type=str, default="../helpfulness_recent_nli/nli/raft-human-2_helpfulness_long.csv", help='path for model response claims')
opt = parser.parse_args()
print (opt)

model_name = "facebook/bart-large-mnli"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForSequenceClassification.from_pretrained(model_name)

df_golden = pd.read_csv(opt.claims_golden)
df_model = pd.read_csv(opt.claims_resps)

refs = df_golden['claims'].to_list()
outputs = df_model['claims'].to_list()
scores = []


for i in range(len(outputs)):
    premise = refs[i].replace("Claim","").replace("claim","").replace(":","")
    hypothesis = outputs[i]
    if "claim" in hypothesis:
        hypothesis = hypothesis.replace("claim","Claim")
    hypothesis = hypothesis.split("Claim")[1:]
    hlen = len(hypothesis)
    s = []
    print(i)
    for j in range(hlen):
        tokens = tokenizer(premise, hypothesis[j], return_tensors="pt")
        out = model(**tokens)
        logits = out.logits
        entail_contradiction_logits = logits[:,[0,2]]
        # print(entail_contradiction_logits)
        probs = entail_contradiction_logits.softmax(dim=1)
        s.append(probs[0][1].detach())
        print(probs[0][1])
        del tokens
        del out
        del logits
        del entail_contradiction_logits
        del probs
        gc.collect()
    scores.append(sum(s) / len(s) )

print("NLI Score: ",sum(scores)/len(scores))