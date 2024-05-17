# RAFT Related Responses

To run RAFT algorithm end-to-end, use the below snippet

```
bash raft_full.sh
```

Various files

```
00_sft.py 
- To perform SFT

reward_scorer.py
- To calculate reward scores for model generated responses

evaluate_helpfulness.py
- To evaluate on helpfulness tasks

prompter.py
- prompter to format the input: prompt template
- prompt template: alpaca.json

generate_responses_for_safe_datasets.py
- to generate repsonses on safe datasets

llama_gaurd_safe_evaluate.py
- to evaluate on safety tasks

push_to_hub.py
- to push to hub
```


```
generate_multiple_responses/generate_multiple_resps.py
- to generate multiple responses by model
```

script to perform Reward model training
```
RM-training/reward_model_trainer.py
```
