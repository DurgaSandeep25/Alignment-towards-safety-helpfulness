# RAFT Related Responses

To run RAFT algorithm end-to-end, use the below snippet

```
bash raft_full.sh
```

Various files

```
scripts/00_sft.py 
- To perform SFT

scripts/reward_scorer.py
- To calculate reward scores for model generated responses

scripts/evaluate_helpfulness.py
- To evaluate on helpfulness tasks

scripts/prompter.py
- prompter to format the input: prompt template
- prompt template: alpaca.json

scripts/generate_responses_for_safe_datasets.py
- to generate repsonses on safe datasets

scripts/llama_gaurd_safe_evaluate.py
- to evaluate on safety tasks

scripts/push_to_hub.py
- to push to hub
```


```
scripts/generate_multiple_responses/generate_multiple_resps.py
- to generate multiple responses by model
```