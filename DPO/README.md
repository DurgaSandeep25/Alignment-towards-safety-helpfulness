SFT LLM using PEFT

Create environment (virtualenv or conda) -  Python 3.9:  conda create -n NLP685 python=3.9 
pip install torch=2.3.0  torchvision=0.18.0 torchaudio=2.3.0
pip install transformers
pip install scikit-learn
pip install pandas
pip install tensorboard
pip install trl
pip install peft bitsandbytes
pip install wanddb



DPO references:
1. https://github.com/mlabonne/llm-course/blob/main/Fine_tune_a_Mistral_7b_model_with_DPO.ipynb
2. https://huggingface.co/blog/dpo-trl
3. https://gist.github.com/alvarobartt/9898c33eb3e9c7108d9ed2330f12a708




Training:
trainer_bash.sh


Evaluation:
gen_eval_resp.sh

Push to hub:
push_to_hub.py 