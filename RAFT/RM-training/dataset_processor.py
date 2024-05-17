from trl import DataCollatorForCompletionOnlyLM
from datasets import load_dataset, Dataset
from tqdm import tqdm

class BaseDatasetProcessor():
    def __init__(self):
        self.dataset = None
    
    def preprocess_for_sft(self, tokenizer):
        pass

    def preprocess_for_rm(self, dataset):
        pass

    def sft_collator(self, tokenizer, mlm=False):
        response_template = "### Response:"
        collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer, mlm=False)
        return collator
    

class RLHFSafe(BaseDatasetProcessor):
    def __init__(self, train_rows = None, test_rows = None, train_type="sft"):
        super().__init__()
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        self.train_dataset = self.dataset['train']
        self.test_dataset = self.dataset['test']
        self.train_dataset = self.train_dataset.to_pandas()
        self.test_dataset = self.test_dataset.to_pandas()

        if train_rows is None:
            train_rows = len(self.train_dataset)
        if test_rows is None:
            test_rows = len(self.test_dataset)
        print(f"training on {train_rows} rows")
        # considering only 1000 for now in training, 100 in testing
        if train_type=="rm":
            self.train_dataset = self.train_dataset[(self.train_dataset['is_response_0_safe'] == True) & (self.train_dataset['is_response_1_safe'] == False)&(self.train_dataset["better_response_id"]==0)][:train_rows]
            # self.train_dataset["desired_response"] = self.train_dataset["response_0"] 
        else:
            self.train_dataset = self.train_dataset[self.train_dataset["is_response_0_safe"]+self.train_dataset["is_response_1_safe"]].reset_index()[:train_rows]
            self.train_dataset["desired_response"] = self.train_dataset.apply(lambda x: x[f'response_{x["safer_response_id"]}'], axis=1)
        self.test_dataset = self.test_dataset[(self.test_dataset['is_response_0_safe'] == True) & (self.test_dataset['is_response_1_safe'] == False)&(self.test_dataset["better_response_id"]==0)][:test_rows]
        self.test_dataset["desired_response"] = self.test_dataset["response_0"]

        self.train_dataset = Dataset.from_pandas(self.train_dataset)
        split = self.train_dataset.train_test_split(test_size=0.2, seed=42)
        self.train_dataset, self.eval_dataset = split['train'], split['test']

        self.test_dataset = Dataset.from_pandas(self.test_dataset)

        # additional variables
        self.prefix = "### Instruction: Answer the following question, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
        self.prefix_resp = "\n### Response: "

        if train_type=="sft":
            self._process_for_sft_train()
        if train_type=="rm":
            self._process_for_rm_train()


    def _process_for_sft_train(self, format_sft=True):
        def _process(dataset):
            dataset = dataset.to_pandas()
            dataset = dataset[["prompt", "desired_response"]]
            dataset["response"] = dataset["desired_response"]
            # print(dataset.columns)
            # print("dataset columns")
            if format_sft:
                # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
                # prefix_resp = "\n### Response: "
                # not using eod token, might have to include it later: TODO##
                dataset["text"] = dataset.apply(lambda x: f"{self.prefix}{x['prompt']}{self.prefix_resp}{x['response']}", axis=1)
            return Dataset.from_pandas(dataset)
        self.train_dataset = _process(self.train_dataset)
        self.test_dataset = _process(self.test_dataset)
        self.eval_dataset = _process(self.eval_dataset)

    def _process_for_rm_train(self):
        def _process(dataset):
            dataset = dataset.to_pandas()
            dataset = dataset[["prompt", "response_0", "response_1"]]
            dataset["chosen"] = dataset.apply(lambda x: f"{x['prompt']} {x['response_0']}", axis=1)
            dataset["rejected"] = dataset.apply(lambda x: f"{x['prompt']} {x['response_1']}", axis=1)
            dataset = dataset[["chosen", "rejected"]]
            return Dataset.from_pandas(dataset)
        
        self.train_dataset = _process(self.train_dataset)
        self.test_dataset = _process(self.test_dataset)
        self.eval_dataset = _process(self.eval_dataset)

    def format_prompts_for_sft_train(self, examples):
        # print("###"*30)
        # print(examples)
        output_texts = []
        # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
        # prefix_resp = "\n### Response: "
        for i in range(len(examples["prompt"])):
            text = f"{self.prefix}{examples['prompt'][i]}{self.prefix_resp}{examples['response'][i]}"
            output_texts.append(text)
        return output_texts

    def process_dataset_for_inferencing(self, dataset):
        # dataset: has 'prompt'
        # returns: Dataset('sft_prompt', 'prompt')
        # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
        # prefix_resp = "\n### Response: "
        dataset = dataset.to_pandas()
        dataset["sft_prompt"] = dataset['prompt'].apply(lambda x: f"{self.prefix}{x}{self.prefix_resp}")
        dataset = Dataset.from_pandas(dataset)
        return dataset
    
    def process_dataset_for_rm(self, dataset):
        """_summary_

        Args:
            dataset (_type_): {'sft_prompt', 'texts', 'responses'}
        """
        def _process(x):
            res = []
            # print("In Process_")
            # print(x)
            for y in x['responses']:
                res.append(f"{x['prompt']} {y}")
            return res

        dataset = dataset.to_pandas()
        dataset['texts'] = dataset.apply(lambda x: _process(x), axis=1)
        dataset = Dataset.from_pandas(dataset)
        return dataset
    
    # def preprocess_for_sft(self, tokenizer):
    #     def formatting_prompts_func(examples):
    #         output_text = []
    #         for i in range(len(examples["prompt"])):
    #             input_text = examples["prompt"][i]

    #             # note that response_0 is safe response (we filtered the data based on that, viceversa can also be done)
    #             response = examples["response"][i]

    #             prompt = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " + input_text + "\n### Response: " + response
    #             prompt += " "+tokenizer.eos_token
    #             output_text.append(prompt)

    #         return output_text
    #     return formatting_prompts_func



class RLHFSafeNew(BaseDatasetProcessor):
    def __init__(self, train_rows = None, test_rows = None, train_type="sft"):
        super().__init__()
        self.dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF")
        self.train_dataset = self.dataset['train']
        self.test_dataset = self.dataset['test']
        self.train_dataset = self.train_dataset.to_pandas()
        self.test_dataset = self.test_dataset.to_pandas()

        if train_rows is None:
            train_rows = len(self.train_dataset)
        if test_rows is None:
            test_rows = len(self.test_dataset)
        
        # considering only 1000 for now in training, 100 in testing
        self.train_dataset = self.train_dataset[(self.train_dataset['is_response_0_safe'] == True) & (self.train_dataset['is_response_1_safe'] == False)&(self.train_dataset["better_response_id"]==0)][:train_rows]
        # self.train_dataset["desired_response"] = self.train_dataset.apply(lambda x: x[f'response_{x["safer_response_id"]}'], axis=1)
        self.test_dataset = self.test_dataset[(self.test_dataset['is_response_0_safe'] == True) & (self.test_dataset['is_response_1_safe'] == False)&(self.test_dataset["better_response_id"]==0)][:test_rows]
        # self.test_dataset["desired_response"] = self.test_dataset["response_0"]

        self.train_dataset = Dataset.from_pandas(self.train_dataset)
        # split = self.train_dataset.train_test_split(test_size=0.2, seed=42)
        # self.train_dataset, self.eval_dataset = split['train'], split['test']
        print(f"training on {len(self.train_dataset)} rows")
        self.test_dataset = Dataset.from_pandas(self.test_dataset)
        self.eval_dataset = self.test_dataset

        # additional variables
        self.prefix = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction: Answer the following question, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n\n\n### Input: " 
        self.prefix_resp = "\n\n### Response: "

        if train_type=="sft":
            self._process_for_sft_train()
        if train_type=="rm":
            self._process_for_rm_train()


    def _process_for_sft_train(self, format_sft=True):
        def _process(dataset):
            dataset = dataset.to_pandas()
            dataset = dataset[["prompt", "response_0"]]
            dataset["response"] = dataset["response_0"]
            # print(dataset.columns)
            # print("dataset columns")
            if format_sft:
                # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
                # prefix_resp = "\n### Response: "
                # not using eod token, might have to include it later: TODO##
                dataset["text"] = dataset.apply(lambda x: f"{self.prefix}{x['prompt']}{self.prefix_resp}{x['response']}", axis=1)
            return Dataset.from_pandas(dataset)
        self.train_dataset = _process(self.train_dataset)
        self.test_dataset = _process(self.test_dataset)
        self.eval_dataset = _process(self.eval_dataset)

    def _process_for_rm_train(self):
        def _process(dataset):
            dataset = dataset.to_pandas()
            dataset = dataset[["prompt", "response_0", "response_1"]]
            dataset["chosen"] = dataset.apply(lambda x: f"{x['prompt']} {x['response_0']}", axis=1)
            dataset["rejected"] = dataset.apply(lambda x: f"{x['prompt']} {x['response_1']}", axis=1)
            dataset = dataset[["chosen", "rejected"]]
            return Dataset.from_pandas(dataset)
        
        self.train_dataset = _process(self.train_dataset)
        self.test_dataset = _process(self.test_dataset)
        self.eval_dataset = _process(self.eval_dataset)

    def format_prompts_for_sft_train(self, examples):
        # print("###"*30)
        # print(examples)
        output_texts = []
        # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
        # prefix_resp = "\n### Response: "
        for i in range(len(examples["prompt"])):
            text = f"{self.prefix}{examples['prompt'][i]}{self.prefix_resp}{examples['response'][i]}"
            output_texts.append(text)
        return output_texts

    def process_dataset_for_inferencing(self, dataset):
        # dataset: has 'prompt'
        # returns: Dataset('sft_prompt', 'prompt')
        # prefix = "### Instruction: Answer the following, while the response provides accurate information and also adherence to ethical standards, respectfulness, avoidance of harmful or inappropriate content.\n\n ### Prompt: " 
        # prefix_resp = "\n### Response: "
        dataset = dataset.to_pandas()
        dataset["sft_prompt"] = dataset['prompt'].apply(lambda x: f"{self.prefix}{x}{self.prefix_resp}")
        dataset = Dataset.from_pandas(dataset)
        return dataset
    
    def process_dataset_for_rm(self, dataset):
        """_summary_

        Args:
            dataset (_type_): {'sft_prompt', 'texts', 'responses'}
        """
        def _process(x):
            res = []
            # print("In Process_")
            # print(x)
            for y in x['responses']:
                res.append(f"{x['prompt']} {y}")
            return res

        dataset = dataset.to_pandas()
        dataset['texts'] = dataset.apply(lambda x: _process(x), axis=1)
        dataset = Dataset.from_pandas(dataset)
        return dataset

DATASET_PREPROCESSOR = {
    "PKU-Alignment/PKU-SafeRLHF":RLHFSafeNew
}