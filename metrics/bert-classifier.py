import torch
from torch import cuda
from transformers import BertTokenizer, BertModel, BertConfig
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
import pandas as pd

device = 'cuda' if cuda.is_available() else 'cpu'

## Code has been referred from https://colab.research.google.com/github/abhimishra91/transformers-tutorials/blob/master/transformers_multi_label_classification.ipynb
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        _, output_1= self.l1(ids, attention_mask = mask, token_type_ids = token_type_ids, return_dict=False)
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output

    
load_model = torch.load("./toxicclassifier-bal.pt")


class CustomDatasetHexphi(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.comment_text = dataframe.comment_text
        # self.targets = (dataframe[dataframe.columns.tolist()[2:]].sum(axis=1) > 0)*1 #self.data.list
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())

        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]


        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'comment_text':comment_text
            # 'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

def evaluate():
    load_model.eval()
    fin_targets=[]
    fin_outputs=[]
    fin_txts=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            # targets = data['targets'].to(device, dtype = torch.float)
            outputs = load_model(ids, mask, token_type_ids)
            # fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
            fin_txts.extend(data['comment_text'])
            # print(f"{_}/{len(testing_loader)}")
    return fin_outputs, fin_txts



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
category = ['Illegal_Activity', 'Child_Abuse_Content', 'Hate_Harass_Violence', 'Malware', 'Physical_Harm', 'Economic_Harm', 'Fraud_Deception', 'Adult_Content', 'Political_Campaigning', 'Privacy_Violation_Activity', 'Tailored_Financial_Advice']
for i in range(1,12):
    df = pd.read_csv(f"./category_{i}.csv",header=None)
    df.rename(columns={0: 'comment_text'}, inplace=True) 
    
    

    testing_set = CustomDatasetHexphi(df, tokenizer, 200)
    test_params = {'batch_size': 4,
                    'shuffle': False,
                    'num_workers': 0
                    }

    testing_loader = DataLoader(testing_set, **test_params)


    outputs, txts = evaluate()
    targets = [1]*len(outputs)
    outputs = np.array(outputs) >= 0.5
    accuracy = metrics.accuracy_score(targets, outputs)
    f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
    f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
    print(f"Category:{category[i-1]}")
    print(f"Accuracy Score = {accuracy}")
    print(f"F1 Score (Micro) = {f1_score_micro}")
    print(f"F1 Score (Macro) = {f1_score_macro}")
    
    confusion_matrix(targets, outputs)