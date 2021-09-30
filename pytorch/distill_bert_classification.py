#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
from sklearn import metrics
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import transformers
from torch import cuda

# 代码来源： https://github.com/goutham794/Bert-text-classification/blob/master/distill_bert_classification.ipynb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)

USERNAME = os.getenv('USERNAME') or os.getenv('USER')

# https://huggingface.co/liam168/c4-zh-distilbert-base-uncased
DISTILBERT_CHINEST_BASE_UNCASED = f'/appcom/apps-data/tmp/{USERNAME}/data/distilbert-base-uncased'
DATA_PATH = rf'/appcom/apps-data/tmp/{USERNAME}/code/bert_distill/data/hotel'

# Setting some configs.
MAX_LEN = 160
BATCH_SIZE = 16
LEARNING_RATE = 1e-05


MODEL_PATH = "pytorch_model"
tokenizer = transformers.DistilBertTokenizer.from_pretrained(
    DISTILBERT_CHINEST_BASE_UNCASED,
    do_lower_case=True
)

# Creating the dataset object
class tweet_Dataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=MAX_LEN):
        with open(dataframe, encoding='utf-8')as f:
            data_set = [t.strip().split('\t') for t in f.readlines() if t and t.strip()]
        self.data = pd.DataFrame([(int(label), text) for label, text in data_set], columns=('target', 'text'))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        text = str(self.data.text[index])
        tokens = self.tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens[:self.max_len - 2] + ["[SEP]"]
        inputs = self.tokenizer.encode_plus(
            tokens,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'targets': torch.tensor(self.data.target[index], dtype=torch.float)
        }

    def __len__(self):
        return len(self.data)

training_set = tweet_Dataset(os.path.join(DATA_PATH, 'train.txt'), tokenizer, MAX_LEN)
testing_set = tweet_Dataset(os.path.join(DATA_PATH, 'test.txt'), tokenizer, MAX_LEN)

train_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

valid_params = {'batch_size': BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

train_dl = DataLoader(training_set, **train_params)
valid_dl = DataLoader(testing_set, **valid_params)


class DistillBERTClass(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distill_bert = transformers.DistilBertModel.from_pretrained(DISTILBERT_CHINEST_BASE_UNCASED) # distilbert-base-uncased
        self.drop = torch.nn.Dropout(0.3)
        self.out = torch.nn.Linear(768, 1)

    def forward(self, ids, mask):
        distilbert_output = self.distill_bert(ids, mask)
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        output_1 = self.drop(pooled_output)
        output = self.out(output_1)
        return output


model = DistillBERTClass()
model.to(device)


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets)


optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

def eval_fn(data_loader, model):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
            ids = d["ids"]
            mask = d["mask"]
            targets = d["targets"]

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(ids=ids, mask=mask)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        fin_outputs = np.array(fin_outputs) >= 0.5
        f1_score = metrics.f1_score(fin_targets, fin_outputs)
    return f1_score

def fit(num_epochs, model, loss_fn, opt, train_dl, valid_dl):
    for epoch in range(num_epochs):
        model.train()
        for _, data in enumerate(train_dl, 0):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)
            outputs = model(ids, mask).squeeze()
            loss = loss_fn(outputs, targets)
            loss.backward()
            opt.step()
            opt.zero_grad()

        valid_acc = eval_fn(valid_dl, model)
        print(
            'Epoch [{}/{}], Train Loss: {:.4f} and Validation acc {:.4f} and loss {:.4f}'.format(epoch + 1, num_epochs,
                                                                                                 loss.item(), valid_acc,
                                                                                                 1.1))
    torch.save(model, './data/DistillBERTClass')

fit(7, model, loss_fn, optimizer, train_dl  , valid_dl)

def sentence_prediction(sentence, max_len = MAX_LEN):
    tokens = self.tokenizer.tokenize(sentence)
    tokens = ["[CLS]"] + tokens[:self.max_len - 2] + ["[SEP]"]

    inputs = tokenizer.encode_plus(
            tokens,
            None,
            add_special_tokens=True,
            max_length=max_len,
            pad_to_max_length=True,
        )

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]


    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)

    outputs = model(ids=ids, mask=mask)

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0] > 0.5

print(sentence_prediction("硬件设施太旧挨着大街，环境很吵价格偏高不符4星"))

print(sentence_prediction("还不错, 房间很大,装修家居都很好,唯一不足没订到海景房间,好像很抢手的说"))

def main():
    pass


if __name__ == '__main__':
    main()