#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 导入库
import os
import torch
import torch.nn as nn
# 处理数据
from torchtext.legacy import data
import torch.optim as optim
import numpy as np
# 中文分词处理工具
import jieba
import re
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# step1 : 读取数据
# 注：data、label均读取成list;可以使用sklearn的train_test_split将训练集分为训练集和测试集

# 数据来源：https://github.com/IAdmireu/ChineseSTS.git
def read_data(filepath):
    texts_1=[]
    texts_2=[]
    labels=[]
    for filename in os.listdir(filepath):
        if filename not in ['simtrain_to05sts.txt', 'simtrain_to05sts_same.txt']:
            continue
        with open(os.path.join(filepath, filename),'r', encoding='utf-8') as f:
            lines=f.readlines()
            for line in lines:
                items=line.replace('\n','').strip().split('\t')
                if float(items[4]) != 0 and float(items[4]) != 5:
                    continue
                texts_1.append(items[1])
                texts_2.append(items[3])
                labels.append(float(items[4])//5)
    return texts_1,texts_2,labels
USERNAME = os.getenv('USERNAME')
data_list = read_data(rf'D:\Users\{USERNAME}\github_project\ChineseSTS')

train_texts_1,test_texts_1, train_texts_2,test_texts_2, train_labels,test_labels = train_test_split(*data_list, test_size=0.2, shuffle=True)
# train_texts_1,train_texts_2,train_labels=read_data('./data/train')
# test_texts_1,test_texts_2,test_labels=read_data('./data/test')
print("训练集:",len(train_texts_1))
print("测试集:",len(test_texts_1))

# step 2 : 将文本处理成bert的输入格式的数据
# 注：tokenizer是可选的bert数据预处理的格式化工具，用于文本处理成bert的输入格式

from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/chinese_roberta_L-4_H-128') # 模型来源： https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main
train_encodings = tokenizer(train_texts_1,train_texts_2, truncation=True, padding=True)
test_encodings = tokenizer(test_texts_1,test_texts_2, truncation=True, padding=True)

# step 3: 将bert的输入格式的数据利用Dataset封装成迭代器
# 注：迭代器是一个可以使用for循环访问的python对象

import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = Dataset(train_encodings, train_labels)
test_dataset = Dataset(test_encodings, test_labels)

# step 4:利用dataloader封装Dataset迭代器
# 注：dataloader二次封装便于按照batchsize来给模型提供数据

from torch.utils.data import DataLoader
#生成训练和测试Dataloader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# step 5:封装微调的模型
# 注：BertModel.from_pretrained(model_name)就是载入预训练的bert架构，BertModel以及对应model_name可以提替换成你想使用的模型和参数版本，具体可以参见：huggingface，官网有详细介绍

from transformers import BertModel, AdamW
class myFinrtuneModel(torch.nn.Module):
    def __init__(self,model_name='bert-base-chinese',freeze_bert=False, hidden_dim=768):
        super(myFinrtuneModel,self).__init__()
        # bert模型
        self.bert = BertModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad=False
        # 定义bert后面要接的网络
        self.class_net = torch.nn.Linear(hidden_dim,1)

    # 微调的具体操作
    def forward(self,input_ids,attention_masks):
        # 输入bert
        outputs = self.bert(input_ids, attention_mask=attention_masks)
        # 获取bert输出的隐藏层特征
        last_hidden_state=outputs.last_hidden_state
        # 把token embedding平均得到sentences_embedding
        sentences_embeddings=torch.mean(last_hidden_state,dim=1)
        sentences_embeddings=sentences_embeddings.squeeze(1)
        # 把sentences_embedding输入分类网络
        out=self.class_net(sentences_embeddings).squeeze(-1)
        return out

# step 6:初始化
# #初始化自定义模型

# model=myFinrtuneModel(model_name='bert-base-chinese')
model=myFinrtuneModel(model_name=rf'D:\Users\{USERNAME}\data/chinese_roberta_L-4_H-128', hidden_dim=128)
#模型参数放在cuda上

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

#调整成训练模式

model.train()

#生成优化器

optim = AdamW(model.parameters(), lr=5e-5)

#最大迭代次数

max_epoch=3

#损失函数

loss_function=torch.nn.BCEWithLogitsLoss()

# step 7: 训练、测试、保存函数
#保存函数

import os
from pathlib import Path
def save(model,optimizer,PATH):
    my_file = Path(PATH)
    if not my_file.exists():
        os.system("mkdir "+PATH)
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict()
    },os.path.join(PATH, 'checkpoint'))
    print("保存模型参数")

#训练函数

def train(model,train_loader,test_loader,optim,loss_function,max_epoch):
    print('-------------- start training ---------------','\n')
    step=0
    for epoch in range(max_epoch):
        print("========= epoch:",epoch,'==============')
        for batch in tqdm(train_loader):
            step+=1
            # 清空优化器
            optim.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            # 将用例输入模型，计算loss
            out=model(input_ids=input_ids,attention_masks=attention_mask)
            loss=loss_function(out,labels)

            if step%100==0:
                print('step ',step,"loss:",format(loss.item(),'.3f'))

            # 反向传播
            loss.backward()
            optim.step()

        # 每一次epoch进行一次测试
        eval(model=model,test_loader=test_loader)

#测试函数

def eval(model,test_loader):
    right=0
    total=0
    for batch in tqdm(test_loader):
        total+=1

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        out=torch.sigmoid(model(input_ids=input_ids,attention_masks=attention_mask))
        # 二分类
        pred_label=0 if out.item()<=0.5 else 1
        if pred_label == labels.item():
            right+=1

    accurcy=format(right/total, '.3f')
    print("= accurcy:",accurcy)
    print("\n")


# step 8: 训练与保存模型参数
#训练模型

train(model=model,train_loader=train_loader,test_loader=test_loader,optim=optim,loss_function=loss_function,max_epoch=max_epoch)

#保存模型

save(model,optim,'save_BertModel_for_text_similarity')

# ------- test_finetune_bert_model.py ----------
# step 9 ：应用
# 注：从finetune_bert_model.py中导入myFinrtuneModel

from transformers import BertTokenizer
# from finetune_bert_model import myFinrtuneModel
import torch

#生成bert的文本输入格式化工具
# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/chinese_roberta_L-4_H-128') # 模型来源： https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main

#载入微调之后的保存参数
checkpoint=torch.load('save_BertModel_for_text_similarity/checkpoint')
model=myFinrtuneModel(model_name=rf'D:\Users\{USERNAME}\data/chinese_roberta_L-4_H-128', hidden_dim=128)
model.load_state_dict(checkpoint['model_state_dict'])

#转换为测试模式
model.eval()

#把文本处理成bert输入格式
inputs = tokenizer("吃饭了么","今天你吃饭了吗", return_tensors="pt")
input_ids=inputs['input_ids']
attention_mask=inputs['attention_mask']

#输入模型
outputs = model(input_ids=input_ids,attention_masks=attention_mask)
#输出score
outputs = torch.sigmoid(outputs).item()
#判断二者是否相似
out=0 if outputs>0.5 else 1
print(out)

def main():
    pass


if __name__ == '__main__':
    main()