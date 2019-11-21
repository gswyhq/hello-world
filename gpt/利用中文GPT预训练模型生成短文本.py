#!/usr/bin/env python
# coding: utf-8

# ## Install the required package

# In[18]:


# get_ipython().system('pip3 install allennlp -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com')


# ## Download the pretrained weights 
# 程序来源: https://colab.research.google.com/drive/1cvBSt2uF7hYL1feDGt0dkCxIeaVXQs5x
# https://github.com/qywu/Chinese-GPT

# encoder.pth
# Encoder Weights: https://drive.google.com/file/d/1Mr2-x_qT2hgyo0RalPjc09NmyNi6a_gs/view?usp=sharing

# model_state_epoch_62.th
# Decoder Weights: https://drive.google.com/file/d/1W6n7Kv6kvHthUX18DhdGSzBYkyzDvxYh/view?usp=sharing

# ## Run the model

# In[2]:


import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import time
import tqdm
import itertools

# uses allennlp modules
from allennlp.nn import util
from allennlp.nn.beam_search import BeamSearch

import sys
sys.path.append('/home/gswyhq/github_projects/Chinese-GPT')
# imports chinese gpt
# https://github.com/qywu/Chinese-GPT
from chinese_gpt import TransformerDecoderLM

# uses bert chinese wordpiece tokenization
from pytorch_pretrained_bert import BertTokenizer


# In[3]:


def top_k_logits(logits, k):
    """Mask logits so that only top-k logits remain
    """
    values, _ = torch.topk(logits, k)
    min_values = values[:, -1].unsqueeze(1).repeat(1, logits.shape[-1])
    return torch.where(logits < min_values, torch.ones_like(logits, dtype=logits.dtype) * -1e10, logits)


# ### Define Bert tokenizer

# In[4]:


tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")


# In[5]:


sentence = "今天北京天气出现小雨,山区还出现了降雪,气温下降,体感十分寒冷。"
print("Tokenize")
print(tokenizer.tokenize(sentence))
print("Tokens to ids")
ids = tokenizer.convert_tokens_to_ids(sentence)
print(ids)
print("Ids to tokens")
print(tokenizer.convert_ids_to_tokens(ids))


# In[6]:


source = '[CLS]' + sentence + '[SEP]'
target = '[CLS]' + sentence + '[SEP]'


# In[7]:


batch_size = 1
# make sure your model is on GPU
#device = torch.device("cuda")

model = TransformerDecoderLM()
#model = model.to(device)


# ### Load weights into the model

# In[8]:


old_state_dict = torch.load("model_state_epoch_62.th", map_location=lambda storage, loc: storage)
new_state_dict = model.state_dict()

for item in new_state_dict.keys():
    new_state_dict[item] = old_state_dict['module.'+item]
    
model.load_state_dict(new_state_dict)


# ### Conditioanl or Unconditional Decoding

# In[9]:


# ask more about news
prompt = tokenizer.tokenize("阿里巴巴集团宣布收购雅虎")
prompt = tokenizer.convert_tokens_to_ids(prompt)


# In[10]:


top_k = 50
temperature = 1.0
length = 0

# start_predictions = torch.LongTensor([[101] + prompt]* batch_size).to(device)
# mask = torch.ones(batch_size, start_predictions.shape[1]).to(device)

start_predictions = torch.LongTensor([[101] + prompt]* batch_size)
mask = torch.ones(batch_size, start_predictions.shape[1])

with torch.no_grad():
    # cache saves in past
    logits, past = model(start_predictions, mask, past=None, past_length=0)
    logits = logits[:, -1, :] / temperature
    logits = top_k_logits(logits, k=top_k)

    sentence = []

    probs = F.softmax(logits, dim=-1)
    prob, prev_pred = torch.topk(probs, k=1, dim=-1)
    sentence.append(prev_pred)
    length += 1

    # decoding loop
    for i in range(500):
        mask = F.pad(mask, (0, 1), "constant", 1.0)
        logits, past = model(prev_pred, mask, past=past, past_length=length)
        logits = logits.squeeze(1) / temperature
        logits = top_k_logits(logits, k=top_k)
        probs = F.softmax(logits, dim=-1)
        prev_pred = torch.multinomial(probs, num_samples=1)
        sentence.append(prev_pred)
        length += 1

    sentence = torch.cat(sentence, dim=-1)

    res = "".join(tokenizer.convert_ids_to_tokens(sentence[0].tolist()))
    
    for i in range(0, 512, 128):
        print(res[i:i+128])

