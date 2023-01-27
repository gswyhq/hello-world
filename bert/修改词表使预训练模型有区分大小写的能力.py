#! -*- coding: utf-8 -*-
# 通过简单修改词表，使得不区分大小写的模型有区分大小写的能力
# 基本思路：将英文单词大写化后添加到词表中，并修改模型Embedding层
# 来源：https://github.com/bojone/bert4keras/tree/master/examples

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.snippets import to_array
import numpy as np

import os
USERNAME = os.getenv("USERNAME")

config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = load_vocab(dict_path)
new_token_dict = token_dict.copy()
compound_tokens = []

###################################################### 未修改词表之前 #####################################################################
tokenizer = Tokenizer(token_dict, do_lower_case=False)
model = build_transformer_model(
    config_path,
    checkpoint_path,
    # compound_tokens=compound_tokens,  # 增加新token，用旧token平均来初始化
)
text = u'Welcome to BEIJING. welcome to beijing'
tokens = tokenizer.tokenize(text)
print(tokens)
# ['[CLS]', 'W', '##el', '##come', 'to', 'B', '##E', '##I', '##J', '##I', '##N', '##G', '.', 'welcome', 'to', 'be', '##i', '##jing', '[SEP]']
# 未修改词表之前, 若区分大小写的话：
# BEIJING -> 'B', '##E', '##I', '##J', '##I', '##N', '##G'
# beijing -> 'be', '##i', '##jing'

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = to_array([token_ids], [segment_ids])
result = model.predict([token_ids, segment_ids])
tokens_emd_dict = {token: emd for token, emd in zip(tokens, result[0])}

###################################################### 未修改词表之后 #####################################################################
for t, i in sorted(token_dict.items(), key=lambda s: s[1]):
    # 这里主要考虑三种情况：1、首字母大写；2、整个单词大写；3、整个单词小写
    tokens = []
    if t.isalpha():  # 判断是否为英文字母
        tokens.extend([t.capitalize(), t.upper(), t.lower()])
    elif t[:2] == '##' and t[2:].isalpha():
        tokens.extend([t.upper(), t.lower()])
    for token in tokens:
        if token not in new_token_dict:
            compound_tokens.append([i])
            new_token_dict[token] = len(new_token_dict)

print("新增token数：", len(compound_tokens))
# 新增token数： 5598
print("原词表：", max(token_dict.values()), len(token_dict))
# 原词表： 21127 21128
print("新词表：", max(new_token_dict.values()), len(new_token_dict))
# 新词表： 26725 26726

tokenizer = Tokenizer(new_token_dict, do_lower_case=False)

model = build_transformer_model(
    config_path,
    checkpoint_path,
    compound_tokens=compound_tokens,  # 增加新token，用旧token平均来初始化
)

text = u'Welcome to BEIJING. welcome to beijing'
tokens = tokenizer.tokenize(text)
print(tokens)
"""
输出：['[CLS]', 'Welcome', 'to', 'BE', '##I', '##JING', '.', 'welcome', 'to', 'be', '##i', '##jing', '[SEP]']
"""

token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = to_array([token_ids], [segment_ids])
result2 = model.predict([token_ids, segment_ids])
tokens_emd_dict2 = {token: emd for token, emd in zip(tokens, result2[0])}

tokens = ['[CLS]', 'W', '##el', '##come', 'to', 'B', '##E', '##I', '##J', '##I', '##N', '##G', '.', 'welcome', 'to', 'be', '##i', '##jing', '[SEP]']
tokens2 = ['[CLS]', 'Welcome', 'to', 'BE', '##I', '##JING', '.', 'welcome', 'to', 'be', '##i', '##jing', '[SEP]']


###################################################### huggingFace transformer 模型实践 #####################################################################
# 有时候想要在bert里面加入一些special token, 以huggingFace transformer为例，需要做两个操作：

# 在tokenizer里面加入special token, 防止tokenizer将special token分词。
# resize embedding, 需要为special token初始化新的word embedding。

import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 模型来源：https://huggingface.co/clue/roberta_chinese_3L312_clue_tiny
tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data\roberta_chinese_3L312_clue_tiny')
model = BertModel.from_pretrained(rf'D:\Users\{USERNAME}\data\roberta_chinese_3L312_clue_tiny')

text = '尊享汇康是一款什么样的保险'
tokenizer.tokenize(text)
# ['尊', '享', '汇', '康', '是', '一', '款', '什', '么', '样', '的', '保', '险']
print(tokenizer.encode(text))
# [101, 1892, 671, 2291, 2088, 2741, 573, 3072, 680, 620, 2916, 3904, 812, 5882, 102]
BatchEncoding = tokenizer.batch_encode_plus([text], return_tensors='pt')
result = model(**BatchEncoding)
last_hidden_state = result['last_hidden_state']
pooler_output = result['pooler_output']
last_hidden_state.shape
# Out[181]: torch.Size([1, 15, 312])
pooler_output.shape
# Out[182]: torch.Size([1, 312])

# 查看某个词的向量
model.embeddings.word_embeddings.weight[2741][:5]
# Out[199]: tensor([ 0.0701,  0.0681, -0.0026,  0.0311, -0.0479], grad_fn=<SliceBackward0>)
model.embeddings.word_embeddings.weight[812][:5]
# Out[206]: tensor([ 0.0407, -0.0863,  0.0092,  0.0452, -0.1608], grad_fn=<SliceBackward0>)
model.embeddings.word_embeddings.weight[5882][:5]
# Out[207]: tensor([ 0.0165, -0.0155,  0.0181,  0.0082,  0.0050], grad_fn=<SliceBackward0>)

special_tokens_dict = {'additional_special_tokens': ['尊享汇康', '保险']}
num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
model.resize_token_embeddings(len(tokenizer))
# Embedding(8023, 312)

# 添加token后，原本存在的词，其向量不会改变；
model.embeddings.word_embeddings.weight[2741][:5]
# Out[201]: tensor([ 0.0701,  0.0681, -0.0026,  0.0311, -0.0479], grad_fn=<SliceBackward0>)

# 新增的词随机初始化；
model.embeddings.word_embeddings.weight[8022][:5]
# Out[202]: tensor([ 0.0044,  0.0049, -0.0198,  0.0108,  0.0163], grad_fn=<SliceBackward0>)

tokenizer.tokenize(text)
# Out[188]: ['尊享汇康', '是', '一', '款', '什', '么', '样', '的', '保险']
print(tokenizer.encode(text))
# [101, 8021, 2741, 573, 3072, 680, 620, 2916, 3904, 8022, 102]
BatchEncoding = tokenizer.batch_encode_plus([text], return_tensors='pt')
result = model(**BatchEncoding)
last_hidden_state2 = result['last_hidden_state']
pooler_output2 = result['pooler_output']
last_hidden_state2.shape
# Out[190]: torch.Size([1, 11, 312])
pooler_output2.shape
# Out[191]: torch.Size([1, 312])

# tokenizer.add_special_tokens可以让tokenizer不给’[C1]’,’[C2]’,’[C3]’,’[C4]'进行分词 (这个tokenizer就可以用于后续的数据预处理使用)
# 而resize_token_embeddings可以将bert的word embedding进行形状变换。
# new_num_tokens大于原先的max word的话，会在末尾pad新随机初始化的vector，至于原本就存在的word,对应向量不会改变;
# 小于原先的max word的话，会把末尾的word embedding vector删掉。
# 不传参数的话，什么也不变，只是return原先的embedding回来。而且这个操作只是pad 新的token或cut已有的token，其他位置token的预训练词向量不会被重新初始化。
#
# 额外要提醒的是，加入specail token的话要注意fine-yune的训练量是不是足以训练这些新加入的specail token的embedding (尤其是做few-shot的)。
# 如果训练的数据量不够，能够用已有的special token就别用新加入的([SEP],etc.)



