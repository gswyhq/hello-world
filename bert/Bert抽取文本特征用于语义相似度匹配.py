#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from bert4keras.models import build_transformer_model  # version '0.10.6'
from bert4keras.tokenizers import Tokenizer
from bert4keras.tokenizers import load_vocab
from bert4keras.snippets import sequence_padding

# 接下来定义四个函数：
# + build_tokenizer ： 加载bert的tokenizer，主要是产生句子bert的输入向量。
# + build_model：加载预训练的bert模型。
# + generate_mask：将句子padding为0的的部分mask掉。
# + extract_emb_feature：抽取句子向量，这里笔者实现的是直接将bert最后一层的句子向量取平均。

import os
USERNAME = os.getenv('USERNAME')

###加载tokenizer
def build_tokenizer(dict_path):
    tokenizer = Tokenizer(dict_path, do_lower_case=True)
    return tokenizer


###加载 bert 模型
def build_model(config_path, checkpoint_path, model='bert'):
    model = build_transformer_model(config_path, checkpoint_path, model=model)
    return model


###生成mask矩阵
def generate_mask(sen_list, max_len):
    len_list = [len(i) if len(i) <= max_len else max_len for i in sen_list]
    array_mask = np.array([np.hstack((np.ones(j), np.zeros(max_len - j))) for j in len_list])
    return np.expand_dims(array_mask, axis=2)


###生成句子向量特征
def extract_emb_feature(model, tokenizer, sentences, max_len, mask_if=False, pooling='MEAN'):
    '''
    获取句子的向量表示；
    # CLS：直接用CLS位置的输出向量作为整个句子向量
    # MEAN：计算所有Token输出向量的平均值作为整个句子向量
    # MAX：取出所有Token输出向量各个维度的最大值作为整个句子向量

    :param model:
    :param tokenizer:
    :param sentences:
    :param max_len:
    :param mask_if:
    :return:
    '''
    mask = generate_mask(sentences, max_len)
    token_ids_list = []
    segment_ids_list = []
    for sen in sentences:
        token_ids, segment_ids = tokenizer.encode(sen, maxlen=max_len)
        token_ids_list.append(token_ids)
        segment_ids_list.append(segment_ids)
    token_ids_list = sequence_padding(token_ids_list, max_len)
    segment_ids_list = sequence_padding(segment_ids_list, max_len)
    result = model.predict([np.array(token_ids_list), np.array(segment_ids_list)])
    if mask_if:
        result = result * mask
    if pooling == 'MEAN':
        # 计算所有Token输出向量的平均值作为整个句子向量
        return np.mean(result, axis=1)
    elif pooling == 'MAX':
        # 取出所有Token输出向量各个维度的最大值作为整个句子向量
        return np.max(result, axis=1)
    elif pooling == 'CLS':
        # 直接用CLS位置的输出向量作为整个句子向量
        return np.array([t[0] for t in result])
    else:
        raise ValueError(f"不支持的句子向量提取策略：{pooling}")


# 你需要定义好预训练模型的词库文件，配置文件，模型参数地址，然后加载bert的tokenizer和模型。

checkpoint_path = rf"D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\bert_model.ckpt"
config_path = rf"D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\bert_config.json"
dict_path = rf"D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\vocab.txt"

tokenizer = build_tokenizer(dict_path)
model = build_model(config_path, checkpoint_path, model='roberta')
# 然后笔者用bert提取了
# "这台苹果真是好用", "这颗苹果真是好吃", "苹果电脑很好", "这台小米真是好用"，四句话的文本向量，计算了一下两两句子间的cosine相似度。

sentences = ["这台苹果真是好用", "这颗苹果真是好吃", "苹果电脑很好", "这台小米真是好用"]
sentence_emb = extract_emb_feature(model, tokenizer, sentences, 200)
from sklearn.metrics.pairwise import cosine_similarity

print(cosine_similarity(sentence_emb))

# 笔者用mask了padding为0的地方后，句子相识度结果排序变化不大，但是相似度数值之间的差距变大了(方差变大)，这说明mask还是对文本向量的提取有比较正向的作用。

sentences = ["这台苹果真是好用", "这颗苹果真好吃", "苹果电脑很好", "这台小米真是好用"]
sentence_emb2 = extract_emb_feature(model, tokenizer, sentences, 200, mask_if=True)

print(cosine_similarity(sentence_emb2))

# 其实句子向量的第一个token[CLS] 的向量经常代表句子向量取做下游分类任务的finetune
# CLS：直接用CLS位置的输出向量作为整个句子向量
# MEAN：计算所有Token输出向量的平均值作为整个句子向量
# MAX：取出所有Token输出向量各个维度的最大值作为整个句子向量

# sentence_emb = extract_emb_feature(model, tokenizer, sentences, 128, pooling='MEAN')
# print(cosine_similarity(sentence_emb))
# [[0.99999964 0.9558396  0.8782223  0.970844  ]
#  [0.9558396  0.99999964 0.89639795 0.91976064]
#  [0.8782223  0.89639795 1.0000002  0.83561945]
#  [0.970844   0.91976064 0.83561945 0.99999994]]
# sentence_emb = extract_emb_feature(model, tokenizer, sentences, 128, pooling='MAX')
# print(cosine_similarity(sentence_emb))
# [[1.         0.96705985 0.9401859  0.98441374]
#  [0.96705985 0.9999997  0.94155896 0.9501133 ]
#  [0.9401859  0.94155896 0.99999994 0.92612237]
#  [0.98441374 0.9501133  0.92612237 0.9999999 ]]
# sentence_emb = extract_emb_feature(model, tokenizer, sentences, 128, pooling='CLS')
# print(cosine_similarity(sentence_emb))
# [[0.99999976 0.9177084  0.83967763 0.9427745 ]
#  [0.9177084  1.0000001  0.8402542  0.8575469 ]
#  [0.83967763 0.8402542  1.0000004  0.7658738 ]
#  [0.9427745  0.8575469  0.7658738  1.0000004 ]]

# 资料来源： https://blog.csdn.net/weixin_39847887/article/details/109917839

###################################################################################################################################
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity

# 模型来源：https://huggingface.co/models?language=zh&sort=downloads&p=5
tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese')
model = BertModel.from_pretrained(rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese')

def text_to_vec(text):
    """将文本转换为向量"""

    # text = "这 个 苹 果 真 好 吃 ."
    text = ' '.join(list(text))
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt')
        outputs = model(**encoded_input)
        # 取最后一层的输出
        encoded_layers = outputs[0]
    return encoded_layers[0][0].data.numpy()

sentences = ["这台苹果真是好用", "这颗苹果真是好吃", "苹果电脑很好", "这台小米真是好用", '今天天气不好', '明天下雨']
sentence_emb = [text_to_vec(text) for text in sentences]

print(cosine_similarity(sentence_emb))

###############################################################################################################################
import torch
from transformers import BertTokenizer, AlbertModel
# https://huggingface.co/clue/albert_chinese_tiny
ALBERT_CHINESE_TINY_PATH = rf'D:\Users\{USERNAME}\data\albert_chinese_tiny'
tokenizer = BertTokenizer.from_pretrained(ALBERT_CHINESE_TINY_PATH)
albert = AlbertModel.from_pretrained(ALBERT_CHINESE_TINY_PATH)

def text_to_vec(text):
    """将文本转换为向量"""

    # text = "这 个 苹 果 真 好 吃 ."
    text = ' '.join(list(text))
    with torch.no_grad():
        encoded_input = tokenizer(text, return_tensors='pt')
        outputs = albert(**encoded_input)
        # 取最后一层的输出
        encoded_layers = outputs[0]
    return encoded_layers[0][0].data.numpy()

sentences = ["这台苹果真是好用", "这颗苹果真是好吃", "苹果电脑很好", "这台小米真是好用", '今天天气不好', '明天下雨']
sentence_emb = [text_to_vec(text) for text in sentences]

print(cosine_similarity(sentence_emb))


def main():
    pass


if __name__ == '__main__':
    main()
