#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_KERAS'] = '1'
import numpy as np
from collections import Counter
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import sequence_padding, AutoRegressiveDecoder

# bert4keras version: '0.11.3'
# keras __version__ = '2.9.0'

maxlen = 32

import os
USERNAME = os.getenv("USERNAME")

# bert配置
# 模型下载于：
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_simbert_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_simbert_L-4_H-312_A-12/vocab.txt'

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器

# 建立加载模型
bert = build_transformer_model(
    config_path,
    checkpoint_path,
    with_pool='linear',
    application='unilm',
    return_keras_model=False,
)

encoder = keras.models.Model(bert.model.inputs, bert.model.outputs[0])
seq2seq = keras.models.Model(bert.model.inputs, bert.model.outputs[1])


class SynonymsGenerator(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(seq2seq).predict([token_ids, segment_ids])

    def generate(self, text, n=1, topp=0.95):
        token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
        output_ids = self.random_sample([token_ids, segment_ids], n,
                                        topp=topp)  # 基于随机采样
        return [tokenizer.decode(ids) for ids in output_ids]


synonyms_generator = SynonymsGenerator(
    start_id=None, end_id=tokenizer._token_end_id, maxlen=maxlen
)


def gen_synonyms(text, n=100, k=20):
    """"含义： 产生sent的n个相似句，然后返回最相似的k个。
    做法：用seq2seq生成，并用encoder算相似度并排序。
    """
    r = synonyms_generator.generate(text, n)
    r = [i for i in set(r) if i != text]
    r = [text] + r
    X, S = [], []
    for t in r:
        x, s = tokenizer.encode(t)
        X.append(x)
        S.append(s)
    X = sequence_padding(X)
    S = sequence_padding(S)
    Z = encoder.predict([X, S])
    Z /= (Z**2).sum(axis=1, keepdims=True)**0.5
    argsort = np.dot(Z[1:], -Z[0]).argsort()
    return [r[i + 1] for i in argsort[:k]]


def main():
    text='苹果多少钱一斤'
    result = gen_synonyms(text, n=100, k=20)
    print(result)

# ['卖苹果多少钱一斤', '苹果上多少钱一斤？', '苹果卖多少钱一斤？', '苹果价格多少钱一斤', '苹果型多少钱一斤', '苹果店买多少钱一斤', '苹果手机多少钱一斤', '苹果手机多少钱一斤？', '苹果企业的多少钱一斤', '苹果1万斤要多少钱一斤', '苹果售后多少钱一斤', '这台苹果多少钱一斤', '苹果手机在多少钱一斤，要钱多少钱', '苹果售后是多少钱一斤', '现在苹果十几年多少钱一斤', '苹果卖卖如何卖多少钱一斤啊？？', '一斤的苹果，苹果按钮要多少钱', '怎么发卖苹果多少钱一斤', '我有多少钱一斤的苹果？', '苹果一斤价格']

if __name__ == '__main__':
    main()
