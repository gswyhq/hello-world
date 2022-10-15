#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
from bert4keras.backend import keras
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.snippets import to_array
import os
USERNAME = os.getenv("USERNAME")
# https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'

tokenizer = Tokenizer(dict_path, do_lower_case=True)  # 建立分词器
model = build_transformer_model(config_path, checkpoint_path)  # 建立模型，加载权重

# 编码测试
text = '好好学习，天天向上'
token_ids, segment_ids = tokenizer.encode(text)
token_ids, segment_ids = to_array([token_ids], [segment_ids])

print('\n ===== 输出每个字的特征向量 =====\n')
print(tokenizer.tokenize(text))
result = model.predict([token_ids, segment_ids])
print(result.shape, result)
"""
['[CLS]', '好', '好', '学', '习', '，', '天', '天', '向', '上', '[SEP]']
(1, 11, 768) [[[ 0.5934874   0.10847043  0.04974978 ...  0.7701504  -0.15371962
   -0.40151072]
  [ 0.87758493  0.25335208 -0.71471214 ...  0.77059495 -0.3238325
   -0.2737752 ]
  [ 1.1124426  -0.5793984  -1.1044426  ...  1.23849    -0.6605567
   -0.04216193]
  ...
  [ 1.0546007  -0.07940374  0.5812258  ...  0.6623699   0.07645733
   -0.4509352 ]
  [ 1.3835579  -0.3811171  -0.6736279  ...  0.43083036  0.27852076
   -0.28636748]
  [ 0.8611593   0.46056855  0.41627747 ... -0.20528266  0.08338898
   -0.5469217 ]]]
"""

def main():
    pass


if __name__ == '__main__':
    main()
