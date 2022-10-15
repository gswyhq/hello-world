#! -*- coding: utf-8 -*-
# 测试代码可用性: 结合MLM的Gibbs采样

# 吉布斯采样(Gibbs Sampling)
# 要完成Gibbs抽样，需要知道条件概率。也就是说，gibbs采样是通过条件分布采样模拟联合分布，再通过模拟的联合分布直接推导出条件分布，以此循环。
# 吉布斯采样（英语：Gibbs sampling）是统计学中用于马尔科夫蒙特卡洛（MCMC）的一种算法，用于在难以直接采样时从某一多变量概率分布中近似抽取样本序列。
# 该序列可用于近似联合分布、部分变量的边缘分布或计算积分（如某一变量的期望值）。某些变量可能为已知变量，故对这些变量并不需要采样。
import random

from tqdm import tqdm
import numpy as np
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
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)  # 建立模型，加载权重

sentences = []
init_sent = u'科学技术是第一生产力。'  # 给定句子或者None
minlen, maxlen = 8, 32
steps = 10000
converged_steps = 1000
vocab_size = tokenizer._vocab_size

if init_sent is None:
    length = np.random.randint(minlen, maxlen + 1)
    tokens = ['[CLS]'] + ['[MASK]'] * length + ['[SEP]']
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
else:
    token_ids, segment_ids = tokenizer.encode(init_sent)
    length = len(token_ids) - 2

for _ in tqdm(range(steps), desc='Sampling'):
    # Gibbs采样流程：随机mask掉一个token，然后通过MLM模型重新采样这个token。
    i = np.random.choice(length) + 1
    token_ids[i] = tokenizer._token_mask_id # 随机mask掉一个token
    probas = model.predict(to_array([token_ids], [segment_ids]))[0, i]  # 预测mask掉的token，可选值及其概率
    token = np.random.choice(vocab_size, p=probas)  # 根据概率分布，随机选择可能的token
    token_ids[i] = token  # 根据选择的token，替换对应的mask掉的token
    sentences.append(tokenizer.decode(token_ids))

print(u'部分随机采样结果：')
for _ in range(10):
    print(np.random.choice(sentences[converged_steps:]))

ts = []
for t in sentences[:300]:
    if t not in ts:
        ts.append(t)
        print(t)

# 科学技术是第一生产力。
# 科学技术是第二生产力。
# 科学技术是第三生产力。
# 科学技术是唯一生产力。
# 科学技术是唯一生产的。
# 化学技术是唯一生产的。
# 科学技术是唯序生产的。
# 科学技术是唯有生产的。
# 科学技术是唯有生命的。
# 科学技术是没有生命的。
# 科学技术是没有寿命的。
# 化学技术是没有生命的。
# 科学技术是否有生命的。
# 科学技术是否有生命的？
# 科学技术是否有生命呢？
# 科学技术是否有生值呢？
# 科学技术是否有价值呢？

###################################################### 按词进行mask #####################################################################
import jieba
from jieba.posseg import dt
from snownlp import SnowNLP
import re
re_han_default = re.compile("([\u4E00-\u9FD5]+)", re.U)

ns_list = [k for k, v in dt.word_tag_tab.items() if v in {'n', 'ns'}]
random_word_weights = {}
for word in ns_list:
    weight = dt.tokenizer.FREQ[word]
    w = word[0]
    random_word_weights.setdefault(w, 0)
    random_word_weights[w] += weight
random_words = list(random_word_weights.keys())
random_words_weight = [min([random_word_weights[word]/10000, 1]) for word in random_words]
random_words_ids = tokenizer.tokens_to_ids(random_words)

text = "深圳是一座海滨城市"
length = len(text)
for _ in tqdm(range(steps), desc='Sampling'):
    # Gibbs采样流程：随机mask掉一个token，然后通过MLM模型重新采样这个token。
    # text = "深圳是一座海滨城市"
    text = SnowNLP(''.join(re.findall(re_han_default, text))).han
    token_ids, segment_ids = tokenizer.encode(text)
    # length = len(token_ids) - 2
    cut_words = jieba.lcut(text)
    if len(token_ids) > length +2:
        cut_words.pop(np.random.choice(len(cut_words)))
        text = ''.join(cut_words)
        token_ids, segment_ids = tokenizer.encode(text)
    mask_words = random.sample(cut_words, 1)
    mask_words_str = ''.join(mask_words)
    for i, token in enumerate(tokenizer.tokenize(text)):
        if token in mask_words_str:
            token_ids[i] = tokenizer._token_mask_id

    # 随机少预测一个mask
    if len(token_ids)-4>length and token_ids.count(tokenizer._token_mask_id) > 2 and random.random() < 0.2:
        # mask在首尾的时候，不要删
        if 2 < token_ids.index(tokenizer._token_mask_id) < len(token_ids)-3:
            token_ids.remove(tokenizer._token_mask_id)
    elif len(token_ids)-4>length and token_ids.count(tokenizer._token_mask_id) > 2 and random.random() < 0.5:
        # mask在首尾的时候，不要删
        if 2 < token_ids.index(tokenizer._token_mask_id) < len(token_ids) - 3:
            token_ids.remove(tokenizer._token_mask_id)
        token_ids[np.random.choice(len(token_ids)-2)+1] = tokenizer._token_mask_id  # 随机mask掉一个token

    # 随机在 mask位置再插入一个 mask
    elif len(token_ids)<=length+2 and random.random() < 0.4:
        token_ids.insert(token_ids.index(tokenizer._token_mask_id), tokenizer._token_mask_id)
    # 随机在 mask位置再插入2个 mask
    elif len(token_ids)<=length+2 and random.random() < 0.4:
        token_ids.insert(token_ids.index(tokenizer._token_mask_id), tokenizer._token_mask_id)
        token_ids.insert(token_ids.index(tokenizer._token_mask_id), tokenizer._token_mask_id)

    mask_indexs = [index for index, value in enumerate(token_ids) if value == tokenizer._token_mask_id]  # 标记token的位置
    segment_ids = [0] * len(token_ids)
    probas = model.predict(to_array([token_ids], [segment_ids]), verbose=0)[0, mask_indexs]
    if all([max(p)>0.5 for p in probas]):
        tokens_choice = []
        weight_choice = []
        # 随机按预测概率选择20个可选的值
        for _ in range(20):
            tokens = [np.random.choice(vocab_size, p=p) for p in probas]
            weight = sum([probas[i][v] for i, v in enumerate(tokens)])  # 多个mask 词，按总权重计其被选概率值
            # print(tokenizer.decode([101] + tokens + [102]), [probas[i][v] for i, v in enumerate(tokens)])
            tokens_choice.append(tokens)
            weight_choice.append(weight)
        # 在20个中按概率选择一个可选值
        tokens = random.choices(tokens_choice, weights=weight_choice)[0]
        weights = []
    else:
        # 每个mask预测的token概率都不大的时候，采用单字排队预测
        tokens = []
        weights = []
        for iter, mask_index in enumerate(mask_indexs):
            if iter == 0:
                if max(probas[iter]) < 0.05:
                    # 当预测概率非常小时候，还不如随机选择的效果
                    # ds = {k: probas[0][v] for k, v in zip(random_words, random_words_ids)}
                    ids_weight = {v: probas[iter][v] for v in random_words_ids}
                    sort_ids = sorted(ids_weight.items(), key=lambda x: x[1])[-20:]
                    token = random.choices([t[0] for t in sort_ids], [t[1] for t in sort_ids])[0]
                else:
                    token = np.random.choice(vocab_size, p=probas[iter])  # 根据概率分布，随机选择可能的token
            else:
                token = sorted(enumerate(probas[iter]), key=lambda x: x[1])[-1][0]
            weight = probas[iter][token]
            token_ids[mask_index] = token  # 根据选择的token，替换对应的mask掉的token
            # print(tokenizer.decode([101, token, 102]), probas[iter][token] )
            tokens.append(token)
            weights.append(weight)
            probas = model.predict(to_array([token_ids], [segment_ids]), verbose=0)[0, mask_indexs]

    for i, token in zip(mask_indexs, tokens):
        token_ids[i] = token  # 根据选择的token，替换对应的mask掉的token
    text = tokenizer.decode(token_ids)
    sentences.append(text)
    print(text, weights)


