#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import math
import random
import time
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.utils import shuffle
import math
from glob import glob
import keras
from keras.layers import Input, Lambda, Dense
from keras.models import Model, load_model
import keras.backend as K
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
from tensorflow.keras.utils import Sequence
from keras.callbacks import History
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from tqdm import tqdm
from keras.optimizers import adam_v2
from keras.models import load_model
import matplotlib.pyplot as plt
from pylab import mpl
import scipy
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# from keras_bert import load_trained_model_from_checkpoint, Tokenizer
# from transformers import AutoTokenizer, AutoModel
USERNAME = os.getenv('USERNAME')

def compute_corrcoef(x, y):
    """Spearman相关系数
    """
    return scipy.stats.spearmanr(x, y).correlation

def l2_normalize(vecs):
    """l2标准化
    """
    norms = (vecs**2).sum(axis=1, keepdims=True)**0.5
    return vecs / np.clip(norms, 1e-8, np.inf)

def generator_test_data():
    data_path = rf"D:\Users\{USERNAME}\data\similarity\chinese_text_similarity.txt"
    test_path = rf"D:\Users\{USERNAME}\data\similarity\senteval_cn\senteval_cn\STS-B\STS-B.test.data"

    df = pd.read_csv(data_path, encoding='utf-8', sep='\t')
    test_df = pd.read_csv(test_path, encoding='utf-8', sep='\t', names=['text_a', 'text_b', 'label'])
    print(test_df.shape)
    train_text_set = set(df['text_a'].unique()) | set(df['text_b'].unique())
    test_df = test_df[(~test_df['text_a'].isin(train_text_set)) & (~test_df['text_b'].isin(train_text_set))]
    print(test_df.shape)

    # 剔除掉相似度不明确数据
    test_df = test_df[test_df['label'].isin([0, 1, 4, 5])]
    test_df['label'] = [1 if t > 3 else 0 for t in test_df['label'].values]
    # test_df['label'].value_counts()
    # Out[20]:
    # 0    434
    # 1    336
    test_df.to_csv(rf"D:\Users\{USERNAME}\data\客户标签相似性\STS-B.test.csv", encoding='utf-8', sep='\t', index=False)

def my_cos(a, b):
    '''计算余弦相似度'''
    cos_ab = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_ab

def get_cosine_similarity(x_vecs, y_vecs):
    '''两组向量，对应位置向量的余弦值'''
    Y_pred = []
    x_vecs  = np.array(x_vecs)
    y_vecs = np.array(y_vecs)
    for x_vec, y_vec in zip(x_vecs, y_vecs):
        # x_vec = l2_normalize(x_vec)
        # y_vec = l2_normalize(y_vec)
        # y_pred = (x_vec * y_vec).sum(1)
        y_pred = my_cos(x_vec, y_vec)
        Y_pred.append(y_pred)
    return Y_pred

def model_evaluate(x_vecs, y_vecs, Y_true):
    """模型评估，先计算两组向量间余弦值，再与标签进行相似度比较"""
    Y_pred = get_cosine_similarity(x_vecs, y_vecs)
    return compute_corrcoef(Y_true, Y_pred)

def load_test_data():
    """加载测试数据"""
    test_df = pd.read_csv(rf"D:\Users\{USERNAME}\data\客户标签相似性\STS-B.test.csv", encoding='utf-8', sep='\t')
    text_a_list = list(test_df['text_a'].values)
    text_b_list = list(test_df['text_b'].values)
    y_true = test_df['label'].values
    return text_a_list, text_b_list, y_true

def cosent_loss(y_true, y_pred):
    """排序交叉熵
    y_true：标签/打分，y_pred：句向量
    """
    y_true = y_true[::2, 0]  # 获取偶数位标签，即取出真实的标签；
    y_true = K.cast(y_true[:, None] < y_true[None, :], K.floatx())  # 取出负例-正例的差值
    y_pred = K.l2_normalize(y_pred, axis=1)  # 对输出的句子向量进行l2归一化   后面只需要对应位相乘  就可以得到cos值了
    y_pred = K.sum(y_pred[::2] * y_pred[1::2], axis=1) * 20  # 奇偶位向量相乘，得到对应cos
    y_pred = y_pred[:, None] - y_pred[None, :]  # 取出负例-正例的差值, # 这里是算出所有位置 两两之间余弦的差值
    # 矩阵中的第i行j列  表示的是第i个余弦值-第j个余弦值
    y_pred = K.reshape(y_pred - (1 - y_true) * 1e12, [-1])  # 乘以e的12次方,要排除掉不需要计算(mask)的部分
    y_pred = K.concatenate([[0], y_pred], axis=0)  # 这里加0是因为e^0 = 1相当于在log中加了1
    return K.logsumexp(y_pred)

def pad_sequences(sequences, maxlen=None, value=0):
    """
    pad sequences (num_samples, num_timesteps) to same length
    """
    if maxlen is None:
        maxlen = max(len(x) for x in sequences)

    outputs = []
    for x in sequences:
        x = x[:maxlen]
        pad_range = (0, maxlen - len(x))
        x = np.pad(array=x, pad_width=pad_range, mode='constant', constant_values=value)
        outputs.append(x)
    return np.array(outputs)

def get_best_f1_thr(y_true, y_score):

    precisions, recalls, thresholds = precision_recall_curve(list(y_true), list(y_score))

    # 拿到最优f1-score结果以及索引
    beta = 1
    f1_scores = ((1+beta*beta) * precisions * recalls) / (beta*beta*precisions + recalls)  # 计算全部f1-score
    best_f1_score = np.max(f1_scores[np.isfinite(f1_scores)])
    best_f1_score_index = np.argmax(f1_scores[np.isfinite(f1_scores)])
    best_f1_thr = thresholds[best_f1_score_index]
    return best_f1_thr

def SimCSE_bert_base():
    from transformers import AutoTokenizer, AutoModel
    # 模型来源：
    # https://huggingface.co/tuhailong/SimCSE-bert-base/tree/main
    model = AutoModel.from_pretrained(rf"D:\Users\{USERNAME}\data\SimCSE-bert-base")
    tokenizer = AutoTokenizer.from_pretrained(fr"D:\Users\{USERNAME}\data\SimCSE-bert-base")

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]
        inputs = tokenizer(sentences, return_tensors="pt", padding='max_length', truncation=True, max_length=32)
        batch_output = model(**inputs)
        outputs.extend(batch_output[0][:,0,:].tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    return y_pred


def RoBERTa_tiny():
    from keras_bert import load_trained_model_from_checkpoint, Tokenizer
    # 要求TensorFlow>=2.2.0
    # https://storage.googleapis.com/cluebenchmark/pretrained_models/RoBERTa-tiny-clue.zip
    config_path = rf'D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\bert_config.json'
    checkpoint_path = rf'D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\bert_model.ckpt'
    dict_path = rf'D:\Users\{USERNAME}\data\RoBERTa-tiny-clue\vocab.txt'

    token_dict = {}
    with open(dict_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = Tokenizer(token_dict)
    cut_words = tokenizer.tokenize(u'今天天气不错')
    print(cut_words)

    bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path, seq_len=None)
    ###########################################################################################################################
    output = Lambda(lambda x: x[:, 0])(bert_model.output)
    encoder2 = Model(bert_model.inputs, output)

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]

        a_token_ids = []
        for word in tqdm(sentences):
            token_ids = tokenizer.encode(word, max_len=16)[0]
            a_token_ids.append(token_ids)
        a_token_ids = np.array(a_token_ids)
        a_vecs = encoder2.predict([a_token_ids,
                                   np.zeros_like(a_token_ids)],
                                  verbose=False)
        outputs.extend(a_vecs.tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    return y_pred


def ERNIE_3_Tiny_Nano_v2_zh():
    # pip3 install Downloads\paddlepaddle-2.4.1-cp39-cp39-win_amd64.whl --user
    # pip3 install Downloads\paddlenlp-2.5.0-py3-none-any.whl --user
    # https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/ernie-tiny
    import paddle
    from paddlenlp.transformers import AutoTokenizer, AutoModelForSequenceClassification, ErnieTokenizer, AutoModel

    from paddlenlp.transformers.ernie.tokenizer import ErnieTokenizer
    from paddlenlp.transformers.ernie.configuration import ERNIE_PRETRAINED_RESOURCE_FILES_MAP

    print('预训练模型文件下载地址：', ERNIE_PRETRAINED_RESOURCE_FILES_MAP['model_state']["ernie-3.0-tiny-nano-v2-zh"])
    print('词表文件下载地址：', ErnieTokenizer.pretrained_resource_files_map['vocab_file']["ernie-3.0-tiny-nano-v2-zh"])
    # https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_nano_v2.pdparams
    # https://bj.bcebos.com/paddlenlp/models/transformers/ernie_3.0/ernie_3.0_tiny_nano_v2_vocab.txt

    # 离线使用的话，需先下载对应的文件到"~/.paddlenlp/models/ernie-3.0-tiny-nano-v2-zh"目录下：
    tokenizer = ErnieTokenizer.from_pretrained("ernie-3.0-tiny-nano-v2-zh")
    model = AutoModel.from_pretrained("ernie-3.0-tiny-nano-v2-zh", 'Model')
    model.eval()

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]
        a_token_ids = [tokenizer.encode(text)['input_ids'] for text in sentences]
        a_token_ids = pad_sequences(a_token_ids, maxlen=16)
        batch_output = model(input_ids=paddle.to_tensor(a_token_ids),
                    token_type_ids=paddle.to_tensor(np.zeros_like(a_token_ids)))
        # sequence_output, pooled_output = model(input_ids=paddle.to_tensor([text['input_ids']]))
        outputs.extend(batch_output[0][:,0,:].tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    return y_pred


def SimBERT_tiny():
    # 环境要求：
    # tensorflow 1.14 + kerar==2.3.1 bert4keras==0.7.7
    # 模型下载于：
    # SimBERT-tiny
    # https://github.com/ZhuiyiTechnology/pretrained-models
    # https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-4_H-312_A-12.zip
    import os
    import numpy as np
    from tqdm import tqdm
    from bert4keras.backend import keras, K
    from bert4keras.tokenizers import Tokenizer
    from bert4keras.models import build_transformer_model
    from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
    # from bert4keras.utils import DataGenerator, pad_sequences
    from bert4keras.layers import Layer
    from bert4keras.models import Model

    config_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/bert_config.json'
    checkpoint_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/bert_model.ckpt'
    dict_path = './result/pre-training/chinese_simbert_L-4_H-312_A-12/vocab.txt'

    # 建立分词器
    tokenizer = Tokenizer(dict_path, do_lower_case=True)

    cut_words = tokenizer.tokenize(u'今天天气不错')
    print(cut_words)

    simbert_model = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
    )

    output = Lambda(lambda x: x[:, 0])(simbert_model.output)
    model = Model(simbert_model.inputs, output)

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]

        a_token_ids = []
        for word in tqdm(sentences):
            token_ids = tokenizer.encode(word, max_length=16)[0]
            a_token_ids.append(token_ids)
        a_token_ids = np.array(a_token_ids)
        a_token_ids = pad_sequences(a_token_ids, maxlen=16)
        a_vecs = model.predict([a_token_ids,
                                   np.zeros_like(a_token_ids)],
                                  verbose=False)
        outputs.extend(a_vecs.tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    test_df['SimBERT_tiny'] = y_pred
    test_df.to_csv("./result/SimBERT_tiny_test_result.csv", encoding='utf-8', sep='\t', index=False)
    return y_pred


def SimBERT_base():
    # 环境要求：
    # tensorflow 2.9.1 + kerar==2.9.0
    #
    # 模型下载于：
    # SimBERT Base
    # https://github.com/ZhuiyiTechnology/pretrained-models
    # https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-12_H-768_A-12.zip
    model = load_model(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-12_H-768_A-12/SimBERT-Base_tf2.hdf5",
                       custom_objects={"PositionEmbedding": PositionEmbedding, "FeedForward": FeedForward,
                                       "LayerNormalization": LayerNormalization,
                                       "MultiHeadAttention": MultiHeadAttention,
                                       "gelu_erf": gelu_erf,
                                       })

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]

        a_token_ids = []
        for word in tqdm(sentences):
            token_ids = tokenizer.encode(word, maxlen=16)[0]
            a_token_ids.append(token_ids)
        a_token_ids = np.array(a_token_ids)
        a_token_ids = pad_sequences(a_token_ids, maxlen=16)
        a_vecs = model.predict([a_token_ids,
                                   np.zeros_like(a_token_ids)],
                                  verbose=False)
        outputs.extend(a_vecs.tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    model_result_df = pd.read_excel(rf"D:\Users\{USERNAME}\data\客户标签相似性\model_test_result.xlsx", dtype=object)
    model_result_df['SimBERT_base'] = y_pred
    model_result_df.to_excel(rf"D:\Users\{USERNAME}\data\客户标签相似性\model_test_result.xlsx", index=False)
    return y_pred


def SimBERT_Small():
    # 环境要求：
    # tensorflow 2.9.1 + kerar==2.9.0
    # 模型下载于：
    # SimBERT Small
    # https://github.com/ZhuiyiTechnology/pretrained-models
    # https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_simbert_L-6_H-384_A-12.zip
    model = load_model(rf"D:\Users\{USERNAME}\data\chinese_simbert_L-6_H-384_A-12/SimBERT-Small_tf2.hdf5",
                       custom_objects={"PositionEmbedding": PositionEmbedding, "FeedForward": FeedForward,
                                       "LayerNormalization": LayerNormalization,
                                       "MultiHeadAttention": MultiHeadAttention,
                                       "gelu_erf": gelu_erf,
                                       })

    text_a_list, text_b_list, y_true = load_test_data()
    sentences_str_list = text_a_list + text_b_list
    batch_size = 32
    num_batches = math.ceil(len(sentences_str_list)/batch_size)
    outputs = []
    for num_batch in tqdm(range(num_batches)):
        sentences = sentences_str_list[num_batch*batch_size: num_batch*batch_size+batch_size]

        a_token_ids = []
        for word in tqdm(sentences):
            token_ids = tokenizer.encode(word, maxlen=16)[0]
            a_token_ids.append(token_ids)
        a_token_ids = np.array(a_token_ids)
        a_token_ids = pad_sequences(a_token_ids, maxlen=16)
        a_vecs = model.predict([a_token_ids,
                                   np.zeros_like(a_token_ids)],
                                  verbose=False)
        outputs.extend(a_vecs.tolist())
    a_vecs = outputs[:len(text_a_list)]
    b_vecs = outputs[len(text_a_list):]
    # cosine_similarity(outputs[0][0][0].reshape(1, -1).tolist(), outputs[0][1][0].reshape(1, -1).tolist(), )
    y_pred = get_cosine_similarity(a_vecs, b_vecs)
    model_result_df = pd.read_excel(rf"D:\Users\{USERNAME}\data\客户标签相似性\model_test_result.xlsx", dtype=object)
    model_result_df['SimBERT_small'] = y_pred
    model_result_df.to_excel(rf"D:\Users\{USERNAME}\data\客户标签相似性\model_test_result.xlsx", index=False)
    return y_pred

def Erlangshen_SimCSE_110M_Chinese():
    # Erlangshen-SimCSE-110M-Chinese
    # 模型下载于：
    # https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese
    import os
    import torch
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import AutoTokenizer, AutoModelForMaskedLM
    USERNAME = os.getenv('USERNAME')
    model = AutoModelForMaskedLM.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')
    tokenizer = AutoTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')
    text_a_list, text_b_list, y_true = load_test_data()

    y_pred = []
    for texta, textb in tqdm(zip(text_a_list, text_b_list)):
        inputs_a = tokenizer(texta, return_tensors="pt")
        inputs_b = tokenizer(textb, return_tensors="pt")

        outputs_a = model(**inputs_a, output_hidden_states=True)
        texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

        outputs_b = model(**inputs_b, output_hidden_states=True)
        textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

        # if you use cuda, the text_embedding should be textb_embedding.cpu().numpy()
        # 或者用torch.no_grad():
        with torch.no_grad():
            silimarity_soce = cosine_similarity(texta_embedding.reshape(1, -1), textb_embedding.reshape(1, -1))[0][0]
        y_pred.append(silimarity_soce)
    return y_pred

def Erlangshen_Roberta_110M_Similarity():
    # Erlangshen-Roberta-110M-Similarity
    # https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Similarity
    from transformers import BertForSequenceClassification
    from transformers import BertTokenizer
    import torch

    tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-Roberta-110M-Similarity')
    model = BertForSequenceClassification.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-Roberta-110M-Similarity')
    text_a_list, text_b_list, y_true = load_test_data()

    y_pred = []
    for texta, textb in tqdm(zip(text_a_list, text_b_list)):
        output = model(torch.tensor([tokenizer.encode(texta, textb)]))
        silimarity_soce = torch.nn.functional.softmax(output.logits, dim=-1).tolist()[0][-1]
        y_pred.append(silimarity_soce)
    return y_pred


def main():
    model_result = {
        "SimCSE-bert-base": SimCSE_bert_base(),
        "RoBERTa-tiny-clue": RoBERTa_tiny(),
        "SimBERT-tiny": SimBERT_tiny,
        "SimBERT-Base": SimBERT_base(),
        "SimBERT-Small": SimBERT_Small(),
        "Erlangshen-SimCSE-110M-Chinese": Erlangshen_SimCSE_110M_Chinese(),
        "Erlangshen-Roberta-110M-Similarity": Erlangshen_Roberta_110M_Similarity(),
        "ERNIE-3.0-Tiny-Nano-v2-zh": ERNIE_3_Tiny_Nano_v2_zh(),
    }

    model_result_df = pd.read_csv(rf"D:\Users\{USERNAME}\data\客户标签相似性\STS-B.test.csv", encoding='utf-8', sep='\t')
    for mode, y_pred in model_result.items():
        model_result_df[mode] = y_pred
    corr_dict = {}
    y_true = model_result_df['label'].values
    for mode in model_result_df.columns:
        if mode in ['text_a', 'text_b', 'label']:
            continue
        y_pred = model_result_df[mode].values
        corr = compute_corrcoef(y_true, y_pred)
        corr_dict[mode] = corr
    print(corr_dict)

    # 按阈值统计：
    count_dict = {}
    y_true = model_result_df['label'].values
    for mode in model_result_df.columns:
        if mode in ['text_a', 'text_b', 'label']:
            continue
        y_pred = model_result_df[mode].values
        # max_score, min_score = max(y_pred), min(y_pred)
        # y_pred = [(y-min_score)/(max_score-min_score) for y in y_pred]
        thr = get_best_f1_thr(y_true, y_pred)
        print("模型：{}，阈值为：{}".format(mode, thr))
        corr = len([1 for k, v in zip(y_true, y_pred) if (v >= thr and k == 1) or (v < thr and k == 0)])
        count_dict[mode] = corr / len(y_true)
    print(count_dict)

# 按相关性统计：
{'SimBERT_tiny': 0.6664634723531726,
 'SimCSE_bert_base': 0.5682280590504014,
 'RoBERTa_tiny_base': 0.47994098547523795,
 'SimBERT_base': 0.6798420568844904,
 'SimBERT_small': 0.6741399885461532,
 'Erlangshen-SimCSE-110M-Chinese': 0.7874373018184543,
 'Erlangshen-Roberta-110M-Similarity': 0.713357279986606,
 'ERNIE-3.0-Tiny-Nano-v2-zh': 0.3551538862901387}

# 模型：SimBERT_tiny，阈值为：0.9431375670981488
# 模型：SimCSE_bert_base，阈值为：0.6238064033092366
# 模型：RoBERTa_tiny_base，阈值为：0.828225386925064
# 模型：SimBERT_base，阈值为：0.9225978780869507
# 模型：SimBERT_small，阈值为：0.9185348173258062
# 模型：Erlangshen-SimCSE-110M-Chinese，阈值为：0.8233329057693481
# 模型：Erlangshen-Roberta-110M-Similarity，阈值为：0.2488364279270172
# 模型：ERNIE-3.0-Tiny-Nano-v2-zh，阈值为：0.9056764231871169
# Out[93]:
# {'SimBERT_tiny': 0.8415584415584415,
#  'SimCSE_bert_base': 0.7571428571428571,
#  'RoBERTa_tiny_base': 0.7012987012987013,
#  'SimBERT_base': 0.8415584415584415,
#  'SimBERT_small': 0.8389610389610389,
#  'Erlangshen-SimCSE-110M-Chinese': 0.8883116883116883,
#  'Erlangshen-Roberta-110M-Similarity': 0.8532467532467533,
#  'ERNIE-3.0-Tiny-Nano-v2-zh': 0.535064935064935}

if __name__ == '__main__':
    main()
