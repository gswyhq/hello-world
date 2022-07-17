#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re
import pandas as pd
import pickle as pkl
import numpy as np
from sklearn.metrics import r2_score

from jieba.analyse.tfidf import TFIDF
tf_idf = TFIDF()

# Out[128]: '2.6.0'
# tf.__version__

USERNAME = os.getenv('USERNAME') or os.getenv('USER')
EXCEL_DATA_FILE = rf'D:\Users\{USERNAME}\data\数据梳理\数据交易平台\京东万象商品目录ipython\京东万象API商品解析_V5.xlsx'

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

# sentences = ["这台苹果真是好用", "这颗苹果真是好吃", "苹果电脑很好", "这台小米真是好用", '今天天气不好', '明天下雨']
# sentence_emb = [text_to_vec(text) for text in sentences]
#
# print(cosine_similarity(sentence_emb))

def get_keyword_to_vec(text):
    """提取关键词，并还原到原句顺序，再求句向量"""
    if len(text) < 128:
        return text_to_vec(text)
    topk_tags = tf_idf.extract_tags(text, topK=40, allowPOS=('ns', 'n', 'vn', 'v','nr', 'nz'))
    words = [w for w in tf_idf.tokenizer.cut(text)]
    topk_tags = [w for w in topk_tags if w in words]
    topk_tags.sort(key=lambda x: words.index(x))
    return text_to_vec(''.join(topk_tags))

def jiage(text):
    '''\d元/次
    0, 按次，1，按月；
    '''
    p1 = '^(?P<num>([\d\.])+)元/次$'
    s1 = re.search(p1, text)
    p2 = '^(?P<num>([\d\.])+)元/月$'
    s2 = re.search(p2, text)
    p3 = '^(?P<num>([\d\.])+)元/年$'
    s3 = re.search(p3, text)
    p4 = '^(?P<num>([\d\.])+)元/半年$'
    s4 = re.search(p4, text)
    if s1:
        return [0, float(s1.groups('num')[0])]  # 按次
    elif s2:
        return [1, float(s2.groups('num')[0])]  # 按月
    elif s3:
        return [1, float(s3.groups('num')[0])/12]  # 转换为按月
    elif s4:
        return [1, float(s4.groups('num')[0])/6]  # 转换为按月
    elif text == "0元数据":
        return [0, 0]
    elif text == "低于0.01元/次":
        return [0, 0.005]
    elif text == "100元":
        return [0, 100]

FIELD_TYPE_DICT = {'标题': text_to_vec,
                     '描述': text_to_vec,
                     '标签': text_to_vec,
                     '店铺': text_to_vec,
                     '价格': jiage,
                     '详细介绍': get_keyword_to_vec,
                     '数据描述': text_to_vec,
                     '应用场景': text_to_vec,
                     '量级及覆盖范围': text_to_vec,
                     '数据来源': get_keyword_to_vec,
                     '商家实力': get_keyword_to_vec
                   }


def read_data(excel_file=EXCEL_DATA_FILE):
    df = pd.read_excel(excel_file)
    columns = df.columns # ['标题', '亮点', '描述', '评分', '标签', '店铺', '服务', '类型', '价格', 'URL', '数据包基本信息', '数据包内容介绍', '价格.1', '版本', '规格', 'API接口', '详细介绍', '数据描述', '应用场景', '计费方式', '量级及覆盖范围', '更新频率', '数据来源', '商家实力', '数据类型']
    df = df[df['类型']=='API']
    df = df.fillna('')
    field_num_dict = {k: len(set(df[k].values)) for k in columns}
    skip_list = ['URL', '价格.1', '规格', '计费方式', '更新频率']
    columns = [t for t in columns if field_num_dict.get(t) > 1 and t not in skip_list]  # ['标题', '亮点', '描述', '评分', '标签', '店铺', '服务', '价格', 'URL', '价格.1', '规格', 'API接口', '详细介绍', '数据描述', '应用场景', '计费方式', '量级及覆盖范围', '更新频率', '数据来源', '商家实力']

    choice_list = ['标题', '描述', '标签', '店铺',  '详细介绍', '数据描述', '应用场景', '量级及覆盖范围', '数据来源', '商家实力', '价格']
    df2 = df[choice_list]

    x_train = []
    y_train = []
    for lines in df2.values:
        x_train.append([FIELD_TYPE_DICT[choice_list[index]](text) for index, text in enumerate(lines[:-1])])
        y_train.append(FIELD_TYPE_DICT[choice_list[-1]](lines[-1]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # x_train.shape
    # Out[8]: (1088, 10, 312)
    # y_train.shape
    # Out[9]: (1088, 2)
    # y_train = y_train[:, 0]
    # X = x_train.sum(axis=1) # (1088, 10, 312) ->  (1088, 312)

    with open(os.path.join(os.path.split(EXCEL_DATA_FILE)[0], 'jd_api_train_dataset.pkl'), 'wb')as f:
        f.write(pkl.dumps({"x_train": x_train, 'y_train': y_train}))

    return x_train, y_train

import keras.backend as K
def r2(y_true, y_pred):
    '''自定义评价函数'''
    a = K.square(y_pred - y_true)
    b = K.sum(a)
    c = K.mean(y_true)
    d = K.square(y_true - c)
    e = K.sum(d)
    f = 1 - b/e
    return f

def train_model():
    '''
    对API接口数据价格进行预测；
    :return:
    '''
    from keras.models import Sequential
    from keras.layers import Dense, Flatten, Embedding, LSTM
    from keras.utils import np_utils
    from keras.models import load_model

    with open(os.path.join(os.path.split(EXCEL_DATA_FILE)[0], 'jd_api_train_dataset.pkl'), 'rb')as f:
        train_dataset = pkl.load(f)
        x_train = train_dataset['x_train']
        y_train = train_dataset['y_train']

    # 只考虑按次计费模式，忽略掉按月计费模式；
    X, Y = [], []
    for x, y in zip(x_train, y_train):
        if y[0] == 1:
            continue
        else:
            X.append(x)
            Y.append(y[1])
    X = np.array(X)
    Y = np.array(Y)
    # X.shape
    # Out[69]: (1056, 10, 312)
    # Y.shape
    # Out[70]: (1056,)
    # Y[:10]
    # Out[71]:
    # array([0.32 , 0.15 , 0.1  , 0.22 , 0.15 , 0.1  , 0.4  , 0.04 , 0.005, 0.005])

    input_shape = (X.shape[1], X.shape[2])
    n_classes = 1  # 输出层的个数（我们只预测数据价格，就1个值所以输出是1）
    model1 = Sequential()
    model1.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    # model1.add(Flatten())
    # model1.add(Dense(128, activation='relu'))
    model1.add(Dense(16, activation='relu'))
    model1.add(Dense(n_classes))  # 因为我们是预测房价，不是分类，所以最后一层可以不用激活函数

    model1.compile(optimizer='rmsprop',
                   loss='mse',  # 这里要用mean_squared_error 可简写成mse
                   metrics=['mae',r2])  # 正确率这里用mae平均绝对误差和我们自定义的r2

    history1 = model1.fit(X,
                          Y,
                          epochs=50,
                          batch_size=8,
                          validation_split=0.1,
                          # validation_data=(x_val, y_val)
                          )

    model1.save(os.path.join(os.path.split(EXCEL_DATA_FILE)[0], 'API接口按次计费价格预测模型.model'))
    model2 = load_model(os.path.join(os.path.split(EXCEL_DATA_FILE)[0], 'API接口按次计费价格预测模型.model'), custom_objects={"r2": r2})

    batch_size = 8
    test_y, pred_test_y = [], []
    for start_index in range(0, X.shape[0], batch_size):
        y_pred = model2.predict(X[start_index:start_index+batch_size])
        y_label = Y[start_index:start_index + batch_size]
        test_y.extend(list(y_label))
        pred_test_y.extend(list(y_pred))

    pred_acc = r2_score(test_y, pred_test_y)

    # 为了更直观我们在进行可视化一下。

    # 绘图
    import matplotlib.pyplot as plt

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 设置图形大小
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(range(len(test_y)), test_y, ls='-.', lw=2, c='r', label='真实值')
    plt.plot(range(len(pred_test_y)), pred_test_y, ls='-', lw=2, c='b', label='预测值')

    # 绘制网格
    plt.grid(alpha=0.4, linestyle=':')
    plt.legend()
    plt.xlabel('number')  # 设置x轴的标签文本
    plt.ylabel('数据价格')  # 设置y轴的标签文本

    # 展示
    plt.show()

    # 按真实值由小到大排序展示；
    pred_label_datas = [[t1, t2] for t1, t2 in zip(test_y, np.array(pred_test_y)[:, 0])]
    pred_label_datas.sort(key=lambda x: x[0])

    # 设置图形大小
    plt.figure(figsize=(8, 4), dpi=80)
    plt.plot(range(len(pred_label_datas)), [t[0] for t in pred_label_datas], ls='-.', lw=2, c='r', label='真实值')
    plt.plot(range(len(pred_label_datas)), [t[1] for t in pred_label_datas], ls='-', lw=2, c='b', label='预测值')

    plt.legend()
    plt.xlabel('测试样本')  # 设置x轴的标签文本
    plt.ylabel('数据价格')  # 设置y轴的标签文本
    plt.ylim(0, 5) # 设置y轴展示范围
    plt.title("API接口数据价格预测")
    plt.show()

def main():
    pass


if __name__ == '__main__':
    main()