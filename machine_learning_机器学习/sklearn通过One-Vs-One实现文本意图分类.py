#!/usr/bin/python3
# coding=utf-8

# 来源： https://scikit-learn.org/stable/modules/multiclass.html
# OneVsOneClassifier每对类构造一个分类器。在预测时，选择获得最多投票的类。如果出现平局（在具有相同票数的两个类别中），则通过对基础二元分类器计算的成对分类置信水平求和来选择具有最高聚类分类置信度的类。

import numpy as np
import pickle
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC

INTENT_DATA_FILE = '/home/gswyhq/data/意图识别数据_all.txt'
MODEL_DATA_FILE = '/home/gswyhq/data/OneVsOneClassifier多分类.pkl'

with open(INTENT_DATA_FILE)as f:
    intent_datas = f.readlines()

question_intent_dict = {}
for line in intent_datas[1:]:
    intent, question = line.strip().split(maxsplit=1)
    question_intent_dict.setdefault(question, intent)

question_intent_list = list(question_intent_dict.items())

question_words_set = set(''.join([w for w, _ in question_intent_list]))
intent_set = set([i for w, i in question_intent_list])

words_char = ''.join(list(question_words_set))
intent_lables = list(intent_set)

def onehot_encoded(question, alphabet = 'abcdefghijklmnopqrstuvwxyz '):
    """
    question='adhij'
    Out[45]: 
    array([1., 0., 0., 1., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    :param question: 
    :param alphabet: 
    :return: 
    """
    inverted = np.zeros(len(alphabet))
    for char in set(question) & set(alphabet):
        _index = alphabet.index(char)
        inverted[_index] = 1
    return inverted

x = []
y = []

for question, intent in question_intent_dict.items():
    inverted = onehot_encoded(question, alphabet=words_char)
    x.append(inverted)
    label = intent_lables.index(intent)
    y.append(label)

x = np.array(x)
y = np.array(y)

# 训练模型
model = OneVsOneClassifier(LinearSVC(random_state=0)).fit(x, y)

def predict_intent(datas, model, intent_lables, words_char):
    """
    对问句的意图进行预测
    :param datas:  问句列表
    :param model: 预训练的模型
    :param intent_lables: 意图标签
    :param words_char: 特征字符串
    :return: 
    """
    inverted_lists = []
    for question in datas:
        inverted = onehot_encoded(question, alphabet=words_char)
        inverted_lists.append(inverted)

    intent_nums = model.predict(np.array(inverted_lists))
    ret_dict = {question: intent_lables[num] for question, num in zip(datas, intent_nums)}
    return ret_dict

# 保存模型到文件
save_model_data = {
    "model": model,
    "intent_lables": intent_lables,
    "words_char": words_char
}

with open(MODEL_DATA_FILE, 'wb')as f:
    print('模型保存于： {}'.format(MODEL_DATA_FILE))
    pickle.dump(save_model_data, f)

datas = ['今天天气怎么样', '你们产品有哪些']
ret = predict_intent(datas, model, intent_lables, words_char)
print('模型预测的结果', ret)

# 模型保存于： /home/gswyhq/data/OneVsOneClassifier多分类.pkl
# 模型预测的结果 {'今天天气怎么样': '闲聊', '你们产品有哪些': '保险推荐'}

def main():
    pass


if __name__ == '__main__':
    main()