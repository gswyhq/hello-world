#!/usr/bin/python3
# coding: utf-8

# pip3 install wmd
# https://github.com/src-d/wmd-relax
# 论文： http://www.cs.cornell.edu/~kilian/papers/wmd_metric.pdf

import time
import numpy
from wmd import WMD
import pickle
embeddings = numpy.array([[0.1, 1], [1, 0.1], [0.8, 0.7]], dtype=numpy.float32)
nbow = {  # key: 序号， 向量， 权重；
        "first":  ("#1", [0, 1, 2], numpy.array([1.5, 0.3, 0.5], dtype=numpy.float32)),
        "你好":  ("#3", [1, 2], numpy.array([1.3, 0.5], dtype=numpy.float32)),
        "second": ("#2", [0, 1], numpy.array([0.75, 0.15], dtype=numpy.float32))}
calc = WMD(embeddings, nbow, vocabulary_min=2)
origin = "first"
print(calc.nearest_neighbors(origin))

model_file = '/home/gswewf/yhb/model/wx_vector_char.pkl'

with open(model_file, "rb")as f:
    w2v_model = pickle.load(f, encoding='iso-8859-1')  # 此处耗内存 60.8 MiB

words_list = []
w_emb = []
for word, emb in w2v_model.items():
    words_list.append(word)
    w_emb.append(emb)

from jieba.analyse.tfidf import TFIDF
tf_idf = TFIDF()
# tf_idf.idf_freq.get('我')

query='发生重疾如何理赔 '
database= ['尊享惠康如何理赔', '患重疾怎么赔', '重疾险如何理赔', '重疾赔付几次', '重疾保额', '重疾赔付次数', '重疾赔付',
           '尊享惠康发生争议如何处理', '重疾保险金怎么赔付', '重疾给付', '重疾包括哪些', '发生争议如何处理', '重疾险保后理赔流程是如何的',
           '医疗事故导致的重疾可以理赔吗', '赔了重疾可以赔轻疾吗', '如何申请理赔', '赔了重疾可以赔轻症吗', '关于轻症重疾确诊就赔吗',
           '如何购买重疾险', '重疾申请理赔流程', '重疾险发生争议如何处理', '重疾种类', '保哪些重疾', '重疾90种',
           '重疾种类解释', '如何理赔', '发生事故如何通知保险公司', '重疾保险金赔付后身故还有的赔吗']

start_time = time.time()
def generator_nbow(question):
    words, weights = [], []
    for word in question:
        if word not in words_list:
            continue
        words.append(words_list.index(word))
        weights.append(tf_idf.idf_freq.get(word, 10))
    sum_weights = sum(weights)
    weights = [w/sum_weights for w in weights]
    words = numpy.array(words, dtype=numpy.uint64)
    return words, weights

nbow = {}
for _index, data in enumerate(database):
    words, weights = generator_nbow(data)
    nbow[data] = (_index, words, weights)

embeddings = numpy.array([list(e) for e in w_emb], dtype=numpy.float32)
calc = WMD(embeddings, nbow, vocabulary_min=2)
origin = generator_nbow(query)

print(calc.nearest_neighbors(origin))

print(time.time()-start_time)

def main():
    pass


if __name__ == '__main__':
    main()

