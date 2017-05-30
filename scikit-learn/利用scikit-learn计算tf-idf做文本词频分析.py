#!/usr/bin/python3
# coding: utf-8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

corpus=["中华 民族 血汗",
        "人民 血汗",
        "共和 国 数学 中华 英语 物理"]

# scikit-learn原生是处理英文的，所以对于单个字母的词（如：a、an、I）会被过滤掉;所以这里的‘国’字会被过滤掉

vectorizer=CountVectorizer()  # CountVectorizer是一个向量计数器

# fit_transform把corpus二维数组转成了一个csr_matrix类型（稀疏矩阵）
# 稀疏矩阵的表示形式，即把二维数组里的所有词语组成的稀疏矩阵的第几行第几列有值
csr_mat = vectorizer.fit_transform(corpus)
print(csr_mat.todense())  # 把稀疏矩阵输出成真实矩阵
# matrix([[1, 0, 0, 0, 1, 0, 0, 1],
#         [0, 1, 0, 0, 0, 0, 0, 1],
#         [1, 0, 1, 1, 0, 1, 1, 0]], dtype=int64)

transformer=TfidfTransformer()
tfidf=transformer.fit_transform(csr_mat)

print(type(tfidf))
print(tfidf)
print(tfidf.todense())

word=vectorizer.get_feature_names()

tfidf_shape = tfidf.get_shape()
for i in range(tfidf_shape[0]):
    for j in range(tfidf_shape[1]):
        if tfidf[i, j] == 0:
            continue
        else:
            print("第{}句，词：{}，的tfidf值是： {}".format(i, word[j], tfidf[i, j]))