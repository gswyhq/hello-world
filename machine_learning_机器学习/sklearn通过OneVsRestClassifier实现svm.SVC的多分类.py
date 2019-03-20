#!/usr/bin/python3
# coding=utf-8

# 来源： https://blog.csdn.net/xiaodongxiexie/article/details/76229042

import numpy as np
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split


digits = load_digits()

x, y = digits.data, digits.target
y = label_binarize(y, classes=list(range(10)))
x_train, x_test, y_train, y_test = train_test_split(x, y)
# 可以将多分类经过多次2分类最终实现多分类，而sklearn中的multiclass包就可以实现这种方式
model = OneVsRestClassifier(svm.SVC(kernel='linear'))
clf = model.fit(x_train, y_train)

clf.score(x_train, y_train)
# Out[236]: 0.97475872308834444

clf.score(x_test, y_test)
# Out[237]: 0.85999999999999999

np.argmax(y_test, axis=1)
# Out[242]: array([0, 0, 2, ..., 5, 6, 7], dtype=int64)

np.argmax(clf.decision_function(x_test), axis=1)
# Out[243]: array([0, 0, 2, ..., 5, 6, 7], dtype=int64)

def main():
    pass


if __name__ == '__main__':
    main()