#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import make_classification
import numpy as np
import pickle
from sklearn2pmml import sklearn2pmml
from sklearn2pmml import PMMLPipeline
import os
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor

# 训练模型
path = r"20220315083639"

model_bbc = BalancedBaggingClassifier(base_estimator=XGBClassifier(objective='binary:logistic'), 
                                  sampling_strategy='majority',
                                  replacement=False,
                                  random_state=0)
model_bbc.fit(Xtrain, Ytrain)

try:
    pipeline = PMMLPipeline([("classifier", model_bbc)])  # xg_classifier 为训练好的模型
    # pipeline.fit(iris.data, iris.target)

    # 导出为PMML
    sklearn2pmml(pipeline, "pipeline.pmml", with_repr=True)

except Exception as e:
    print('导出为PMML失败：{}'.format(e))

# 导出为PMML失败：'BalancedBaggingClassifier' object has no attribute 'sampler'

# balance模型转换
dim = 41
has_con = 0
dim = dim + has_con
X, y = make_classification(n_samples=100, n_features=dim,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = BaggingClassifier(base_estimator=SVC(),
                         n_estimators=10, random_state=0)


clf.fit(X, y)
for idx in range(len(clf.estimators_)):
    clf.estimators_[idx] = model_bbc.estimators_[idx][1]
clf.base_estimator = model_bbc.base_estimator
clf.base_estimator_ = model_bbc.base_estimator
# clf.estimators_

a = [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 36, 37, 38, 39, 40]
test_idx_x = np.array(a)

test_x = np.zeros(shape=[1, dim])
test_x[0, [int(tmp) for tmp in test_idx_x[:-has_con] if tmp != -1 and tmp != '-1' ]] = 1
test_x[0, -has_con:] = test_idx_x[-has_con:]

print(model_bbc.predict_proba(test_x))
print(clf.predict_proba(test_x))

try:
    pipeline = PMMLPipeline([("classifier", clf)])  # xg_classifier 为训练好的模型
    # pipeline.fit(iris.data, iris.target)

    # 导出为PMML
    sklearn2pmml(pipeline, os.path.join(path, "pipeline.pmml"), with_repr=True)

except Exception as e:
    print('导出为PMML失败：{}'.format(e))

# 这时就可以成功导出了；

def main():
    pass


if __name__ == '__main__':
    main()

