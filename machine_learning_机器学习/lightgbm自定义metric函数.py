#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sklearn
from sklearn import datasets
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,mean_squared_error
import numpy as np
from matplotlib import pyplot as plt
import lightgbm as lgb
import pickle
from scipy.stats import rankdata


boston_price = datasets.load_breast_cancer()

data = boston_price.data
target = boston_price.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=1024)
print("Train data length:", len(X_train))
print("Test data length:", len(X_test))

# 转换为Dataset数据格式
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# 参数
params = {
    'boosting_type': 'bbdt',  # 设置提升类型
    'objective': 'binary',  # 目标函数
    'metric': {'f1_score'},  # 评估函数
    'first_metric_only': True, # 若设为True, 则仅检查第一个指标 metric
    'num_leaves': 64,  # 叶子节点数
    'learning_rate': 0.05,  # 学习速率
    'nthread': 20,
    'verbose': -1, # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
    'bagging_fraction': 0.8 #	每次迭代时用的数据比例
}

### custom metric test
print("custom metric test ")
# 自定义metric
def custom_auc(preds, train_data):
    """
    :param preds:  array, 预测值
    :param train_data:  lgb Dataset, lgb的传入数据集
    :return:返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    """
    labels = train_data.get_label()
    # preds = 1. / (1. + np.exp(-preds))
    return 'my_auc', roc_auc_score(labels, preds,), True

def wubao_eval(preds, train_data):
    y_true, y_pred = train_data.get_label(), preds
    threshold = 0.5
    a, b, c, d = 0, 0, 0, 0
    for pred, label in zip(y_pred, y_true):
        if label == 0 and pred < threshold:
            a += 1
        elif label == 0 and pred >= threshold:
            b += 1
        elif label == 1 and pred < threshold:
            c += 1
        elif label == 1 and pred >= threshold:
            d += 1
    wubao = b/(b+d) if (b+d) > 0 else 1
    ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return "误报率", wubao, False

def loubao_eval(preds, train_data):
    y_true, y_pred = train_data.get_label(), preds
    threshold = 0.5
    a, b, c, d = 0, 0, 0, 0
    for pred, label in zip(y_pred, y_true):
        if label == 0 and pred < threshold:
            a += 1
        elif label == 0 and pred >= threshold:
            b += 1
        elif label == 1 and pred < threshold:
            c += 1
        elif label == 1 and pred >= threshold:
            d += 1
    loubao = c/(c+d) if (c+d) > 0 else 1
    ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return "漏报率", loubao, False

def fbeta_eval(preds, train_data):
    beta = 2
    y_true, y_pred = train_data.get_label(), preds
    threshold = 0.5
    a, b, c, d = 0, 0, 0, 0
    for pred, label in zip(y_pred, y_true):
        if label == 0 and pred < threshold:
            a += 1
        elif label == 0 and pred >= threshold:
            b += 1
        elif label == 1 and pred < threshold:
            c += 1
        elif label == 1 and pred >= threshold:
            d += 1
    if (b + d) == 0 or (c + d) == 0:
        fbeta_mean = 0
    else:
        precision = d / (b + d)  # 精确率(precision) = (真正)/(真正+假正)
        recall = d / (c + d)  # recall = (真正)/(真正+假负)
        fbeta_mean = (1+beta*beta) * precision * recall / (beta*beta*precision + recall)
        ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return "f-{}".format(beta), fbeta_mean, True

def custom_f1_eval(preds, train_data):
    y_true, y_pred = train_data.get_label(), preds
    threshold = 0.5
    a, b, c, d = 0, 0, 0, 0
    for pred, label in zip(y_pred, y_true):
        if label == 0 and pred < threshold:
            a += 1
        elif label == 0 and pred >= threshold:
            b += 1
        elif label == 1 and pred < threshold:
            c += 1
        elif label == 1 and pred >= threshold:
            d += 1
    if (b + d) == 0 or (c + d) == 0:
        f1_mean = 0
    else:
        precision = d / (b + d)  # 精确率(precision) = (真正)/(真正+假正)
        recall = d / (c + d)  # recall = (真正)/(真正+假负)
        f1_mean = 2 * precision * recall / (precision + recall)
        ###  返回 (评估指标名称, 评估计算值, 是否评估值越大模型性能越好)
    return "f1", f1_mean, True


evals_result = {}  # to record eval results
gbm = lgb.train(params, lgb_train,
                num_boost_round=100, # 迭代次数
                valid_sets=lgb_eval,
                early_stopping_rounds=5, # 如果一次验证数据的一个度量在最近的early_stopping_round 回合中没有提高，模型将停止训练
                evals_result=evals_result,
                feval=[fbeta_eval, custom_f1_eval, wubao_eval, loubao_eval, custom_auc],
                # feval=[ wubao_eval, loubao_eval]
                )

print("最佳迭代次数", gbm.best_iteration)

# 模型预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

print(evals_result)

def main():
    pass


if __name__ == '__main__':
    main()