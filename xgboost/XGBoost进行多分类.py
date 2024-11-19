#!/usr/bin/env python
# coding=utf-8

# xgboost官方给出的例子是根据34个特征识别6种皮肤病。
# 这是一个6分类问题。

# 第一种情况：给出属于哪个类别
# multi:softmax

import os
USERNAME = os.getenv("USERNAME")
import xgboost as xgb  # 直接导入陈天奇的开源项目，而不是sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import plot_importance
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import xgboost

# 33: lambda x:int(x == '?') 将第33列?转化为0 ，对应第34列数值-1
# https://github.com/datasets/dermatology
data_file = rf'D:/Users/{USERNAME}/github_project/dermatology/data/dermatology.csv'
data = pd.read_csv(data_file, dtype=str)
data = data.fillna('-1').astype(int)
sz = data.shape  # 数据的结构

X = [list(t[:34]) for t in data.values] # 前面列是特征
Y = [t[34]-1 for t in data.values]  # 最后一列是标签label

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                    test_size=0.3,
                                                    random_state=0)

# 加载numpy的数组到DMatrix对象
xg_train = xgb.DMatrix(X_train, label=y_train)
xg_test = xgb.DMatrix(X_test, label=y_test)

# 准备参数
param = {}
param['objective'] = 'multi:softmax'
param['num_class'] = 6  # 6个类别
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
# 这里的并行是指：在同一棵树中，同一个节点选择哪个特征进行划分的时候，可以并行计算gini系数或者mse均方差

watchlist = [(xg_train, 'train'), (xg_val, 'test')]
num_round = 1000
early_stopping_rounds = 3

early_stopping = xgboost.callback.EarlyStopping(
    rounds=early_stopping_rounds,
    min_delta=1e-3,
    save_best=True,
    maximize=False,
    metric_name="mlogloss",
)
evals_result = {}

# 训练模型
bst = xgb.train(param,  # 参数
                xg_train,  # 训练数据
                num_round,  # 弱学习器的个数
                evals=watchlist,
                early_stopping_rounds=early_stopping_rounds,
                evals_result=evals_result,
                callbacks=[early_stopping]);

# 通过测试数据，检测模型的优劣
pred = bst.predict(xg_test);
print('predicting, classification error=%f' % (
            sum(int(pred[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))


bst.save_model(f"model-{bst.best_iteration:02d}-{bst.best_score:.3f}.json")
best_model_file = f"model-{bst.best_iteration:02d}-{bst.best_score:.3f}.json"
print('最佳模型：', best_model_file)

# 加载模型
booster = xgb.Booster()
booster.load_model(best_model_file)
y_pred = booster.predict(xg_test)
print("整体准确率(accuracy)：",  accuracy_score(y_test, y_pred))


# 第二种情况：给出属于每个类别的概率
# multi:softprob
param['objective'] = 'multi:softprob'

bst = xgb.train(param,
                xg_train,
                num_round,
                watchlist);

# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
yprob = np.array(bst.predict(xg_test))
# 从预测的6组中选择最大的概率进行输出
ylabel = np.argmax(yprob, axis=1)  # return the index of the biggest pro

print('predicting, classification error=%f' % (
            sum(int(ylabel[i]) != y_test[i] for i in range(len(y_test))) / float(len(y_test))))

# 最小二乘方差
mse2 = mean_squared_error(y_test, ylabel)

print(mse2)

from sklearn import metrics

print('ACC: %.4f' % metrics.accuracy_score(y_test, ylabel))
print(metrics.confusion_matrix(y_test, ylabel))

# 显示重要特征
plot_importance(bst)
plt.show()

# 值越大，越重要
print('各特征的权重：', bst.get_score(importance_type = 'weight', fmap=''))

# 链接：https://juejin.cn/post/7033320187575140382

def main():
    pass


if __name__ == "__main__":
    main()

