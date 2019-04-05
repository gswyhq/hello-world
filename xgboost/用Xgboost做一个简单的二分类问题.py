#!/usr/bin/python3
# coding=utf-8

# https://blog.csdn.net/qq_36142114/article/details/80515721
# 下面这个数据为例，来判断病人是否会在5年内患糖尿病，这个数据前8列是变量，最后一列是预测值为0或1。

# 数据描述： 
# https: // archive.ics.uci.edu / ml / datasets / Pima + Indians + Diabetes

# 下载数据集，并保存为 “pima-indians-diabetes.csv“ 文件： 
# 数据来源：百度网盘(https://pan.baidu.com/s/10ehnADv9ibhJyHbfbvXBBA);提取码：wp4y

# 基础应用

# 引入xgboost等包

from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 分出变量和标签

dataset = loadtxt('/home/gswyhq/data/pima-indians-diabetes.csv', delimiter=",")

X = dataset[:, 0:8]
Y = dataset[:, 8]
# 将数据分为训练集和测试集，测试集用来预测，训练集用来学习模型

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
# xgboost有封装好的分类器和回归器，可以直接用XGBClassifier建立模型，这里是XGBClassifier的文档：
# http://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn

model = XGBClassifier()
model.fit(X_train, y_train)
# xgboost的结果是每个样本属于第一类的概率，需要用round将其转换为01值

y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]
# 得到Accuracy: 77.95 %

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# 监控模型表现

# xgboost可以在模型训练时，评价模型在测试集上的表现，也可以输出每一步的分数，只需要将

# model = XGBClassifier()
# model.fit(X_train, y_train)
# 变为：

model = XGBClassifier()
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, early_stopping_rounds=10, eval_metric="logloss", eval_set=eval_set, verbose=True)
# 那么它会在每加入一颗树后打印出logloss

# [31]    validation_0-logloss:0.487867
# [32]    validation_0-logloss:0.487297
# [33]    validation_0-logloss:0.487562

# 并打印出 Early Stopping 的点：
# Stopping. Best iteration:
# [32]    validation_0-logloss:0.487297

# 3. 输出特征重要度

# gradient
# boosting还有一个优点是可以给出训练好的模型的特征重要性， 
# 这样就可以知道哪些变量需要被保留，哪些可以舍弃。

# 需要引入下面两个类：

# from xgboost import plot_importance
# from matplotlib import pyplot

# 和前面的代码相比，就是在
# fit
# 后面加入两行画出特征的重要性

# model.fit(X, y)

# plot_importance(model)
# pyplot.show()

# 4. 调参

# 如何调参呢，下面是三个超参数的一般实践最佳值，可以先将它们设定为这个范围，然后画出 learning curves，再调解参数找到最佳模型：

# learning_rate ＝ 0.1 或更小，越小就需要多加入弱学习器；
# tree_depth ＝ 2～8；
# subsample ＝ 训练集的 30%～80%；

# 接下来我们用 GridSearchCV 来进行调参会更方便一些：

# 可以调的超参数组合有：

# 树的个数和大小 (n_estimators and max_depth). 
# 学习率和树的个数 (learning_rate and n_estimators). 
# 行列的 subsampling rates (subsample, colsample_bytree and colsample_bylevel).


# 下面以学习率为例：

# 先引入这两个类

# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import StratifiedKFold

# 设定要调节的 learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]，和原代码相比就是在 model 后面加上 grid search 这几行：

# model = XGBClassifier()
# learning_rate = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3]
# param_grid = dict(learning_rate=learning_rate)
# kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
# grid_search = GridSearchCV(model, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=kfold)
# grid_result = grid_search.fit(X, Y)

# 最后会给出最佳的学习率为 0.1

# Best: -0.483013 using {‘learning_rate’: 0.1}

# print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# 我们还可以用下面的代码打印出每一个学习率对应的分数：

# means = grid_result.cv_results_['mean_test_score']
# stds = grid_result.cv_results_['std_test_score']
# params = grid_result.cv_results_['params']
# for mean, stdev, param in zip(means, stds, params):
#     print("%f (%f) with: %r" % (mean, stdev, param))

def main():
    pass


if __name__ == '__main__':
    main()