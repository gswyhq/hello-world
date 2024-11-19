#!/usr/bin/env python
# coding=utf-8

import numpy as np
import xgboost
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import metrics
from xgboost.sklearn import XGBClassifier
from sklearn2pmml import PMMLPipeline,sklearn2pmml, make_pmml_pipeline
from pypmml import Model
from nyoka import xgboost_to_pmml

iris = load_iris()
# 提取特征名称（数据的名称）
feature = iris['feature_names']
# feature = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# 目标名称
target = 'class'
# 获取数据
data = iris['data']
# 获取标签
label = iris['target']
# 获取训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(data, label, train_size=0.8)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, train_size=0.5)

# 加载XGBoost
num_round = 1000
early_stopping_rounds = 5

early_stopping = xgboost.callback.EarlyStopping(
    rounds=early_stopping_rounds,
    min_delta=0.0,
    save_best=True,
    maximize=False,
    metric_name="mlogloss",
)

params = {'objective': 'multi:softmax',
         'num_class': len(set(y_train)),
         'n_estimators': num_round,
         'eta': 0.1,
         'importance_type': 'weight',
         'max_depth': 10,
         'min_child_weight': 3,
         'gamma': 0.1,
         'eval_metric': ['merror', 'mlogloss'],
         'nthread': 4,
         'early_stopping_rounds': early_stopping_rounds,
         'callbacks': [early_stopping]}


clt = XGBClassifier(**params)
clt.fit(x_train,y_train, eval_set=[(x_val, y_val)], )

print('特征重要性：', clt.feature_importances_)
y_pred1 = clt.predict(x_test)
# pmml = PMMLPipeline([("class", clt)])
pmml = make_pmml_pipeline(clt, active_fields = feature, target_fields = ['y'])
y_pred2 = pmml.predict(x_test)

# 转为PMML文件
sklearn2pmml(pmml, "xgboost.pmml", with_repr = True)
# xgboost_to_pmml(pmml, feature, "class", "xgboost.pmml")  # 若使用xgboost_to_pmml转换，可能存在极个别样例在转换前后预测结果不一致
# 加载模型
model = Model.fromFile('xgboost.pmml')
y_pred3 = model.predict(x_test)

print("训练后的模型预测正确率：", metrics.accuracy_score(y_test, y_pred1))
print("转换为pmml模型预测正确率：", metrics.accuracy_score(y_test, y_pred2))
print('加载pmml模型预测正确率：', metrics.accuracy_score(y_test, np.array(y_pred3).argmax(axis=-1)))

#########################################################################################################################



def main():
    pass


if __name__ == "__main__":
    main()
