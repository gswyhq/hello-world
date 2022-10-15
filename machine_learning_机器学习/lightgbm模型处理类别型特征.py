
类别型特征编码由于是字符串类型，所以一般需要经过编码处理转换成数值型。
本文主要想说的是直接将字符串值传到lightgbm中训练。注意：xgboost模型也需要提前one-hot编码转换才能入模。

from random import randint
import lightgbm
import pandas as pd
import numpy as np

a = [i for i in range(1000)]
b = ["tag","bga","efd","rfh","esg","tyh"]
c = [b[randint(0,5)] for i in range(1000)]
d = [randint(0,1) for i in range(1000)]
tmp = []
for i in range(1000):
    tmp.append([a[i],c[i],d[i]])
df = pd.DataFrame(tmp,columns=["a","b","label"])   # 造数据


import lightgbm

df["b"] = df["b"].astype('category')   # 必须有，不然报错
cf = lightgbm.LGBMClassifier(max_depth=3)
cf.fit(df[["a","b"]],df["label"],categorical_feature="b")  # 记得加上这个参数

from sklearn.metrics import accuracy_score
print(accuracy_score(df["label"].values, cf.predict(df[["a","b"]])))


XgBoost和Random Forest，不能直接处理categorical feature，必须先编码成为numerical feature。
lightgbm和CatBoost，可以直接处理categorical feature。
lightgbm： 需要先做label encoding。用特定算法（On Grouping for Maximum Homogeneity）找到optimal split，效果优于ONE。
也可以选择采用one-hot encoding。https://lightgbm.readthedocs.io/en/latest/Features.html?highlight=categorical#optimal-split-for-categorical-features

CatBoost： 不需要先做label encoding。可以选择采用one-hot encoding，target encoding (with regularization)。
https://catboost.ai/en/docs/concepts/algorithm-main-stages_cat-to-numberic

category_encoders(https://github.com/scikit-learn-contrib/category_encoders),是一个编码工具集，里头提供了多种编码方法；
类别特征转换为数值无监督(Unsupervised)方法:
Backward Difference Contrast
BaseN
Binary
Count
Hashing
Helmert Contrast
Ordinal
One-Hot
Polynomial Contrast
Sum Contrast

类别特征转换为数值监督(Supervised)方法:
CatBoost
Generalized Linear Mixed Model
James-Stein Estimator
LeaveOneOut
M-estimator
Target Encoding
Weight of Evidence
Quantile Encoder
Summary Encoder



