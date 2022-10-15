#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 基于树模型的特征排序

import pandas as pd
import matplotlib.pyplot as plt

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
from lightgbm import plot_importance
from lightgbm import LGBMClassifier
from sklearn.datasets import load_iris

x, y = load_iris(return_X_y=True)
model = LGBMClassifier()
model.fit(x, y, feature_name=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])

plot_importance(model,  max_num_features=40, figsize=(16,10),importance_type='split', title='特征在模型中使用的次数')
plt.show()

plot_importance(model,  max_num_features=40, figsize=(16,10),importance_type='gain', title='特征的分割的总增益')
plt.show()


feature_importance = pd.DataFrame({
        'feature': model.booster_.feature_name(),
        'gain': model.booster_.feature_importance('gain'), # 包含使用该特征的分割的总增益
        'split': model.booster_.feature_importance('split')  # 该特征在模型中使用的次数。
    }).sort_values('gain',ascending=False)

feature_importance.head()

def main():
    pass


if __name__ == '__main__':
    main()