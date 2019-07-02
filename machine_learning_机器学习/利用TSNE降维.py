#!/usr/bin/python3

from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
iris = load_iris() # 使用sklearn自带的测试文件
iris.data.shape
Out[4]: (150, 4)

# 降到2维
X_tsne = TSNE(n_components=2,learning_rate=100).fit_transform(iris.data)
X_tsne.data.shape
Out[6]: (150, 2)

# 降到3维
X_tsne = TSNE(n_components=3,learning_rate=100).fit_transform(iris.data)
X_tsne.data.shape
Out[8]: (150, 3)
X_tsne.shape
Out[9]: (150, 3)


