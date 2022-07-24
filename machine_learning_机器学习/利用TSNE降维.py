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



# 示例：共5条数据，每个数据的特征均不一样；
origin_data = [[1,3,5], [2, 7], [2, 4, 8], [6, 9], [0, 3, 5, 2]]
# 转为为0,1编码，指定位置置1
encoded_test = np.zeros((len(origin_data),max(sum(origin_data, []))+1 ))
for index, data in enumerate(origin_data):
    for col in data:
        encoded_test[index, col] = 1

embedded_test = TSNE(n_components=3).fit_transform(encoded_test)
print(encoded_test.shape, embedded_test.shape)
# (5, 10) (5, 3)
# encoded_test
# Out[37]:
# array([[0., 1., 0., 1., 0., 1., 0., 0., 0., 0.],
#        [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
#        [0., 0., 1., 0., 1., 0., 0., 0., 1., 0.],
#        [0., 0., 0., 0., 0., 0., 1., 0., 0., 1.],
#        [1., 0., 1., 1., 0., 1., 0., 0., 0., 0.]])
# embedded_test
# Out[38]:
# array([[ -57.4828  ,  -70.48181 ,   29.336174],
#        [-397.9627  ,  200.90654 ,  142.95302 ],
#        [ 276.9897  , -161.48886 ,  295.89194 ],
#        [ 446.2278  ,   76.39137 ,  -81.55921 ],
#        [-307.54205 ,  -57.483055, -333.50143 ]], dtype=float32)