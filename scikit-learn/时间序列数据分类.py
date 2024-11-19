#!/usr/bin/env python
# coding=utf-8

#####################################示例1： 单个时间序列进行分类#####################################################################################

from sktime.classification.kernel_based import RocketClassifier
from sktime.datasets import load_unit_test
from sklearn.metrics import accuracy_score

X_train, y_train = load_unit_test(split="train", return_X_y=True)
X_test, y_test = load_unit_test(split="test", return_X_y=True)
# X_train是一个多层级索引数据，第一层级索引维度为样本维度，即多少个样本；第二层级索引维度是为时间维度；
print("查看样例数据：{}".format((X_train.shape, X_train.iloc[0, 0].shape )))
X_train.iloc[0, 0].plot()

clf = RocketClassifier(num_kernels=500, rocket_transform='minirocket')
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("预测acc: %s".format(accuracy_score(y_test, y_pred)))

# 资料来源：https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.kernel_based.RocketClassifier.html

####################################示例2： 单个时间序列进行分类######################################################################################

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_arrow_head  # univariate dataset
from sktime.datasets import load_basic_motions  # multivariate dataset
from sktime.datasets import (
    load_japanese_vowels,  # multivariate dataset with unequal length
)
from sktime.transformations.panel.rocket import (
    MiniRocket,
    MiniRocketMultivariate,
    MiniRocketMultivariateVariable,
)

X_train, y_train = load_arrow_head(split="train", return_X_y=True)
# 可视化时间维度
X_train.iloc[0, 0].plot()

minirocket = MiniRocket()  # by default, MiniRocket uses ~10_000 kernels
minirocket.fit(X_train)
X_train_transform = minirocket.transform(X_train)
# test shape of transformed training data -> (n_instances, 9_996)
X_train_transform.shape

scaler = StandardScaler(with_mean=False)
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))

X_train_scaled_transform = scaler.fit_transform(X_train_transform)
classifier.fit(X_train_scaled_transform, y_train)

X_test, y_test = load_arrow_head(split="test", return_X_y=True)
X_test_transform = minirocket.transform(X_test)

X_test_scaled_transform = scaler.transform(X_test_transform)
classifier.score(X_test_scaled_transform, y_test)

# https://sktime-backup.readthedocs.io/en/stable/examples/transformation/minirocket.html

####################################示例3： 单个时间序列进行分类，通过管道，无需数据转换####################################################################################################################
# 我们可以在管道中将 MiniRocket 与 RidgeClassifierCV（或其他分类器）一起使用。然后，我们可以像使用独立分类器一样使用管道，只需调用一次即可拟合，而无需单独转换数据等。

minirocket_pipeline = make_pipeline(
    MiniRocket(),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
X_train, y_train = load_arrow_head(split="train")

minirocket_pipeline.fit(X_train, y_train)
X_test, y_test = load_arrow_head(split="test")

minirocket_pipeline.score(X_test, y_test)

# https://sktime-backup.readthedocs.io/en/stable/examples/transformation/minirocket.html

######################################示例4： 多个时间序列进行分类##########################################################################################

X_train, y_train = load_basic_motions(split="train", return_X_y=True)
# 有6个时间维度， X_train.iloc[0].shape
minirocket_multi = MiniRocketMultivariate()
minirocket_multi.fit(X_train)
X_train_transform = minirocket_multi.transform(X_train)

scaler = StandardScaler(with_mean=False)
X_train_scaled_transform = scaler.fit_transform(X_train_transform)

classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
classifier.fit(X_train_scaled_transform, y_train)

X_test, y_test = load_basic_motions(split="test", return_X_y=True)
X_test_transform = minirocket_multi.transform(X_test)

X_test_scaled_transform = scaler.transform(X_test_transform)
classifier.score(X_test_scaled_transform, y_test)


# https://sktime-backup.readthedocs.io/en/stable/examples/transformation/minirocket.html
########################################示例6： 变长时间序列进行分类########################################################################################
# 具有 MiniRocketMultivariateVariable 和不等长时间序列数据的管道示例
# 使用 MiniRocket 的扩展版本，MiniRocketMultivariateVariable 用于可变/不等长时间序列数据。
# 将其与 RidgeClassifierCV 组合在一个 sklearn 管道中。然后，我们可以像使用独立分类器一样使用管道，只需调用一次即可拟合，而无需单独转换数据等。

X_train_jv, y_train_jv = load_japanese_vowels(split="train", return_X_y=True)
# lets visualize the first three voice recordings with dimension 0-11

print("number of samples training: ", X_train_jv.shape[0])
print("series length of recoding 0, dimension 5: ", X_train_jv.iloc[0, 5].shape)
print("series length of recoding 1, dimension 5: ", X_train_jv.iloc[1, 0].shape)

X_train_jv.head(3)
# additional visualizations
number_example = 153
for i in range(12):
    X_train_jv.loc[number_example, f"dim_{i}"].plot()
print("Speaker ID: ", y_train_jv[number_example])

minirocket_mv_var_pipeline = make_pipeline(
    MiniRocketMultivariateVariable(
        pad_value_short_series=-10.0, random_state=42, max_dilations_per_kernel=16
    ),
    StandardScaler(with_mean=False),
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
)
print(minirocket_mv_var_pipeline)

minirocket_mv_var_pipeline.fit(X_train_jv, y_train_jv)

X_test_jv, y_test_jv = load_japanese_vowels(split="test", return_X_y=True)

minirocket_mv_var_pipeline.score(X_test_jv, y_test_jv)

################################示例7，针对时间序列数据，标准分类器与时间序列分类器效果对比 ########################################################################################################################
# 对于单变量、等长分类问题，可以使用标准的 sklearn 分类器，但它不太可能像定制的时间序列分类器那样好，因为监督表格分类器会忽略变量中的序列信息。
# 要直接应用 sklearn 分类器，需要将数据重塑为与 sklearn 兼容的 2D 数据格式之一。sklearn 不能直接用于多变量或不等长数据集，除非选择如何将数据插入到 2D 结构中。

# 标准的sklearn分类器
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

classifier = RandomForestClassifier(n_estimators=100)
arrow_train_X_2d, arrow_train_y_2d = load_arrow_head(
    split="train", return_type="numpy2d"
)
arrow_test_X_2d, arrow_test_y_2d = load_arrow_head(split="test", return_type="numpy2d")
classifier.fit(arrow_train_X_2d, arrow_train_y_2d)
y_pred = classifier.predict(arrow_test_X_2d)

accuracy_score(arrow_test_y_2d, y_pred)

# 使用时间序列分类器，效果有明显提升
from sktime.classification.kernel_based import RocketClassifier
arrow_train_X, arrow_train_y = load_arrow_head(split="train", return_type="numpy2d")
arrow_test_X, arrow_test_y = load_arrow_head(split="test", return_type="numpy2d")
print(arrow_train_X.shape, arrow_train_y.shape, arrow_test_X.shape, arrow_test_y.shape)
rocket = RocketClassifier(num_kernels=2000, rocket_transform='minirocket')
rocket.fit(arrow_train_X, arrow_train_y)
y_pred = rocket.predict(arrow_test_X)

accuracy_score(arrow_test_y, y_pred)

# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

###############################示例8，多分类器分类 #########################################################################################################################
# "basic motions" dataset
motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(
    split="train", return_type="numpy3d"
)
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")
print(type(motions_train_X))
print(
    motions_train_X.shape,
    motions_train_y.shape,
    motions_test_X.shape,
    motions_test_y.shape,
)
plt.title(" First and second dimensions of the first instance in BasicMotions data")
plt.plot(motions_train_X[0][0])
plt.plot(motions_train_X[0][1])

from sktime.classification.kernel_based import RocketClassifier

rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

from sktime.classification.hybrid import HIVECOTEV2

HIVECOTEV2(time_limit_in_minutes=0.2)
hc2.fit(motions_train_X, motions_train_y)
y_pred = hc2.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)

# 通过 ColumnConcatenator 将时间序列列串联成一个长时间序列列，并将分类器应用于串联数据，
from sktime.classification.interval_based import DrCIF
from sktime.transformations.panel.compose import ColumnConcatenator

clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
clf.fit(motions_train_X, motions_train_y)
y_pred = clf.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


# 通过 ColumnEnsembleClassifier 进行维度组装，其中为时间序列的每个时间序列列/维度拟合一个分类器，并通过投票方案组合它们的预测。
from sktime.classification.compose import ColumnEnsembleClassifier
from sktime.classification.interval_based import DrCIF
from sktime.classification.kernel_based import RocketClassifier

col = ColumnEnsembleClassifier(
    estimators=[
        ("DrCIF0", DrCIF(n_estimators=10, n_intervals=5), [0]),
        ("ROCKET3", RocketClassifier(num_kernels=1000), [3]),
    ]
)

col.fit(motions_train_X, motions_train_y)
y_pred = col.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)


# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

########################### 示例9， 不定长数据分类 ##############################################################################################
from sktime.registry import all_estimators
# 查看所有支持不定长数据的分类器
all_estimators(
    filter_tags={"capability:unequal_length": True}, estimator_types="classifier"
)

from sktime.classification.feature_based import RandomIntervalClassifier
from sktime.transformations.panel.padder import PaddingTransformer

plaid_X, plaid_y = load_plaid()
plaid_train_X, plaid_train_y = load_plaid(split="train")
plaid_test_X, plaid_test_y = load_plaid(split="test")
print(type(plaid_X))

plt.title(" Four instances of PLAID dataset")
plt.plot(plaid_X.iloc[0, 0])
plt.plot(plaid_X.iloc[1, 0])
plt.plot(plaid_X.iloc[2, 0])
plt.plot(plaid_X.iloc[3, 0])
plt.show()

padded_clf = PaddingTransformer() * RandomIntervalClassifier(n_intervals=5)
padded_clf.fit(plaid_train_X, plaid_test_y)
y_pred = padded_clf.predict(plaid_test_X)

accuracy_score(plaid_test_y, y_pred)

# https://www.sktime.net/en/v0.19.2/examples/02_classification.html

#########################################################################################################################


def main():
    pass


if __name__ == "__main__":
    main()
