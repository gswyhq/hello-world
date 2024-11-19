#!/usr/bin/env python
# coding=utf-8

# 安装tsfresh：
# # pip安装
# $ pip install -U tsfresh

#########################################################################################################################
from tsfresh import extract_relevant_features
from tsfresh.examples.robot_execution_failures import load_robot_execution_failures

# 加载示例数据
timeseries, y = load_robot_execution_failures()  # 加载样例数据，共88个id，每个id数据的时间维度长度为15,1320行(88*15) 共8列，除去id,及时间维度，还有6个特征维度；
# 自动进行相关特征提取
features = extract_relevant_features(timeseries, y, column_id="id", column_sort="time")
# 提取完特征后,共88行，682个特征列

#########################################################################################################################
# tsfresh支持与scikit-learn 兼容，可以合并到现有的机器学习管道中：

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd

# 下载数据
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures()

#
pipeline = Pipeline([
    ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
    ('classifier', RandomForestClassifier()),
])

df_ts, y = load_robot_execution_failures()
X = pd.DataFrame(index=y.index)

pipeline.set_params(augmenter__timeseries_container=df_ts)
pipeline.fit(X, y)

#########################################################################################################################
# 当然也可以将 序列数据与常规特征数据结合使用
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
import pandas as pd

# 下载数据
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures()

#
pipeline = Pipeline([
    ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
    ('classifier', RandomForestClassifier()),
])

df_ts, y = load_robot_execution_failures()

# 随机删除一些数据，使之每个样本的时间序列长度不一致；即验证不强制要求时间序列长度是一致；
# 计算要保留的样本数量
keep_count = int((len(df_ts) - int(0.05 * len(df_ts))))

# 随机选取并删除5%的数据
random_indexes = df_ts.sample(n=keep_count).index
df_ts2 = df_ts[df_ts.index.isin(random_indexes)]
y2 = y[y.index.isin(random_indexes)]

d2 = np.random.random(size=(y2.shape[0], 3))
X3 = pd.DataFrame(d2, index=y2.index)
X3.columns = ['t1', 't2', 't3']
pipeline.set_params(augmenter__timeseries_container=df_ts2)
classifier = pipeline.fit(X3, y2)

#########################################################################################################################
# 特征筛选
from tsfresh.examples import robot_execution_failures

robot_execution_failures.download_robot_execution_failures()
# 加载数据集
df, y = robot_execution_failures.load_robot_execution_failures()

print(df)
print(y)
# 样例数据对比
df[df.id == 3][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Success example (id 3)', figsize=(12, 6))
df[df.id == 20][['time', 'F_x', 'F_y', 'F_z', 'T_x', 'T_y', 'T_z']].plot(x='time', title='Failure example (id 20)', figsize=(12, 6))

# 自动提取特征：
from tsfresh import extract_features
extracted_features = extract_features(df, column_id="id", column_sort="time")

# 特征自动提取后，特征由之前的6个特征变为了4698个特征：
extracted_features.shape
# Out[74]: (88, 4698)

# 当然，特征提取时候，也可以设置一些参数，如：
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
extraction_settings = ComprehensiveFCParameters()

X = extract_features(df, column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     # impute就是自动移除所有NaN的特征
                     impute_function=impute)

# 特征选择
# 可见进行特征提取之后，数据量多了，因为原始和特征函数有不同的组装方式，使用数据量多了。
# Tsfresh将对每一个特征进行假设检验，以检查它是否与给定的目标相关。
from tsfresh import select_features

X_filtered = select_features(X, y)
X_filtered.shape
# Out[84]: (88, 682)
# 经过特征筛选后，特征数由筛选前的4698，变成了682个特征；


# 同时特征提取、选择、过滤
# 此外，甚至可以使用以下tsfresh.extract_relevant_features()功能同时执行提取、选择( imputing)和过滤 。
from tsfresh import extract_relevant_features

features_filtered_direct = extract_relevant_features(df, y, column_id='id', column_sort='time')
features_filtered_direct.shape
# Out[87]: (88, 682)


#########################################################################################################################
# 训练模型评估模型
# 让我们在过滤后的以及提取的全部特征集上训练一个增强的决策树。

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
import matplotlib.pylab as plt

from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 加载数据集
df, y = robot_execution_failures.load_robot_execution_failures()

# 抽取特征
X = extract_features(df, column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     # impute就是自动移除所有NaN的特征
                     impute_function=impute)

# 特征筛选
X_filtered = select_features(X, y)

# 拆分数据集：
X_full_train, X_full_test, y_train, y_test = train_test_split(X, y, test_size=.4)

# 进行特征选择（也可以直接使用特征选择后的数据而不用到这里再选择）
X_filtered_train, X_filtered_test = X_full_train[X_filtered.columns],X_full_test[X_filtered.columns]

# 没进行选择特征之前 训练评估模型：

classifier_full = DecisionTreeClassifier()
classifier_full.fit(X_full_train, y_train)
print(classification_report(y_test, classifier_full.predict(X_full_test)))

# 进行选择特征之后 训练评估模型：
classifier_filtered = DecisionTreeClassifier()
classifier_filtered.fit(X_filtered_train, y_train)
print(classification_report(y_test, classifier_filtered.predict(X_filtered_test)))


#########################################################################################################################
# 除了直接输入特征X和标签Y进行训练
# 还可以输入索引映射和设置PPL的set_params进行训练
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute

from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures()
df_ts, y = load_robot_execution_failures()


X = pd.DataFrame(index=y.index)

# Split data into train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y)

#构建管道
ppl = Pipeline([
        ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
        ('classifier', RandomForestClassifier())
      ])

# 第一种训练评估方法：一次性声明时间序列容器

#声明时间序列容器
ppl.set_params(augmenter__timeseries_container=df_ts)
ppl.fit(X_train, y_train)
# 测试集评估
y_pred = ppl.predict(X_test)
print(classification_report(y_test, y_pred))

# 第二种训练评估方法：分别声明训练集、测试集的 时间序列容器

#分别获取训练集、测试集的数据索引
df_ts_train = df_ts[df_ts["id"].isin(y_train.index)]
df_ts_test = df_ts[df_ts["id"].isin(y_test.index)]

# 设置 时间序列容器为训练集，并训练模型
ppl.set_params(augmenter__timeseries_container=df_ts_train);
#关键一步：输入训练集的数据索引作为X，输入y标签作为Y进行训练。期间索引会在管道里面找到df_ts_train的数据。
ppl.fit(X_train, y_train);
import pickle
with open("pipeline.pkl", "wb") as f:
    pickle.dump(ppl, f)



# 设置 时间序列容器为测试集，并测试评估模型；【如无模型，先加载模型】
import pickle
with open("pipeline.pkl", "rb") as f:
    ppk = pickle.load(f)
ppl.set_params(augmenter__timeseries_container=df_ts_test);
# 关键一步：输入测试集的数据索引作为X进行预测标签y
y_pred = ppl.predict(X_test)
print(classification_report(y_test, y_pred))

#########################################################################################################################
# Tsfresh提供了三种不同的选项来指定用于函数Tsfresh .extract_features()的时间序列数据格式。（当然，包括所有需要时间序列的实用函数，例如tsfresh.utilities.dataframe_functions.roll_time_series()）

#########################################################################################################################
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import tsfresh
from tsfresh.utilities.dataframe_functions import roll_time_series
from tsfresh.examples import load_robot_execution_failures
from tsfresh.transformers import RelevantFeatureAugmenter
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures
download_robot_execution_failures()
df_ts, y = load_robot_execution_failures()

X = pd.DataFrame(index=y.index)

# 拆分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y)
ppl = Pipeline([
        ('augmenter', RelevantFeatureAugmenter(column_id='id', column_sort='time')),
        ('classifier', RandomForestClassifier())
      ])
ppl.set_params(augmenter__timeseries_container=df_ts)

# 特征提取方法汇总：
# https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html

#########################################################################################################################
# 使用tsfresh select features方法选择相关特征的子集。然而，它只适用于二元分类或回归任务。
# 对于一个6个标签的多分类，因此将选择问题分割成6个二进制的一个-而不是所有的分类问题。对于它们中的每一个，可以做一个二元分类特征选择

relevant_features = set()

for label in y.unique():
    y_train_binary = y_train == label
    X_train_filtered = select_features(X_train, y_train_binary)
    print("Number of relevant features for class {}: {}/{}".format(label, X_train_filtered.shape[1], X_train.shape[1]))
    relevant_features = relevant_features.union(set(X_train_filtered.columns))

#########################################################################################################################
# 改进的多类特征选择
# 为了通过筛选过程，可以指定一个特征应该作为相关预测器的类的数量。这与将multiclass参数设置为True和将n_significant设置为所需的类数量一样简单。将尝试与5类相关的要求。可以看到相关功能的数量比之前的实现要少。

X_train_filtered_multi = select_features(X_train, y_train, multiclass=True, n_significant=5)
print(X_train_filtered_multi.shape)


classifier_selected_multi = DecisionTreeClassifier()
classifier_selected_multi.fit(X_train_filtered_multi, y_train)
X_test_filtered_multi = X_test[X_train_filtered_multi.columns]
print(classification_report(y_test, classifier_selected_multi.predict(X_test_filtered_multi)))

#########################################################################################################################
from tsfresh.examples.har_dataset import download_har_dataset, load_har_dataset, load_har_classes

# fetch dataset from uci
download_har_dataset()

df = load_har_dataset()
y = load_har_classes()
df.head()

df["id"] = df.index
df = df.melt(id_vars="id", var_name="time").sort_values(["id", "time"]).reset_index(drop=True)

df.head()

# 可视化：

plt.title('accelerometer reading')
plt.plot(df[df["id"] == 0].set_index("time").value)
plt.show()

# 只使用前700个id来操作，加快处理速度
X = extract_features(df[df["id"] < 700], column_id="id", column_sort="time", impute_function=impute)

X.head()

# 拆分数据集

X_train, X_test, y_train, y_test = train_test_split(X, y[:700], test_size=.2)
classifier_full = DecisionTreeClassifier()
classifier_full.fit(X_train, y_train)
print(classification_report(y_test, classifier_full.predict(X_test)))

#########################################################################################################################
# 资料来源：https://blog.csdn.net/qq_42658739/article/details/122358303#%E4%BA%86%E8%A7%A3%E3%80%81%E5%AE%89%E8%A3%85tsfresh

def main():
    pass


if __name__ == "__main__":
    main()
