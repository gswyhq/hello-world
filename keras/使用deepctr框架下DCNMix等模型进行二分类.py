#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 链接：https://zhuanlan.zhihu.com/p/377900375
# 更多示例代码见：https://deepctr-doc.readthedocs.io/en/latest/Examples.html
# DeepCTR框架
# 主要是对目前的一些基于深度学习的点击率预测算法进行了实现，如PNN,WDL,DeepFM,MLR,DeepCross,AFM,NFM,DIN,DIEN,xDeepFM,AutoInt等,并且对外提供了一致的调用接口。

import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from deepctr.models import DeepFM, DCNMix
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names

# 第一步：读取数据
data = pd.read_csv('./examples/criteo_sample.txt') #读取数据, 数据来源：https://github.com/shenweichen/DeepCTR.git

sparse_features = ['C' + str(i) for i in range(1, 27)] # 字符型，稀疏特征一般是类别特征
dense_features = ['I'+str(i) for i in range(1, 14)]  # 数值型
# # 注意：历史行为序列特征名称必须以“hist_”开头。故而，其他非历史行为序列，最好不要以hist_开头


data[sparse_features] = data[sparse_features].fillna('-1', ) # fillna是对空值的填充处理函数
data[dense_features] = data[dense_features].fillna(0,)
target = ['label']

# 第二步 数据预处理
# 神经网络输入的都是数字，因此需要对类别特征编码，如标签编码或者哈希编码：
# 标签编码 下面的代码是遍历每个类别特征，并对每个类别特征编码
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
# 假设数据里有两个类别特征，分别是性别和种族：从编码后不难可以看出标签编码的原理，即用0~len(#unique)-1中的数字去代替特征中的各个类别。
# 比如性别特征只有两个不同的类，Male和Female，那么标签编码就是用数字1表示Male，0表示Female。

# 哈希编码哈希编码可以在训练前完成，比如下面的例子；也可以在训练过程中调用函数SparseFeat或者VarlenSparseFeat 时令use_hase=True。
# 哈希编码需要注意的是编码后特征数的选择，太小容易引起冲突，太大则容易导致编码后特征维度过大。
from sklearn.feature_extraction import FeatureHasher
import pandas as pd

# h = FeatureHasher(n_features=5, input_type='string')
# f = h.transform(data[feat])
# f.toarray()
#数值特征我们往往需要对其做归一化处理，消除量纲的影响：
mms = MinMaxScaler(feature_range=(0,1))
data[dense_features] = mms.fit_transform(data[dense_features])

# 第三步 生成特征列将稀疏特征进一步通过嵌入技术将其转成稠密向量，将稠密特征拼接到全连接神经网络的输入向量：
# 标签编码
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]
# 或者哈希编码
fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=len(data[feat].unique())+1,embedding_dim=4, use_hash=True, dtype='int64')
                              for feat in sparse_features] + [DenseFeat(feat, 1, )
                          for feat in dense_features]
#生成特征列
dnn_feature_columns = fixlen_feature_columns  #用做dnn的输入向量
linear_feature_columns = fixlen_feature_columns #用在线性模型的输入特征

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns) #获取所有特征的名字

# 第四步 生成训练样本和模型
train, test = train_test_split(data, test_size=0.2) #按照8:2的比例划分数据集为训练集合测试集

train_model_input = {name:train[name].values for name in feature_names}
test_model_input = {name:test[name].values for name in feature_names}


# model = DeepFM(linear_feature_columns,dnn_feature_columns,task='binary') #调用deepctr库中的DeepFM模型，执行二分类任务
model = DCNMix(linear_feature_columns,dnn_feature_columns,task='binary') #调用deepctr库中的DCNMix模型，执行二分类任务
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], ) #设置优化器，损失函数类型和评估指标

history = model.fit(train_model_input, train[target].values,
                    batch_size=32, epochs=2, verbose=1, validation_split=0.2, ) #fit数据
pred_ans = model.predict(test_model_input, batch_size=32) # 预测


#####################################################示例二#####################################################################

import numpy as np
import pandas as pd
from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_auc_score

##  https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
train_file = 'dataset/adult.data'

# Define the column names for the data sets.
COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
  "marital_status", "occupation", "relationship", "race", "gender",
  "capital_gain", "capital_loss", "hours_per_week", "native_country", "income_bracket"]
target = 'label'
sparse_features= ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]
dense_features = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

data = pd.read_csv(train_file, names=COLUMNS, skipinitialspace=True)

data[target] = (data['income_bracket'].apply(lambda x: '>50K' in x)).astype(int)
data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])

fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].max() + 1,embedding_dim=4)
                       for i,feat in enumerate(sparse_features)] + [DenseFeat(feat, 1,)
                      for feat in dense_features]

print('==============================================fixlen_feature_columns============================================\n')
print(fixlen_feature_columns)
print('\n\n')
dnn_feature_columns = fixlen_feature_columns
linear_feature_columns = fixlen_feature_columns

feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)
print('===========================================feature_names===========================================================\n')
print(feature_names)
print('\n\n')
train, test = train_test_split(data, test_size=0.2, random_state=2020)
train_model_input = {name:train[name] for name in feature_names}
test_model_input = {name:test[name] for name in feature_names}

# 4.Define Model,train,predict and evaluate
model = DeepFM(linear_feature_columns, dnn_feature_columns, task='binary')
model.compile("adam", "binary_crossentropy",
              metrics=['binary_crossentropy'], )

history = model.fit(train_model_input, train[target].values,
                    batch_size=256, epochs=10, verbose=0, validation_split=0.2, )
pred_ans = model.predict(test_model_input, batch_size=256)
print('============================================test LogLoss & test AUC=============================================\n')
print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))

#####################################################示例三#####################################################################

# 链接：https://zhuanlan.zhihu.com/p/133138798

# 下面使用一个简单的数据集，实践一下YouTubeNet召回模型。
# 该模型的实现主要参考：python软件的DeepCtr和DeepMatch模块。
import pandas as pd
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from deepctr.inputs import SparseFeat, VarLenSparseFeat
from deepmatch.models import YoutubeDNN
from deepmatch.utils import sampledsoftmaxloss
from keras.utils import plot_model

# 1. 数据处理
# 加载数据samples.txt
# 数据可以从百度网盘下载：链接: https://pan.baidu.com/s/1Gy0TIFIlmh7W6lexVKum9w 提取码: p8fn
samples_data = pd.read_csv("samples.txt", sep="\t", header = None)
samples_data.columns = ["user_id", "movie_id", "gender", "age", "hist_movie_id", "hist_len", "label"]

samples_data.head()
# 	user_id	movie_id	gender	age	hist_movie_id	hist_len	label
# 0	1	112	1	1	186,0,0,0,0...	1	1
# 1	1	84	1	1	112,186,0,0...	2	1
# 2	1	52	1	1	84,112,186,0...	3	1

# 本示例中包含：6个特征。
# user端特征有5个，分别为["user_id", "gender", "age", "hist_movie_id", "hist_len"]；
# user_id 为 用户ID特征，离散特征，从1-3表示；
# gender 为 用户性别特征，离散特征，从1-2表示；
# age 为 用户年龄特征，离散特征，从1-3表示；
# hist_movie_id 为 用户观看的movie序列特征，根据观看的时间倒排，即最新观看的movieID排在前面；该字段有多个值通过逗号连接组成；
# hist_len 为 用户观看的movie序列长度特征，连续特征；
# movie端特征有1个，为 ["movie_id"]；movie_id 为 movieID特征，离散特征，从1-208表示；

# 分割数据为训练集和验证集
samples_data = shuffle(samples_data)

X = samples_data[["user_id", "movie_id", "gender", "age", "hist_movie_id", "hist_len"]]
y = samples_data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2020)

# 转换数据存储格式
X_train = {"user_id": np.array(X_train["user_id"]), \
              "gender": np.array(X_train["gender"]), \
              "age": np.array(X_train["age"]), \
              "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X_train["hist_movie_id"]]), \
              "hist_len": np.array(X_train["hist_len"]), \
              "movie_id": np.array(X_train["movie_id"])}

y_train = np.array(y_train)

X_test = {"user_id": np.array(X_test["user_id"]), \
              "gender": np.array(X_test["gender"]), \
              "age": np.array(X_test["age"]), \
              "hist_movie_id": np.array([[int(i) for i in l.split(',')] for l in X_test["hist_movie_id"]]), \
              "hist_len": np.array(X_test["hist_len"]), \
              "movie_id": np.array(X_test["movie_id"])}

y_test = np.array(y_test)

# 2. 构建模型
# 统计每个离散特征的词频量，构造特征参数
embedding_dim = 16
SEQ_LEN = 50

user_feature_columns = [SparseFeat('user_id', max(X['user_id'])+1, embedding_dim),
                        SparseFeat("gender", max(X['gender'])+1, embedding_dim),
                        SparseFeat("age", max(X['age'])+1, embedding_dim),
                        VarLenSparseFeat(SparseFeat('hist_movie_id', max(X['movie_id'])+1, embedding_dim,
                                                    embedding_name="movie_id"), SEQ_LEN, 'mean', 'hist_len'),
                        ]

item_feature_columns = [SparseFeat('movie_id', max(X['movie_id'])+1, embedding_dim)]

# 构建模型及训练模型
model = YoutubeDNN(user_feature_columns, item_feature_columns, num_sampled=5, user_dnn_hidden_units=(64, 16))
model.compile(optimizer="adam", loss=sampledsoftmaxloss)

history = model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test))

##########################################################################################################################

# ~/deepctr$ tree
# .
# +--- contrib
# |   +--- rnn.py
# |   +--- rnn_v2.py
# |   +--- utils.py
# |   +--- __init__.py
# +--- estimator
# |   +--- feature_column.py
# |   +--- inputs.py
# |   +--- models
# |   |   +--- afm.py
# |   |   +--- autoint.py
# |   |   +--- ccpm.py
# |   |   +--- dcn.py
# |   |   +--- deepfefm.py
# |   |   +--- deepfm.py
# |   |   +--- fibinet.py
# |   |   +--- fnn.py
# |   |   +--- fwfm.py
# |   |   +--- nfm.py
# |   |   +--- pnn.py
# |   |   +--- wdl.py
# |   |   +--- xdeepfm.py
# |   |   +--- __init__.py
# |   +--- utils.py
# |   +--- __init__.py
# +--- feature_column.py
# +--- inputs.py
# +--- layers
# |   +--- activation.py
# |   +--- core.py
# |   +--- interaction.py
# |   +--- normalization.py
# |   +--- sequence.py
# |   +--- utils.py
# |   +--- __init__.py
# +--- models
# |   +--- afm.py
# |   +--- autoint.py
# |   +--- ccpm.py
# |   +--- dcn.py
# |   +--- dcnmix.py
# |   +--- deepfefm.py
# |   +--- deepfm.py
# |   +--- difm.py
# |   +--- fgcnn.py
# |   +--- fibinet.py
# |   +--- flen.py
# |   +--- fnn.py
# |   +--- fwfm.py
# |   +--- ifm.py
# |   +--- mlr.py
# |   +--- multitask
# |   |   +--- esmm.py
# |   |   +--- mmoe.py
# |   |   +--- ple.py
# |   |   +--- sharedbottom.py
# |   |   +--- __init__.py
# |   +--- nfm.py
# |   +--- onn.py
# |   +--- pnn.py
# |   +--- sequence
# |   |   +--- bst.py
# |   |   +--- dien.py
# |   |   +--- din.py
# |   |   +--- dsin.py
# |   |   +--- __init__.py
# |   +--- wdl.py
# |   +--- xdeepfm.py
# |   +--- __init__.py
# +--- utils.py
# +--- __init__.py

# feature_column.py和inputs.py用于构造特征列和处理输入；
# models模块则包含了各个CTR算法，比如FM、DFM、DIN等，我们可以直接调用这些方法用在具体任务上
# feature_column.py中的类SparseFeat、DenseFeat、VarLenSparseFeat 就是分别处理类别特征、数值特征和变长序列特征
# VarLenSparseFeat,可以出来变长的类别特征，而变长的浮点型特征用DenseFeat处理就好，只需要设置好对应的维度即好(一般是＞1)
# DenseFeat(name, dimension, dtype)、参数含义
# name：特征名称
# dimension：密集特征向量的维度
# dtype	默认float32。dtype of input tensor(张量)

# VarLenSparseFeat(sparsefeat, maxlen, combiner, length_name)
# 参数	含义
# sparsefeat	一个系数特征的实例
# maxlen	在所有的样本当中，此特征的最大长度
# combiner	池化方法，可以是 sum、mean 、max
# length_name	特征长度名称，如果None，用 0 填充

# feature_columns.py还包含了四个函数：
# def get_feature_names作用：获取所有特征列的名字，以列表形式返回
# def build_input_features作用：为所有的特征列构造keras tensor，结果以OrderDict形式返回
# def get_linear_logit作用：获取linear_logit（线性变换）的结果
# def input_from_feature_columns：为所有特征列创建嵌入矩阵，并分别返回包含SparseFeat和VarLenSparseFeat的嵌入矩阵的字典，以及包含DenseFeat的数值特征的字典
# 具体实现是通过调用inputs中的create_embedding_matrix、embedding_lookup、varlen_embedding_lookup等函数完成

# 链接：https://zhuanlan.zhihu.com/p/377916768
# 输入模块inputs.py
# SparseFeat和VarLenSparseFeat对象需要创建嵌入矩阵，嵌入矩阵的构造和查表等操作都是通过inputs.py模块实现的，该模块包含9个方法：
# 1. get_inputs_listfilter函数过滤输入中的空值map函数是取每个元素x的valuechain构建了一个迭代器，循环处理输入中的每条样本最后返回一个list
# 作用：过滤输入中的空值并返回列表形式的输入

# 2. create_embedding_dict
# 作用：为每个稀疏特征创建可训练的嵌入矩阵，使用字典存储所有特征列的嵌入矩阵，并返回该字典

# 3.get_embedding_vec_list
# 作用：从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以列表形式返回查询结果
# 关键参数：
#     embedding_dict：type：dict；存储着所有特征列的嵌入矩阵的字典
#     input_dict：type：dict；存储着特征列和对应的嵌入矩阵索引的字典，在没有使用hash查询时使用
#     sparse_feature_columns：type：list；所有稀疏特征列
#     return_feat_list:需要查询的特征列，默认为空，为空则返回所有稀疏特征列的嵌入矩阵，不为空则仅返回该元组中的特征列的嵌入矩阵

# 4.create_embedding_matrix
# 作用：从所有特征列中筛选出SparseFeat和VarLenSparseFeat，然后调用函数create_embedding_dict为筛选的特征列创建嵌入矩阵
#
# 5. embedding_lookup
# 作用：从所有稀疏特征列中查询指定稀疏特征列(参数return_feat_list）的嵌入矩阵，以字典形式返回查询结果
# 参数：
#     sparse_embedding_dict：存储稀疏特征列的嵌入矩阵的字典
#     sparse_input_dict：存储稀疏特征列的名字和索引的字典
#     sparse_feature_columns：稀疏特征列列表，元素为SparseFeat
#     return_feat_list：需要查询的稀疏特征列，如果元组为空，默认返回所有特征列的嵌入矩阵
#     mask_feat_list：用于哈希查询
#     to_list：是否以列表形式返回查询结果，默认是False

# 6.varlen_embedding_lookup
# 作用：获取varlen_sparse_feature_columns的嵌入矩阵

# 7.get_varlen_pooling_list
# 作用：获取varlen_sparse_feature_columns池化后的嵌入向量

# 8.get_dense_input
# 作用：从所有特征列中选出DenseFeat，并以列表形式返回结果

# 9. def mergeDict
#     作用：将a、b两个字典合并


def main():
    pass


if __name__ == '__main__':
    main()

