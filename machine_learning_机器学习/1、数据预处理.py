#Day 1: Data Prepocessing

#Step 1: Importing the libraries
import numpy as np
import pandas as pd

#第2步：导入数据集
# https://github.com/Avik-Jain/100-Days-Of-ML-Code/blob/master/datasets/Data.csv
dataset = pd.read_csv('/home/gswyhq/github_projects/100-Days-Of-ML-Code/datasets/Data.csv')
X = dataset.iloc[ : , :-1].values
Y = dataset.iloc[ : , 3].values
print("第2步：导入数据集")
print("X")
print(X)
print("Y")
print(Y)

#第3步：处理丢失数据
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(X[ : , 1:3])
X[ : , 1:3] = imputer.transform(X[ : , 1:3])
print("---------------------")
print("第3步：处理丢失数据")
print("step2")
print("X")
print(X)

#第4步：解析分类数据
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[ : , 0] = labelencoder_X.fit_transform(X[ : , 0])
#创建虚拟变量
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y =  labelencoder_Y.fit_transform(Y)
print("---------------------")
print("第4步：解析分类数据")
print("X")
print(X)
print("Y")
print(Y)

#第5步：拆分数据集为训练集合和测试集合
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X , Y , test_size = 0.2, random_state = 0)
print("---------------------")
print("第5步：拆分数据集为训练集合和测试集合")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("Y_train")
print(Y_train)
print("Y_test")
print(Y_test)

#第6步：特征量化
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
print("---------------------")
print("第6步：特征量化")
print("X_train")
print(X_train)
print("X_test")
print(X_test)
