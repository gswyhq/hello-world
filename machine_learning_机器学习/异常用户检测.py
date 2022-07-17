#!/usr/bin/env python
# coding: utf-8


# import numpy and pandas
import numpy as np
import pandas as pd

# to plot within notebook
import seaborn as sns
import matplotlib.pyplot as plt

import os
# machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import homogeneity_score
from sklearn.metrics import silhouette_score
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

import sklearn
print(sklearn.__version__)

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# In[2]:

USERNAME = os.getenv('USERNAME')
# load the dataset, https://github.com/limchiahooi/fraud-detection
df = pd.read_csv(rf"D:\Users\{USERNAME}\github_project\fraud-detection\chapter_1\creditcard_sampledata_3.csv")

df.head()


# In[3]:


# explore the features available in the dataframe
print(df.info())


# In[4]:


# 汇总统计(summary statistics)
df.describe()


# In[5]:


# 检查缺失值(check for missing values)
df.isnull().sum()


# In[6]:


# 统计欺诈发生及没发生的用户数量；count the occurrences of fraud and no fraud cases
df["Class"].value_counts()


# In[7]:


# 统计欺诈发生及没发生的用户数量，按百分比显示；ratio of fraud and no fraud cases
df["Class"].value_counts(normalize=True)


# 传统方法
# 我们将首先以“传统方法”在信用卡数据集中查找欺诈案例。
# 首先，我们将使用通用统计数据定义阈值，以区分欺诈和非欺诈。
# 然后，在特征上使用这些阈值来检测欺诈。

# 使用 groupby() 对 Class 上的 df 进行分组并获得特征的均值。 get the mean for each group
df.groupby("Class").mean()


# In[9]:


# 创建小于 -3 的条件 V1 和小于 -5 的 V3 作为标记欺诈案例的条件。
# implement a rule for stating which cases are flagged as fraud
df["flag_as_fraud"] = np.where(np.logical_and(df["V1"] < -3, df["V3"] < -5), 1, 0)
df["flag_as_fraud"].head(10)


# In[10]:


# 作为性能的衡量标准，使用 pandas 的交叉表函数将标记的欺诈案例与实际欺诈案例进行比较。
# create a crosstab of flagged fraud cases versus the actual fraud cases
print(pd.crosstab(df.Class, df.flag_as_fraud, rownames=["Actual Fraud"], colnames=["Flagged Fraud"]))
# Flagged Fraud     0   1
# Actual Fraud
# 0              4984  16
# 1                28  22

# 使用此规则，我们检测到 50 个欺诈案例中的 22 个，但无法检测到其他 28 个，并得到 16 个误报。


# 监督机器学习
# 机器学习模型来捕捉欺诈
# 当我们有标记数据时，我们可以使用监督机器学习技术来标记欺诈交易。 我们可以使用分类器，调整它们并比较它们以找到最有效的欺诈检测模型。

# create input and target variable
X = df.drop(["Unnamed: 0", "Class", "flag_as_fraud"], axis=1)
y = df["Class"]

# create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# 通过逻辑回归模型进行拟合fit a logistic regression model to the data
model = LogisticRegression(solver="liblinear")
model.fit(X_train, y_train)


# In[14]:


# 获取模型的预测结果，obtain model predictions
predicted = model.predict(X_test)


# In[15]:


# 模型的预测概率 predict probabilities
probs = model.predict_proba(X_test)


# In[16]:


# print the accuracy score
print("Accuracy Score: {}".format(accuracy_score(y_test, predicted)))


# In[17]:


# print the ROC score
print("ROC score: {}\n".format(roc_auc_score(y_test, probs[:,1])))

# print the classifcation report and confusion matrix
print("分类报告(Classification report):\n{}\n".format(classification_report(y_test, predicted)))

# print confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print("混淆矩阵(Confusion matrix):\n{}\n".format(conf_mat))


# 混淆矩阵(Confusion matrix):
# [[1504    1]
#  [   2    8]]


# 混淆矩阵可视化 plot the confusion matrix
sns.heatmap(conf_mat, annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("cm.png", bbox_inches="tight")
plt.show()

# 如上所示，我们设法在 10 个欺诈案例中捕获了 8 个，只有 1 个误报和 2 个漏报，这对于我们的第一个机器学习模型来说还不错。

show_roc_curve(y_test, probs[:,1])  # ROC曲线
show_roc_pr_curve(y_test, probs[:,1])  # 展示ROC曲线，召回率-准确率曲线；

def show_roc_curve(y_true, probas_pred):
    """展示ROC曲线"""
    fpr, tpr, thresholds = roc_curve(y_true, probas_pred)
    fig = plt.figure(figsize=(12, 8))

    # plot Random Forest ROC
    plt.plot(fpr, tpr,
             label="随机森林Random Forest (AUC = {:1.4f})".format(roc_auc_score(y_true, probas_pred)))

    # plot Baseline ROC
    plt.plot([0, 1], [0, 1], label="基线Baseline (AUC = 0.5000)", linestyle="--")

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("假阳性率(False Positive Rate)", fontsize=14)
    plt.ylabel("真阳性率(True Positive Rate)", fontsize=14)
    plt.title("ROC曲线(Curve)", fontsize=16)
    plt.legend(loc="lower right")
    # plt.savefig("roc.png", bbox_inches="tight")
    plt.show()

def show_roc_pr_curve(y_true, probas_pred):
    '''展示ROC曲线，和精确率-召回率曲线'''
    fpr, tpr, thresholds = roc_curve(y_true, probas_pred)
    precision, recall, thresholds = precision_recall_curve(y_true, probas_pred)
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(1, 2, 1)
    ax1 = fig.add_subplot(1, 2, 2)

    # plot Random Forest ROC
    ax.plot(fpr, tpr,
            label="随机森林Random Forest (AUC = {:1.4f})".format(roc_auc_score(y_true, probas_pred)))
    ax1.plot(recall, precision, label='召回率-准确率曲线')
    ax1.legend(loc="upper right")
    # plot Baseline ROC
    ax.plot([0, 1], [0, 1], label="基线Baseline (AUC = 0.5000)", linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("假阳性率(False Positive Rate)", fontsize=14)
    ax.set_ylabel("真阳性率(True Positive Rate)", fontsize=14)
    ax.set_title("ROC曲线(Curve)", fontsize=16)
    ax1.set_xlabel("召回率(recall)", fontsize=14)
    ax1.set_ylabel("精确率(precision)", fontsize=14)
    ax1.set_title("召回率-准确率曲线(Curve)", fontsize=16)
    ax.legend(loc="lower right")
    # plt.savefig("roc.png", bbox_inches="tight")
    fig.show()

# ### 数据重采样
# 为了处理类别不平衡，我们可以对多数类别（非欺诈案例）进行欠采样或对少数类别（欺诈案例）进行过采样。但也有缺点。由于欠采样，我们丢弃了大量数据和信息。
# 通过过采样，我们正在复制数据并创建重复项。 SMOTE 或 Synthetic Minority Oversampling Technique 可能是通过过采样少数类来调整类不平衡的更好方法。
# 使用 SMOTE，我们不仅仅是复制监控类，SMOTE 使用欺诈案件最近邻居的特征来创建新的合成欺诈案件并避免重复。但只有在欺诈案件彼此非常相似的情况下才能很好地发挥作用。
# 如果欺诈分布在数据上并且不是很明显，则使用最近邻居创建更多欺诈案例会在数据中引入一些噪音，因为最近邻居可能不一定是欺诈案例。

# 要记住的一件事：在训练集上使用重采样方法，不要在测试集上使用。始终确保测试集没有重复或合成数据。

#在[19]：


# import SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# Define the resampling method
method = SMOTE(random_state=1)

# Create the resampled feature set
X_resampled, y_resampled = method.fit_resample(X_train, y_train)


# In[20]:


# check before and after resample
print("重采用前（Before resampling）:\n{}\n".format(y_train.value_counts()))
print("重采用后（After resampling）:\n{}\n".format(pd.Series(y_resampled).value_counts()))

# 上面的结果显示了两个类之间的平衡如何随着 SMOTE 发生变化。
# 使用 SMOTE 可以让我们更多地观察少数类。
# 与随机过采样不同，SMOTE 不会创建观察的精确副本，而是创建新的、合成的样本，这些样本与少数类中的现有观察非常相似。
# 因此，SMOTE 比仅仅复制观察稍微复杂一些。 然后我们可以将重采样的训练数据拟合到机器学习模型中，并对非重采样的测试数据进行预测。

# In[21]:


# fit the model
model = LogisticRegression(solver="liblinear")
model.fit(X_resampled, y_resampled)

# make predictions
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# print the accuracy score
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))

# print the ROC score
print("ROC score: {}\n".format(roc_auc_score(y_test, probs[:,1])))

# print the classifcation report and confusion matrix
print("分类报告Classification report:\n{}\n".format(classification_report(y_test, predicted)))

# print confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print("混淆矩阵Confusion matrix:\n{}\n".format(conf_mat))

# 混淆矩阵Confusion matrix:
# [[1499    6]
#  [   0   10]]

# 从结果可以看出，10个案例都检测出来了，漏检率为0（召回率recall_score=1），但误报的却高达6个；

##### 管道
# 我们也可以使用将重采样方法与模型一次性结合的管道。 首先，我们需要定义我们将要使用的管道。 Pipeline() 需要两个参数。 我们需要在相应的参数中声明我们希望将重采样与模型结合起来。
#
# 在我们定义了我们的管道之后，也就是通过将逻辑回归与 SMOTE 方法相结合，我们可以在数据上运行它。 我们可以将管道视为单个机器学习模型。

# 在[22]：

# import SMOTE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE

# this is the pipeline module we need for this from imblearn
from imblearn.pipeline import Pipeline 

# define which resampling method and which ML model to use in the pipeline
resampling = BorderlineSMOTE(kind="borderline-2")
model = LogisticRegression(solver="liblinear")

# define the pipeline, tell it to combine SMOTE with the Logistic Regression model
pipeline = Pipeline([("SMOTE", resampling), ("Logistic Regression", model)])


# In[23]:


# fit your pipeline onto your training set and obtain predictions by fitting the model onto the test data 
pipeline.fit(X_train, y_train) 
predicted = pipeline.predict(X_test)
probs = pipeline.predict_proba(X_test)

# print the accuracy score
print("准确率Accuracy Score: {}\n召回率：{}".format(accuracy_score(y_test, predicted), recall_score(y_test, predicted)))

# print the ROC score
print("ROC score: {}\n".format(roc_auc_score(y_test, probs[:,1])))

# print the classifcation report and confusion matrix
print("分类报告Classification report:\n{}\n".format(classification_report(y_test, predicted)))

# print confusion matrix
conf_mat = confusion_matrix(y_true=y_test, y_pred=predicted)
print("混淆矩阵Confusion matrix:\n{}\n".format(conf_mat))


# 混淆矩阵Confusion matrix:
# [[1499    6]
#  [   0   10]]


# plot the confusion matrix
sns.heatmap(conf_mat, annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()

# 如我们所见，SMOTE 略微改善了我们的结果。我们现在设法找到所有欺诈案例，但我们的误报数量略高，尽管只有 6 个案例。
# 请记住，并非在所有情况下重新采样都一定会导致更好的结果。当欺诈案例非常分散且分散在数据中时，使用 SMOTE 可能会引入一些偏差。
# 最近的邻居不一定也是欺诈案例，因此合成样本可能会稍微“混淆”模型。

# ### 随机森林

# 在欺诈检测的情况下，准确性可能会产生误导。一个没有预测能力并且只预测一切为非欺诈的模型将具有很高的准确性。这并不意味着它是一个好模型。
# 对于高度不平衡的欺诈数据，我们需要查看 Precision、Recall 和 AUC ROC 曲线。这些是更可靠的性能指标，用于比较不同的分类器。
# 要了解 Precision 和 Recall，我们需要了解 False Negative 和 False positive。假阴性 - 预测没有欺诈，但实际上存在欺诈。误报 - 误报，预测欺诈但实际上没有欺诈。
# 实施欺诈检测的不同公司可能侧重于不同的方面。例如，银行和保险公司可能有不同的侧重点。
#
# 银行可能希望尽量减少误报，尽可能多地捕获欺诈，因为欺诈信用卡交易会花费很多钱，他们不介意误报，因为这只是意味着停止交易。
# 因此，银行可能希望针对召回进行优化，尽可能高，意味着在所有实际欺诈案件中，尽可能多地被标记，从而在所有实际欺诈案件中实现高比例的预测欺诈案件。
#
# 另一方面，保险公司可能希望最大限度地减少误报，即最大限度地减少误报，因为高误报意味着需要花费大量资源组建调查团队来处理每个标记的欺诈案件。
# 因此，保险公司可能希望针对 Precision 进行尽可能高的优化，从而在所有预测的欺诈案件中实现较高比例的实际欺诈案件。
#
# 精度和召回率成反比，随着精度的增加，召回率下降，反之亦然。这就是精确召回权衡。需要在您的模型中实现这两者之间的平衡，否则您最终可能会出现许多误报，或者没有捕获足够的实际欺诈案例。
# 为了实现这一点并比较性能，精确召回曲线派上用场。然而，更好的指标是 AUC ROC（接收者操作特征曲线下的面积）。
# AUC ROC 回答了这个问题：“在各种不同的基线概率下，这个分类器的总体表现如何？”但精确度和召回率没有。
#
# 因为对于欺诈检测，我们最感兴趣的是尽可能多地捕获欺诈案例，我们可以优化我们的模型设置以获得最佳的召回分数。如果我们还关心减少误报的数量，我们可以优化 F1-score，这给了我们很好的 Precision-Recall 权衡。
# 为了决定哪种最终模型最好，我们需要考虑不抓住欺诈者有多糟糕，以及欺诈分析团队可以处理多少误报。最终，这个最终决定应该由我们和欺诈团队共同做出。
#
# 话虽如此，准确度是一个很好的衡量标准，因为我们需要知道“自然准确度”是什么，如果我们要将一切预测为非欺诈。重要的是要了解我们需要“击败”哪个级别的“准确性”才能获得比什么都不做更好的预测。


# 我们将创建第一个用于欺诈检测的随机森林分类器。这将作为我们将尝试改进的“基线”模型。


# load dataset
df = pd.read_csv(rf"D:\Users\{USERNAME}\github_project\fraud-detection\chapter_2\creditcard_sampledata_2.csv")

df.head()

# explore the features available in the dataframe
print(df.info())

# summary statistics
df.describe()

# check for missing values
df.isnull().sum()

# count the occurrences of fraud and no fraud
df["Class"].value_counts()

# calculate the ratio of fraud and no fraud
df["Class"].value_counts(normalize=True)

# create input and target variable
X = df.drop(["Unnamed: 0", "Class"], axis=1)
y = df["Class"]

# create the training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

#模型调整
# 调整随机森林模型以处理高度不平衡的欺诈数据的一种简单方法是在定义 sklearn 模型时使用 ```class_weights``` 选项。
# ```"balanced"``` 模式使用 y 的值来自动调整与输入数据中的类频率成反比的权重，如 n_samples / (n_classes * np.bincount(y))。
# ```"balanced_subsample"``` 模式与“平衡”模式相同，除了权重是根据每棵生长的树的引导样本计算的。

# 将模型定义为随机森林
model = RandomForestClassifier(class_weight="balanced_subsample", random_state=0)

# fit the model to our training set
model.fit(X_train, y_train)

# obtain predictions from the test data 
predicted = model.predict(X_test)

# predict probabilities
probs = model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("准确率Accuracy Score: {}\n召回率: {}".format(accuracy_score(y_test, predicted), recall_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("分类报告Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("混淆矩阵Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))


# In[34]:


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()

# GridSearchCV 寻找最优参数
#
# 使用 GridSearchCV，我们可以定义对选项进行评分的性能指标。
# 由于对于欺诈检测，我们最感兴趣的是尽可能多地捕获欺诈案例，因此我们可以优化模型设置以获得最佳的召回分数。
# 如果我们还关心减少误报的数量，我们可以优化 F1-score，这给了我们很好的 Precision-Recall 权衡。

# 在[35]:

# define the parameter sets to test
param_grid = {"n_estimators": [10, 50],
              "max_features": ["auto", "log2"],
#               "min_samples_leaf": [1, 10],
              "max_depth": [4, 8],
              "criterion": ["gini", "entropy"],
              "class_weight": [None, {0:1, 1:12}]
}
# define the model to use
model = RandomForestClassifier(random_state=0)
# combine the parameter sets with the defined model
CV_model = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring="recall", n_jobs=-1)
# fit the model to our training data and obtain best parameters
CV_model.fit(X_train, y_train)


# In[36]:


# show best parameters
print("最优参数",CV_model.best_params_)


# In[37]:


# obtain predictions from the test data 
predicted = CV_model.predict(X_test)

# predict probabilities
probs = CV_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("准确率：Accuracy Score: {}\n召回率：{}".format(accuracy_score(y_test, predicted), recall_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("分类报告：Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("混淆矩阵：Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))


# In[38]:


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()


# In[39]:


# 或者手动输入最优参数到模型
model = RandomForestClassifier(class_weight={0:1,1:12}, criterion='entropy',
            n_estimators=50, max_features='log2', max_depth=4, n_jobs=-1, random_state=0)

model.fit(X_train, y_train)
predicted = model.predict(X_test)
probs = model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))

# Confusion Matrix:
# [[2098    1]
#  [  16   75]]

# 逻辑回归

# define the Logistic Regression model with weights
lr_model = LogisticRegression(class_weight={0:1, 1:15}, random_state=5, solver="liblinear")

# fit the model to our training data
lr_model.fit(X_train, y_train)

# obtain predictions from the test data 
predicted = lr_model.predict(X_test)

# predict probabilities
probs = lr_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))


# In[41]:


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()

# Confusion Matrix:
# [[2055   44]
#  [  11   80]]
# Logistic Regression 的性能与随机森林截然不同。 更多的误报，但也有更好的召回。 因此，它将成为集成模型中随机森林的有用补充。

# 决策树
# 定义权重平衡的决策树模型
tree_model = DecisionTreeClassifier(random_state=0, class_weight="balanced")

# fit the model to our training data
tree_model.fit(X_train, y_train)

# obtain predictions from the test data 
predicted = tree_model.predict(X_test)

# predict probabilities
probs = tree_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))

# Confusion Matrix:
# [[2078   21]
#  [  15   76]]


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()

# 与随机森林相比，决策树也有更多的误报，但有更好的召回率。 因此，它将成为集成模型中随机森林的有用补充。

# ### 投票分类器
#
# 现在让我们将三个机器学习模型合二为一，以改进我们之前的随机森林欺诈检测模型。 我们将把我们常用的随机森林模型与上一节中的逻辑回归和决策树模型结合起来。

# import the package
from sklearn.ensemble import VotingClassifier

# define the three classifiers to use in the ensemble
clf1 = LogisticRegression(class_weight={0:1, 1:15}, random_state=0, solver="liblinear")
clf2 = RandomForestClassifier(class_weight={0:1,1:12}, criterion='entropy', n_estimators=50, max_features='log2', max_depth=4, n_jobs=-1, random_state=0)
clf3 = DecisionTreeClassifier(random_state=0, class_weight="balanced")


# 在集成模型中组合分类器
ensemble_model = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2), ("dt", clf3)], voting="hard")
# voting = 'hard'：表示最终决策方式为 Hard Voting Classifier,根据少数服从多数来定最终结果
# voting = 'soft'：表示最终决策方式为 Soft Voting Classifier,将所有模型预测样本为某一类别的概率的平均值作为标准，概率最高的对应的类型为最终的预测结果；
ensemble_model.fit(X_train, y_train)
predicted = ensemble_model.predict(X_test)
# probs = ensemble_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
# print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))


# Confusion Matrix:
# [[2090    9]
#  [  13   78]]


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()

# 在投票分类器中调整权重
# 我们刚刚看到投票分类器允许我们通过结合多个模型的优点来提高欺诈检测性能。
# 现在让我们尝试调整我们赋予这些模型的权重。 通过增加或减少权重，我们可以玩弄我们对特定模型相对于其他模型的重视程度。
# 当某个模型的整体性能优于其他模型时，这会派上用场，但我们仍然希望结合其他模型的各个方面来进一步改进我们的结果。

# 组合集成模型中的分类器
ensemble_model = VotingClassifier(estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)], voting="soft", weights=[1,4,1], flatten_transform=True)

ensemble_model.fit(X_train, y_train)
predicted = ensemble_model.predict(X_test)
probs = ensemble_model.predict_proba(X_test)

# print the accuracy score, ROC score, classification report and confusion matrix
print("Accuracy Score: {}\n".format(accuracy_score(y_test, predicted)))
print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:,1])))
print("Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
print("Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))


# Confusion Matrix:
# [[2094    5]
#  [  13   78]]


# plot the confusion matrix
sns.heatmap(confusion_matrix(y_test, predicted), annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of the classifier", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("log_cm.png", bbox_inches="tight")
plt.show()


# In[49]:


# create ROC curves
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, CV_model.predict_proba(X_test)[:,1])
lr_fpr, lr_tpr, lr_thresholds = roc_curve(y_test, lr_model.predict_proba(X_test)[:,1])
tree_fpr, tree_tpr, tree_thresholds = roc_curve(y_test, tree_model.predict_proba(X_test)[:,1])
ensemble_fpr, ensemble_tpr, ensemble_thresholds = roc_curve(y_test, ensemble_model.predict_proba(X_test)[:,1])
plt.figure(figsize=(12, 8))

# plot Random Forest ROC
plt.plot(fpr, tpr, label="Random Forest (AUC = {:1.4f})".format(roc_auc_score(y_test, CV_model.predict_proba(X_test)[:,1])))
# plot Linear Regression ROC
plt.plot(lr_fpr, lr_tpr, label="Logistic Regression (AUC = {:1.4f})".format(roc_auc_score(y_test, lr_model.predict_proba(X_test)[:,1])))
# plot Decision Tree ROC
plt.plot(tree_fpr, tree_tpr, label="Decision Tree (AUC = {:1.4f})".format(roc_auc_score(y_test, tree_model.predict_proba(X_test)[:,1])))
# plot Voting Classifier ROC
plt.plot(ensemble_fpr, ensemble_tpr, label="Voting Classifier (AUC = {:1.4f})".format(roc_auc_score(y_test, ensemble_model.predict_proba(X_test)[:,1])))
# plot Baseline ROC
plt.plot([0,1], [0,1],label="基线Baseline (AUC = 0.5000)", linestyle="--")

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("假阳性率(False Positive Rate)", fontsize=14)
plt.ylabel("真阳性率(True Positive Rate)", fontsize=14)
plt.title("ROC曲线(Curve)", fontsize=16)
plt.legend(loc="lower right")
# plt.savefig("roc.png", bbox_inches="tight")
plt.show()


#  讨论
# 通过组合分类器，我们可以取多个模型中的优点。随机森林作为一个独立的模型在 Precision 方面表现不错，但在误报方面却很糟糕。
# Logistic 回归在 Recall 方面表现良好，但在误报方面非常糟糕。决策树在中间。通过将这些模型组合在一起，我们确实设法提高了性能。
# 我们将捕获的欺诈案例从 75 个增加到 78 个，并将误报减少了 3 个，而我们只有 4 个额外的误报作为回报。
# 如果我们确实关心捕获尽可能多的欺诈案件，同时保持低误报，这是一个很好的权衡。

# 无监督机器学习
# 当我们没有欺诈案件的标签时（通常在现实生活中），我们必须利用无监督机器学习。
# 当使用无监督学习技术进行欺诈检测时，我们希望区分正常和异常（因此可能是欺诈）行为。作为欺诈分析师，要了解什么是“正常”，我们需要对数据及其特征有很好的了解。

# load the dataset
df = pd.read_csv(rf"D:\Users\{USERNAME}\github_project\fraud-detection\chapter_3\banksim.csv")

print(df.shape)
df.head()

print(df.info())

df["fraud"].value_counts()

df["fraud"].value_counts(normalize=True)

# 对 category 的值进行统计分析
df.groupby("category").mean().sort_values(by="fraud", ascending=False)

# 如上所示，大部分欺诈发生在休闲(leisure)、旅游(travel)和体育(sportsandtoys)相关交易中。
#
# ### 客户细分
#
# 我们将检查这些数据中的客户端是否有任何明显的模式，因此我们是否需要将数据分组，或者数据是否相当同质。
#
# 不幸的是，我们没有很多可用的客户信息； 例如，我们无法区分不同客户的财富水平。 但是，有可用的年龄数据，所以让我们看看不同年龄段的行为是否存在显着差异。


# group by age groups and get the mean
df.groupby("age").mean()


# In[56]:


df["age"].value_counts().sort_index()

# 基于以上结果，在运行欺诈检测算法之前将数据划分为年龄段是否有意义？ 正如我们所见，各组的平均花费金额和欺诈发生率相当相似。
# 年龄组“0”很突出，但由于只有 40 个案例，将它们分成一个单独的组并在它们上运行单独的模型是没有意义的。

#
# 用欺诈和非欺诈数据创建两个数据
df_fraud = df.loc[df.fraud == 1] 
df_non_fraud = df.loc[df.fraud == 0]

# plot histograms of the amounts in fraud and non-fraud data 
plt.hist(df_fraud.amount, alpha=0.5, label='欺诈用户(fraud)')
plt.hist(df_non_fraud.amount, alpha=0.5, label='正常用户(nonfraud)')
plt.title("Fraud vs Non-fraud by Amount")
plt.xlabel("金额(Amount)", fontsize=12)
plt.ylabel("交易数量(No of transaction)", fontsize=12)
plt.legend()
plt.show()

# 展示欺诈用户和正常用户对应的年龄分布图；
age_xlabel = ['0', '1', '2', '3', '4', '5', '6', 'U']
plt.bar(age_xlabel, [df_fraud.age.value_counts().get(x, 0) for x in age_xlabel], alpha=0.5, label='欺诈用户(fraud)')
plt.bar(age_xlabel, [df_non_fraud.age.value_counts().get(x, 0) for x in age_xlabel], alpha=0.5, label='正常用户(nonfraud)')
plt.title("Fraud vs Non-fraud by Amount")
plt.xlabel("金额(Amount)", fontsize=12)
plt.ylabel("交易数量(No of transaction)", fontsize=12)
plt.legend()
plt.show()



# 由于欺诈观察的数量要小得多，因此很难看到完整的分布。 尽管如此，我们可以看到，相对于正常观察，欺诈交易往往处于较大的一边。
# 这是个好消息，因为它有助于我们稍后从非欺诈中检测欺诈。 接下来我们将实现一个聚类模型来区分正常和异常交易，
# **当欺诈标签不再可用时。**


# load the dataset
df = pd.read_csv(rf"D:\Users\{USERNAME}\github_project\fraud-detection\chapter_3\banksim_adj.csv")

print(df.shape)
df.head()


# In[59]:


# 统计欺诈和无欺诈的发生情况
df["fraud"].value_counts()

# count the occurrences of fraud and no fraud
df["fraud"].value_counts(normalize=True)

# 缩放数据
#
# 对于使用基于距离的度量的 ML 算法，始终缩放我们的数据至关重要，因为使用不同比例的特征会扭曲我们的结果。
# K-means 使用欧几里得距离来评估到聚类质心的距离，因此我们首先需要在继续实现算法之前缩放数据。 让我们先这样做。

# 创建输入和目标变量
# 将特征转换为一个 numpy 数组

X = np.array(df.drop(["Unnamed: 0", "fraud"], axis=1)).astype(np.float)
y = df["fraud"].values

from sklearn.preprocessing import MinMaxScaler

# 定义缩放器并应用于数据，默认 feature_range=(0, 1)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ### K-means 聚类
#
# 一个非常常用的聚类算法是 K-means 聚类。对于欺诈检测，K-means 聚类实施起来很简单，并且在预测可疑案例方面相对强大。在处理欺诈检测问题时，这是一个很好的算法。但是，欺诈数据通常非常大，尤其是在我们处理交易数据时。 MiniBatch K-means 是在大型数据集上实现 K-means 的一种有效方式（更快），我们将在本项目中使用。

# Import MiniBatchKmeans 
from sklearn.cluster import MiniBatchKMeans

# Define the model 
# default: n_clusters=8, init='k-means++', max_iter=100, batch_size=100,
kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)

# 将模型拟合到缩放数据 Fit the model to the scaled data
kmeans.fit(X_scaled)

# 我们现在已经将 MiniBatch K-means 模型拟合到数据中。默认的 n_clusters 是 8。但是我们需要弄清楚要使用的正确集群数量是多少。
#
# 肘部法则–Elbow Method
# 我们知道k-means是以最小化样本与质点平方误差作为目标函数，将每个簇的质点与簇内样本点的平方距离误差和称为畸变程度(distortions)，
# 那么，对于一个簇，它的畸变程度越低，代表簇内成员越紧密，畸变程度越高，代表簇内结构越松散。
# 畸变程度会随着类别的增加而降低，但对于有一定区分度的数据，在达到某个临界点时畸变程度会得到极大改善，之后缓慢下降，这个临界点就可以考虑为聚类性能较好的点。

# 我们已经用 8 个集群实现了 MiniBatch K-means，但实际上并没有检查正确数量的集群应该是多少。
# 对于我们的第一个欺诈检测方法，获得正确的集群数量很重要，尤其是当我们想使用这些集群的异常值作为欺诈预测时。
# 为了决定我们要使用多少集群，让我们应用 Elbow 方法，看看基于这种方法的最佳集群数量应该是多少。


# define the range of clusters to try
clustno = range(1, 10)

# run MiniBatch Kmeans over the number of clusters
kmeans = [MiniBatchKMeans(n_clusters=i) for i in clustno]

# obtain the score for each model
score = [kmeans[i].fit(X_scaled).score(X_scaled) for i in range(len(kmeans))]

# plot the models and their respective score 
plt.plot(clustno , score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# 另一种方法：
# 对于每个 k 值，我们将初始化 k-means 并使用惯性属性来识别样本到最近聚类中心的距离平方和。

# kmeans = MiniBatchKMeans(n_clusters=8, random_state=0)
Sum_of_squared_distances = []
# Cluster_centers = []
clustno = range(1,16)
for k in clustno:
    km = MiniBatchKMeans(n_clusters=k, random_state=0)
    km = km.fit(X_scaled)
    Sum_of_squared_distances.append(km.inertia_)
#     Cluster_centers.append(km.cluster_centers_)
    
# print("There are {} clusters.".format(len(Cluster_centers), "\n"))
plt.plot(clustno, Sum_of_squared_distances, "bx-")
plt.xlabel("k")
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
plt.ylabel("Sum_of_squared_distances")
plt.title("Elbow Method For Optimal k")
plt.show()

# 现在我们可以看到最佳的簇数应该是大约 3 个簇，因为那是肘部在曲线中的位置。 我们将使用它作为我们的基准模型，看看它在检测欺诈方面的效果如何。
#
# ### 检测异常值
#
# 我们将使用 K-means 算法来预测欺诈，并将这些预测与保存的实际标签进行比较，以检测我们的结果。
#
# 欺诈交易通常被标记为离集群质心最远的观察。 我们需要确定截止点。


# split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=0)

# define K-means model 
kmeans = MiniBatchKMeans(n_clusters=3, random_state=42).fit(X_train)

# obtain predictions and calculate distance from cluster centroid
X_test_clusters = kmeans.predict(X_test)

# save the cluster centroids
X_test_clusters_centers = kmeans.cluster_centers_

# calculate the distance to the cluster centroids for each point
dist = [np.linalg.norm(x-y) for x, y in zip(X_test, X_test_clusters_centers[X_test_clusters])]

# create fraud predictions based on outliers on clusters 
# define the boundary between fraud and non fraud to be at 95% of distance distribution and higher
km_y_pred = np.array(dist)
km_y_pred[dist >= np.percentile(dist, 95)] = 1
km_y_pred[dist < np.percentile(dist, 95)] = 0


# 现在我们有了来自 kmeans 模型的预测，让我们检查它与实际标签的比较情况。
#
# ### 检查模型结果
#
# 在上一节中，如果所有观察值与聚类质心的距离位于前 5 个百分位，我们已将所有观察值标记为欺诈。
# 这些是三个集群的异常值。 我们已经将缩放数据和标签分成训练和测试集，因此 y_test 可用。 预测 km_y_pred 也可用。 让我们创建一些性能指标，看看我们做得如何。

#在[67]中：


# check the results
km_y_pred[:100]


# In[68]:


# print the accuracy score, ROC score, classification report and confusion matrix
print("准确率Accuracy Score: {}\n召回率： {}".format(accuracy_score(y_test, km_y_pred), recall_score(y_test, km_y_pred)))

# obtain the ROC score
print("ROC score = {}\n".format(roc_auc_score(y_test, km_y_pred)))

# create a confusion matrix
km_cm = confusion_matrix(y_test, km_y_pred)
print('混乱矩阵', km_cm)
# [2029,   70],
#        [  20,   38]

# plot the confusion matrix in a figure to visualize results 
sns.heatmap(km_cm, annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("Confusion matrix of K-means", fontsize=14)
plt.ylabel("Actual label", fontsize=12)
plt.xlabel("Predicted label", fontsize=12)

# plt.savefig("_cm.png", bbox_inches="tight")
plt.show()

# 如果我们将上一个练习中用作截止点的百分位数减少到 93% 而不是 95%，发现的欺诈案件数量增加，但误报也在增加。通过降低标记为欺诈的案例的阈值，我们总体上标记了更多的案例，但也因此得到了更多的误报。
#
# DBSCAN
# DBSCAN - 基于密度的噪声应用空间聚类。找到高密度的核心样本并从中扩展集群。适用于包含相似密度集群的数据。 DBSCAN 根据距离测量（通常是欧几里德距离）和最小点数将彼此靠近的点组合在一起。它还将低密度区域中的点标记为异常值。
#
# 参数：
# DBSCAN算法基本上需要2个参数：
# - eps：两点之间的最小距离。这意味着如果两点之间的距离小于或等于该值 (eps)，则这些点被视为邻居。
# - minPoints：形成密集区域的最小点数。例如，如果我们将 minPoints 参数设置为 5，那么我们至少需要 5 个点才能形成一个密集区域。

# DBSCAN vs K-means
# - 无需预定义集群数量
# - 调整簇内点之间的最大距离
# - 在集群中分配最小数量的样本
# - 在形状怪异的数据（即非凸）上有更好的表现
# - 但更高的计算成本

# 这一次，我们不会将集群的异常值用于欺诈，而是将数据中最小的集群标记为欺诈。


# import DBSCAN
from sklearn.cluster import DBSCAN

# initialize and fit the DBscan model
db = DBSCAN(eps=0.9, min_samples=10, n_jobs=-1).fit(X_scaled)

# obtain the predicted labels 
pred_labels = db.labels_

# calculate number of clusters
# Cluster labels for each point in the dataset given to fit(). Noisy samples are given the label -1.
n_clusters = len(set(pred_labels)) - (1 if -1 in y else 0)

# print performance metrics for DBscan
print("Estimated number of clusters: {}".format(n_clusters))
print("Homogeneity: {:0.4f}".format(homogeneity_score(y, pred_labels)))
print("Silhouette Coefficient: {:0.4f}".format(silhouette_score(X_scaled, pred_labels)))

# 使用 DBSCAN (23) 的集群数量远高于使用 K-means (3)。 对于欺诈检测，这暂时没问题，因为我们只对最小的集群感兴趣，因为它们被认为是异常的。 现在让我们看看这些集群并决定将哪个集群标记为欺诈。
# ### 评估最小的集群

# 我们将查看来自 DBSCAN 的集群，并将某些集群标记为欺诈：
#
# - 首先，我们需要弄清楚集群有多大，并过滤掉最小的
# - 然后，我们将取最小的并将其标记为欺诈
# - 最后，我们将检查原始标签是否确实在检测欺诈方面做得很好


# 计算每个簇号中的观察值
counts = np.bincount(pred_labels[pred_labels>=0])

# 对簇的样本数进行排序，并取前 3 个最小的簇
smallest_clusters = np.argsort(counts)[:3]
print('前 3 个最小的簇', smallest_clusters)

# 最小簇的计数
print("Their counts are:")      
print(counts[smallest_clusters])


# 所以现在我们知道我们可以将哪些最小的集群标记为欺诈。
# 如果我们要采用更多最小的集群，我们会撒得更广，捕获更多欺诈，但很可能也会出现更多误报。
# 欺诈分析师需要找到合适数量的案例进行标记和调查。
#
#### 使用实际标签检查结果
#
# 我们将检查 DBSCAN 欺诈检测模型的结果。实际上，我们通常没有可靠的标签，而欺诈分析师可以在这里帮助我们验证结果。
# 他/她可以检查我们的结果，看看我们标记的案例是否确实可疑。我们还可以检查历史上已知的欺诈案例，看看我们的模型是否标记了它们。


# 创建预测聚类数和欺诈标签的数据框
df_result = pd.DataFrame({'clusternr':pred_labels,'fraud':y})

# 为最小的集群创建一个条件标记欺诈
df_result['predicted_fraud'] = np.where((df_result.clusternr==21)|(df_result.clusternr==17)|(df_result.clusternr==9), 1 , 0)

# 对结果运行交叉表
print(pd.crosstab(y, df_result['predicted_fraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud']))


# In[72]:


# 在图中绘制混淆矩阵以可视化结果
testabc = pd.crosstab(y, df_result['predicted_fraud'], rownames=['Actual Fraud'], colnames=['Flagged Fraud'])
sns.heatmap(testabc, annot=True, annot_kws={"size":16}, fmt="d", cbar=False, linewidths=0.1, cmap="Blues")
plt.title("DBSCAN 混淆矩阵(Confusion matrix)", fontsize=14)
plt.ylabel("实际标签(Actual label)", fontsize=12)
plt.xlabel("预测标签(Predicted label)", fontsize=12)

# plt.savefig("_cm.png", bbox_inches="tight")
plt.show()


# 在所有标记的案例（40）中，大约 2/3 实际上是欺诈（24）！由于我们只采用三个最小的集群，根据定义，我们标记的欺诈案例较少，因此我们捕获的较少但误报也较少。
# 但是，我们遗漏了很多欺诈案例。增加我们标记的最小集群的数量可以改善这一点，当然代价是更多的误报。

# ### 验证模型结果
# 在现实中，我们通常没有实际欺诈案例的可靠标签（即真实情况），因此很难用正常的性能指标（例如准确性）来验证模型结果。
# 但是还有其他方法可以做到这一点：
# - 与欺诈分析师核对
# - 更详细地调查和描述被标记的案例
# - 与过去已知的欺诈案例进行比较（使用过去已知欺诈案例的模型来查看该模型是否能够真正正确地检测到那些历史欺诈案例）


# 讨论和结论
# 我们使用了有监督和无监督的机器学习技术来检测欺诈案例。
# 当我们遇到带有标签的欺诈案件时，我们会使用监督机器学习。
# 通过组合分类器，我们可以利用多个模型中的优点。随机森林作为一个独立的模型在 Precision 方面表现不错，但在误报方面却很糟糕。
# Logistic 回归在 Recall 方面表现良好，但在误报方面非常糟糕。
# 决策树在中间。通过将这些模型组合在一起，我们确实设法提高了性能。我们将捕获的欺诈案例从 75 个增加到 78 个，并将误报减少了 3 个，而我们只有 4 个额外的误报作为回报。
# 如果我们确实关心捕获尽可能多的欺诈案件，同时保持低误报，这是一个很好的权衡。

# 当我们没有欺诈案例的标签时（通常在现实生活中），我们可以使用无监督机器学习技术来区分正常和异常（因此可能是欺诈）行为。
# 这需要了解什么是“正常”，我们需要对数据及其特征有很好的了解。需要指出的是，很难用正常的性能指标（例如准确性、预测、召回）来验证无监督机器学习模型的结果，因为我们没有实际的欺诈标签或基本事实。
# 但是还有其他方法可以做到这一点，例如与欺诈分析师核对以帮助我们验证并查看我们标记的案例是否确实可疑，更详细地调查和描述标记的案例并使用过去已知欺诈案例的模型来查看是否该模型实际上可以正确检测那些历史欺诈案例。







