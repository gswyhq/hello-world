#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, ShuffleSplit
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# to plot within notebook
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
import os
from collections import defaultdict
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
# machine learning modules
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, average_precision_score
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
from imblearn.over_sampling import SMOTE  #  imbalanced-learn
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
import matplotlib

from sklearn.inspection import permutation_importance
from sklearn.utils.fixes import parse_version

try:
    from sklearn.datasets import load_boston
except Exception as e:
    print('最新版本sklearn中boston数据集已经被移除，故而需要通过如下方法获取')

    def load_boston():
        boston = {}
        # data_url = "http://lib.stat.cmu.edu/datasets/boston"  # 数据集下载地址
        raw_df = pd.read_csv(rf"D:\Users\{USERNAME}\Downloads\boston.txt", sep="\s+", skiprows=22, header=None)
        data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
        target = raw_df.values[1::2, 2]
        feature_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
        data.shape, target.shape
        # Out[10]: ((506, 13), (506,))
        boston["data"] = data
        boston["target"] = target
        boston["feature_names"] = feature_names
        return boston

# 使用imlbearn库中上采样方法中的SMOTE接口
from imblearn.over_sampling import SMOTE
import sklearn
print(sklearn.__version__)

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

x = np.array([[1., 0.], [2., 1.], [0., 0.]])
y = np.array([0, 1, 2])
# 同时打乱多个数组（同时打乱数据集、标签集），并保持原有的映射关系；
from sklearn.utils import shuffle
x, y = shuffle(x, y, random_state=0)


def linear_model_filter_feature(X, Y, names=None, sort=False):
    '''线性模型筛选特征；
    它的主要思想是在不同的数据子集和特征子集上运行特征选择算法，不断的重复，最终汇总特征选择结果，
    比如可以统计某个特征被认为是重要特征的频率（被选为重要特征的次数除以它所在的子集被测试的次数）。
    理想情况下，重要特征的得分会接近100%。稍微弱一点的特征得分会是非0的数，而最无用的特征得分将会接近于0。
    用回归模型的系数来选择特征。
    '''
    lr = LinearRegression()
    lr.fit(X, Y)

    # A helper method for pretty-printing linear models
    coefs = lr.coef_
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return [(name, round(coef, 3)) for coef, name in lst]


def ridge_model_filter_feature(X, Y, names=None, sort=False):
    '''线性模型筛选特征；ridge是一个线性回归器。
    用回归模型的系数来选择特征。越是重要的特征在模型中对应的系数就会越大，而跟输出变量越是无关的特征对应的系数就会越接近于0。
    Ridge()函数是具有l2正则化的线性最小二乘法。
    '''

    ridge = Ridge(alpha=10)
    ridge.fit(X, Y)

    coefs = ridge.coef_
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst, key=lambda x: -np.abs(x[0]))
    return [(name, round(coef, 3)) for coef, name in lst]

def mean_decrease_accuracy_filter_feature(X, Y, names):
    '''平均精确率减少 Mean decrease accuracy
    另一种常用的特征选择方法就是直接度量每个特征对模型精确率的影响。
    主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(X), pd.DataFrame(Y), test_size=0.3, random_state=0,
                                                        shuffle=True)

    rf = RandomForestRegressor()
    scores = defaultdict(list)
    r = rf.fit(X_train, Y_train)
    acc = r2_score(Y_test, rf.predict(X_test))
    for i in range(len(X[0])):
        X_t = np.array(X_test.copy(), type(float))
        np.random.shuffle(X_t[:, i])
        shuff_acc = r2_score(Y_test, rf.predict(X_t))
        scores[names[i]].append((acc - shuff_acc) / acc)
    return sorted([(round(np.mean(score), 4), feat) for
			  feat, score in scores.items()], reverse=True)


def mean_decrease_precision_filter_feature(X, Y, names):
    '''平均精确度减少 Mean decrease precision
    另一种常用的特征选择方法就是直接度量每个特征对模型精确率的影响。
    主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
    按正常特征训练模型，预测时候，依次打乱每个特征值与未打乱的预测结果进行对比
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(pd.DataFrame(X), pd.DataFrame(Y), test_size=0.2, random_state=0,
                                                        shuffle=True)

    # rf = RandomForestRegressor()
    rf = RandomForestClassifier(class_weight="balanced_subsample")
    scores = defaultdict(list)
    r = rf.fit(X_train, Y_train)
    precision = average_precision_score(Y_test, rf.predict_proba(X_test)[:,1])
    for i in range(len(X[0])):
        X_t = np.array(X_test.copy(), type(float))
        np.random.shuffle(X_t[:, i])
        shuff_precision = average_precision_score(Y_test, rf.predict_proba(X_t)[:,1])
        scores[names[i]].append((precision - shuff_precision) / precision)
    return sorted([(round(np.mean(score), 4), feat) for
			  feat, score in scores.items()], reverse=True)

def rfe_filter_feature(X, Y, names):
    '''
    递归特征消除 Recursive feature elimination (RFE)
    递归特征消除的主要思想是反复的构建模型（如SVM或者回归模型）然后选出最好的（或者最差的）的特征（可以根据系数来选），把选出来的特征放到一遍，然后在剩余的特征上重复这个过程，直到所有特征都遍历了。
    这个过程中特征被消除的次序就是特征的排序。因此，这是一种寻找最优特征子集的贪心算法。
    RFE的稳定性很大程度上取决于在迭代的时候底层用哪种模型。
    例如，假如RFE采用的普通的回归，没有经过正则化的回归是不稳定的，那么RFE就是不稳定的；假如采用的是Ridge，而用Ridge正则化的回归是稳定的，那么RFE就是稳定的。
    对于RFE来说，由于它给出的是顺序而不是得分
    :param X:
    :param Y:
    :param names:
    :return:
    '''
    # use linear regression as the model
    lr = LinearRegression()
    # rank all features, i.e continue the elimination until the last one
    rfe = RFE(lr, n_features_to_select=1)  # 将最好的 n_features_to_select 个的得分定为1，其他的特征的得分均匀的分布在0-1之间。
    rfe.fit(X, Y)

    # 越重要的特征，排序越靠前；
    return sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))


def mean_decrease_impurity():
    '''平均精确率减少,筛选特征'''

    from sklearn.model_selection import ShuffleSplit
    from sklearn.metrics import r2_score
    from collections import defaultdict
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]

    rf = RandomForestRegressor()
    scores = defaultdict(list)
    scaler = StandardScaler()

    # 随机排列，交叉验证
    ss = ShuffleSplit(n_splits=10, test_size=100, train_size=.3)  # 循环 n_splits 次；测试集大小绝对值：100，训练集大小比例：len(X)*0.3
    for train_idx, test_idx in ss.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        Y_train, Y_test = Y[train_idx], Y[test_idx]
        # X_train = scaler.fit_transform(X[train_idx], Y_train)  # 训练集进行了均一化，对结果影响很大; fit_transform方法是fit和transform的结合，fit_transform(X_train) 意思是找出X_train的均值和​​​​​​​标准差，并应用在X_train上。
        r = rf.fit(X_train, Y_train)
        acc = r2_score(Y_test, rf.predict(X_test))
        for i in range(X.shape[1]):
            X_t = X_test.copy()
            np.random.shuffle(X_t[:, i])
            shuff_acc = r2_score(Y_test, rf.predict(X_t))
            scores[names[i]].append((acc - shuff_acc) / acc)
    print("根据分数排序特征:")
    print(sorted([(round(np.mean(score), 4), feat) for
                  feat, score in scores.items()], reverse=True))

def train_show(X, Y, field_list):

    # 基于随机森林筛选特征
    # 用随机森林进行特征重要性评估的思想比较简单，主要是看每个特征在随机森林中的每棵树上做了多大的贡献，然后取平均值，最后比较不同特征之间的贡献大小。
    # 随机森林提供了两种特征选择的方法：平均不纯度减少(mean decrease impurity)和平均精确率减少(mean decrease accuracy)。
    # 平均不纯度减少----mean decrease impurity
    # 随机森林由多个决策树构成。决策树中的每一个节点都是关于某个特征的条件，为的是将数据集按照不同的响应变量一分为二。
    # 利用不纯度可以确定节点（最优条件），对于分类问题，通常采用基尼不纯度或者信息增益，对于回归问题，通常采用的是方差或者最小二乘拟合。
    # 当训练决策树的时候，可以计算出每个特征减少了多少树的不纯度。对于一个决策树森林来说，可以算出每个特征平均减少了多少不纯度，并把它平均减少的不纯度作为特征选择的值。
    # 使用基于不纯度的方法的时候，要记住：
    # 1、这种方法存在偏向，对具有更多类别的变量会更有利；
    # 2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降，
    # 因为不纯度已经被选中的那个特征降下来了，其他的特征就很难再降低那么多不纯度了，这样一来，只有先被选中的那个特征重要度很高，其他的关联特征重要度往往较低。
    # 在理解数据时，这就会造成误解，导致错误的认为先被选中的特征是很重要的，而其余的特征是不重要的，但实际上这些特征对响应变量的作用确实非常接近的。
    # 特征随机选择方法稍微缓解了这个问题，但总的来说并没有完全解决。
    #
    # 平均精确率减少----Mean decrease accuracy
    # 另一种常用的特征选择方法就是直接度量每个特征对模型精确率的影响。主要思路是打乱每个特征的特征值顺序，并且度量顺序变动对模型的精确率的影响。
    # 很明显，对于不重要的变量来说，打乱顺序对模型的精确率影响不会太大，但是对于重要的变量来说，打乱顺序就会降低模型的精确率。
    # 当两个特征相关联并且其中一个特征被随机重排时，模型仍然可以通过其相关特征来访问此特征。这将导致两个特征的重要性指标降低，而这两个特征实际上可能很重要。
    # 当特征是共线特征时，置换一个特征对模型性能的影响很小，因为它可以从相关特征中获得相同的信息。

    #  尽管我们在所有特征上进行了训练得到了模型，然后才得到了每个特征的重要性测试，这并不意味着我们扔掉某个或者某些重要特征后模型的性能就一定会下降很多，因为即便某个特征删掉之后，其关联特征一样可以发挥作用，让模型性能基本上不变。

    # rf = RandomForestRegressor()
    rf = RandomForestClassifier(random_state=42, class_weight="balanced_subsample", criterion="gini")  # 权重随机森林的应用（用于增加小样本的识别概率，从而提高总体的分类准确率）；默认是采用基尼不纯度
    rf.fit(X, Y)
    sort_feature = sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), field_list), reverse=True)
    print(f"按分数排序特征:{sort_feature}")
    x_df = pd.DataFrame(X, columns=field_list)
    X = x_df[[col for _, col in sort_feature[:int(len(sort_feature)*0.8)]]].values
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)

    # 将模型定义为随机森林
    model = RandomForestClassifier(class_weight="balanced_subsample", random_state=0)

    # 定义SMOTE模型，random_state相当于随机数种子的作用
    smo = SMOTE(sampling_strategy={1: max([sum(y_train), int(len([t for t in y_train if t==0])*0.3)])}, random_state=42)
    X_smo, y_smo = smo.fit_resample(X_train, y_train)
    X_train, y_train = X_smo, y_smo

    # fit the model to our training set
    model.fit(X_train, y_train)

    # obtain predictions from the test data
    predicted = model.predict(X_test)

    # predict probabilities
    probs = model.predict_proba(X_test)

    # # xgboost模型
    # import xgboost as xgb
    # xg_classifier = xgb.XGBClassifier(objective='binary:logistic')
    # xg_classifier.fit(np.array(X_train), np.array(y_train))
    # xg_classifier.score(np.array(X_test), np.array(y_test))
    # probs = xg_classifier.predict_proba(np.array(X_test))
    # predicted = xg_classifier.predict(np.array(X_test))

    # 调整阈值为0.1
    # predicted = [1 if v > 0.1 else 0 for v in probs[:,1]]

    # print the accuracy score, ROC score, classification report and confusion matrix
    print("准确率Accuracy Score: {}\n召回率: {}".format(accuracy_score(y_test, predicted), recall_score(y_test, predicted)))
    print("平均精度(AP, average precision)", average_precision_score(y_test, predicted))
    print("ROC score = {}\n".format(roc_auc_score(y_test, probs[:, 1])))
    print("分类报告Classification Report:\n{}\n".format(classification_report(y_test, predicted)))
    print("混淆矩阵Confusion Matrix:\n{}\n".format(confusion_matrix(y_test, predicted)))

    fpr, tpr, thresholds = roc_curve(y_test, probs[:, 1])
    plt.figure(figsize=(12, 8))

    # plot Random Forest ROC
    plt.plot(fpr, tpr,
             label="随机森林Random Forest (AUC = {:1.4f})".format(roc_auc_score(y_test, probs[:, 1])))

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




def plot_permutation_importance(clf, X, y, ax):
    result = permutation_importance(clf, X, y, n_repeats=10, random_state=42, n_jobs=2)
    perm_sorted_idx = result.importances_mean.argsort()

    tick_labels_parameter_name = (
        "tick_labels"
        if parse_version(matplotlib.__version__) >= parse_version("3.9")
        else "labels"
    )
    tick_labels_dict = {tick_labels_parameter_name: X.columns[perm_sorted_idx]}
    ax.boxplot(result.importances[perm_sorted_idx].T, vert=False, **tick_labels_dict)
    ax.axvline(x=0, color="k", linestyle="--")
    return ax

def plot_permutation_importance_multicollinear():
    '''
    当特征是共线特征时，置换一个特征对模型性能的影响很小，因为它可以从相关特征中获得相同的信息。
    处理多线性特征的一种方法是对Spearman秩序相关性执行分层聚类，选择一个阈值，并从每个聚类中保留一个特征。
    :return:
    '''

    X, y = load_breast_cancer(return_X_y=True, as_frame=True) # 二分类数据集；
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    names = X_train.columns.to_list()

    # 首先，我们绘制了相关特征的热图：
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    corr = spearmanr(X).correlation

    # 确保矩阵是对称的
    corr = (corr + corr.T) / 2
    np.fill_diagonal(corr, 1)

    # 将相关矩阵转换为距离矩阵
    # 使用 Ward's linkage 进行分层聚类
    distance_matrix = 1 - np.abs(corr)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    dendro = hierarchy.dendrogram(
        dist_linkage, labels=names, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))

    ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.show()

    # 接下来，我们通过对树状图的可视化检查来手动选择一个阈值，将我们的特征分组成簇，并从每个簇中选择一个特征来保存，从我们的数据集中选择这些特征，并训练一个新的随机森林。
    cluster_ids = hierarchy.fcluster(dist_linkage, 1, criterion="distance")  # 此处的参数1，即为对树状图的可视化检查来手动选择的一个阈值
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features_names = X.columns[selected_features]

    X_train_sel = X_train[selected_features_names]
    X_test_sel = X_test[selected_features_names]

    clf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_sel.fit(X_train_sel, y_train)
    print(
        f"剔除共线特征后的测试结果: {clf_sel.score(X_test_sel, y_test):.2}"
    )

    fig, ax = plt.subplots()
    plot_permutation_importance(clf_sel, X_test_sel, y_test, ax)
    ax.set_title("特征重要性")
    ax.set_xlabel("准确度减少")
    ax.figure.tight_layout()
    plt.show()

def sort_features_train():
    # 基于学习模型的特征排序 (Model based ranking)
    # 这种方法的思路是直接使用你要用的机器学习算法，针对每个单独的特征和响应变量建立预测模型。其实Pearson相关系数等价于线性回归里的标准化回归系数。
    # 假如某个特征和响应变量之间的关系是非线性的，可以用基于树的方法（决策树、随机森林）、或者扩展的线性模型等。基于树的方法比较易于使用，因为他们对非线性关系的建模比较好，并且不需要太多的调试。但要注意过拟合问题，因此树的深度最好不要太大，再就是运用交叉验证。

    # 在 波士顿房价数据集 上使用sklearn的 随机森林回归 给出一个单变量选择的例子：
    # 下边的例子是sklearn中基于随机森林的特征重要度度量方法：
    boston = load_boston()
    X = boston["data"]
    Y = boston["target"]
    names = boston["feature_names"]
    rf = RandomForestRegressor(warm_start=True) #  warm_start 参数取值为 True，允许随机森林进行增量学习
    rf.fit(X, Y)
    # print "Features sorted by their score:"
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

    # 这里特征得分实际上采用的是 Gini Importance 。
    # 使用基于不纯度的方法的时候，要记住：
    # 1、这种方法存在 偏向 ，对具有更多类别的变量会更有利；
    # 2、对于存在关联的多个特征，其中任意一个都可以作为指示器（优秀的特征），并且一旦某个特征被选择之后，其他特征的重要度就会急剧下降，
    # 因为不纯度已经被选中的那个特征降下来了，其他的特征就很难再降低那么多不纯度了，这样一来，只有先被选中的那个特征重要度很高，其他的关联特征重要度往往较低。
    # 在理解数据时，这就会造成误解，导致错误的认为先被选中的特征是很重要的，而其余的特征是不重要的，但实际上这些特征对响应变量的作用确实非常接近的（这跟Lasso是很像的）。


    breast = load_breast_cancer()  # 二分类数据集；
    X = breast["data"]
    Y = breast["target"]
    names = breast["feature_names"]
    train_show(X, Y, names)

def test_permutation_importance():
    '''基于排列的重要性
    '''
    breast = load_breast_cancer()  # 二分类数据集；
    X = breast["data"]
    Y = breast["target"]
    names = breast["feature_names"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0, shuffle=True)
    # 将模型定义为随机森林
    model = RandomForestClassifier(class_weight="balanced_subsample", random_state=0)
    model.fit(X_train, y_train)
    print("交叉验证结果：", model.score(X_test, y_test))

    # permutation_importance函数可以计算给定数据集的估计器的特征重要性。n_repeats参数设置特征取值随机重排的次数，并返回样本的特征重要性。
    r = permutation_importance(model, X_test, y_test,
                               n_repeats=30,
                               random_state=0)
    for i in r.importances_mean.argsort()[::-1]:
        if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
            print(f"{names[i]:<8}"
                  f"{r.importances_mean[i]:.3f}"
                  f" +/- {r.importances_std[i]:.3f}")
    print("特征及其重要性：", [(r.importances_mean[i], names[i]) for i in r.importances_mean.argsort()[::-1]])

def show_feature_importance():
    '''xgboost展示特征重要性'''
    breast = load_breast_cancer()  # 二分类数据集；
    X = breast["data"]
    Y = breast["target"]
    names = breast["feature_names"]
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1898)
    xgb_model = xgb.XGBClassifier(objective="binary:logistic", random_state=1898)
    xgb_model.fit(X_train, y_train)
    xgb_model.get_booster().feature_names = list(names)
    y_pred = xgb_model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    # [[72  2]
    #  [0 97]]
    xgb_model.score(X_test, y_test)
    # Out[73]: 0.9883040935672515
    print('特征及其重要性：', xgb_model.get_booster().get_score())  # 基于不纯度的重要性倾向于高基数(取值很多)特征
    xgb.plot_importance(xgb_model)


def main():
    sort_features_train()


if __name__ == '__main__':
    main()

