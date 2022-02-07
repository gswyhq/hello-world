#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import Counter
import math
import matplotlib.pyplot as plt

from numpy.random import RandomState
from imblearn.over_sampling import RandomOverSampler  # 随机过采样
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, KMeansSMOTE, SMOTENC, SVMSMOTE  # 过采样
from imblearn.under_sampling import ClusterCentroids, RandomUnderSampler, NearMiss  # 欠采样
from imblearn.under_sampling import TomekLinks, EditedNearestNeighbours, RepeatedEditedNearestNeighbours, CondensedNearestNeighbour, OneSidedSelection, NeighbourhoodCleaningRule, InstanceHardnessThreshold  # 欠采样，无法控制数量；
from imblearn.ensemble import EasyEnsembleClassifier
from imblearn.combine import SMOTEENN, SMOTETomek

from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor


# 构造数据
from sklearn.datasets import make_classification
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

def random_over_sampler():
    """
    随机过采样（上采样），针对类别少的数据，重复采样，使之与类别多的数据数据差不多；
    原理：从样本少的类别中随机抽样，再将抽样得来的样本添加到数据集中。
    缺点：重复采样往往会导致严重的过拟合
    :return:
    """
    X, y = make_classification(n_samples=50, n_features=2, n_informative=2,
                               n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.05, 0.2, 0.75],
                               class_sep=0.8, random_state=0)

    print('原始', Counter(y))  # Counter({2: 47, 1: 2, 0: 1})

    ros = RandomOverSampler(random_state=0)
    # 过采样，（上采样）随机过采样
    X_resampled, y_resampled = ros.fit_resample(X, y)
    print('随机过采样', Counter(y_resampled))         # Counter({2: 47, 1: 47, 0: 47})
    show_image(X, y, X_resampled, y_resampled, title2='随机过采样(与原始数据重复，故看不出差异)')


def show_image(X=None, y=None, X_resampled=None, y_resampled=None, title2='', x_list=None, y_list=None, title_list=None):
    fig = plt.figure()  # 定义一个画布；以便一张画布上面，画多个子图；
    if x_list is None:
        x_list = [X, X_resampled]
        y_list = [y, y_resampled]
        title_list = ['原始数据', title2]
    if len(title_list) == 2:
        row_num = 1
        col_num = 2
    elif 2< len(title_list)<9:
        row_num = 2
        col_num = math.ceil(len(title_list)/row_num)
    else:
        row_num = math.ceil(math.sqrt(len(title_list)))
        col_num = row_num
    for index, title in enumerate(title_list, 1):
        ax0 = fig.add_subplot(row_num, col_num, index)  # row_num行, col_num 列个子图上面，添加第index个子图
        ax0.set_title(title)
        # 散点图
        X = x_list[index-1]
        y = y_list[index-1]
        scatter0=ax0.scatter(X[:, 0], X[:, 1], c=y)

        legend0 = ax0.legend(*scatter0.legend_elements(),loc="upper left", title="Classes")  # 分类标签在左上角显示；
        ax0.add_artist(legend0)

    fig.show()
    plt.pause(60)
    return fig

def smote_over_sampler():
    """
    SMOTE方法
    原理：在少数类样本之间进行插值来产生额外的样本。对于少数类样本a, 随机选择一个最近邻的样本b, 从a与b的连线上随机选取一个点c作为新的少数类样本;
    具体地，对于一个少数类样本xi使用K近邻法(k值需要提前指定)，求出离xi距离最近的k个少数类样本，其中距离定义为样本之间n维特征空间的欧氏距离。然后从k个近邻点中随机选取一个
    SMOTE会随机选取少数类样本用以合成新样本，而不考虑周边样本的情况，这样容易带来两个问题：
    如果选取的少数类样本周围也都是少数类样本，则新合成的样本不会提供太多有用信息。这就像支持向量机中远离margin的点对决策边界影响不大。
    如果选取的少数类样本周围都是多数类样本，这类的样本可能是噪音，则新合成的样本会与周围的多数类样本产生大部分重叠，致使分类困难。

    :return:
    """

    X, y = make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    print('原始', Counter(y))  # Counter({2: 930, 1: 57, 0: 13})
    X_resampled_smote, y_resampled_smote = SMOTE().fit_resample(X, y)
    print('SMOTE过采样', Counter(y_resampled_smote))  # Counter({2: 930, 1: 930, 0: 930})
    # plt.scatter(X_resampled_smote[:, 0], X_resampled_smote[:, 1], c=y_resampled_smote)
    # plt.show()

    show_image(X, y, X_resampled_smote, y_resampled_smote, title2='SMOTE过采样')

def BorderlineSMOTE_sampler():
    '''
    Border-line SMOTE
    Border-line SMOTE 算法主要是解决了，我们希望新合成的少数类样本能处于两个类别的边界附近的问题，这样往往能提供足够的信息用以分类。
    这个算法会先将所有的少数类样本分成三类，如下图所示：
    "noise" ： 所有的k近邻个样本都属于多数类
    "danger" ： 超过一半的k近邻样本属于多数类
    "safe"： 超过一半的k近邻样本属于少数类

    Border-line SMOTE算法只会从处于”danger“状态的样本中随机选择，然后用SMOTE算法产生新的样本。
    处于”danger“状态的样本代表靠近”边界“附近的少数类样本，而处于边界附近的样本往往更容易被误分类。
    因而 Border-line SMOTE 只对那些靠近”边界“的少数类样本进行人工合成样本，而 SMOTE 则对所有少
    数类样本一视同仁。

    Border-line SMOTE 分为两种: Borderline-1 SMOTE 和 Borderline-2 SMOTE。
     Borderline-1 SMOTE 在合成样本时所选的近邻是一个少数类样本，而 Borderline-2 SMOTE
     中所选的k近邻中的是任意一个样本。
    :return:
    '''
    X, y = make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    print('原始', Counter(y))  # Counter({2: 930, 1: 57, 0: 13})

    X_resampled_smote, y_resampled_smote = BorderlineSMOTE(kind="borderline-1").fit_resample(X, y)
    show_image(X, y, X_resampled_smote, y_resampled_smote, title2='borderline-1(近邻是一个少数类样本)SMOTE过采样')
    print('borderline-1(近邻是一个少数类样本)SMOTE过采样', Counter(y_resampled_smote))

    X_resampled_smote, y_resampled_smote = BorderlineSMOTE(kind="borderline-2").fit_resample(X, y)
    print('borderline-2(k近邻中的是任意一个样本)SMOTE过采样', Counter(y_resampled_smote))
    show_image(X, y, X_resampled_smote, y_resampled_smote, title2='borderline-2(k近邻中的是任意一个样本)SMOTE过采样')

def ADASYN_sampler():
    '''ADASYN-自适应合成采样
    原理：采用某种机制自动决定每个少数类样本需要产生多少合成样本，而不是像SMOTE那样对每个少数类样本合成同数量的样本。
    先确定少数样本需要合成的样本数量（与少数样本周围的多数类样本数呈正相关），然后利用SMOTE合成样本。
    缺点：ADASYN的缺点是易受离群点的影响，如果一个少数类样本的K近邻都是多数类样本，则其权重会变得相当大，进而会在其周围生成较多的样本。
    用 SMOTE 合成的样本分布比较平均，而Border-line SMOTE合成的样本则集中在类别边界处。
    ADASYN的特性是一个少数类样本周围多数类样本越多，则算法会为其生成越多的样本，
    从图中也可以看到生成的样本大都来自于原来与多数类比较靠近的那些少数类样本。

    '''
    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    print('原始', Counter(y))           # Counter({2: 930, 1: 57, 0: 13})

    X_resampled_adasyn, y_resampled_adasyn = ADASYN().fit_resample(X, y)
    print('ADASYN-自适应合成采样', Counter(y_resampled_adasyn))      # Counter({2: 930, 0: 928, 1: 923})
    show_image(X, y, X_resampled_adasyn, y_resampled_adasyn, title2='ADASYN-自适应合成采样')

def KMeansSMOTE_sampler():
    """
    原理：在使用SMOTE进行过采样之前应用KMeans聚类。
    KMeansSMOTE包括三个步骤：聚类、过滤和过采样。在聚类步骤中，使用k均值聚类为k个组。
    过滤选择用于过采样的簇，保留具有高比例的少数类样本的簇。
    然后，它分配合成样本的数量，将更多样本分配给少数样本稀疏分布的群集。
    最后，过采样步骤，在每个选定的簇中应用SMOTE以实现少数和多数实例的目标比率。
    :return:
    """
    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    print('原始', Counter(y))           # Counter({2: 930, 1: 57, 0: 13})

    X_resampled, y_resampled = KMeansSMOTE().fit_resample(X, y)
    print('KMeansSMOTE过采样', Counter(y_resampled))      # Counter({2: 930, 0: 928, 1: 923})
    show_image(X, y, X_resampled, y_resampled, title2='KMeansSMOTE过采样')

def SMOTENC_sampler():
    """
    可处理离散数据特征
    """
    X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

    print(f'原始数据分类 {{Counter(y)}}')

    # 将最后两列模拟为分类特征
    X[:, -2:] = RandomState(10).randint(0, 4, size=(1000, 2))
    sm = SMOTENC(random_state=42, categorical_features=[18, 19])
    X_res, y_res = sm.fit_resample(X, y)

    print('SMOTENC过采样', Counter(y_res))      # Counter({2: 930, 0: 928, 1: 923})
    show_image(X, y, X_res, y_res, title2='SMOTENC过采样')

def SVMSMOTE_sampler():
    """
    使用支持向量机分类器产生支持向量然后再生成新的少数类样本，然后使用SMOTE合成样本
    """
    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    print('原始', Counter(y))           # Counter({2: 930, 1: 57, 0: 13})

    X_resampled, y_resampled = SVMSMOTE().fit_resample(X, y)
    print('SVMSMOTE过采样', Counter(y_resampled))      # Counter({2: 930, 0: 928, 1: 923})
    show_image(X, y, X_resampled, y_resampled, title2='SVMSMOTE过采样')

def ClusterCentroids_sampler():
    '''下采样-降采样,欠采样：类别少的，全部采样，类别多的样本，采样一部分，最终使二者数量差不多；
    ClusterCentroids（可控制欠采样数量）
    原理：利用kmeans将对各类样本分别聚类，利用质心替换整个簇的样本。
    给定数据集S, 原型生成算法将生成一个子集S’, 其中|S’| < |S|, 但是子集并非来自于原始数据集.
    意思就是说: 原型生成方法将减少数据集的样本数量, 剩下的样本是由原始数据集生成的, 而不是直接来源于原始数据集
    '''
    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0, n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    cc = ClusterCentroids(random_state=0)
    X_resampled, y_resampled = cc.fit_resample(X, y)
    print('原始', Counter(y))
    print('ClusterCentroids欠采样', Counter(y_resampled))
    show_image(X, y, X_resampled, y_resampled, title2='ClusterCentroids欠采样')

def RandomUnder_sampler():
    """随机欠采样,可控制欠采样数量
    原理：从多数类样本中随机选取一些剔除掉。
    缺点：被剔除的样本可能包含着一些重要信息，致使学习出来的模型效果不好。
    """

    X, y = make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print('原始', Counter(y))
    print('随机欠采样', Counter(y_resampled))
    show_image(X, y, X_resampled, y_resampled, title2='随机欠采样')

def NearMiss_sampler():
    '''
    原理：从多数类样本中选取最具代表性的样本用于训练，主要是为了缓解随机欠采样中的信息丢失问题。可控制欠采样数量。
    NearMiss采用一些启发式的规则来选择样本，根据规则的不同可分为3类,通过设定version参数来确定：
    NearMiss-1：选择到最近的K个少数类样本平均距离最近的多数类样本
    NearMiss-2：选择到最远的K个少数类样本平均距离最近的多数类样本
    NearMiss-3：对于每个少数类样本选择K个最近的多数类样本，目的是保证每个少数类样本都被多数类样本包围
    NearMiss-1和NearMiss-2的计算开销很大，因为需要计算每个多类别样本的K近邻点。另外，NearMiss-1易受离群点的影响
    '''

    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    # 不同的NearMiss类别通过version参数来设置
    nml = NearMiss(version=1)
    X_resampled1,y_resampled1 = nml.fit_resample(X,y)
    print('原始', Counter(y))
    print('NearMiss-1欠采样', Counter(y_resampled1))
    nml = NearMiss(version=2)
    X_resampled2,y_resampled2 = nml.fit_resample(X,y)
    print('原始', Counter(y))
    print('NearMiss-2欠采样', Counter(y_resampled2))
    nml = NearMiss(version=3)
    X_resampled3,y_resampled3 = nml.fit_resample(X,y)
    print('原始', Counter(y))
    print('NearMiss-3欠采样', Counter(y_resampled3))

    show_image(x_list=[X, X_resampled1, X_resampled2, X_resampled3], y_list=[y, y_resampled1, y_resampled2, y_resampled3], title_list=['原始数据',
                    'NearMiss-1欠采样\n选择到最近的K个少数类样本平均距离最近的多数类样本',
                    'NearMiss-2欠采样\n选择到最远的K个少数类样本平均距离最近的多数类样本',
                    'NearMiss-3欠采样\n对于每个少数类样本选择K个最近的多数类样本，目的是保证每个少数类样本都被多数类样本包围'])

def TomekLinks_sampler():
    """
    数据清洗方法，无法控制欠采样数量
    原理：Tomek Link表示不同类别之间距离最近的一对样本，即这两个样本互为最近邻且分属不同类别。
    这样如果两个样本形成了一个Tomek Link，则要么其中一个是噪音，要么两个样本都在边界附近。
    这样通过移除Tomek Link就能“清洗掉”类间重叠样本，使得互为最近邻的样本皆属于同一类别，从而能更好地进行分类。
    TomekLinks函数中的auto参数控制Tomek’s links中的哪些样本被剔除.
     默认的 sampling_strategy =‘auto’ 移除多数类的样本, 当 sampling_strategy ='all' 时, 两个样本均被移除.
    :return:
    """

    X, y = make_classification(n_samples=300, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = TomekLinks(sampling_strategy='auto')
    X_resampled, y_resampled = rus.fit_resample(X, y)

    rus1 = TomekLinks(sampling_strategy='all')
    X_resampled1, y_resampled1 = rus1.fit_resample(X, y)
    print('原始', Counter(y))
    print('TomekLinks欠采样，移除多数类的样本', Counter(y_resampled))
    print('TomekLinks欠采样，两个样本均被移除', Counter(y_resampled1))
    show_image(x_list=[X, X_resampled, X_resampled1], y_list=[y, y_resampled,y_resampled1], title_list=['原始', 'TomekLinks欠采样(移除多数类的样本)', 'TomekLinks欠采样(两个样本均被移除)'])

def EditedNearestNeighbours_sampler():
    """
    数据清洗方法，无法控制欠采样数量
    原理：kind_sel='mode'时，如果多数类样本其K个近邻点有超过一半不属于多数类，则该样本被删除；针对多数类样本其周围少数类样本占多数才会删除。
    kind_sel='all'时，如果多数类样本其K个近邻点不全部属于多数类，则该样本被删除；针对多数类样本其周围只有有少数类样本就会删除。
    策略"all"将不如"mode"那么保守。 因此，一般在`kind_sel="all"`时会移除更多的样本。
    :return:
    """

    X, y = make_classification(n_samples=300, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = EditedNearestNeighbours(kind_sel='all')
    X_resampled, y_resampled = rus.fit_resample(X, y)

    rus1 = EditedNearestNeighbours(kind_sel='mode')
    X_resampled1, y_resampled1 = rus1.fit_resample(X, y)
    print('原始', Counter(y))
    print('EditedNearestNeighbours欠采样，k近邻全部不属于多数类则剔除', Counter(y_resampled))
    print('EditedNearestNeighbours欠采样，k近邻有超过一半不属于多数类则剔除', Counter(y_resampled1))
    show_image(x_list=[X, X_resampled, X_resampled1], y_list=[y, y_resampled,y_resampled1], title_list=['原始',
            'EditedNearestNeighbours欠采样(k近邻全部不属于多数类则剔除)', 'EditedNearestNeighbours欠采样(k近邻有超过一半不属于多数类则剔除)'])

def RepeatedEditedNearestNeighbours_sampler():
    """
    数据清洗方法，无法控制欠采样数量
    原理：重复EditedNearestNeighbours多次（参数 max_iter 控制迭代次数）
    kind_sel='mode'时，如果多数类样本其K个近邻点有超过一半不属于多数类，则该样本被删除；针对多数类样本其周围少数类样本占多数才会删除。
    kind_sel='all'时，如果多数类样本其K个近邻点不全部属于多数类，则该样本被删除；针对多数类样本其周围只有有少数类样本就会删除。
    策略"all"将不如"mode"那么保守。 因此，一般在`kind_sel="all"`时会移除更多的样本。
    :return:
    """

    X, y = make_classification(n_samples=150, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = RepeatedEditedNearestNeighbours(kind_sel='all', max_iter=200)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    rus1 = RepeatedEditedNearestNeighbours(kind_sel='mode', max_iter=200)
    X_resampled1, y_resampled1 = rus1.fit_resample(X, y)
    print('原始', Counter(y))
    print('RepeatedEditedNearestNeighbours欠采样，k近邻不全部属于多数类则剔除', Counter(y_resampled))
    print('RepeatedEditedNearestNeighbours欠采样，k近邻有超过一半不属于多数类则剔除', Counter(y_resampled1))
    show_image(x_list=[X, X_resampled, X_resampled1], y_list=[y, y_resampled,y_resampled1], title_list=['原始',
            'RepeatedEditedNearestNeighbours欠采样\n(k近邻全部不属于多数类则剔除)', 'RepeatedEditedNearestNeighbours欠采样\n(k近邻有超过一半不属于多数类则剔除)'])

def CondensedNearestNeighbour_sampler():
    """数据清洗方法，无法控制欠采样数量，数据量大时耗时较久
    压缩最近邻(Condensed Nearest Neighbour)是一种贪心算法，删除那些不参入定义判别式的样本而不增加训练误差；旨在最小化训练误差和存放子集规模度量的复杂度。
    使用1近邻的方法来进行迭代, 来判断一个样本是应该保留还是剔除, 具体的实现步骤如下:
    1)集合C: 所有的少数类样本;
    2)以随机次序逐个扫描多数类样本，扫描样本x与集合C组合成集合S=C+x;
    3)使用集合S训练一个1-NN的分类器, 对集合S中的样本进行分类;
    4)将集合S中错分的样本加入集合C, 即若1NN分类结果不是x,则将x加入C;
    5)重复上述2、3、4过程, 直到没有样本再加入到集合C.
    使用该方法时，应当扫描数据多遍，知道没有实例再添加到C中。
    CondensedNearestNeighbour方法对噪音数据是很敏感的, 也容易加入噪音数据到集合C中.
    """

    X, y = make_classification(n_samples=150, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = CondensedNearestNeighbour()
    X_resampled, y_resampled = rus.fit_resample(X, y)

    print('原始', Counter(y)) # Counter({2: 103, 1: 29, 0: 18})
    print('压缩最近邻(CondensedNearestNeighbour)欠采样', Counter(y_resampled))  # Counter({2: 19, 0: 18, 1: 6})
    show_image(x_list=[X, X_resampled], y_list=[y, y_resampled], title_list=['原始', '压缩最近邻(CondensedNearestNeighbour)欠采样'])

def OneSidedSelection_sampler():
    """ （数据清洗方法，无法控制欠采样数量）
    原理：在压缩最近邻CondensedNearestNeighbour的基础上使用 TomekLinks 方法来剔除噪声数据(多数类样本).
    """

    X, y = make_classification(n_samples=150, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = OneSidedSelection()
    X_resampled, y_resampled = rus.fit_resample(X, y)

    print('原始', Counter(y)) # Counter({2: 103, 1: 29, 0: 18})
    print('OneSidedSelection欠采样', Counter(y_resampled))  # Counter({2: 19, 0: 18, 1: 6})
    show_image(x_list=[X, X_resampled], y_list=[y, y_resampled], title_list=['原始', 'OneSidedSelection欠采样'])

def NeighbourhoodCleaningRule_sampler():
    """ （数据清洗方法，无法控制欠采样数量）
    该算法是利用k近邻的方法寻找重叠区域，具体步骤如下：
    1，在训练集中选择样本E;
    2，寻找样本E的k个近邻；
    3，若E的样本类别，在k个近邻样本中不是最多，则将E剔除。
    如此反复，直至无可删除的样本。
    策略"mode"将不如"all"那么保守。 因此，一般在`kind_sel="mode"`时会移除更多的样本。
    """
    X, y = make_classification(n_samples=50, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.05, 0.1, 0.85],
                               random_state=500)

    rus = NeighbourhoodCleaningRule(kind_sel='all')
    X_resampled, y_resampled = rus.fit_resample(X, y)

    rus1 = NeighbourhoodCleaningRule(kind_sel='mode')
    X_resampled1, y_resampled1 = rus1.fit_resample(X, y)
    print('原始', Counter(y))   # Counter({2: 74, 1: 21, 0: 5})
    print('NeighbourhoodCleaningRule欠采样，all模式', Counter(y_resampled))  # Counter({2: 67, 1: 12, 0: 5})
    print('NeighbourhoodCleaningRule欠采样，mode模式', Counter(y_resampled1))  # Counter({2: 62, 1: 12, 0: 5})
    show_image(x_list=[X, X_resampled, X_resampled1], y_list=[y, y_resampled,y_resampled1], title_list=['原始',
            'NeighbourhoodCleaningRule欠采样(all模式)', 'NeighbourhoodCleaningRule欠采样(mode模式)'])

def InstanceHardnessThreshold_sampler():
    """ （数据清洗方法，无法控制欠采样数量）
    在数据上运用一种分类器, 然后将概率低于阈值的样本剔除掉
    """

    X, y = make_classification(n_samples=150, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                               n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.1, 0.2, 0.7],
                               random_state=500)

    rus = InstanceHardnessThreshold()
    X_resampled, y_resampled = rus.fit_resample(X, y)

    print('原始', Counter(y)) # Counter({2: 103, 1: 29, 0: 18})
    print('InstanceHardnessThreshold欠采样', Counter(y_resampled))  # Counter({2: 29, 0: 18, 1: 6})
    show_image(x_list=[X, X_resampled], y_list=[y, y_resampled], title_list=['原始', 'InstanceHardnessThreshold欠采样'])

def SMOTEENN_sampler():
    '''
    过采样与下采样结合
    1）SMOTEENN
    # 先过采样后清洗
    :return:
    '''

    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=3,
                               n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    smote_enn = SMOTEENN(random_state=0)
    X_resampled,y_resampled = smote_enn.fit_resample(X,y)
    print('原始', Counter(y))  # Counter({2: 930, 1: 57, 0: 13})
    print('先SMOTEENN过采样后清洗', Counter(y_resampled))             # Counter({1: 783, 0: 763, 2: 627})
    show_image(x_list=[X, X_resampled, ],
               y_list=[y, y_resampled],
               title_list=['原始数据', '先SMOTEENN过采样后清洗'])

def SMOTETomek_sampler():
    '''SMOTETomek采样'''

    X, y =  make_classification(n_samples=1000, n_features=2,
                               n_informative=2, n_redundant=0, n_repeated=0,
                                n_classes=3,n_clusters_per_class=1,
                               weights=[0.01, 0.05, 0.94],
                               random_state=500)

    smote_tomek = SMOTETomek(random_state=0)

    X_resampled,y_resampled = smote_tomek.fit_resample(X,y)

    print('原始', Counter(y))  # Counter({2: 930, 1: 57, 0: 13})
    print('SMOTETomek采样', Counter(y_resampled))             # Counter({0: 890, 1: 885, 2: 863})
    show_image(x_list=[X, X_resampled, ],
               y_list=[y, y_resampled],
               title_list=['原始数据', 'SMOTETomek采样'])

def train3():
    ''' 针对不平衡样本，也直接在模型训练的时候，采用过采样模型训练'''
    model_bbc = BalancedBaggingClassifier(base_estimator=XGBClassifier(objective='binary:logistic'),
                                      n_estimators=15,
                                      sampling_strategy='majority',
                                      replacement=False,
                                      random_state=0)
    model_bbc.fit(X, Y)

    probs2 = model_bbc.predict_proba(np.array(X_test))
    y_score2 = probs2[:, 1]
    print("重采样检测出阳性数量：{}， 总阳性数量：{}, 检出比例：{:.4f}".format(len([y for y in y_score2 if y >=0.5]), len(y_score2), len([y for y in y_score2 if y >=0.5])/ len(y_score2) ))

def main():
    # random_over_sampler()
    # smote_over_sampler()
    # BorderlineSMOTE_sampler()
    # ADASYN_sampler()
    # KMeansSMOTE_sampler()
    # SMOTENC_sampler()
    # SVMSMOTE_sampler()
    # ClusterCentroids_sampler()
    # RandomUnder_sampler()
    # NearMiss_sampler()
    # TomekLinks_sampler()
    EditedNearestNeighbours_sampler()
    # RepeatedEditedNearestNeighbours_sampler()
    # CondensedNearestNeighbour_sampler()
    # OneSidedSelection_sampler()
    # NeighbourhoodCleaningRule_sampler()
    # InstanceHardnessThreshold_sampler()

    # SMOTEENN_sampler()
    # SMOTETomek_sampler()

if __name__ == '__main__':
    main()
