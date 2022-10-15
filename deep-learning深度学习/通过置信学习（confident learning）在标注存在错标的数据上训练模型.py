#!/usr/bin/python3
# coding: utf-8

# 很多时候训练模型所用的数据都是人工标注的，那么往往甚至不可避免的存在一些错标的数据，尤其是标注的准则或者流程不是很完善时，错标就更常见。
# 如果直接用这些存在错标的数据训练模型，那么模型的上限将受限与标注的准确率。
# 如何区分正确的label与错误的label上，目前主要有三种方法：
# 直接建模：建立一个概率模型，直接估计每个样本标注正确或错误的概率，剔除正确率低的数据；
# 迭代法：根据模型预测的损失初选一些正确或错误的样本，然后过滤掉错误的样本，在此基础上重新训练并进行反复迭代；
# 加权法：接受所有样本，只是根据每个样本的正确率赋予不同的权重，构建一个加权的loss function进行训练。
#
# 置信学习综合运用了上述处理三种方法，主要是通过估计noisy label(噪音标签)与真实label的联合分布实现的，实现正确与错误样本的区分的。
# 其基本假设为：数据错标的概率与类别有关，但与数据本身无关，如美洲豹可能被错标为美洲虎的可能性较高，但不大可能被错标为浴缸。
# 在进行统计建模时，假设存在一个潜在的真实label，然后使用计数法估计真实label与观察到的noisy label的联合分布。
# 得到每个类别错标的联合分布后，根据每个样本的prediction score进行过滤，然后基于每个样本所属的类别的confidence score计算权重进行训练，
# 主要包含以下几个步骤：
# 1.初步训练：先用原始数据通过交叉验证的方式训练模型，并对每一个样本进行打分，得到每个样本对每个类别的分数；
# 2.计数（Count）：这一步主要是用每个样本的得分来预估其标注正确的置信度；
# 3.净化（Clean）：过滤掉标注正确的置信度较低的样本；
# 进行K轮交叉验证计算每个样本的类别概率
# 把数据分为K份（这里我使用的是5。最好大于5）
# 其中选一份为测试集， 其余K-1份为 训练集，训练一个模型
# 把测试集 输入 训练的模型， 得到测试集每个样本的预测每个类的概率 得到我们需要的 每个样本在每个类别的概率
# 测试集 每个样本 自身的label 则是 每个样本实际属于哪个类别
# 4.重新训练（Re-train）：用过滤过后的样本重新训练模型。

# 通过cleanlab库寻找噪声标签from cleanlab.filter import find_label_issues
#
# ranked_label_issues = find_label_issues(
#     y,
#     pred_probs,
#     filter_by='prune_by_class', # 可以通过输入filter_by参数选择筛选方法, 如： predicted_neq_given
# )

# 来源： https://github.com/cleanlab/examples/blob/master/classifier_comparison/classifier_comparison.ipynb

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from cleanlab.classification import CleanLearning
from cleanlab.benchmarking.noise_generation import generate_noise_matrix_from_trace
from cleanlab.benchmarking.noise_generation import generate_noisy_labels
from cleanlab.internal.util import print_noise_matrix
import copy
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

# Silence neural network SGD convergence warnings.
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import keras

from keras.models import Model
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, TimeDistributed

# 一行代码识别训练数据中的脏数据（比如标记错误的数据）
# from cleanlab.classification import CleanLearning
# # labels = 噪声标签
# issues = CleanLearning(yourFavoriteModel).find_label_issues(data, labels)

def make_meshgrid(x, y, h=0.02):
    """Create a mesh of points to plot in

    Parameters
    ----------
    x: data to base x-axis meshgrid on
    y: data to base y-axis meshgrid on
    h: stepsize for meshgrid, optional

    Returns
    -------
    xx, yy : ndarray
    """
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(ax, clf, xx, yy, **params):
    """Plot the decision boundaries for a classifier.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def make_linearly_seperable_dataset(n_classes=3, n_samples=300):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        random_state=1,
        n_clusters_per_class=1,
        n_classes=n_classes,
    )
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    return (X, y)
#
# def build_model():
#     input_1 = Input(shape=(2), name="input_1")
#     # input_2 = Input(shape=(1,), name="input_2")
#
#     # concat = Concatenate()([input_1, input_2])
#     dense1 = Dense(32, kernel_initializer="uniform", activation='relu')(input_1)
#     dense2 = Dense(1, kernel_initializer="uniform", activation='sigmoid', name="output")(dense1)
#     inputs = [input_1]
#     outputs = [dense2]
#     model = Model(inputs, outputs)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     model.summary()
#     return model
#
# keras.backend.clear_session()
# dnn_model = build_model()
# dnn_model.fit({"input_1": np.array([t[0] for t in X_train]), "input_2": np.array([t[1] for t in X_train])}, y_train)
# 下面的代码将生成非稀疏噪声矩阵（所有非零噪声率）。

# 初始化.
# 设置噪声矩阵(noise matrix)的稀疏度.
FRAC_ZERO_NOISE_RATES = 0.0  # 可以增加到 0.5
# 正确标签的比例.
avg_trace = 0.65  # ~35% 错误标签。增加会使问题变得更容易.
# 数据集的数量.
dataset_size = 400  # 可以尝试更少或更多数据.
# 网格中步长.
h = 0.02

names = [
    "Naive Bayes",
    "LogisticReg",
    "K-NN (K=3)",
    "Linear SVM",
    "RBF SVM",
    "Rand Forest",
    "Neural Net",
    "AdaBoost",
    "QDA",
]

classifiers = [
    GaussianNB(),
    LogisticRegression(random_state=0, solver="lbfgs", multi_class="auto"),
    KNeighborsClassifier(n_neighbors=3),
    SVC(kernel="linear", C=0.025, probability=True, random_state=0),
    SVC(gamma=2, C=1, probability=True, random_state=0),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(
        alpha=1,
        random_state=0,
    ),
    AdaBoostClassifier(random_state=0),
    QuadraticDiscriminantAnalysis(),
]

dataset_names = [
    "Linear (m = 4)",
    "Linear (m = 3)",
    "Moons (m = 2)",
    "Circles (m = 2)",
]

# Hyper-parameters for CleanLearning() classifier
params = {
    "cv_n_folds": [5],  # Default. Keep as default for fair comparison.
    "prune_method": ["prune_by_noise_rate", "prune_by_class", "both"],
    "converge_latent_estimates": [False, True],
}

experiments = [
    "no_label_errors",
    "label_errors_no_cl",
    "label_errors_with_cl",
]

datasets = [
    make_linearly_seperable_dataset(n_classes=4, n_samples=4 * dataset_size),
    make_linearly_seperable_dataset(n_classes=3, n_samples=3 * dataset_size),
    make_moons(n_samples=2 * dataset_size, noise=0.3, random_state=0),  # 2 classes
    make_circles(
        n_samples=2 * dataset_size, noise=0.2, factor=0.5, random_state=1
    ),  # 2 classes
]

results = []
# 遍历数据集
for ds_cnt, ds in enumerate(datasets):
    # 数据集预处理，划分为训练集、测试集
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=0
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=1
    )
    num_classes = len(np.unique(y_train))
    print(
        "运行数据集：",
        ds_cnt + 1,
        "\n类别数： =",
        num_classes,
        "\n训练集样本数： =",
        len(X_train),
    )

    # CONFIDENT LEARNING COMPONENT

    np.random.seed(seed=0)

    py = np.bincount(y_train) / float(len(y_train))
    # 根据描述的错误标签比例，生成噪声矩阵
    noise_matrix = generate_noise_matrix_from_trace(
        K=num_classes,
        trace=num_classes * avg_trace,
        py=py,
        frac_zero_noise_rates=FRAC_ZERO_NOISE_RATES,
    )
    print_noise_matrix(noise_matrix)
    np.random.seed(seed=1)
    # 创建噪声标签(noisy labels). 这种是确切的噪声矩阵 noise_matrix.
    y_train_w_errors = generate_noisy_labels(y_train, noise_matrix)

    clf_results = {}
    # 遍历分类器
    for name, clf in zip(names, classifiers):
        # 创建分类器的四个副本
        # perf_label_clf - 无噪标签(noise-free labels)上训练
        # noisy_clf - 噪声标签(noisy labels)上训练
        # noisy_clf_w_cl - 使用CleanLearning训练噪声标签(noisy labels)

        clfs = [copy.deepcopy(clf) for i in range(len(experiments))]
        perf_label_clf, noisy_clf, noisy_clf_w_cl = clfs
        # 分类器 (用没有错误标签数据训练)
        perf_label_clf.fit(X_train, y_train)
        perf_label_score = perf_label_clf.score(X_test, y_test)
        # 分类器 (用含错误标签数据训练)
        noisy_clf.fit(X_train, y_train_w_errors)
        noisy_score = noisy_clf.score(X_test, y_test)
        # 分类器 + CL (用含错误标签数据训练)
        cl = CleanLearning(noisy_clf_w_cl, verbose=False)  # 使用清理过的数据进行机器学习
        cl.fit(X_train, y_train_w_errors)
        noisy_score_w_cl = cl.clf.score(X_test, y_test)

        # 将每个分类器的结果存于字典中 key = clf_name.
        clf_results[name] = {
            "clfs": clfs,
            "perf_label_score": perf_label_score,
            "noisy_score": noisy_score,
            "noisy_score_w_cl": noisy_score_w_cl,
        }

    results.append(
        {
            "X": X,
            "X_train": X_train,
            "y_train": y_train,
            "y_train_w_errors": y_train_w_errors,
            "num_classes": num_classes,
            "py": py,
            "noise_matrix": noise_matrix,
            "clf_results": clf_results,
        }
    )

# 展示各个数据集，各个分类器效果
fig = plt.figure(figsize=[300, 100])
def sub_ax(ax, idx, ds):
    clf_name = names[idx]
    clf_results = ds['clf_results']
    perf_label_score = clf_results[clf_name]["perf_label_score"]
    noisy_score = clf_results[clf_name]["noisy_score"]
    noisy_score_w_cl = clf_results[clf_name]["noisy_score_w_cl"]

    ax.scatter(['noisy_score', 'noisy_score_w_cl', 'perf_label_score'], [noisy_score, noisy_score_w_cl, perf_label_score], cmap='coolwarm', s=30)
    ax.set_title("数据集：{}， 分类器：{}".format(ds_idx, clf_name))

idx = 0
for row_idx, col in enumerate(names, 0):
    for ds_idx, ds in enumerate(results):
        ax = fig.add_subplot(len(names), len(results), idx+1)
        idx += 1
        sub_ax(ax, row_idx, ds)

plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=0.5) # 这个是调整宽度距离和高度距离；调整子图间的间隙


# 图表展示：
save_figures = False  # Save figure as png

for e_i, experiment in enumerate(
    [
        "no_label_errors",
        "label_errors_no_cl",
        "label_errors_with_cl",
    ]
):
    print(
        "实验： " + str(e_i + 1) + ":",
        "绘制的决策边界是: ",
        " ".join(experiment.split("_")).capitalize(),
    )
    print("=" * 80)
    figure = plt.figure(figsize=(27, 12))
    i = 1

    # 遍历数据集
    for ds_cnt, ds in enumerate(datasets):
        # 获取数据用于绘图.
        for key, val in results[ds_cnt].items():
            exec(key + "=val")

        # 先绘制数据集
        X0, X1 = X[:, 0], X[:, 1]
        xx, yy = make_meshgrid(X0, X1)
        cm = plt.cm.coolwarm
        cm = plt.cm.nipy_spectral
        cm = plt.cm.Spectral
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        ax.set_ylabel(dataset_names[ds_cnt], fontsize=18)
        if ds_cnt == 0:
            ax.set_title("Dataset", fontsize=18)
        # 绘制训练集的点
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm, edgecolors="k")
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        # 遍历分类器
        for name, clf in zip(names, classifiers):
            # 获取分类器计算结果
            for key, val in clf_results[name].items():
                exec(key + "=val")

            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

            # 绘制边界的条件
            clf = clfs[e_i]

            # 为网格中点 [x_min, x_max]x[y_min, y_max]，绘制决策边界
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])

            plot_contours(ax, clf, xx, yy, cmap=cm, alpha=0.5)

            # Plot the training points
            ax.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c=y_train if experiment == "no_label_errors" else y_train_w_errors,
                cmap=cm,
                edgecolors="k",
            )
            if experiment != "no_label_errors":
                # Plot the label errors
                ax.scatter(
                    X_train[y_train != y_train_w_errors][:, 0],
                    X_train[y_train != y_train_w_errors][:, 1],
                    edgecolors="lime",
                    s=60,
                    facecolors="none",
                    alpha=0.55,
                    linewidth=2,
                )

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name, fontsize=18)
            ax.text(
                xx.min() + 1.5,
                yy.max() - 0.7,
                ("%.2f" % perf_label_score).lstrip("0"),
                size=20,
                horizontalalignment="right",
                color="black",
            )
            ax.text(
                xx.max() - 0.2,
                yy.max() - 0.7,
                ("%.2f" % noisy_score).lstrip("0"),
                size=20,
                horizontalalignment="right",
                color="white",
            )
            ax.text(
                xx.mean() + 0.75,
                yy.max() - 0.7,
                ("%.2f" % noisy_score_w_cl).lstrip("0"),
                size=20,
                horizontalalignment="right",
                color="blue",
            )
            i += 1

    plt.tight_layout()
    if save_figures:
        _ = plt.savefig(
            "./img/{}.png".format(experiment), pad_inches=0.0, bbox_inches="tight"
        )

    plt.show()

def main():
    pass


if __name__ == '__main__':
    main()