#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 对图谱上的节点进行分类；
# 对图谱数据要求：
# 1.节点间的连接关系；
# 2.节点对应的特征向量；
# 3.节点的主题标签；

# 更多示例见：https://github.com/danielegrattarola/spektral/tree/master/examples/node_prediction

import os
import networkx as nx
import scipy.sparse as sp
import numpy as np
import pandas as pd
from collections import defaultdict
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from spektral.data.loaders import SingleLoader
from spektral.datasets.citation import Citation
from spektral.data import Dataset, Graph
from spektral.layers import GCNConv
from spektral.models.gcn import GCN
from spektral.transforms import LayerPreprocess
from spektral.datasets.utils import DATASET_FOLDER
from spektral.utils.io import load_binary

learning_rate = 1e-2
seed = 0
epochs = 200
patience = 10
data = "cora"

tf.random.set_seed(seed=seed)  # make weight initialization reproducible

def _idx_to_mask(idx, l):
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def _preprocess_features(features):
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def _read_file(path, name, suffix):
    full_fname = os.path.join(path, "ind.{}.{}".format(name, suffix))
    if suffix == "test.index":
        return np.loadtxt(full_fname)

    return load_binary(full_fname)

class MyDataset(Dataset):

    def __init__(
        self, name='cora', random_split=False, normalize_x=False, dtype=np.float32, **kwargs
    ):
        self.name = name.lower()
        self.random_split = random_split
        self.normalize_x = normalize_x
        self.mask_tr = self.mask_va = self.mask_te = None
        self.dtype = dtype
        super().__init__(**kwargs)

    def read(self):
        USERNAME = os.getenv("USERNAME")
        data_dir = rf"D:\Users\{USERNAME}\data\cora"

        # 读取论文引用关系数据；
        citations = pd.read_csv(
            os.path.join(data_dir, "cora.cites"),
            sep="\t",
            header=None,
            names=["target", "source"],
        )

        # 读取每篇论文特征及其主题
        papers = pd.read_csv(
            os.path.join(data_dir, "cora.content"),
            sep="\t",
            header=None,
            names=["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"],
        )

        class_values = sorted(papers["subject"].unique())
        class_idx = {name: id for id, name in enumerate(class_values)}
        paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

        papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
        citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
        citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
        papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])

        # 拆分数据集, [140, 500, 1000]
        y = to_categorical(papers.sort_values("paper_id")["subject"].to_numpy())

        idx_tr = np.arange(140)
        idx_va = np.arange(140, 140 + 500)
        idx_te = np.arange(1708, 2708)
        # Define graph, namely an edge tensor and a node feature tensor
        # edges = tf.convert_to_tensor(citations[["target", "source"]])
        edges = [(s, t) for s, t in citations[["source", "target"]].values]
        x = papers.sort_values("paper_id").iloc[:, 1:-1]

        if self.normalize_x:
            print("Pre-processing node features")
            x = _preprocess_features(x.astype("float32"))

        # 邻接矩阵(Adjacency matrix)
        graph1 = defaultdict(list)
        [graph1.setdefault(source, []) for source in range(0, 2708)]  # 要求 graph 的key是由小到大排序，否则存在问题；
        for source, target in edges:
            graph1[source].append(target)
        a = nx.adjacency_matrix(nx.from_dict_of_lists(graph1))  # CSR
        a.setdiag(0)
        a.eliminate_zeros()

        # Train/valid/test masks
        self.mask_tr = _idx_to_mask(idx_tr, y.shape[0])
        self.mask_va = _idx_to_mask(idx_va, y.shape[0])
        self.mask_te = _idx_to_mask(idx_te, y.shape[0])

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]


class MyCitation(Dataset):
    suffixes = ["x", "y", "tx", "ty", "allx", "ally", "graph", "test.index"]

    def __init__(
        self, name='cora', random_split=False, normalize_x=False, dtype=np.float32, **kwargs
    ):
        self.name = name.lower()
        self.random_split = random_split
        self.normalize_x = normalize_x
        self.mask_tr = self.mask_va = self.mask_te = None
        self.dtype = dtype
        super().__init__(**kwargs)

    @property
    def path(self):
        return os.path.join(DATASET_FOLDER, "Citation", self.name)

    def read(self):
        objects = [_read_file(self.path, self.name, s) for s in self.suffixes]
        objects = [o.A if sp.issparse(o) else o for o in objects]
        x, y, tx, ty, allx, ally, graph, idx_te = objects

        # Public Planetoid splits. This is the default
        idx_tr = np.arange(y.shape[0])
        idx_va = np.arange(y.shape[0], y.shape[0] + 500)
        idx_te = idx_te.astype(int)
        idx_te_sort = np.sort(idx_te)

        x = np.vstack((allx, tx))
        y = np.vstack((ally, ty))
        x[idx_te, :] = x[idx_te_sort, :]
        y[idx_te, :] = y[idx_te_sort, :]

        # Adjacency matrix
        a = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # CSR
        a.setdiag(0)
        a.eliminate_zeros()

        # Train/valid/test masks
        self.mask_tr = _idx_to_mask(idx_tr, y.shape[0])
        self.mask_va = _idx_to_mask(idx_va, y.shape[0])
        self.mask_te = _idx_to_mask(idx_te, y.shape[0])

        return [
            Graph(
                x=x.astype(self.dtype),
                a=a.astype(self.dtype),
                y=y.astype(self.dtype),
            )
        ]

# Load data
# https://github.com/tkipf/gcn/tree/master/gcn/data
# dataset = Citation(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
# dataset = MyCitation(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
# Epoch 46/200
# 1/1 [==============================] - 0s 79ms/step - loss: 0.2086 - acc: 0.9929 - val_loss: 0.8236 - val_acc: 0.7680
dataset = MyDataset(data, normalize_x=True, transforms=[LayerPreprocess(GCNConv)])
# Epoch 221/500
# 1/1 [==============================] - 0s 92ms/step - loss: 0.5527 - acc: 0.9714 - val_loss: 1.1864 - val_acc: 0.7340

# 我们将二进制掩码转换为样本权重，以便计算节点上的平均损失
def mask_to_weights(mask):
    '''
    若mask值为真，则其权重为1/所有为真的总数，若mask值为假，则其权重为0；
    '''
    return mask.astype(np.float32) / np.count_nonzero(mask)

# train_mask：训练集的mask向量，标识哪些节点属于训练集。
# val_mask：验证集的mask向量，标识哪些节点属于验证集。
# test_mask：测试集的mask向量，表示哪些节点属于测试集。
# x：输入的特征矩阵。y：节点标签。
weights_tr, weights_va, weights_te = (
    mask_to_weights(mask)
    for mask in (dataset.mask_tr, dataset.mask_va, dataset.mask_te)
)

# dataset.n_labels, 类别数，共7类；
model = GCN(n_labels=dataset.n_labels)
model.compile(
    optimizer=Adam(learning_rate),
    loss=CategoricalCrossentropy(reduction="sum"),
    weighted_metrics=["acc"],
)

# Train model
loader_tr = SingleLoader(dataset, sample_weights=weights_tr)  # 通过 sample_weights 参数将训练集对应的样本权重设置为非零，其他的样本（验证集、测试集）权重设置为0；
loader_va = SingleLoader(dataset, sample_weights=weights_va)  # 通过 sample_weights 参数将验证集对应的样本权重设置为非零，其他的样本（训练集、测试集）权重设置为0；
model.fit(
    loader_tr.load(),
    steps_per_epoch=loader_tr.steps_per_epoch,
    validation_data=loader_va.load(),
    validation_steps=loader_va.steps_per_epoch,
    epochs=epochs,
    callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)],
)

# Evaluate model
print("Evaluating model.")
loader_te = SingleLoader(dataset, sample_weights=weights_te)
eval_results = model.evaluate(loader_te.load(), steps=loader_te.steps_per_epoch)
print("Done.\n" "Test loss: {}\n" "Test accuracy: {}".format(*eval_results))


def main():
    pass


if __name__ == '__main__':
    main()
