#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/graph/mpnn-molecular-graphs/

# 实现一种称为消息传递神经网络 (MPNN) 的图神经网络 (GNN) 来预测图属性。具体来说，我们将实施 MPNN 来预测称为 血脑屏障通透性(BBBP) 的分子特性。

# RDKit是用 C++ 和 Python 编写的化学信息学和机器学习软件的集合。在本教程中，RDKit 用于方便高效地将 SMILES转换为分子对象，然后从中获得原子和键的集合。
#
# SMILES 以 ASCII 字符串的形式表示给定分子的结构。
# SMILES 字符串是一种紧凑的编码，对于较小的分子而言，它相对易于人类阅读。
# 将分子编码为字符串既减轻并促进了给定分子的数据库和/或网络搜索。
# RDKit 使用算法将给定的 SMILES 准确地转换为分子对象，然后可用于计算大量分子属性/特征。

# pip -q install rdkit

import os

# Temporary suppress tf logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
USERNAME = os.getenv("USERNAME")

# rdkit安装成功了，但导入rdkit包报错：ImportError: DLL load failed while importing rdBase: 找不到指定的模块。
# 解决方法：
from ctypes import WinDLL
libs_dir = os.path.abspath(fr'D:\Users\{USERNAME}\AppData\Roaming\Python\Python39\site-packages\rdkit.libs')
with open(os.path.join(libs_dir, '.load-order-rdkit-2022.3.5')) as file:
    load_order = file.read().split()
for lib in load_order:
    WinDLL(os.path.join(libs_dir, lib))

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Draw import MolsToGridImage

# Temporary suppress warnings and RDKit logs
warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

np.random.seed(42)
tf.random.set_seed(42)

# 数据集
# 该数据集包含2,050 个分子。每个分子都有一个名称、标签 和SMILES字符串。
# SMILES（Simplified molecular input line entry system），简化分子线性输入规范，是一种用ASCII字符串明确描述分子结构的规范。
# 由于SMILES用一串字符来描述一个三维化学结构，它必然要将化学结构转化成一个生成树，此系统采用纵向优先遍历树算法。
# 转化时，先要去掉氢，还要把环打开。表示时，被拆掉的键端的原子要用数字标记，支链写在小括号里。
# SMILES字符串可以被大多数分子编辑软件导入并转换成二维图形或分子的三维模型。转换成二维图形可以使用Helson的“结构图生成算法”（Structure Diagram Generation algorithms）。

# 血脑屏障 (BBB) 是将血液与脑细胞外液隔开的膜，因此阻止了大多数药物（分子）到达大脑。
# 正因为如此，BBBP 对于研究针对中枢神经系统的新药的开发非常重要。
# 该数据集的标签是二进制的（1 或 0），表示分子的渗透性。

csv_path = rf"D:\Users\{USERNAME}\data\BBBP\BBBP.csv"
# "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv"


df = pd.read_csv(csv_path, usecols=[1, 2, 3])
df.iloc[96:104]

# 定义特征
# 为了编码原子和键的特征（我们稍后需要），我们将定义两个类：AtomFeaturizer和BondFeaturizer。
# 为了减少代码行数，即保持本教程简短和简洁，将仅考虑少数（原子和键）特征：[原子特征] 符号（元素）、 价电子数、 氢键数， 轨道杂交，[键特征] （共价）键类型和 共轭。

class Featurizer:
    def __init__(self, allowable_sets):
        self.dim = 0
        self.features_mapping = {}
        for k, s in allowable_sets.items():
            s = sorted(list(s))
            self.features_mapping[k] = dict(zip(s, range(self.dim, len(s) + self.dim)))
            self.dim += len(s)

    def encode(self, inputs):
        output = np.zeros((self.dim,))
        for name_feature, feature_mapping in self.features_mapping.items():
            feature = getattr(self, name_feature)(inputs)
            if feature not in feature_mapping:
                continue
            output[feature_mapping[feature]] = 1.0
        return output


class AtomFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)

    def symbol(self, atom):
        return atom.GetSymbol()

    def n_valence(self, atom):
        return atom.GetTotalValence()

    def n_hydrogens(self, atom):
        return atom.GetTotalNumHs()

    def hybridization(self, atom):
        return atom.GetHybridization().name.lower()


class BondFeaturizer(Featurizer):
    def __init__(self, allowable_sets):
        super().__init__(allowable_sets)
        self.dim += 1

    def encode(self, bond):
        output = np.zeros((self.dim,))
        if bond is None:
            output[-1] = 1.0
            return output
        output = super().encode(bond)
        return output

    def bond_type(self, bond):
        return bond.GetBondType().name.lower()

    def conjugated(self, bond):
        return bond.GetIsConjugated()


atom_featurizer = AtomFeaturizer(
    allowable_sets={
        "symbol": {"B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"},
        "n_valence": {0, 1, 2, 3, 4, 5, 6},
        "n_hydrogens": {0, 1, 2, 3, 4},
        "hybridization": {"s", "sp", "sp2", "sp3"},
    }
)

bond_featurizer = BondFeaturizer(
    allowable_sets={
        "bond_type": {"single", "double", "triple", "aromatic"},
        "conjugated": {True, False},
    }
)

# 生成图表
# 在我们可以从 SMILES 生成完整的图之前，我们需要实现以下功能：
# molecule_from_smiles，它将一个 SMILES 作为输入并返回一个分子对象。这一切都由 RDKit 处理。
# graph_from_molecule，它将分子对象作为输入并返回一个图，表示为一个三元组（atom_features、bond_features、pair_indices）。为此，我们将使用之前定义的类。
# 最后，我们现在可以实现函数graphs_from_smiles，它将函数 (1) 和随后的 (2) 应用于训练、验证和测试数据集的所有 SMILES。
#
# 注意：虽然建议对该数据集进行脚手架拆分（参见 此处），但为简单起见，执行了简单的随机拆分。

def molecule_from_smiles(smiles):
    # MolFromSmiles(m, sanitize=True) should be equivalent to
    # MolFromSmiles(m, sanitize=False) -> SanitizeMol(m) -> AssignStereochemistry(m, ...)
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)

    # If sanitization is unsuccessful, catch the error, and try again without
    # the sanitization step that caused the error
    flag = Chem.SanitizeMol(molecule, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        Chem.SanitizeMol(molecule, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ flag)

    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    return molecule


def graph_from_molecule(molecule):
    # Initialize graph
    atom_features = []
    bond_features = []
    pair_indices = []

    for atom in molecule.GetAtoms():
        atom_features.append(atom_featurizer.encode(atom))

        # Add self-loops
        pair_indices.append([atom.GetIdx(), atom.GetIdx()])
        bond_features.append(bond_featurizer.encode(None))

        for neighbor in atom.GetNeighbors():
            bond = molecule.GetBondBetweenAtoms(atom.GetIdx(), neighbor.GetIdx())
            pair_indices.append([atom.GetIdx(), neighbor.GetIdx()])
            bond_features.append(bond_featurizer.encode(bond))

    return np.array(atom_features), np.array(bond_features), np.array(pair_indices)


def graphs_from_smiles(smiles_list):
    # Initialize graphs
    atom_features_list = []
    bond_features_list = []
    pair_indices_list = []

    for smiles in smiles_list:
        molecule = molecule_from_smiles(smiles)
        atom_features, bond_features, pair_indices = graph_from_molecule(molecule)

        atom_features_list.append(atom_features)
        bond_features_list.append(bond_features)
        pair_indices_list.append(pair_indices)

    # Convert lists to ragged tensors for tf.data.Dataset later on
    return (
        tf.ragged.constant(atom_features_list, dtype=tf.float32),
        tf.ragged.constant(bond_features_list, dtype=tf.float32),
        tf.ragged.constant(pair_indices_list, dtype=tf.int64),
    )


# Shuffle array of indices ranging from 0 to 2049
permuted_indices = np.random.permutation(np.arange(df.shape[0]))

# Train set: 80 % of data
train_index = permuted_indices[: int(df.shape[0] * 0.8)]
x_train = graphs_from_smiles(df.iloc[train_index].smiles)
y_train = df.iloc[train_index].p_np

# Valid set: 19 % of data
valid_index = permuted_indices[int(df.shape[0] * 0.8) : int(df.shape[0] * 0.99)]
x_valid = graphs_from_smiles(df.iloc[valid_index].smiles)
y_valid = df.iloc[valid_index].p_np

# Test set: 1 % of data
test_index = permuted_indices[int(df.shape[0] * 0.99) :]
x_test = graphs_from_smiles(df.iloc[test_index].smiles)
y_test = df.iloc[test_index].p_np

# 函数测试
print(f"Name:\t{df.name[100]}\nSMILES:\t{df.smiles[100]}\nBBBP:\t{df.p_np[100]}")
molecule = molecule_from_smiles(df.iloc[100].smiles)
print("Molecule:")



graph = graph_from_molecule(molecule)
print("Graph (including self-loops):")
print("\tatom features\t", graph[0].shape)  # (13, 29)
print("\tbond features\t", graph[1].shape) # (39, 7)
print("\tpair indices\t", graph[2].shape)  # (39, 2)


# 创建一个tf.data.Dataset
# 在本教程中，MPNN 实现将采用单个图作为输入（每次迭代）。
# 因此，给定一批（子）图（分子），我们需要将它们合并为一个图（我们将此图称为全局图）。
# 这个全局图是一个断开的图，其中每个子图都与其他子图完全分离。

def prepare_batch(x_batch, y_batch):
    """Merges (sub)graphs of batch into a single global (disconnected) graph
    """

    atom_features, bond_features, pair_indices = x_batch

    # Obtain number of atoms and bonds for each graph (molecule)
    num_atoms = atom_features.row_lengths()
    num_bonds = bond_features.row_lengths()

    # Obtain partition indices (molecule_indicator), which will be used to
    # gather (sub)graphs from global graph in model later on
    molecule_indices = tf.range(len(num_atoms))
    molecule_indicator = tf.repeat(molecule_indices, num_atoms)

    # Merge (sub)graphs into a global (disconnected) graph. Adding 'increment' to
    # 'pair_indices' (and merging ragged tensors) actualizes the global graph
    gather_indices = tf.repeat(molecule_indices[:-1], num_bonds[1:])
    increment = tf.cumsum(num_atoms[:-1])
    increment = tf.pad(tf.gather(increment, gather_indices), [(num_bonds[0], 0)])
    pair_indices = pair_indices.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    pair_indices = pair_indices + increment[:, tf.newaxis]
    atom_features = atom_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()
    bond_features = bond_features.merge_dims(outer_axis=0, inner_axis=1).to_tensor()

    return (atom_features, bond_features, pair_indices, molecule_indicator), y_batch


def MPNNDataset(X, y, batch_size=32, shuffle=False):
    dataset = tf.data.Dataset.from_tensor_slices((X, (y)))
    if shuffle:
        dataset = dataset.shuffle(1024)
    return dataset.batch(batch_size).map(prepare_batch, -1).prefetch(-1)

# 模型
# MPNN 模型可以采用各种形状和形式。
# 在本教程中，我们将基于原始论文 Neural Message Passing for Quantum Chemistry和 DeepChem 的 MPNNModel 实现一个 MPNN。
# 本教程的 MPNN 包括三个阶段：消息传递、读出和分类。

# 消息传递
# 消息传递步骤本身由两部分组成：
# 1.边缘网络，根据它们之间的边缘特征，将消息从 v 的 1 跳邻居 w_{i} 传递到 v，从而产生更新的节点（状态）v'。 w_{i} 表示 v 的第 i 个邻居。
# 2.门控循环单元 (GRU)，它将最近的节点状态作为输入，并根据之前的节点状态对其进行更新。
# 换句话说，最近的节点状态作为 GRU 的输入，而先前的节点状态被合并到 GRU 的内存状态中。
# 这允许信息从一个节点状态（例如，v）传播到另一个（例如，v''）。
#
# 重要的是，步骤 (1) 和 (2) 重复 k 步，并且在每一步 1...k 处，来自 v 的聚合信息的半径（或跳数）增加 1。

class EdgeNetwork(layers.Layer):
    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.kernel = self.add_weight(
            shape=(self.bond_dim, self.atom_dim * self.atom_dim),
            initializer="glorot_uniform",
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.atom_dim * self.atom_dim), initializer="zeros", name="bias",
        )
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Apply linear transformation to bond features
        bond_features = tf.matmul(bond_features, self.kernel) + self.bias

        # Reshape for neighborhood aggregation later
        bond_features = tf.reshape(bond_features, (-1, self.atom_dim, self.atom_dim))

        # Obtain atom features of neighbors
        atom_features_neighbors = tf.gather(atom_features, pair_indices[:, 1])
        atom_features_neighbors = tf.expand_dims(atom_features_neighbors, axis=-1)

        # Apply neighborhood aggregation
        transformed_features = tf.matmul(bond_features, atom_features_neighbors)
        transformed_features = tf.squeeze(transformed_features, axis=-1)
        aggregated_features = tf.math.unsorted_segment_sum(
            transformed_features,
            pair_indices[:, 0],
            num_segments=tf.shape(atom_features)[0],
        )
        return aggregated_features


class MessagePassing(layers.Layer):
    def __init__(self, units, steps=4, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.steps = steps

    def build(self, input_shape):
        self.atom_dim = input_shape[0][-1]
        self.message_step = EdgeNetwork()
        self.pad_length = max(0, self.units - self.atom_dim)
        self.update_step = layers.GRUCell(self.atom_dim + self.pad_length)
        self.built = True

    def call(self, inputs):
        atom_features, bond_features, pair_indices = inputs

        # Pad atom features if number of desired units exceeds atom_features dim.
        # Alternatively, a dense layer could be used here.
        atom_features_updated = tf.pad(atom_features, [(0, 0), (0, self.pad_length)])

        # Perform a number of steps of message passing
        for i in range(self.steps):
            # Aggregate information from neighbors
            atom_features_aggregated = self.message_step(
                [atom_features_updated, bond_features, pair_indices]
            )

            # Update node state via a step of GRU
            atom_features_updated, _ = self.update_step(
                atom_features_aggregated, atom_features_updated
            )
        return atom_features_updated

# 读出
# 当消息传递过程结束时，k-step-aggregated 节点状态将被划分为子图（对应于批次中的每个分子），然后减少到图级嵌入。
# 在 原始论文中，为此目的使用了一个 set-to-set 层。然而，在本教程中，将使用transformer编码器 + 平均池。具体来说：
# 1,k 步聚合的节点状态将被划分为子图（对应于批次中的每个分子）；
# 2,然后将填充每个子图以匹配具有最大节点数的子图，然后是 a tf.stack(...);
# 3,（堆叠填充）张量，编码子图（每个子图包含一组节点状态）被屏蔽以确保填充不会干扰训练；
# 4,最后，张量被传递给转换器，然后是平均池化。

class PartitionPadding(layers.Layer):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    def call(self, inputs):

        atom_features, molecule_indicator = inputs

        # Obtain subgraphs
        atom_features_partitioned = tf.dynamic_partition(
            atom_features, molecule_indicator, self.batch_size
        )

        # Pad and stack subgraphs
        num_atoms = [tf.shape(f)[0] for f in atom_features_partitioned]
        max_num_atoms = tf.reduce_max(num_atoms)
        atom_features_stacked = tf.stack(
            [
                tf.pad(f, [(0, max_num_atoms - n), (0, 0)])
                for f, n in zip(atom_features_partitioned, num_atoms)
            ],
            axis=0,
        )

        # Remove empty subgraphs (usually for last batch in dataset)
        gather_indices = tf.where(tf.reduce_sum(atom_features_stacked, (1, 2)) != 0)
        gather_indices = tf.squeeze(gather_indices, axis=-1)
        return tf.gather(atom_features_stacked, gather_indices, axis=0)


class TransformerEncoderReadout(layers.Layer):
    def __init__(
        self, num_heads=8, embed_dim=64, dense_dim=512, batch_size=32, **kwargs
    ):
        super().__init__(**kwargs)

        self.partition_padding = PartitionPadding(batch_size)
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.average_pooling = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        x = self.partition_padding(inputs)
        padding_mask = tf.reduce_any(tf.not_equal(x, 0.0), axis=-1)
        padding_mask = padding_mask[:, tf.newaxis, tf.newaxis, :]
        attention_output = self.attention(x, x, attention_mask=padding_mask)
        proj_input = self.layernorm_1(x + attention_output)
        proj_output = self.layernorm_2(proj_input + self.dense_proj(proj_input))
        return self.average_pooling(proj_output)

# 消息传递神经网络 (MPNN)
# 现在是完成 MPNN 模型的时候了。除了消息传递和读出之外，还将实施一个两层分类网络来对 BBBP 进行预测。

def MPNNModel(
    atom_dim,
    bond_dim,
    batch_size=32,
    message_units=64,
    message_steps=4,
    num_attention_heads=8,
    dense_units=512,
):

    atom_features = layers.Input((atom_dim), dtype="float32", name="atom_features")
    bond_features = layers.Input((bond_dim), dtype="float32", name="bond_features")
    pair_indices = layers.Input((2), dtype="int32", name="pair_indices")
    molecule_indicator = layers.Input((), dtype="int32", name="molecule_indicator")

    x = MessagePassing(message_units, message_steps)(
        [atom_features, bond_features, pair_indices]
    )

    x = TransformerEncoderReadout(
        num_attention_heads, message_units, dense_units, batch_size
    )([x, molecule_indicator])

    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(
        inputs=[atom_features, bond_features, pair_indices, molecule_indicator],
        outputs=[x],
    )
    return model


mpnn = MPNNModel(
    atom_dim=x_train[0][0][0].shape[0], bond_dim=x_train[1][0][0].shape[0],
)

mpnn.compile(
    loss=keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=5e-4),
    metrics=[keras.metrics.AUC(name="AUC")],
)

keras.utils.plot_model(mpnn, show_dtype=True, show_shapes=True)


# 训练
train_dataset = MPNNDataset(x_train, y_train)
valid_dataset = MPNNDataset(x_valid, y_valid)
test_dataset = MPNNDataset(x_test, y_test)

history = mpnn.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=40,
    verbose=1,
    class_weight={0: 2.0, 1: 0.5},
)

plt.figure(figsize=(10, 6))
plt.plot(history.history["AUC"], label="train AUC")
plt.plot(history.history["val_AUC"], label="valid AUC")
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("AUC", fontsize=16)
plt.legend(fontsize=16)


# 测试模型效果
# rdkit库中提供了一些方法可以返回模型的图，绘制预测结果如下；
molecules = [molecule_from_smiles(df.smiles.values[index]) for index in test_index]
y_true = [df.p_np.values[index] for index in test_index]
y_pred = tf.squeeze(mpnn.predict(test_dataset), axis=1)

legends = [f"y_true/y_pred = {y_true[i]}/{y_pred[i]:.2f}" for i in range(len(y_true))]
MolsToGridImage(molecules, molsPerRow=4, legends=legends)

# 结论
# 在本教程中，我们演示了一个消息传递神经网络 (MPNN) 来预测许多不同分子的血脑屏障通透性 (BBBP)。
# 我们首先必须从 SMILES 构建图，然后构建可以在这些图上运行的 Keras 模型，最后训练模型进行预测。


def main():
    pass


if __name__ == '__main__':
    main()
