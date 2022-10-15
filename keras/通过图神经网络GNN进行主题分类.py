#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Graph Neural Networks(图神经网络，GNN)
# https://keras.io/examples/graph/gnn_citations/


# 此示例演示了图神经网络 (GNN) 模型的简单实现。该模型用于Cora 数据集上的节点预测任务， 以根据其单词和引文网络预测论文的主题。

import os
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

USERNAME = os.getenv("USERNAME")

# 准备数据集
# Cora 数据集包含 2,708 篇科学论文，分为七个类别之一。引文网络由 5,429 个链接组成。每篇论文都有一个大小为 1,433 的二进制词向量，表示存在相应的词。
#
# 下载数据集
# https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
# 该数据集有两个分接头文件：cora.cites和cora.content.

# cora.cites包括具有两列的引文记录：（ 目标cited_paper_id）和citing_paper_id（来源）。
# cora.content包括具有 1,435 列的论文内容记录： 、paper_id和subject1,433 个二进制特征。
# 让我们下载数据集。


data_dir = rf"D:\Users\{USERNAME}\data\cora"
# 处理和可视化数据集
# 然后我们将引文数据加载到 Pandas DataFrame 中。

citations = pd.read_csv(
    os.path.join(data_dir, "cora.cites"),
    sep="\t",
    header=None,
    names=["target", "source"],
)
print("引文 Citations shape:", citations.shape)

# 引文 Citations shape: (5429, 2)


# 现在我们显示citationsDataFrame 的示例。该target列包括列中的论文 ID 引用的论文 ID source。
citations.sample(frac=1).head()

# 现在让我们将论文数据加载到 Pandas DataFrame 中。
# 每篇论文都有一个大小为 1,433 的二进制词向量，表示存在相应的词。
column_names = ["paper_id"] + [f"term_{idx}" for idx in range(1433)] + ["subject"]
papers = pd.read_csv(
    os.path.join(data_dir, "cora.content"), sep="\t", header=None, names=column_names,
)
print("论文 Papers shape:", papers.shape)
# 论文 Papers shape: (2708, 1435)


# 现在我们显示papersDataFrame 的示例。
# DataFrame 包括paper_id 和subject列，以及表示论文中是否存在术语的 1,433 个二进制列。

print(papers.sample(5).T)

# 让我们显示每个主题的论文数。
print(papers.subject.value_counts())


# 我们将论文 ID 和主题转换为从零开始的索引。

class_values = sorted(papers["subject"].unique())
class_idx = {name: id for id, name in enumerate(class_values)}
paper_idx = {name: idx for idx, name in enumerate(sorted(papers["paper_id"].unique()))}

papers["paper_id"] = papers["paper_id"].apply(lambda name: paper_idx[name])
citations["source"] = citations["source"].apply(lambda name: paper_idx[name])
citations["target"] = citations["target"].apply(lambda name: paper_idx[name])
papers["subject"] = papers["subject"].apply(lambda value: class_idx[value])


# 现在让我们可视化引文图。
# 图中的每个节点代表一篇论文，节点的颜色对应其主题。请注意，我们仅显示数据集中的论文样本。
plt.figure(figsize=(10, 10))
colors = papers["subject"].tolist()
cora_graph = nx.from_pandas_edgelist(citations.sample(n=1500))
subjects = list(papers[papers["paper_id"].isin(list(cora_graph.nodes))]["subject"])
nx.draw_spring(cora_graph, node_size=15, node_color=subjects)


# 将数据集拆分为分层训练集和测试集
train_data, test_data = [], []

for _, group_data in papers.groupby("subject"):
    # Select around 50% of the dataset for training.
    random_selection = np.random.rand(len(group_data.index)) <= 0.5
    train_data.append(group_data[random_selection])
    test_data.append(group_data[~random_selection])

train_data = pd.concat(train_data).sample(frac=1)
test_data = pd.concat(test_data).sample(frac=1)

print("Train data shape:", train_data.shape)
print("Test data shape:", test_data.shape)
# Train data shape: (1360, 1435)
# Test data shape: (1348, 1435)


# 实施训练和评估实验
hidden_units = [32, 32]
learning_rate = 0.01
dropout_rate = 0.5
num_epochs = 100
batch_size = 64


# 此函数使用给定的训练数据编译和训练输入模型。

def run_experiment(model, x_train, y_train):
    # Compile the model.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")],
    )
    # Create an early stopping callback.
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", patience=50, restore_best_weights=True
    )
    # Fit the model.
    history = model.fit(
        x=x_train,
        y=y_train,
        epochs=num_epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stopping],
    )

    return history


# 该函数显示模型在训练过程中的损失和准确率曲线。

def display_learning_curves(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history["loss"])
    ax1.plot(history.history["val_loss"])
    ax1.legend(["train", "test"], loc="upper right")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")

    ax2.plot(history.history["acc"])
    ax2.plot(history.history["val_acc"])
    ax2.legend(["train", "test"], loc="upper right")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")
    plt.show()


# 实施前馈网络 (FFN) 模块
# 我们将在基线和 GNN 模型中使用这个模块。

def create_ffn(hidden_units, dropout_rate, name=None):
    fnn_layers = []

    for units in hidden_units:
        fnn_layers.append(layers.BatchNormalization())
        fnn_layers.append(layers.Dropout(dropout_rate))
        fnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

    return keras.Sequential(fnn_layers, name=name)


# 构建基线神经网络模型
# 为基线模型准备数据
feature_names = set(papers.columns) - {"paper_id", "subject"}
num_features = len(feature_names)
num_classes = len(class_idx)

# Create train and test features as a numpy array.
x_train = train_data[feature_names].to_numpy()
x_test = test_data[feature_names].to_numpy()
# Create train and test targets as a numpy array.
y_train = train_data["subject"]
y_test = test_data["subject"]


# 实现基线分类器
# 我们添加了五个带有跳跃连接的 FFN 块，以便我们生成一个基线模型，其参数数量与稍后要构建的 GNN 模型大致相同。

def create_baseline_model(hidden_units, num_classes, dropout_rate=0.2):
    inputs = layers.Input(shape=(num_features,), name="input_features")
    x = create_ffn(hidden_units, dropout_rate, name=f"ffn_block1")(inputs)
    for block_idx in range(4):
        # Create an FFN block.
        x1 = create_ffn(hidden_units, dropout_rate, name=f"ffn_block{block_idx + 2}")(x)
        # Add skip connection.
        x = layers.Add(name=f"skip_connection{block_idx + 2}")([x, x1])
    # Compute logits.
    logits = layers.Dense(num_classes, name="logits")(x)
    # Create the model.
    return keras.Model(inputs=inputs, outputs=logits, name="baseline")


baseline_model = create_baseline_model(hidden_units, num_classes, dropout_rate)
baseline_model.summary()
# Model: "baseline"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_features (InputLayer)    [(None, 1433)]       0           []
#
#  ffn_block1 (Sequential)        (None, 32)           52804       ['input_features[0][0]']
#
#  ffn_block2 (Sequential)        (None, 32)           2368        ['ffn_block1[0][0]']
#
#  skip_connection2 (Add)         (None, 32)           0           ['ffn_block1[0][0]',
#                                                                   'ffn_block2[0][0]']
#
#  ffn_block3 (Sequential)        (None, 32)           2368        ['skip_connection2[0][0]']
#
#  skip_connection3 (Add)         (None, 32)           0           ['skip_connection2[0][0]',
#                                                                   'ffn_block3[0][0]']
#
#  ffn_block4 (Sequential)        (None, 32)           2368        ['skip_connection3[0][0]']
#
#  skip_connection4 (Add)         (None, 32)           0           ['skip_connection3[0][0]',
#                                                                   'ffn_block4[0][0]']
#
#  ffn_block5 (Sequential)        (None, 32)           2368        ['skip_connection4[0][0]']
#
#  skip_connection5 (Add)         (None, 32)           0           ['skip_connection4[0][0]',
#                                                                   'ffn_block5[0][0]']
#
#  logits (Dense)                 (None, 7)            231         ['skip_connection5[0][0]']
#
# ==================================================================================================
# Total params: 62,507
# Trainable params: 59,065
# Non-trainable params: 3,442
# __________________________________________________________________________________________________


# 训练基线分类器
history = run_experiment(baseline_model, x_train, y_train)

# 让我们绘制学习曲线。
display_learning_curves(history)


# 现在我们在测试数据拆分上评估基线模型。

_, test_accuracy = baseline_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"基线模型 Test accuracy: {round(test_accuracy * 100, 2)}%")
# 基线模型 Test accuracy: 73.52%


# 检查基线模型预测
# 让我们通过随机生成关于单词存在概率的二进制单词向量来创建新的数据实例。

def generate_random_instances(num_instances):
    # 随机生成几个示例
    token_probability = x_train.mean(axis=0)
    instances = []
    for _ in range(num_instances):
        probabilities = np.random.uniform(size=len(token_probability))
        instance = (probabilities <= token_probability).astype(int)
        instances.append(instance)

    return np.array(instances)


def display_class_probabilities(probabilities):
    for instance_idx, probs in enumerate(probabilities):
        print(f"示例 Instance {instance_idx + 1}:")
        for class_idx, prob in enumerate(probs):
            print(f"- {class_values[class_idx]}: {round(prob * 100, 2)}%")

# 现在我们展示给定这些随机生成的实例的基线模型预测。
new_instances = generate_random_instances(num_classes)
logits = baseline_model.predict(new_instances)
probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
display_class_probabilities(probabilities)

# 构建图神经网络模型
# 为图模型准备数据
# 准备图数据并将其加载到模型中进行训练是 GNN 模型中最具挑战性的部分，专门的库以不同的方式解决了这一问题。
# 在此示例中，我们展示了一种用于准备和使用图形数据的简单方法，该方法适用于您的数据集由一个完全适合内存的图形组成的情况。

# 图数据由graph_info元组表示，元组由以下三个元素组成：
# node_features：这是一个[num_nodes, num_features]包含节点特征的 NumPy 数组。
#     在这个数据集中，节点是论文，并且是每篇论文node_features的单词存在二进制向量。
# edges：这是[num_edges, num_edges]NumPy 数组，表示节点之间链接的稀疏 邻接矩阵 。在这个例子中，链接是论文之间的引用。
# edge_weights（可选）：这是一个[num_edges]包含边权重的 NumPy 数组，用于量化 图中节点之间的关系。在此示例中，论文引用没有权重。


# 创建一个形状为[2, num_edges]的数组(稀疏邻接矩阵)
edges = citations[["source", "target"]].to_numpy().T
# 创建一个边权重数组
edge_weights = tf.ones(shape=edges.shape[1])
# 创建一个形状为[num_nodes, num_features]的节点特征数组；
node_features = tf.cast(
    papers.sort_values("paper_id")[feature_names].to_numpy(), dtype=tf.dtypes.float32
)
# 创建带有节点特征、边和边权重的图信息元组。
graph_info = (node_features, edges, edge_weights)

print("边形状 Edges shape:", edges.shape)
print("节点形状 Nodes shape:", node_features.shape)


# 实现图卷积层
# 我们将图卷积模块实现为Keras 层。我们GraphConvLayer执行以下步骤：
# 准备：使用 FFN 处理输入节点表示以产生消息。您可以通过仅对表示应用线性变换来简化处理。
# 聚合：每个节点的邻居的消息是相对于edge_weights使用置换不变池操作聚合的，例如sum、mean和max，为每个节点准备单个聚合消息。
# 更新：node_repesentations和aggregated_messages- 两者的形状[num_nodes, representation_dim]- 被组合和处理以产生节点表示的新状态（节点嵌入）。
# 如果combination_type是gru，则node_repesentations和aggregated_messages被堆叠以创建一个序列，然后由 GRU 层处理。
# 否则，node_repesentationsandaggregated_messages被添加或连接，然后使用 FFN 进行处理。

# 该技术使用来自图卷积网络、 GraphSage、图同构网络、 简单图网络和 门控图序列神经网络的思想。未涵盖的另外两个关键技术是图注意网络 和消息传递神经网络。

class GraphConvLayer(layers.Layer):
    def __init__(
        self,
        hidden_units,
        dropout_rate=0.2,
        aggregation_type="mean",
        combination_type="concat",
        normalize=False,
        *args,
        **kwargs,
    ):
        super(GraphConvLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.ffn_prepare = create_ffn(hidden_units, dropout_rate)
        if self.combination_type == "gated":
            self.update_fn = layers.GRU(
                units=hidden_units,
                activation="tanh",
                recurrent_activation="sigmoid",
                dropout=dropout_rate,
                return_state=True,
                recurrent_dropout=dropout_rate,
            )
        else:
            self.update_fn = create_ffn(hidden_units, dropout_rate)

    def prepare(self, node_repesentations, weights=None):
        # node_repesentations shape is [num_edges, embedding_dim].
        messages = self.ffn_prepare(node_repesentations)
        if weights is not None:
            messages = messages * tf.expand_dims(weights, -1)
        return messages

    def aggregate(self, node_indices, neighbour_messages, node_repesentations):
        # node_indices shape is [num_edges].
        # neighbour_messages shape: [num_edges, representation_dim].
        # node_repesentations shape is [num_nodes, representation_dim]
        num_nodes = node_repesentations.shape[0]
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(
                neighbour_messages, node_indices, num_segments=num_nodes
            )
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_repesentations, aggregated_messages):
        # node_repesentations shape is [num_nodes, representation_dim].
        # aggregated_messages shape is [num_nodes, representation_dim].
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_repesentations and aggregated_messages.
            h = tf.concat([node_repesentations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_repesentations and aggregated_messages.
            h = node_repesentations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def call(self, inputs):
        """Process the inputs to produce the node_embeddings.

        inputs: a tuple of three elements: node_repesentations, edges, edge_weights.
        Returns: node_embeddings of shape [num_nodes, representation_dim].
        """

        node_repesentations, edges, edge_weights = inputs
        # Get node_indices (source) and neighbour_indices (target) from edges.
        node_indices, neighbour_indices = edges[0], edges[1]
        # neighbour_repesentations shape is [num_edges, representation_dim].
        neighbour_repesentations = tf.gather(node_repesentations, neighbour_indices)

        # Prepare the messages of the neighbours.
        neighbour_messages = self.prepare(neighbour_repesentations, edge_weights)
        # Aggregate the neighbour messages.
        aggregated_messages = self.aggregate(
            node_indices, neighbour_messages, node_repesentations
        )
        # Update the node embedding with the neighbour messages.
        return self.update(node_repesentations, aggregated_messages)

# 实现一个图神经网络节点分类器
# GNN 分类模型遵循Design Space for Graph Neural Networks方法，如下：
# 1.使用 FFN 对节点特征应用预处理以生成初始节点表示。
# 2.将一个或多个带有跳跃连接的图卷积层应用于节点表示以产生节点嵌入。
# 3.使用 FFN 对节点嵌入应用后处理以生成最终节点嵌入。
# 4.在 Softmax 层中输入节点嵌入以预测节点类别。

# 添加的每个图卷积层都从更高级别的邻居捕获信息。但是，添加许多图卷积层会导致过度平滑，其中模型会为所有节点生成相似的嵌入。

# 请注意，graph_info传递给 Keras 模型的构造函数，并用作 Keras 模型对象的属性 ，而不是用于训练或预测的输入数据。
# 该模型将接受一批，node_indices用于从 中查找节点特征和邻居graph_info。

class GNNNodeClassifier(tf.keras.Model):
    def __init__(
        self,
        graph_info,
        num_classes,
        hidden_units,
        aggregation_type="sum",
        combination_type="concat",
        dropout_rate=0.2,
        normalize=True,
        *args,
        **kwargs,
    ):
        super(GNNNodeClassifier, self).__init__(*args, **kwargs)

        # Unpack graph_info to three elements: node_features, edges, and edge_weight.
        node_features, edges, edge_weights = graph_info
        self.node_features = node_features
        self.edges = edges
        self.edge_weights = edge_weights
        # Set edge_weights to ones if not provided.
        if self.edge_weights is None:
            self.edge_weights = tf.ones(shape=edges.shape[1])
        # Scale edge_weights to sum to 1.
        self.edge_weights = self.edge_weights / tf.math.reduce_sum(self.edge_weights)

        # Create a process layer.
        self.preprocess = create_ffn(hidden_units, dropout_rate, name="preprocess")
        # Create the first GraphConv layer.
        self.conv1 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv1",
        )
        # Create the second GraphConv layer.
        self.conv2 = GraphConvLayer(
            hidden_units,
            dropout_rate,
            aggregation_type,
            combination_type,
            normalize,
            name="graph_conv2",
        )
        # Create a postprocess layer.
        self.postprocess = create_ffn(hidden_units, dropout_rate, name="postprocess")
        # Create a compute logits layer.
        self.compute_logits = layers.Dense(units=num_classes, name="logits")

    def call(self, input_node_indices):
        # Preprocess the node_features to produce node representations.
        x = self.preprocess(self.node_features)
        # Apply the first graph conv layer.
        x1 = self.conv1((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x1 + x
        # Apply the second graph conv layer.
        x2 = self.conv2((x, self.edges, self.edge_weights))
        # Skip connection.
        x = x2 + x
        # Postprocess node embedding.
        x = self.postprocess(x)
        # Fetch node embeddings for the input node_indices.
        node_embeddings = tf.gather(x, input_node_indices)
        # Compute logits
        return self.compute_logits(node_embeddings)

# 让我们测试实例化和调用 GNN 模型。
# 请注意，如果您提供N节点索引，则输出将是 shape 的张量[N, num_classes]，而与图的大小无关。

gnn_model = GNNNodeClassifier(
    graph_info=graph_info,
    num_classes=num_classes,
    hidden_units=hidden_units,
    dropout_rate=dropout_rate,
    name="gnn_model",
)

print("GNN output shape:", gnn_model([1, 10, 100]))

gnn_model.summary()
# Model: "gnn_model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #
# =================================================================
#  preprocess (Sequential)     (2708, 32)                52804
#
#  graph_conv1 (GraphConvLayer  multiple                 5888
#  )
#
#  graph_conv2 (GraphConvLayer  multiple                 5888
#  )
#
#  postprocess (Sequential)    (2708, 32)                2368
#
#  logits (Dense)              multiple                  231
#
# =================================================================
# Total params: 67,179
# Trainable params: 63,481
# Non-trainable params: 3,698
# _________________________________________________________________

# 训练 GNN 模型
# 请注意，我们使用标准的监督交叉熵损失来训练模型。但是，我们可以为生成的节点嵌入添加另一个自监督损失项，以确保图中的相邻节点具有相似的表示，而远处的节点具有不同的表示。

x_train = train_data.paper_id.to_numpy()
history = run_experiment(gnn_model, x_train, y_train)

# 让我们绘制学习曲线

display_learning_curves(history)


# 现在我们在测试数据拆分上评估 GNN 模型。结果可能因训练样本而异，但 GNN 模型在测试准确性方面始终优于基线模型。

x_test = test_data.paper_id.to_numpy()
_, test_accuracy = gnn_model.evaluate(x=x_test, y=y_test, verbose=0)
print(f"GNN模型 Test accuracy: {round(test_accuracy * 100, 2)}%")
# GNN模型 Test accuracy: 80.19%


# 检查 GNN 模型预测
# 让我们将新实例作为节点添加到 中node_features，并生成到现有节点的链接（引用）。

# 首先将 N 个new_instances 作为节点添加到图中；
# 添加 new_instance 到 node_features.
num_nodes = node_features.shape[0]
new_node_features = np.concatenate([node_features, new_instances])
# 将 M 条边 (引用) 添加到集合中
# 特定主体中的现有节点数
new_node_indices = [i + num_nodes for i in range(num_classes)]
new_citations = []
for subject_idx, group in papers.groupby("subject"):
    subject_papers = list(group.paper_id)
    # 随机选择x篇论文主体
    selected_paper_indices1 = np.random.choice(subject_papers, 5)
    # 随便选择y篇论文 (其中 y < x).
    selected_paper_indices2 = np.random.choice(list(papers.paper_id), 2)
    # 合并选择.
    selected_paper_indices = np.concatenate(
        [selected_paper_indices1, selected_paper_indices2], axis=0
    )
    # 在论文 idx 和选定的被引论文之间创建边。
    citing_paper_indx = new_node_indices[subject_idx]
    for cited_paper_idx in selected_paper_indices:
        new_citations.append([citing_paper_indx, cited_paper_idx])

new_citations = np.array(new_citations).T
new_edges = np.concatenate([edges, new_citations], axis=1)


# 现在让我们更新 GNN 模型中的node_features和edges。

print("Original node_features shape:", gnn_model.node_features.shape)
print("Original edges shape:", gnn_model.edges.shape)
gnn_model.node_features = new_node_features
gnn_model.edges = new_edges
gnn_model.edge_weights = tf.ones(shape=new_edges.shape[1])
print("New node_features shape:", gnn_model.node_features.shape)
print("New edges shape:", gnn_model.edges.shape)

logits = gnn_model.predict(tf.convert_to_tensor(new_node_indices))
probabilities = keras.activations.softmax(tf.convert_to_tensor(logits)).numpy()
display_class_probabilities(probabilities)

# 与基线模型相比，预期主题（添加了几个引用）的概率更高。

def main():
    pass


if __name__ == '__main__':
    main()
