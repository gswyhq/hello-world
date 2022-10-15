#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# https://keras.io/examples/graph/node2vec_movielens/
# 图表示学习旨在从结构化为图形的对象中学习图节点的向量表示(嵌入)
# node2vec 的流程如下：
# 构建无向电影图谱（依据评分，若某人同时评价了两个电影，则在对应电影间构建边）
# 使用（有偏的）随机游走生成节点序列。
# 从这些序列中创建正面和负面的训练数据，具体是通过跳字模型在生成的随机游走序列中取样，若上下文相同（来自同一序列）则label为1，否则label为0。
# 训练word2vec模型（skip-gram）来学习节点的嵌入向量。

# 通过将电影视为节点，并在用户评分相似的电影之间创建边，可以将此类数据集表示为图形。学习到的电影嵌入可用于电影推荐或电影类型预测等任务。
import os
from collections import defaultdict
import math
import networkx as nx
import random
from tqdm import tqdm
from zipfile import ZipFile
from urllib.request import urlretrieve
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

USERNAME = os.getenv("USERNAME")

# MovieLens 数据集的小型版本包括来自 610 位用户对 9,742 部电影的大约 100k 评分。
# 数据来源：http://files.grouplens.org/datasets/movielens/ml-latest-small.zip

# 数据预处理
# 加载电影数据 Load movies to a DataFrame.
movies = pd.read_csv(rf"D:\Users\{USERNAME}\data\ml-latest-small\movies.csv")
# 创建电影ID, Create a `movieId` string.
movies["movieId"] = movies["movieId"].apply(lambda x: f"movie_{x}")

# 加载评分数据，Load ratings to a DataFrame.
ratings = pd.read_csv(rf"D:\Users\{USERNAME}\data\ml-latest-small\ratings.csv")
# 转换为浮点数， Convert the `ratings` to floating point
ratings["rating"] = ratings["rating"].apply(lambda x: float(x))
# 转换为字符串，Create the `movie_id` string.
ratings["movieId"] = ratings["movieId"].apply(lambda x: f"movie_{x}")

print("电影数据 Movies data shape:", movies.shape)
print("评分数据 Ratings data shape:", ratings.shape)

def get_movie_title_by_id(movieId):
    return list(movies[movies.movieId == movieId].title)[0]


def get_movie_id_by_title(title):
    return list(movies[movies.title == title].movieId)[0]

# 构建电影图
# 如果两部电影都由同一用户评分，且评分>= min_rating，我们会在图中的两个电影节点之间创建一条边。
# 边缘的权重将基于 两部电影之间的逐点互信息,计算公式：log(xy) - log(x) - log(y) + log(D)，其中：
# xy：是有多少用户对电影x和电影y同时进行评分, 且评分≥min_rating
# x：是有多少用户评价了电影x，且评分>= min_rating。
# y：是有多少用户评价了电影y，且评分>= min_rating。
# D：>= min_rating电影的评分总数。

# 第 1 步：创建电影之间的边权重。
min_rating = 5
pair_frequency = defaultdict(int)  # 电影对，一同被评分次数
item_frequency = defaultdict(int)  # 每个电影被评分了多少次

# 评分小于min_rating的忽略不计； Filter instances where rating is greater than or equal to min_rating.
rated_movies = ratings[ratings.rating >= min_rating]
# 按用户分组，Group instances by user.
movies_grouped_by_users = list(rated_movies.groupby("userId"))
for group in tqdm(
    movies_grouped_by_users,
    position=0,
    leave=True,
    desc="计算电影评分频率Compute movie rating frequencies",
):
    # 获取用户评分的电影列表Get a list of movies rated by the user.
    current_movies = list(group[1]["movieId"])

    for i in range(len(current_movies)):
        item_frequency[current_movies[i]] += 1
        for j in range(i + 1, len(current_movies)):
            x = min(current_movies[i], current_movies[j])
            y = max(current_movies[i], current_movies[j])
            pair_frequency[(x, y)] += 1

# 第 2 步：创建包含节点和边的图
# 为了减少节点之间的边数，我们只在电影边的权重大于min_weight之间添加边。

min_weight = 10
D = math.log(sum(item_frequency.values()))

# 创建电影无向图 Create the movies undirected graph.
movies_graph = nx.Graph()
# 在电影直接添加边的权重；Add weighted edges between movies.
# This automatically adds the movie nodes to the graph.
for pair in tqdm(
    pair_frequency, position=0, leave=True, desc="创建电影图 Creating the movie graph"
):
    x, y = pair
    xy_frequency = pair_frequency[pair]
    x_frequency = item_frequency[x]
    y_frequency = item_frequency[y]
    pmi = math.log(xy_frequency) - math.log(x_frequency) - math.log(y_frequency) + D
    weight = pmi * xy_frequency
    # 只考虑 weight >= min_weight.
    if weight >= min_weight:
        movies_graph.add_edge(x, y, weight=weight)

# 让我们在图中显示节点和边的总数。
# 请注意，节点的数量小于电影的总数，因为只添加了与其他电影有边的电影。

print("节点总数(Total number of graph nodes):", movies_graph.number_of_nodes())
print("边总数(Total number of graph edges):", movies_graph.number_of_edges())


# 让我们在图中显示平均节点度（邻居数）。
# 节点度是指和该节点相关联的边的条数，又称关联度。
degrees = []
for node in movies_graph.nodes:
    degrees.append(movies_graph.degree[node])

print("平均节点数 Average node degree:", round(sum(degrees) / len(degrees), 2))


# 第 3 步：创建词汇表和从标记到整数索引的映射
# 词汇表是图中的节点（电影 ID）。

vocabulary = ["NA"] + list(movies_graph.nodes)
vocabulary_lookup = {token: idx for idx, token in enumerate(vocabulary)} # movie_id -> int


# 实施有偏随机游走
# 随机游走从给定节点开始，并随机选择要移动到的相邻节点。
# 如果边缘被加权，则根据当前节点与其邻居之间的边缘的权重概率地选择邻居。
# 重复此过程num_steps次以生成一系列相关节点。

# 注意：当出现了普遍的超级节点后，会导致图上游走采样困难、噪声加剧。这个时候可以对节点进行细化（即考虑能不能将一个节点拆分为多个），并且对于边权有所限制；

# 通过引入以下两个参数，有偏随机游走在广度优先采样（仅访问本地邻居）和深度优先采样（访问远邻居）之间 取得平衡：
# 返回参数(p)：控制在 walk 中立即重新访问节点的可能性。将其设置为较高的值会鼓励适度的探索，而将其设置为较低的值将保持步行本地化。
# In-out 参数(q)：允许搜索区分向内节点和向外节点。将其设置为高值会使随机游走偏向本地节点，而将其设置为低值会使游走偏向于访问更远的节点。

def next_step(graph, previous, current, p, q):
    '''
    计算下一个要访问的节点
    previous: 上一步节点
    current: 当前节点
    p:随机游走返回参数, 该值越大，越不容易往回走
    q:随机游走进出参数，该值越小，越容易往外走(即不往回走)
    '''
    neighbors = list(graph.neighbors(current))  # 获取当前节点的所有邻居

    weights = []
    # 调整权重 Adjust the weights of the edges to the neighbors with respect to p and q.
    for neighbor in neighbors:
        if neighbor == previous:
            # 控制返回前一个节点概率；Control the probability to return to the previous node.
            weights.append(graph[current][neighbor]["weight"] / p)
        elif graph.has_edge(neighbor, previous):  # 判断两个点是否有边相连
            # 控制返回本节点概率； The probability of visiting a local node.
            weights.append(graph[current][neighbor]["weight"])
        else:
            # 控制前进概率 Control the probability to move forward.
            weights.append(graph[current][neighbor]["weight"] / q)

    # 计算访问每个邻居概率 Compute the probabilities of visiting each neighbor.
    weight_sum = sum(weights)
    probabilities = [weight / weight_sum for weight in weights]
    # 概率性地选择要访问的邻居。Probabilistically select a neighbor to visit.
    next = np.random.choice(neighbors, size=1, p=probabilities)[0]
    return next


def random_walk(graph, num_walks, num_steps, p, q):
    '''
    num_walks:随机游走迭代次数
    num_steps:每个随机游走的步数.
    p:随机游走返回参数, 该值越大，越不容易往回走
    q:随机游走进出参数，该值越小，越容易往外走(即不往回走)
    '''
    walks = []
    nodes = list(graph.nodes())
    # 执行随机游走，多次迭代； Perform multiple iterations of the random walk.
    for walk_iteration in range(num_walks):
        random.shuffle(nodes)

        for node in tqdm(
            nodes,
            position=0,
            leave=True,
            desc=f"随机游走迭代Random walks iteration {walk_iteration + 1} of {num_walks}",
        ):
            # 从图中的随机的一个节点开始 Start the walk with a random node from the graph.
            walk = [node]
            # 游走num_steps步 Randomly walk for num_steps.
            while len(walk) < num_steps:
                current = walk[-1]
                previous = walk[-2] if len(walk) > 1 else None
                # 计算下一个要访问的节点 Compute the next node to visit.
                next = next_step(graph, previous, current, p, q)
                walk.append(next)
            # 将节点movie_ids替换为 token_ids Replace node ids (movie ids) in the walk with token ids.
            walk = [vocabulary_lookup[token] for token in walk]
            # 将游走步添加到生成的序列中； Add the walk to the generated sequence.
            walks.append(walk)

    return walks

# 使用有偏随机游走生成训练数据
# 可以设置不同的p和q来活动相关电影的不同结果。

# 随机游走返回参数
p = 1
# 随机游走进出参数.
q = 1
# 随机游走迭代次数
num_walks = 5
# 每个随机游走的步数.
num_steps = 10
walks = random_walk(movies_graph, num_walks, num_steps, p, q)

print("生成的行走次数:", len(walks))

# walks[:5]
# Out[157]:
# [[155, 147, 864, 147, 677, 1015, 307, 774, 338, 1057],
#  [1380, 18, 556, 149, 1319, 149, 260, 226, 227, 1320],
#  [1330, 641, 907, 230, 721, 737, 379, 673, 272, 1029],
#  [458, 44, 17, 940, 845, 740, 18, 85, 49, 543],
#  [260, 238, 22, 62, 33, 643, 940, 1364, 266, 26]]

# 生成正面和负面的例子
# 为了训练 skip-gram 模型，我们使用生成的游走来创建正负训练示例。每个示例都包含以下功能：
# target: 步行序列中的电影。
# context: 步行序列中的另一部电影。
# weight：这两部电影在步行序列中出现了多少次。
# label：如果这两部电影是步行序列的样本，则标签为 1，否则（即，如果随机采样）标签为 0。

# 生成示例
def generate_examples(sequences, window_size, num_negative_samples, vocabulary_size):
    example_weights = defaultdict(int)
    # 遍历所有序列（步行） Iterate over all sequences (walks).
    for sequence in tqdm(
        sequences,
        position=0,
        leave=True,
        desc=f"生成正负样本",
    ):
        # 生成正负 skip-gram 对.
        # tf.keras.preprocessing.sequence.skipgrams()是Tensorflow预处理模块的一个函数，其功能是根据输入条件生成词汇对。因为可能是跳n个词生成的词汇对，所以也叫跳字模型。
        # 若最后pairs中两个id都是来自sequence，则label为1，否则label为0
        pairs, labels = keras.preprocessing.sequence.skipgrams(
            sequence,  # 词汇索引数组，整数; 这里指的每一轮游走步电影id序列, 如：[155, 147, 864, 147, 677, 1015, 307, 774, 338, 1057]
            vocabulary_size=vocabulary_size, # 词汇表大小
            window_size=window_size,  # 正样本对之间的距离
            negative_samples=num_negative_samples, # 大于0的浮点数，等于0代表没有负样本，等于1代表负样本与正样本数目相同，以此类推（即负样本的数目是正样本的negative_samples倍）
        )
        # sequence
        # Out[177]: [155, 147, 864, 147, 677, 1015, 307, 774, 338, 1057]
        # pairs[:6]
        # Out[176]: [[307, 774], [864, 147], [774, 45], [864, 774], [864, 1239], [147, 403]]
        # labels[:6]
        # Out[175]: [1, 1, 0, 1, 0, 0]
        for idx in range(len(pairs)):
            pair = pairs[idx]
            label = labels[idx]
            target, context = min(pair[0], pair[1]), max(pair[0], pair[1])
            if target == context:
                continue
            entry = (target, context, label)
            example_weights[entry] += 1

    targets, contexts, labels, weights = [], [], [], []
    for entry in example_weights:
        weight = example_weights[entry]
        target, context, label = entry
        targets.append(target)
        contexts.append(context)
        labels.append(label)
        weights.append(weight)

    return np.array(targets), np.array(contexts), np.array(labels), np.array(weights)


num_negative_samples = 4
targets, contexts, labels, weights = generate_examples(
    sequences=walks,
    window_size=num_steps,
    num_negative_samples=num_negative_samples,
    vocabulary_size=len(vocabulary),
)

# 让我们显示输出的形状

print(f"Targets shape: {targets.shape}")
print(f"Contexts shape: {contexts.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Weights shape: {weights.shape}")
# Targets shape: (880017,)
# Contexts shape: (880017,)
# Labels shape: (880017,)
# Weights shape: (880017,)


# 将数据转换为tf.data.Dataset对象
batch_size = 1024


def create_dataset(targets, contexts, labels, weights, batch_size):
    inputs = {
        "target": targets,
        "context": contexts,
    }
    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels, weights))
    dataset = dataset.shuffle(buffer_size=batch_size * 2)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


dataset = create_dataset(
    targets=targets,
    contexts=contexts,
    labels=labels,
    weights=weights,
    batch_size=batch_size,
)


# 训练skip-gram模型
# 我们的 skip-gram 是一个简单的二元分类模型，其工作原理如下：
#
# target为电影查找嵌入。
# context为电影查找嵌入。
# 在这两个嵌入之间计算点积。
# 结果（在 sigmoid 激活之后）与标签进行比较。
# 使用二元交叉熵损失。

learning_rate = 0.001
embedding_dim = 50
num_epochs = 10
# 实施模型
def create_model(vocabulary_size, embedding_dim):

    inputs = {
        "target": layers.Input(name="target", shape=(), dtype="int32"),
        "context": layers.Input(name="context", shape=(), dtype="int32"),
    }
    # 向量初始化
    embed_item = layers.Embedding(
        input_dim=vocabulary_size,
        output_dim=embedding_dim,
        embeddings_initializer="he_normal",
        embeddings_regularizer=keras.regularizers.l2(1e-6),
        name="item_embeddings",
    )
    # 查找target向量.
    target_embeddings = embed_item(inputs["target"])
    # 查找context向量
    context_embeddings = embed_item(inputs["context"])
    # 计算target向量和 context 向量相似度.
    logits = layers.Dot(axes=1, normalize=False, name="dot_similarity")(
        [target_embeddings, context_embeddings]
    )
    # 构建模型.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model


# 训练模型
# 我们实例化模型并编译它。

model = create_model(len(vocabulary), embedding_dim)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
)
# 让我们绘制模型。

keras.utils.plot_model(
    model,
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
)

# 训练模型
history = model.fit(dataset, epochs=num_epochs)

# 最后我们绘制学习历史。
plt.plot(history.history["loss"])
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()


# 分析学习的嵌入。
movie_embeddings = model.get_layer("item_embeddings").get_weights()[0]
print("Embeddings shape:", movie_embeddings.shape)
# Embeddings shape: (1406, 50)

# 查找相关电影
# 定义一个包含一些名为 的电影的列表query_movies。

query_movies = [
    "Matrix, The (1999)",
    "Star Wars: Episode IV - A New Hope (1977)",
    "Lion King, The (1994)",
    "Terminator 2: Judgment Day (1991)",
    "Godfather, The (1972)",
]

# 获取电影的嵌入query_movies。

query_embeddings = []

for movie_title in query_movies:
    movieId = get_movie_id_by_title(movie_title)
    token_id = vocabulary_lookup[movieId]
    movie_embedding = movie_embeddings[token_id]
    query_embeddings.append(movie_embedding)

query_embeddings = np.array(query_embeddings)

# 计算嵌入和所有其他电影之间的余弦相似度query_movies ，然后为每个电影选择前 k 个。
# 向量内积/向量的模 = 余弦相似性
# normalize_L2=向量的各自分量/向量的模; normalize_L2 -》向量的内积 -》 余弦相似性
# 余弦相似度与两个特征在经过L2归一化之后的矩阵内积等价
# tf.linalg.matmul：将矩阵 a 乘以矩阵 b ，产生 a * b 。
similarities = tf.linalg.matmul(
    tf.math.l2_normalize(query_embeddings),
    tf.math.l2_normalize(movie_embeddings),
    transpose_b=True,
)

_, indices = tf.math.top_k(similarities, k=5)
indices = indices.numpy().tolist()

# 显示最相关的电影query_movies。
for idx, title in enumerate(query_movies):
    print(title)
    print("".rjust(len(title), "-"))
    similar_tokens = indices[idx]
    for token in similar_tokens:
        similar_movieId = vocabulary[token]
        similar_title = get_movie_title_by_id(similar_movieId)
        print(f"- {similar_title}")
    print()

# # 使用嵌入投影仪可视化嵌入
# import io
#
# out_v = io.open("embeddings.tsv", "w", encoding="utf-8")
# out_m = io.open("metadata.tsv", "w", encoding="utf-8")
#
# for idx, movie_id in enumerate(vocabulary[1:]):
#     movie_title = list(movies[movies.movieId == movie_id].title)[0]
#     vector = movie_embeddings[idx]
#     out_v.write("\t".join([str(x) for x in vector]) + "\n")
#     out_m.write(movie_title + "\n")
#
# out_v.close()
# out_m.close()


def main():
    pass


if __name__ == '__main__':
    main()
