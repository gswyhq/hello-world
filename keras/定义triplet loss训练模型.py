#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Concatenate
from keras.layers import Input, LSTM, Dense, Lambda
from keras.layers.merging import dot
from keras.models import Model
from keras import backend as K
from keras.layers import Input, Dense, Concatenate, Reshape, Dropout, TimeDistributed, Flatten
from keras.layers import Embedding
from sklearn.metrics.pairwise import cosine_similarity

# ranking loss在很多不同的领域，任务和神经网络结构（比如siamese net或者Triplet net）中被广泛地应用。
# 其广泛应用但缺乏对其命名标准化导致了其拥有很多其他别名，比如对比损失Contrastive loss，边缘损失Margin loss，铰链损失hinge loss和我们常见的三元组损失Triplet loss等。
#
# ranking loss函数：不像其他损失函数，比如交叉熵损失和均方差损失函数，这些损失的设计目的就是学习如何去直接地预测标签，或者回归出一个值，又或者是在给定输入的情况下预测出一组值。
# ranking loss的目的是去预测输入样本之间的相对距离。这个任务经常也被称之为度量学习(metric learning)。
# 在使用ranking loss的过程中，我们首先从两个样本(或三个样本)输入数据中提取出特征，并且得到其各自的嵌入表达(embedded representation)。
# 然后，我们定义一个距离度量函数用以度量这些表达之间的相似度，比如说欧式距离。
# 最终，我们训练这个特征提取器，以对于特定的样本对（sample pair）产生特定的相似度度量。
# 我们并不需要关心这些表达的具体值是多少，只需要关心样本之间的距离是否足够接近或者足够远离。
# ranking loss的表达式，主要针对以下两种不同的设置:
# 1、使用一对的训练数据点（即是两个一组）
# 2、使用三元组的训练数据点（即是三个数据点一组）
# 在两个一组的这个设置中，由训练样本中采样到的正样本和负样本组成的两种样本对作为训练输入使用。
# 正样本对(xa,xp)由两部分组成，一个锚点样本xa和 另一个和之标签相同的样本xp，这个样本xp与锚点样本在我们需要评价的度量指标上应该是相似的（经常体现在标签一样）；
# 负样本对(xa,xn)由一个锚点样本xa和一个标签不同的样本xn组成，xn在度量上应该和xa不同。（体现在标签不一致）。
# 现在，我们的目标就是学习出一个特征表征，这个表征使得正样本对中的度量距离d尽可能的小，而在负样本对中，这个距离应该要大于一个人为设定的超参数——阈值m。
# 成对样本的ranking loss强制样本的表征在正样本对中拥有趋向于0的度量距离，而在负样本对中，这个距离则至少大于一个阈值。
# 用ra,rp,rn分别表示这些样本的特征表征，我们可以有以下的式子：
# 正样本对(xa,xp)时：loss = d(ra, rp)
# 负样本对(xa,xn)时：loss = max(0, m-d(ra, rn))
# 对于正样本对来说，这个loss随着样本对输入到网络生成的表征之间的距离的减小而减少，增大而增大，直至变成0为止。
# 对于负样本来说，这个loss只有在所有负样本对的元素之间的表征的距离都大于阈值m的时候才能变成0。
# 当实际负样本对的距离小于阈值的时候，这个loss就是个正值，因此网络的参数能够继续更新优化，以便产生更适合的表征。
# 这里设置阈值的目的是，当某个负样本对中的表征足够好，体现在其距离足够远的时候，就没有必要在该负样本对中浪费时间去增大这个距离了，因此进一步的训练将会关注在其他更加难分别的样本对中。
#
# 三元组样本对的ranking loss称之为triplet loss。
# 在这个设置中，与二元组不同的是，输入样本对是一个从训练集中采样得到的三元组。
# 这个三元组(xa,xp,xn)由一个锚点样本xa，一个正样本xp，一个负样本xn组成。
# 其目标是锚点样本与负样本之间的距离d(ra,rn)与锚点样本和正样本之间的距离d(ra,rp)之差大于一个阈值m，可以表示为：
# L(ra,rp,rn)=max(0,m+d(ra,rp)−d(ra,rn))



# triplet loss 的最大特点是，它不会去过分要求同样标签的样本在向量空间中的距离要相近。
# Loss = max(d(a, p) - d(a, n) + margin, 0)
# a 是训练集中的一个随机的样本（anchor）
# p 是训练集中与 a 属于同一标签的一个随机样本（positive）
# n 是训练集中与 a 属于不同标签的一个随机样本（negative）
# d 是自定义的一个表征两个样本之间的距离函数
# margin 是一个大于0的常数。
# 最终的优化目标是拉近a和p的距离，拉远a和n的距离。

# pairwise hinge loss
# 其衡量的是pairwise场景下正负样本的差异，公式如下所示，其中margin代表的是预设的阈值，u代表输入query，d+代表的是正样本，d−代表的是负样本，
# <>代表的是两个向量之间的相似度，该公式代表的含义是：当相似样本时，只有当输入query与正样本足够相似时，loss才会降为0；
# 当非相似样本时，样本距离大于阈值margin时，loss降为0。
# loss= <u, d+>  如果相似样本时，样本约相似，Loss越小
# loss= max(0,margin−<u,d−>)  如果非相似样本时，样本越不相似，loss越小

# loss= max(0,margin - pos_score + neg_score)
# 即我们希望正样本分数(pos_score)越高越好，负样本分数(neg_score)越低越好，但二者得分之差最多到margin就足够了，差距增大并不会有任何奖励。
# 比如，我们想训练词向量，我们希望经常同时出现的词，他们的向量内积越大越好；不经常同时出现的词，他们的向量内积越小越好。则我们的hinge loss function可以是：
# l(w,w+,w−)=max(0,1− wT⋅w+ + wT⋅w−)
# 其中，w是当前正在处理的词，wT为w的转置.
# w+: 是w在文中前3个词和后3个词中的某一个词，
# w−: 是随机选的一个词。

# hinge loss
# Loss的目的是让预测值y_pred 和y_true 相等的时候，返回0，否则返回一个线性值
# y_true = np.random.choice([-1, 1], size=(2, 3))
# y_pred = np.random.random(size=(2, 3))
# y_pred = tf.convert_to_tensor(y_pred)
# y_true = tf.cast(y_true, y_pred.dtype)
# y_true = _maybe_convert_labels(y_true)  # 二进制（0或1）标签，它们将被转换为-1或1。
# return backend.mean(tf.maximum(1. - y_true * y_pred, 0.), axis=-1)  # maximum逐位比较取其大者
def hinge_loss(
    positive_scores: torch.tensor,
    negative_scores: torch.tensor,
) -> torch.tensor:
    """
    https://github.com/ShopRunner/collie/blob/main/collie/loss/hinge.py
    Parameters
    ----------
    positive_scores: torch.tensor, 1-d
        Tensor containing scores for known positive items
    negative_scores: torch.tensor, 1-d
        Tensor containing scores for a single sampled negative item
    """
    score_difference = (positive_scores - negative_scores)

    ideal_difference = 1

    loss = torch.clamp((ideal_difference - score_difference), min=0)

    return (loss.sum() + loss.pow(2).sum()) / len(positive_scores)

# bpr loss
# 其同样衡量的是pairwise场景下正负样本的差异，公式如下，可以看出其整体的含义和pairwise hinge lossl类似，但少了 margin参数，这使得模型变得更加鲁棒。
# loss=log(1+exp(<u,d−>−<u,d+>))
def bpr_loss(
    positive_scores: torch.tensor,
    negative_scores: torch.tensor,
):
    # https://github.com/ShopRunner/collie/blob/main/collie/loss/bpr.py
    preds = positive_scores - negative_scores
    ideal_difference = 1
    loss = (ideal_difference - torch.sigmoid(preds))
    return (loss.sum() + loss.pow(2).sum()) / len(positive_scores)

# triplet loss
# Triplet loss的输入是一个三元组<a,p,n>，a表示本体Anchor，p表示与a是同一类别的样本Positive，n表示与a不同类别的样本Negative，其目的是通过学习后使得a与p的距离变小，而和n的距离变大
# L=max(d(a,p)−d(a,n)+margin,0)
# 从名称上可以看出，该损失函数的输入由三部分构成，这三部分分别是anchor(锚点)、positive(正例)以及negative(负例)。triplet loss的核心思想由有如下三部分构成：
# 1、anchor与negative差异越大越好
# 2、anchor与positive差异越小越好
# 3、positive与negative差异越大越好
# 基于上面三个子思想，写出triplet loss的公式，其中anc、pos以及neg是anchor、positive以及negative在模型中的表示，可以理解为 anc=f(anchor)， pos=f(positive)， neg=f(negative)。
# loss =
# 2∗∥pos−anc∥2−∥pos−neg∥2−∥anc−neg∥2+margin    if label=1
# 2∗∥neg−anc∥2−∥neg−pos∥2−∥anc−pos∥2+margin    if label=0
# 上述公式的含义为：
# 如果label为1，则pos与anc差异越小越好 + pos与neg差异越大越好 + anc与neg差异越大越好
# 如果label为0，则neg与anc差异越小越好 + pos与neg差异越大越好 + anc与pos差异越大越好
# margin存在的意义为提升模型学习的难度，因为如果不加margin，则模型很容易把anc、pos以及neg弄成0，这样loss也会很小。
# 这里给出triplet loss的tf代码实现：
# def triplet_loss(self, pos, neg, anc, label):
#     part1 = tf.reduce_sum(tf.square(pos - anc), axis=-1)
#     part2 = tf.reduce_sum(tf.square(pos - neg), axis=-1)
#     part3 = tf.reduce_sum(tf.square(anc - neg), axis=-1)
#
#     part1_1 = tf.reduce_sum(tf.square(neg - anc), axis=-1)
#     part2_1 = tf.reduce_sum(tf.square(neg - pos), axis=-1)
#     part3_1 = tf.reduce_sum(tf.square(anc - pos), axis=-1)
#
#     loss1 = tf.expand_dims(2 * part1 - part2 - part3 + self.triplet_loss_margin, axis=-1)
#     loss2 = tf.expand_dims(2 * part1_1 - part2_1 - part3_1 + self.triplet_loss_margin, axis=-1)
#     loss = tf.where(tf.equal(label, 1.0), loss1, loss2)
#
#     return loss

# 原文链接：https://blog.csdn.net/weixin_37688445/article/details/117917746

def build_model3(time_step, feature_dim=44, margin=0.1):
    emb_encoder = Embedding(737, 32, input_length=(time_step, feature_dim))
    time_encoder = TimeDistributed(Dense(32), input_shape=(time_step, feature_dim, 32))
    reshape_encoder = Reshape(target_shape=(time_step, feature_dim * 32,))
    drop_encoder = Dropout(0.2)
    concat_encoder = Concatenate()
    lstm_encoder = LSTM(256, input_shape=(time_step, feature_dim * 32 + 1), return_sequences=False)
    dense_encoder = Dense(32, kernel_initializer="uniform", activation='relu')

    def encode(input_1, input_2):
        embedding = emb_encoder(input_1)
        time_embedding = time_encoder(embedding)
        embedding = reshape_encoder(time_embedding)
        embedding = drop_encoder(embedding)

        concat = concat_encoder([embedding, input_2])
        #     print(concat)

        lstm2 = lstm_encoder(concat)
        lstm2 = drop_encoder(lstm2)

        dense1 = dense_encoder(lstm2)
        return dense1

    input_1 = Input(shape=(time_step, feature_dim,), name="input_1")
    input_2 = Input(shape=(time_step, 1,), name="input_2")

    input_q1 = Input(shape=(time_step, feature_dim,), name="input_q1")
    input_q2 = Input(shape=(time_step, 1,), name="input_q2")

    input_a1 = Input(shape=(time_step, feature_dim,), name="input_a1")
    input_a2 = Input(shape=(time_step, 1,), name="input_a2")

    q_encoded = encode(input_1, input_2)
    a_right_encoded = encode(input_q1, input_q2)
    a_wrong_encoded = encode(input_a1, input_a2)

    # normalize: 是否在点积之前对即将进行点积的轴进行 L2 标准化。 如果设置成 True，那么输出两个样本之间的余弦相似值。
    right_cos = dot([q_encoded, a_right_encoded], -1, normalize=True)
    wrong_cos = dot([q_encoded, a_wrong_encoded], -1, normalize=True)

    loss = Lambda(lambda x: K.relu(margin + x[0] - x[1]))([wrong_cos, right_cos])
    # loss=Lambda(lambda x: K.log(1+K.exp(x[0]-x[1])))([wrong_cos,right_cos])

    model_train = Model(inputs=[input_1, input_2, input_q1, input_q2, input_a1, input_a2, ], outputs=loss)

    model_train.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)

    print(model_train.count_params())

    return model_train

time_step = 6
feature_dim = 44
batch_size = 32
margin=0.1

keras.backend.clear_session()
model = build_model3(time_step, feature_dim=feature_dim, margin=margin)
print(model.count_params())

def generator_data(batch_size=32):
    while True:
        yield {
                  "input_1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_2": np.random.random(size=(batch_size, time_step, 1)),
                  "input_q1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_q2": np.random.random(size=(batch_size, time_step, 1)),
                  "input_a1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_a2": np.random.random(size=(batch_size, time_step, 1)),
              }, np.reshape([1 for _ in range(batch_size)], (batch_size, 1))

train_size = 200
dev_size = 40
historys = model.fit(generator_data(batch_size=batch_size),
                    steps_per_epoch=train_size//batch_size,  # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                    validation_data=generator_data(batch_size=batch_size),
                    validation_steps = dev_size//batch_size,
                    epochs=2,
                    verbose=1,
                    shuffle=True,
         )
print(historys.history)

test_data = {
                  "input_1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_2": np.random.random(size=(batch_size, time_step, 1)),
                  "input_q1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_q2": np.random.random(size=(batch_size, time_step, 1)),
                  "input_a1": np.random.randint(1, 737, size=(batch_size, time_step, feature_dim)),
                  "input_a2": np.random.random(size=(batch_size, time_step, 1)),
              }
y_preds = model.predict(test_data)

# 提取中间层，重构模型
representation_model = Model(inputs=[model.get_layer('input_1').input,
                                     model.get_layer('input_2').input],
                             outputs=model.get_layer('dense_1').output)

# 使用重构模型预测
y_pred1 = representation_model({
                  "input_1": test_data["input_1"],
                  "input_2": test_data["input_2"]})

y_pred2 = representation_model({
                  "input_1": test_data["input_q1"],
                  "input_2": test_data["input_q2"]})

y_pred3 = representation_model({
                  "input_1": test_data["input_a1"],
                  "input_2": test_data["input_a2"]})

def my_cos(a, b):
    '''计算余弦相似度'''
    cos_ab = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_ab

# 对比三种方法计算余弦相似度的结果：
cosine_similarity(np.array(y_pred1[:1,:]), np.array(y_pred1[1:2, :]))
tf.keras.layers.Dot(axes=1, normalize=True)([y_pred1[:1, :], y_pred1[1:2, :]])
my_cos(np.array(y_pred1[0,:]), np.array(y_pred1[1, :]))

# 重构模型计算loss
loss_list = []
for idx in range(y_pred3.shape[0]):
    q_encoded = y_pred1[idx, :]
    a_right_encoded = y_pred2[idx, :]
    a_wrong_encoded = y_pred3[idx, :]

    right_cos = my_cos(np.array(q_encoded), np.array(a_right_encoded))
    wrong_cos = my_cos(np.array(q_encoded), np.array(a_wrong_encoded))

    loss = K.relu(margin + wrong_cos - right_cos)
    loss_list.append(loss.numpy())

# 对比原有模型结果，及重构模型的结果；
[k-v[0] for k, v in zip(loss_list, y_preds)]

def main():
    pass


if __name__ == '__main__':
    main()
