#!/usr/bin/env python3
# -*- coding: utf-8 -*-

################################################################################# 第三方shap 包实现 #################################################################################
# pip3 install shap
# https://github.com/slundberg/shap
import transformers
import shap

# load a transformers pipeline model
model = transformers.pipeline('sentiment-analysis', return_all_scores=True)

# explain the model on two sample inputs
explainer = shap.Explainer(model)
shap_values = explainer(["What a great movie! ...if you have no taste."])

# visualize the first prediction's explanation for the POSITIVE output class
shap.plots.text(shap_values[0, :, "POSITIVE"])

##################################################################################################################################################################
# 通过积分梯度（Integrated Gradients）来给输入进行重要性排序
# 原始论文：https://arxiv.org/abs/1703.01365
# 博客介绍：https://kexue.fm/archives/7533
# 来源：https://github.com/bojone/bert4keras/tree/master/examples
# 模型下载于：
# https://github.com/brightmart/albert_zh
# https://storage.googleapis.com/albert_zh/albert_small_zh_google.zip
# albert_small_google_zh(累积学习10亿个样本,google版本)，


import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import random
import numpy as np
from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

from keras.layers import Layer, Input
from bert4keras.backend import K, batch_gather
from keras.models import Model
from bert4keras.snippets import uniout
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve

set_gelu('tanh')  # 切换gelu版本

num_classes = 2
maxlen = 128
batch_size = 32
config_path = rf'D:/Users/{USERNAME}/data/albert_small_zh_google/albert_config_small_google.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/albert_small_zh_google/albert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/albert_small_zh_google/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：(文本, 标签id)
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(rf'D:/Users/{USERNAME}/data/sentiment/sentiment.train.data')
valid_data = load_data(rf'D:/Users/{USERNAME}/data/sentiment/sentiment.valid.data')
test_data = load_data(rf'D:/Users/{USERNAME}/data/sentiment/sentiment.test.data')

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def data_generator(train_data, batch_size=32, shuffle=True, train=True):
    while True:
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if shuffle and train:
            random.shuffle(train_data)
        for text, label in train_data:
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == batch_size:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        if not train:
            if batch_token_ids:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
            break

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
# 带有分段线性学习率的优化器
#         其中schedule是形如{1000: 1, 2000: 0.1}的字典，
#         表示0～1000步内学习率线性地从零增加到100%，然后
#         1000～2000步内线性地降到10%，2000步以后保持10%
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    # optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
    #     1000: 1,
    #     2000: 0.1
    # }),
    metrics=['accuracy'],
)

# 转换数据集
# train_generator = data_generator(train_data, batch_size)
# valid_generator = data_generator(valid_data, batch_size)
# test_generator = data_generator(test_data, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total




############################################### 模型训练及评估 ###################################################################################

historys = model.fit(data_generator(train_data, batch_size),
                     steps_per_epoch=len(train_data) // batch_size,
                     # 在声明一个 epoch 完成并开始下一个 epoch 之前从 generator 产生的总步数（批次样本）。 它通常应该等于你的数据集的样本数量除以批量大小。len(train_data)/batch_size
                     validation_data=data_generator(valid_data, batch_size),
                     validation_steps=len(valid_data) // batch_size,
                     # 仅当 validation_data 是一个生成器时才可用。 在停止前 generator 生成的总步数（样本批数）。len(dev_data)/batch_size
                     epochs=50,
                     verbose=1,
                     shuffle=True,
                     )
print(historys.history)

y_pred = model.predict(data_generator(test_data, batch_size, shuffle=False, train=False))
y_score = y_pred[:, -1]
y_true = [t[-1] for t in test_data]
confusion_matrix(y_true, [1 if t >= 0.5 else 0 for t in y_score])
# 0.844

################################################# 通过积分梯度（Integrated Gradients）来给输入进行重要性排序 #################################################################################

def get_model_grad(model, token_ids, segment_ids, label_in):
    """获取某个标签对向量层的梯度"""
    with tf.GradientTape() as tape:
        input = model.get_layer('Embedding-Token').embeddings
        y_pred = model([token_ids, segment_ids], training=False)
        output = y_pred[:, label_in]
        gradients = tape.gradient(output, [input])[0]  # Embedding梯度
        grads = gradients * input
    return grads

# 获取原始embedding层
embeddings = model.get_layer('Embedding-Token').embeddings
values = K.eval(embeddings)  # tf.Tensor -> np.array

result_list = []
for text in [ u'这家店真黑心', u'图太乱了 有点看不懂重点', '讲故事的时候很难让孩子集中', u'这是一本很好看的书', u'这是一本很糟糕的书']:
    print('-'*30, text)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = sequence_padding([token_ids], length=maxlen), sequence_padding([segment_ids], length=maxlen)
    preds = model.predict([token_ids, segment_ids], verbose=0)
    label = np.argmax(preds[0])

    pred_grads = []
    n = 20
    for i in range(n):
        # nlp任务中参照背景通常直接选零向量，所以这里
        # 让embedding层从零渐变到原始值，以实现路径变换。
        alpha = 1.0 * i / (n - 1)
        K.set_value(embeddings, alpha * values)
        pred_grad = get_model_grad(model, token_ids, segment_ids, label)
        pred_grads.append(pred_grad)

    # 然后求平均
    mean_pred_grads = np.mean(pred_grads, 0)

    # 这时候我们得到形状为(seq_len, hidden_dim)的矩阵，我们要将它变换成(seq_len,)
    # 这时候有两种方案：1、直接求模长；2、取绝对值后再取最大。两者效果差不多。
    scores = np.sqrt((mean_pred_grads**2).sum(axis=1))
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = scores.round(4)
    ids = [_id for _id in token_ids[0] if _id > 0]
    results1 = [(w, scores[v]) for w, v in zip(tokenizer.ids_to_tokens(ids), ids)]
    print(results1[1:-1])

    # 方案 2、取绝对值后再取最大
    scores = np.abs(mean_pred_grads).max(axis=1)
    scores = (scores - scores.min()) / (scores.max() - scores.min())
    scores = scores.round(4)
    ids = [_id for _id in token_ids[0] if _id > 0]
    results2 = [(w, scores[v]) for w, v in zip(tokenizer.ids_to_tokens(ids), ids)]
    print(results2[1:-1])
    result_list.append([results1[1:-1], results2[1:-1]])

# ------------------------------ 这家店真黑心
# [('这', 1.0), ('家', 0.5968), ('店', 0.5769), ('真', 0.7146), ('黑', 0.6856), ('心', 0.4891)]
# [('这', 1.0), ('家', 0.4878), ('店', 0.4308), ('真', 0.5372), ('黑', 0.6127), ('心', 0.3741)]
# ------------------------------ 图太乱了 有点看不懂重点
# [('图', 0.3615), ('太', 0.51), ('乱', 0.5184), ('了', 0.7016), ('有', 0.5052), ('点', 0.7518), ('看', 0.5037), ('不', 1.0), ('懂', 0.5813), ('重', 0.5008), ('点', 0.7518)]
# [('图', 0.2857), ('太', 0.4945), ('乱', 0.4426), ('了', 0.9089), ('有', 0.4122), ('点', 0.7033), ('看', 0.4722), ('不', 1.0), ('懂', 0.7469), ('重', 0.8275), ('点', 0.7033)]
# ------------------------------ 讲故事的时候很难让孩子集中
# [('讲', 0.3337), ('故', 0.4751), ('事', 0.4856), ('的', 0.4192), ('时', 0.3636), ('候', 0.4403), ('很', 1.0), ('难', 0.9525), ('让', 0.6668), ('孩', 0.4725), ('子', 0.1931), ('集', 0.2623), ('中', 0.3146)]
# [('讲', 0.2119), ('故', 0.4284), ('事', 0.4665), ('的', 0.3327), ('时', 0.2989), ('候', 0.4666), ('很', 1.0), ('难', 0.8644), ('让', 0.6868), ('孩', 0.5733), ('子', 0.1824), ('集', 0.1948), ('中', 0.2494)]
# ------------------------------ 这是一本很好看的书
# [('这', 1.0), ('是', 0.899), ('一', 0.5212), ('本', 0.7363), ('很', 0.9652), ('好', 0.8309), ('看', 0.9523), ('的', 0.6179), ('书', 0.4557)]
# [('这', 0.7845), ('是', 0.787), ('一', 0.3255), ('本', 0.5876), ('很', 0.8018), ('好', 0.6167), ('看', 1.0), ('的', 0.3475), ('书', 0.2783)]
# ------------------------------ 这是一本很糟糕的书
# [('这', 0.8185), ('是', 0.5494), ('一', 0.3393), ('本', 0.5586), ('很', 0.976), ('糟', 1.0), ('糕', 0.4005), ('的', 0.5), ('书', 0.43)]
# [('这', 0.7943), ('是', 0.4889), ('一', 0.2518), ('本', 0.4996), ('很', 1.0), ('糟', 0.7303), ('糕', 0.2435), ('的', 0.3136), ('书', 0.35)]

######################################################### 将结果可视化 #########################################################################################################
# result_list = [[[('这', 1.0), ('家', 0.5968), ('店', 0.5769), ('真', 0.7146), ('黑', 0.6856), ('心', 0.4891)], [('这', 1.0), ('家', 0.4878), ('店', 0.4308), ('真', 0.5372), ('黑', 0.6127), ('心', 0.3741)]], [[('图', 0.3615), ('太', 0.51), ('乱', 0.5184), ('了', 0.7016), ('有', 0.5052), ('点', 0.7518), ('看', 0.5037), ('不', 1.0), ('懂', 0.5813), ('重', 0.5008), ('点', 0.7518)], [('图', 0.2857), ('太', 0.4945), ('乱', 0.4426), ('了', 0.9089), ('有', 0.4122), ('点', 0.7033), ('看', 0.4722), ('不', 1.0), ('懂', 0.7469), ('重', 0.8275), ('点', 0.7033)]], [[('讲', 0.3337), ('故', 0.4751), ('事', 0.4856), ('的', 0.4192), ('时', 0.3636), ('候', 0.4403), ('很', 1.0), ('难', 0.9525), ('让', 0.6668), ('孩', 0.4725), ('子', 0.1931), ('集', 0.2623), ('中', 0.3146)], [('讲', 0.2119), ('故', 0.4284), ('事', 0.4665), ('的', 0.3327), ('时', 0.2989), ('候', 0.4666), ('很', 1.0), ('难', 0.8644), ('让', 0.6868), ('孩', 0.5733), ('子', 0.1824), ('集', 0.1948), ('中', 0.2494)]], [[('这', 1.0), ('是', 0.899), ('一', 0.5212), ('本', 0.7363), ('很', 0.9652), ('好', 0.8309), ('看', 0.9523), ('的', 0.6179), ('书', 0.4557)], [('这', 0.7845), ('是', 0.787), ('一', 0.3255), ('本', 0.5876), ('很', 0.8018), ('好', 0.6167), ('看', 1.0), ('的', 0.3475), ('书', 0.2783)]], [[('这', 0.8185), ('是', 0.5494), ('一', 0.3393), ('本', 0.5586), ('很', 0.976), ('糟', 1.0), ('糕', 0.4005), ('的', 0.5), ('书', 0.43)], [('这', 0.7943), ('是', 0.4889), ('一', 0.2518), ('本', 0.4996), ('很', 1.0), ('糟', 0.7303), ('糕', 0.2435), ('的', 0.3136), ('书', 0.35)]]]

import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题
import numpy as np
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import pandas as pd
ax = plt.subplot(111)  # 注意:一般都在ax中设置,不再plot中设置

# 颜色由红到白
color_list = ['#ff0000', '#ff1111', '#ff2222', '#ff3333', '#ff4444', '#ff5555', '#ff6666', '#ff7777', '#ff8888', '#ff9999', '#ffaaaa', '#ffbbbb', '#ffcccc', '#ffdddd', '#ffeeee', '#ffffff']

for row, result in enumerate(result_list):
    x2 = np.linspace(row, row+1, 1)
    ax.fill_between(x2, row, row+1, facecolor=color_list[row])
    max_pred = max([t[-1] for t in result[0]])
    for col, ret in enumerate(result[0]):
        word = ret[0]
        pred = ret[1]
        # facecolor='green', '#ff0000'
        facecolor = (1, 1-pred/max_pred, 1-pred/max_pred)
        ax.text(col, row, word, bbox=dict(facecolor=facecolor, alpha=0.5))

plt.xlim(0, 15)
plt.ylim(0, 5)
ax.xaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.yaxis.set_major_locator(MultipleLocator(1))  # 设置y主坐标间隔 1
ax.xaxis.grid(True, which='major')  # major,color='black'
ax.yaxis.grid(True, which='major')  # major,color='black'
plt.show()

def main():
    pass

if __name__ == '__main__':
    main()
