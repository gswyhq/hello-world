#!/usr/bin/env python
# coding=utf-8


import os 
from sentence_transformers import SentenceTransformer
USERNAME = os.getenv("USERNAME")


import pickle
import numpy as np 
import pandas as pd 
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
import tensorflow as tf
from tensorflow.keras.layers import Input,Embedding,LSTM, Bidirectional,Dense,Dropout,BatchNormalization,Reshape, Flatten, Concatenate, Lambda, Add, Conv2D, MaxPooling2D, LSTM, RepeatVector
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from tensorflow import keras 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils import shuffle
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import callbacks
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from scipy.optimize import linear_sum_assignment as linear_assignment
from keras.utils import plot_model
from tensorflow.keras.models import Model, load_model


from pylab import mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体   
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题


################################################################################################################################
# 以 toutiao的新闻主题 数据集为例，训练 K-Means 模型来进行聚类为 10 个组：

metrics_nmi = normalized_mutual_info_score
metrics_ari = adjusted_rand_score


def metrics_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
        
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size 

data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'label'])
df = shuffle(df)

texts = [' '.join(list(text)) for text in df['title'].values]
tokenizer = Tokenizer(num_words=10000)
# 根据文本列表更新内部词汇表。
tokenizer.fit_on_texts(texts)

# 将文本中的每个文本转换为整数序列。
# 只有最常出现的“num_words”字才会被考虑在内。
# 只有分词器知道的单词才会被考虑在内。
sequences = tokenizer.texts_to_sequences(texts)
# dict {word: index}
word_index = tokenizer.word_index

print('tokens数量：', len(word_index))

data = pad_sequences(sequences, maxlen=48)
print('Shape of data tensor:', data.shape)

code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = np.array([code2id[label] for label in df['label'].values])
n_clusters = len(code2id)

del texts, sequences

x = data


kmeans = KMeans(n_clusters=n_clusters, n_init=20)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.
metrics_acc(y, y_pred_kmeans)
# 正确率：0.11221412743540429

################################################################################################################################
# 基于自动编码器的聚类步骤：
# 1、预训练一个自动编码器，以学习无标签数据集的初始压缩后的特征表示.
# 2、在编码器上堆积聚类层(clustering)，以分配编码器输出到一个聚类组. 聚类层的权重初始化采用的是基于当前得到的 K-Means 聚类中心.
# 3、聚类模型的训练，以同时改善聚类层和编码器。

def build_autoencoder3(input_shape, encoding_dim, vocab_size):
    # 编码器
    inputs = Input(shape=input_shape, name='input')
    embedded = Embedding(vocab_size, encoding_dim, )(inputs)
    lstm1 = LSTM(encoding_dim, return_sequences=True, name='lstm1')(embedded)
    lstm2 = LSTM(encoding_dim, return_sequences=False, name='lstm2')(lstm1)
    dense1 = Dense(10, activation='relu', )(lstm2)
    dense2 = Dense(encoding_dim, activation='relu')(dense1)
    repeat = RepeatVector(input_shape[-1], name='repeat')(dense2)  # RepeatVector层将输入重复n次
    # 解码器
    lstm3 = LSTM(encoding_dim, return_sequences=True, name='lstm3')(repeat)
    lstm4 = LSTM(encoding_dim, return_sequences=True, name='lstm4')(lstm3)

    outputs = Dense(vocab_size, activation='softmax')(lstm4)
    # 构建自动编码器模型
    autoencoder = Model(inputs, outputs, name='autoencoder')
    # 编码器模型，用于预测
    encoder = Model(inputs, dense1, name='encoder')
    # 解码器模型，用于生成
    decoder = Model(dense1, outputs, name='decoder')
    autoencoder.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy')
    return autoencoder, encoder, decoder


def build_autoencoder4(input_shape, encoding_dim, vocab_size):
    encoder_inputs = Input(shape=(input_shape[-1],), name='Encoder-Input')
    x = Embedding(vocab_size, encoding_dim, input_length=input_shape[-1], mask_zero=False)(encoder_inputs)
    state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
    encoder = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
    seq2seq_encoder_out = encoder(encoder_inputs)

    # 第三步：解码器模型（Decoder Model）
    decoded = RepeatVector(input_shape[-1])(seq2seq_encoder_out)
    decoder_lstm_output = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))(decoded)
    decoder_outputs = Dense(vocab_size, activation='softmax', name='Final-Output-Dense-before')(decoder_lstm_output)
    decoder= Model(seq2seq_encoder_out, decoder_outputs, name='decoder')
    # 第四步：构建模型及训练(Combining Model and Training)
    autoencoder = Model(encoder_inputs, decoder_outputs)
    autoencoder.compile(optimizer=optimizers.Nadam(learning_rate=0.001), loss='sparse_categorical_crossentropy')
    return autoencoder,  encoder, decoder


data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'label'])
df = shuffle(df)

texts = [' '.join(list(text)) for text in df['title'].values]

save_tokenizer_file = './results/tokenizer.pkl' 
if os.path.exists(save_tokenizer_file):
    with open(save_tokenizer_file, 'rb') as f:
        tokenizer = pickle.load(f)
    num_words = 4991
else:
    word_counts = {}
    for text in df['title'].values:
        for word in list(text):
            word_counts.setdefault(word, 0)
            word_counts[word] += 1
    filters = ''.join([k for k, v in word_counts.items() if v<3])
    num_words = len([k for k, v in word_counts.items() if v>=3]) + 1
    tokenizer = Tokenizer(num_words=num_words, filters=filters)
    
    # 根据文本列表更新内部词汇表。
    tokenizer.fit_on_texts(texts)
    
    with open(save_tokenizer_file, 'wb') as f:
        pickle.dump(tokenizer, f)

# texts[:3]
# Out[75]: 
# ['河 南 理 工 大 学 会 成 为 2 1 1 大 学 吗 ？',
#  '鲜 卑 族 和 朝 鲜 族 有 关 系 吗 ？',
#  '云 南 边 境 与 缅 甸 通 婚 的 现 象 有 多 普 遍 ？']

sequences = tokenizer.texts_to_sequences(texts)
# dict {word: index}
word_index = tokenizer.word_index

print('tokens数量：', len(word_index))  
# tokens数量： 4973

data = pad_sequences(sequences, maxlen=48)
print('Shape of data tensor:', data.shape)
# Shape of data tensor: (382688, 48)

code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = np.array([code2id[label] for label in df['label'].values])
n_clusters = len(code2id)

batch_size = 64
save_dir = './results'
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

model_checkpoint = ModelCheckpoint(filepath=save_dir + '/ae_weights.h5', save_best_only=True, save_weights_only=False)

# 使用示例
input_shape = (48, )
encoding_dim = 128  # 编码维度
max_length = 48
vocab_size = num_words

keras.backend.clear_session()
autoencoder, encoder, decoder = build_autoencoder3(input_shape, encoding_dim, vocab_size)
print(autoencoder.summary())

# Model: "autoencoder"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input (InputLayer)          [(None, 48)]              0         
#                                                                  
#  embedding (Embedding)       (None, 48, 128)           638848    
#                                                                  
#  lstm1 (LSTM)                (None, 48, 128)           131584    
#                                                                  
#  lstm2 (LSTM)                (None, 128)               131584    
#                                                                  
#  dense (Dense)               (None, 10)                1290      
#                                                                  
#  dense_1 (Dense)             (None, 128)               1408      
#                                                                  
#  repeat (RepeatVector)       (None, 48, 128)           0         
#                                                                  
#  lstm3 (LSTM)                (None, 48, 128)           131584    
#                                                                  
#  lstm4 (LSTM)                (None, 48, 128)           131584    
#                                                                  
#  dense_2 (Dense)             (None, 48, 4991)          643839    
#                                                                  
# =================================================================
# Total params: 1,811,721
# Trainable params: 1,811,721
# Non-trainable params: 0
# _________________________________________________________________

# X_train, X_val = train_test_split(data, test_size=0.1)

autoencoder.fit(data, data, batch_size=batch_size, epochs=100,
                verbose=1,
                # validation_split=0.2,
                shuffle=True,
                # validation_data=([x_test, y_test], y_test),
                callbacks=[early_stopping, model_checkpoint]
                )

# 
# Epoch 1/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.4213

# 5980/5980 [==============================] - 3396s 568ms/step - loss: 3.4213
# Epoch 2/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.4135

# 5980/5980 [==============================] - 3794s 634ms/step - loss: 3.4135
# Epoch 3/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.4079

# 5980/5980 [==============================] - 4594s 768ms/step - loss: 3.4079
# Epoch 4/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.4039

# 5980/5980 [==============================] - 4702s 786ms/step - loss: 3.4039
# Epoch 5/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.4010

# 5980/5980 [==============================] - 4878s 816ms/step - loss: 3.4010
# Epoch 6/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3994

# 5980/5980 [==============================] - 5146s 861ms/step - loss: 3.3994
# Epoch 7/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3971

# 5980/5980 [==============================] - 5423s 907ms/step - loss: 3.3971
# Epoch 8/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3961

# 5980/5980 [==============================] - 5558s 930ms/step - loss: 3.3961
# Epoch 9/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3952

# 5980/5980 [==============================] - 5711s 955ms/step - loss: 3.3952
# Epoch 10/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3941

# 5980/5980 [==============================] - 6475s 1s/step - loss: 3.3941
# Epoch 11/100
# 5980/5980 [==============================] - ETA: 0s - loss: 3.3934

# 5980/5980 [==============================] - 8161s 1s/step - loss: 3.3934
# Epoch 12/100

autoencoder.save(save_dir + '/ae_weights_epoch_12_loss_3.3938.h5')
# y_pred = autoencoder.predict(data[:100])
# sparse_categorical_crossentropy(data[:100], y_pred).numpy().mean()
# 3.3750305

# y_pred = autoencoder.predict(data[:3])
# sparse_categorical_crossentropy(data[:3], y_pred).numpy().mean()
# Out[76]: 2.3406591

################################################################################################################################
# 提取自编码器模型中间层的表示
# 使用自编码器将高维的文本数据压缩为低维的中间层表示，并利用这些中间层表示进行聚类。
# 从自编码模型中获取编码模型，获取低维向量 
autoencoder = load_model(save_dir + '/ae_weights.h5')
encoder = Model(autoencoder.inputs, autoencoder.get_layer('dense').output, name='encoder')
decoder = Model(autoencoder.get_layer('dense_1').input, autoencoder.outputs, name='decoder')

data_vec = encoder.predict(data)
print(data_vec.shape)
(382688, 10)

kmeans = KMeans(n_clusters=n_clusters, n_init=20, )
y_pred_kmeans = kmeans.fit_predict(data_vec)
print(metrics_acc(y, y_pred_kmeans))
# 0.12830556484655908

# 也可以选择最佳分类数
Sum_of_squared_distances = []
K = range(1,30)
for k in K:
    km = KMeans(init='k-means++', n_clusters=k, n_init=10)
    km.fit(data_vec)
    Sum_of_squared_distances.append(km.inertia_)

plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.vlines(ymin=0, ymax=0.08, x=8, colors='red')
plt.text(x=8.2, y=0.0145, s="optimal K=8")
plt.xlabel('Number of Clusters K')
plt.ylabel('Sum of squared distances')
plt.title('Elbow Method For Optimal K')
plt.show()

################################################################################################################################
vec_dim = 10
num_clusters = n_clusters

# x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.15)

# 构建聚类模型：
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

from keras.utils import plot_model
plot_model(model, to_file='model.png', show_shapes=True)
from IPython.display import Image
Image(filename='model.png')

model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

print(model.summary())
# Model: "model"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  input (InputLayer)          [(None, 48)]              0         
#                                                                  
#  embedding_24 (Embedding)    (None, 48, 128)           638848    
#                                                                  
#  lstm1 (LSTM)                (None, 48, 128)           131584    
#                                                                  
#  lstm2 (LSTM)                (None, 128)               131584    
#                                                                  
#  dense_29 (Dense)            (None, 10)                1290      
#                                                                  
#  clustering (ClusteringLayer  (None, 15)               150       
#  )                                                               
#                                                                  
# =================================================================
# Total params: 903,456
# Trainable params: 903,456
# Non-trainable params: 0
# _________________________________________________________________
# None

# 初始化聚类中心
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(data))
y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# 计算分布
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

x = data 
# x.shape (382688, 48)

loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])

tol = 0.001 # tolerance threshold to stop training

# 训练模型：
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics_acc(y, y_pred), 5)
            nmi = np.round(metrics_nmi(y, y_pred), 5)
            ari = np.round(metrics_ari(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/DEC_model_final.h5')


# 加载聚类模型的权重：
model.load_weights(save_dir + '/DEC_model_final.h5')

# Eval.
q = model.predict(x, verbose=1)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
if y is not None:
    acc = np.round(metrics_acc(y, y_pred), 5)
    nmi = np.round(metrics_nmi(y, y_pred), 5)
    ari = np.round(metrics_ari(y, y_pred), 5)
    loss = np.round(loss, 5)
    print('Acc = %.5f, nmi = %.5f, ari = %.5f' % (acc, nmi, ari), ' ; loss=', loss)

# Acc = 0.12566, nmi = 0.01876, ari = 0.00961  ; loss= 0.0

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(40, 30))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("混淆矩阵(Confusion matrix)", fontsize=10)
plt.ylabel('真实标签(True label)', fontsize=10)
plt.xlabel('聚类标签(Clustering label)', fontsize=10)
plt.show()

from scipy.optimize import linear_sum_assignment as linear_assignment

y_true = y.astype(np.int64)
D = max(y_pred.max(), y_true.max()) + 1
w = np.zeros((D, D), dtype=np.int64)
# Confusion matrix.
for i in range(y_pred.size):
    w[y_pred[i], y_true[i]] += 1
ind = linear_assignment(-w)

print("acc: ", sum([w[i, j] for i, j in zip(ind[0], ind[1])]) * 1.0 / y_pred.size)
# acc: 0.12566111296931182

# 结论：使用自编码，针对此任务聚类效果压根就没有提升；

################################################################################################################################

def main():
    pass


if __name__ == "__main__":
    main()
