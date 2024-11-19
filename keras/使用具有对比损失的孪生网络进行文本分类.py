#!/usr/bin/env python
# coding=utf-8

# 来源：https://keras.io/examples/vision/siamese_contrastive/

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow import keras
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import RepeatVector, Permute, Reshape
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, mnist

from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import plot_model


import os
USERNAME = os.getenv("USERNAME")
import pandas as pd
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Input, BatchNormalization, Lambda, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow.keras.regularizers import l2
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve

# 交叉熵损失函数
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits_v2

epochs = 10
batch_size = 32
margin = 1

#############################################################################################################################
data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'keyword', 'label'])
df = shuffle(df)
df = df.fillna('')
texts = [' '.join(list(text + keyword.replace(',', ''))) for text, keyword in df[['title', 'keyword']].values]
tokenizer = Tokenizer(num_words=8000)
# 根据文本列表更新内部词汇表。
tokenizer.fit_on_texts(texts)

# 将文本中的每个文本转换为整数序列。
# 只有最常出现的“num_words”字才会被考虑在内。
# 只有分词器知道的单词才会被考虑在内。
sequences = tokenizer.texts_to_sequences(texts)
# dict {word: index}
word_index = tokenizer.word_index

print('tokens数量：', len(word_index))
maxlen=48
data = pad_sequences(sequences, maxlen=maxlen)
print('Shape of data tensor:', data.shape)

code2id = {label: idx for idx, label in enumerate(sorted(df['label'].unique()))}
y = np.array([code2id[label] for label in df['label'].values])
num_classes = len(code2id) # 15类

del texts, sequences

x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5)


def make_pairs(x, y):
    """“”“创建一个元组，其中包含具有相应标签的图像对。

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]  # 获取相同类别的数据索引

    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]  # 相同类别的标签为0

        # add a non-matching example
        label2 = random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = random.randint(0, num_classes - 1)

        idx2 = random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]  # 不同类别的标签为1

    return np.array(pairs), np.array(labels).astype("float32")

# df = pd.read_csv(rf"D:\Users\{USERNAME}/data/similarity/chinese_text_similarity.txt", sep='\t')

# make train pairs
pairs_train, labels_train = make_pairs(x_train, y_train)

# make validation pairs
pairs_val, labels_val = make_pairs(x_val, y_val)

# make test pairs
pairs_test, labels_test = make_pairs(x_test, y_test)


x_train_1 = pairs_train[:, 0]  # x_train_1.shape is (535762, 48)
x_train_2 = pairs_train[:, 1]

x_val_1 = pairs_val[:, 0]  # x_val_1.shape = (114806, 48)
x_val_2 = pairs_val[:, 1]

x_test_1 = pairs_test[:, 0]  # x_test_1.shape = (114808, 48)
x_test_2 = pairs_test[:, 1]

#############################################################################################################################


# 对数据对及其标签进行可视化：

def visualize(pairs, labels, to_show=6, num_col=3, predictions=None, test=False):
    for i in range(to_show):
        text_a, text_b = tokenizer.sequences_to_texts([pairs[i][0], pairs[i][1]])
        label = labels[i]
        print(f"{label}\t{text_a.replace(' ', '')}\t{text_b.replace(' ', '')}")

visualize(pairs_train[:-1], labels_train[:-1], to_show=4, num_col=4)
visualize(pairs_val[:-1], labels_val[:-1], to_show=4, num_col=4)
visualize(pairs_test[:-1], labels_test[:-1], to_show=4, num_col=4)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1, keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))

def contrastive_loss(y, preds, margin=1):
    # explicitly cast the true class label data type to the predicted
    # class label data type (otherwise we run the risk of having two
    # separate data types, causing TensorFlow to error out)
    y = tf.cast(y, preds.dtype)
    # calculate the contrastive loss between the true labels and
    # the predicted labels
    squaredPreds = K.square(preds)
    squaredMargin = K.square(K.maximum(margin - preds, 0))
    loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
    # return the computed contrastive loss to the calling function
    return loss


keras.backend.clear_session()
input = keras.layers.Input((maxlen,))
x = Embedding(8000, 32)(input)
x = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(x)
# x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)

x = keras.layers.Dense(num_classes, activation="tanh")(x)
x = Lambda(lambda x: K.l2_normalize(x, axis=-1))(x)

embedding_network = keras.Model(input, x)


input_1 = keras.layers.Input((48, ))
input_2 = keras.layers.Input((48, ))

tower_1 = embedding_network(input_1)
tower_2 = embedding_network(input_2)

distance = Lambda(euclidean_distance)([tower_1, tower_2])

# Connect the inputs with the outputs
model = Model([input_1, input_2], distance)

model.compile(loss=contrastive_loss, optimizer=Adam(0.001), metrics=["accuracy"])

# 使用准确率作为评估指标可能不是最佳选择，因为对比损失函数的目标是尽可能减小正样本对的距离并增大负样本对的距离。

# merge_layer = keras.layers.Lambda(euclidean_distance, output_shape=(1,))(
#     [tower_1, tower_2]
# )
# normal_layer = keras.layers.BatchNormalization()(merge_layer)
# output_layer = keras.layers.Dense(1, activation="sigmoid")(normal_layer)
# model = keras.Model(inputs=[input_1, input_2], outputs=output_layer)

# 模型结构可视化
plot_model(model, show_shapes=True, show_layer_names=True, to_file="./images/euclidean_distance2.png")

print(model.summary())

# Model: "model_1"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to                     
# ==================================================================================================
#  input_2 (InputLayer)           [(None, 48)]         0           []                               
#                                                                                                   
#  input_3 (InputLayer)           [(None, 48)]         0           []                               
#                                                                                                   
#  model (Functional)             (None, 15)           621583      ['input_2[0][0]',                
#                                                                   'input_3[0][0]']                
#                                                                                                   
#  lambda_1 (Lambda)              (None, 1)            0           ['model[0][0]',                  
#                                                                   'model[1][0]']                  
#                                                                                                   
# ==================================================================================================
# Total params: 621,583
# Trainable params: 621,583
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 在Keras中，`compile()`方法的`metrics`参数只能接受字符串、可调用对象（如函数）或两者组成的列表。
# 由于精度、召回率、F1分数和ROC曲线等评估指标不是简单的字符串或可调用对象，因此无法直接在`compile()`方法中使用它们作为`metrics`参数。
# 但是，您可以使用Keras回调（Callback）来实现这个功能。Keras回调是在训练过程中执行的可自定义操作，例如在每个epoch结束时保存模型或记录日志。
# 您可以创建一个自定义回调类，该类在每个epoch结束时计算所需的评估指标，并将它们记录到日志中。以下是使用自定义回调类计算精度、召回率、F1分数和ROC曲线


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, validation_data):
        self.validation_data = validation_data

    def on_epoch_end(self, epoch, logs=None):
        (x_val_1, x_val_2), y_val = self.validation_data
        y_pred = self.model.predict([x_val_1, x_val_2])
        y_pred_binary = np.where(y_pred < 0.5, 0, 1)

        accuracy = accuracy_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        f1 = f1_score(y_val, y_pred_binary)
        roc_auc = roc_auc_score(y_val, y_pred)

        fpr, tpr, thresholds = roc_curve(y_val, y_pred)

        logs['accuracy'] = accuracy
        logs['recall'] = recall
        logs['f1_score'] = f1
        logs['roc_auc'] = roc_auc


custom_callback = CustomCallback(validation_data=([x_val_1, x_val_2], labels_val))
history = model.fit(
    [x_train_1, x_train_2],
    labels_train,
    validation_data=([x_val_1, x_val_2], labels_val),
    batch_size=batch_size,
    epochs=epochs,
    callbacks=[custom_callback],
)
print(history.history)

# Epoch 1/50
# 16743/16743 [==============================] - 13158s 786ms/step - loss: 0.2595 - accuracy: 0.4938 - val_loss: 0.2574 - val_accuracy: 0.4797
# Epoch 2/50
# 16743/16743 [==============================] - 13177s 787ms/step - loss: 0.2544 - accuracy: 0.4732 - val_loss: 0.2561 - val_accuracy: 0.4826
# Epoch 3/50
# 16743/16743 [==============================] - 13264s 792ms/step - loss: 0.2512 - accuracy: 0.4616 - val_loss: 0.2550 - val_accuracy: 0.4748
# Epoch 4/50
# 16743/16743 [==============================] - 13308s 795ms/step - loss: 0.2488 - accuracy: 0.4505 - val_loss: 0.2541 - val_accuracy: 0.4762
# Epoch 5/50
# 16743/16743 [==============================] - 13490s 806ms/step - loss: 0.2465 - accuracy: 0.4413 - val_loss: 0.2554 - val_accuracy: 0.4788
# Epoch 6/50
# 16743/16743 [==============================] - 13636s 814ms/step - loss: 0.2448 - accuracy: 0.4337 - val_loss: 0.2540 - val_accuracy: 0.4731
# Epoch 7/50
# 16743/16743 [==============================] - 13721s 820ms/step - loss: 0.2433 - accuracy: 0.4259 - val_loss: 0.2547 - val_accuracy: 0.4756
# Epoch 8/50
# 16743/16743 [==============================] - 13722s 820ms/step - loss: 0.2419 - accuracy: 0.4201 - val_loss: 0.2546 - val_accuracy: 0.4729
# Epoch 9/50
# 16743/16743 [==============================] - 13740s 821ms/step - loss: 0.2405 - accuracy: 0.4142 - val_loss: 0.2543 - val_accuracy: 0.4732
# Epoch 10/50
# 16743/16743 [==============================] - 13803s 824ms/step - loss: 0.2394 - accuracy: 0.4088 - val_loss: 0.2550 - val_accuracy: 0.4781
# Epoch 11/50
# 16743/16743 [==============================] - 13923s 832ms/step - loss: 0.2384 - accuracy: 0.4051 - val_loss: 0.2544 - val_accuracy: 0.4737
# Epoch 12/50
# 16743/16743 [==============================] - 14065s 840ms/step - loss: 0.2374 - accuracy: 0.4010 - val_loss: 0.2541 - val_accuracy: 0.4745
# Epoch 13/50
# 16743/16743 [==============================] - 14161s 846ms/step - loss: 0.2367 - accuracy: 0.3984 - val_loss: 0.2548 - val_accuracy: 0.4735
# Epoch 14/50
# 16743/16743 [==============================] - 14142s 845ms/step - loss: 0.2358 - accuracy: 0.3945 - val_loss: 0.2550 - val_accuracy: 0.4784
# Epoch 15/50
# 16743/16743 [==============================] - 14155s 845ms/step - loss: 0.2352 - accuracy: 0.3926 - val_loss: 0.2557 - val_accuracy: 0.4783
# Epoch 16/50
# 16743/16743 [==============================] - 14144s 845ms/step - loss: 0.2347 - accuracy: 0.3900 - val_loss: 0.2551 - val_accuracy: 0.4765

# 可视化模型训练结果：
def plt_metric(history, metric, title, has_valid=True):
    """Plots the given 'metric' from 'history'.

    Arguments:
        history: history attribute of History object returned from Model.fit.
        metric: Metric to plot, a string value present as key in 'history'.
        title: A string to be used as title of plot.
        has_valid: Boolean, true if valid data was passed to Model.fit else false.

    Returns:
        None.
    """
    plt.plot(history[metric])
    if has_valid:
        plt.plot(history["val_" + metric])
        plt.legend(["train", "validation"], loc="upper left")
    plt.title(title)
    plt.ylabel(metric)
    plt.xlabel("epoch")
    plt.show()


# Plot the accuracy
plt_metric(history=history.history, metric="accuracy", title="Model accuracy")
plt_metric(history=history.history, metric="loss", title="Contrastive Loss")

# 评估模型：
results = model.evaluate([x_test_1, x_test_2], labels_test)
print("test loss, test acc:", results)
# test loss, test acc: [0.01498295459896326, 0.9803500175476074]

# 可视化预测效果：
predictions = model.predict([x_test_1, x_test_2])
visualize(pairs_test, labels_test, to_show=3, predictions=predictions, test=True)


#############################################################################################################################
# 总结：以上是使用具有对比损失的孪生网络进行文本分类的一个失败尝试，至于为何在图像分类中可以，迁移到文本分类模型就不收敛，原因暂时不明。
# 但可以改成下面这样进行模型训练,是正常的
#############################################################################################################################
# 对模型损失函数进行修改：


class_idx_to_train_idxs = defaultdict(list)
for y_train_idx, y in enumerate(y_train):
    class_idx_to_train_idxs[y].append(y_train_idx)

class_idx_to_val_idxs = defaultdict(list)
for y_val_idx, y in enumerate(y_val):
    class_idx_to_val_idxs[y].append(y_val_idx)
    
class_idx_to_test_idxs = defaultdict(list)
for y_test_idx, y in enumerate(y_test):
    class_idx_to_test_idxs[y].append(y_test_idx)


class AnchorPositivePairs(keras.utils.Sequence):
    def __init__(self, num_batches):
        super().__init__()
        self.num_batches = num_batches

    def __len__(self):
        return self.num_batches

    def __getitem__(self, _idx):
        # 每个分类中，都取一个相同类别的两个不同样本作为训练数据
        x1, x2 = [], []
        y = []
        for class_idx in range(num_classes):
            examples_for_class = class_idx_to_train_idxs[class_idx]
            anchor_idx = random.choice(examples_for_class)
            positive_idx = random.choice(examples_for_class)
            while positive_idx == anchor_idx:
                positive_idx = random.choice(examples_for_class)
            x1.append(x_train[anchor_idx])
            x2.append(x_train[positive_idx])
            y.append(class_idx)
        return (np.array(x1), np.array(x2)), np.array(y)

    
keras.backend.clear_session()
input = keras.layers.Input((maxlen,))
x = Embedding(8000, 32)(input)
x = LSTM(64, dropout=0.2, recurrent_dropout=0.2)(x)

x = Dense(32, activation="relu")(x)
x = Dropout(0.2)(x)
# x = keras.layers.Dense(num_classes, activation=None)(x)
embeddings = keras.layers.Dense(units=8, activation=None)(x)

embeddings = keras.layers.UnitNormalization()(embeddings)
# embeddings = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)

embedding_network = keras.Model(input, embeddings)


input1=Input(shape=(maxlen,))
input2=Input(shape=(maxlen,))
anchor_embeddings = embedding_network(input1)
positive_embeddings = embedding_network(input2)

similarities = Lambda(lambda x:tf.einsum("ae,pe->ap", x[0], x[1])/0.2)([anchor_embeddings, positive_embeddings])

# Connect the inputs with the outputs
model = Model([input1, input2], similarities)

model.compile(loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=Adam(0.001),
              # metrics=["accuracy"], # 使用准确率作为评估指标可能不是最佳选择，因为对比损失函数的目标是尽可能减小正样本对的距离并增大负样本对的距离。
              )
print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(AnchorPositivePairs(num_batches=1000), epochs=10)
print(history.history)

plt.plot(history.history["loss"])
plt.show()

# Epoch 1/10
# 1000/1000 [==============================] - 115s 94ms/step - loss: 2.3651
# Epoch 2/10
# 1000/1000 [==============================] - 90s 90ms/step - loss: 1.7984
# Epoch 3/10
# 1000/1000 [==============================] - 115s 115ms/step - loss: 1.5867
# Epoch 4/10
# 1000/1000 [==============================] - 107s 107ms/step - loss: 1.4300
# Epoch 5/10
# 1000/1000 [==============================] - 75s 75ms/step - loss: 1.3408
# Epoch 6/10
# 1000/1000 [==============================] - 61s 61ms/step - loss: 1.2643
# Epoch 7/10
# 1000/1000 [==============================] - 59s 59ms/step - loss: 1.1729
# Epoch 8/10
# 1000/1000 [==============================] - 56s 56ms/step - loss: 1.1339
# Epoch 9/10
# 1000/1000 [==============================] - 53s 53ms/step - loss: 1.1015
# Epoch 10/10
# 1000/1000 [==============================] - 53s 53ms/step - loss: 1.0740

# {'loss': [2.365140438079834, 1.7983804941177368, 1.5866962671279907, 1.4299558401107788, 1.3407514095306396, 1.2642648220062256, 1.1728661060333252, 1.1339274644851685, 1.1014622449874878, 1.0740418434143066]}

#############################################################################################################################
# 若上模型收敛，还可以进一步的训练一个分类模型；
emb_model = model.get_layer('model')
for l in emb_model.layers:
    l.trainable = False
    
emb_model.summary()

keras.backend.clear_session()

x = Dense(64, activation="relu", name="dense_2")(emb_model.output)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax', name='output')(x)
cls_model = Model(emb_model.input, outputs)
cls_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

print(cls_model.summary())

# 训练模型
hist = cls_model.fit(x_train, y_train, epochs=10, validation_data=[x_val, y_val], batch_size=32)
print(hist.history)

# Epoch 1/10
# 8372/8372 [==============================] - 197s 23ms/step - loss: 0.7443 - accuracy: 0.8048 - val_loss: 0.6903 - val_accuracy: 0.8092
# Epoch 2/10
# 8372/8372 [==============================] - 206s 25ms/step - loss: 0.6907 - accuracy: 0.8113 - val_loss: 0.6785 - val_accuracy: 0.8105
# Epoch 3/10
# 8372/8372 [==============================] - 205s 25ms/step - loss: 0.6832 - accuracy: 0.8113 - val_loss: 0.6760 - val_accuracy: 0.8101
# Epoch 4/10
# 8372/8372 [==============================] - 212s 25ms/step - loss: 0.6789 - accuracy: 0.8121 - val_loss: 0.6719 - val_accuracy: 0.8104
# Epoch 5/10
# 8372/8372 [==============================] - 211s 25ms/step - loss: 0.6757 - accuracy: 0.8124 - val_loss: 0.6698 - val_accuracy: 0.8115
# Epoch 6/10
# 8372/8372 [==============================] - 211s 25ms/step - loss: 0.6738 - accuracy: 0.8124 - val_loss: 0.6671 - val_accuracy: 0.8117
# Epoch 7/10
# 8372/8372 [==============================] - 208s 25ms/step - loss: 0.6716 - accuracy: 0.8121 - val_loss: 0.6664 - val_accuracy: 0.8119
# Epoch 8/10
# 8372/8372 [==============================] - 207s 25ms/step - loss: 0.6717 - accuracy: 0.8125 - val_loss: 0.6652 - val_accuracy: 0.8118
# Epoch 9/10
# 8372/8372 [==============================] - 214s 26ms/step - loss: 0.6703 - accuracy: 0.8121 - val_loss: 0.6660 - val_accuracy: 0.8126
# Epoch 10/10
# 8372/8372 [==============================] - 218s 26ms/step - loss: 0.6697 - accuracy: 0.8128 - val_loss: 0.6633 - val_accuracy: 0.8120
# {'loss': [0.7442697882652283, 0.690725564956665, 0.6832072138786316, 0.6788511872291565, 0.6756582856178284, 0.6737834215164185, 0.6715915203094482, 0.6717283725738525, 0.6702909469604492, 0.6696801781654358], 'accuracy': [0.8048050999641418, 0.8113415837287903, 0.8113266825675964, 0.8120919466018677, 0.8124129772186279, 0.8124428391456604, 0.812076985836029, 0.812517523765564, 0.812136709690094, 0.8127564191818237], 'val_loss': [0.6902925372123718, 0.6785123348236084, 0.6760433912277222, 0.6718730330467224, 0.6697819232940674, 0.6671063303947449, 0.6663568615913391, 0.6651512384414673, 0.6660482287406921, 0.6632565855979919], 'val_accuracy': [0.8091562986373901, 0.8105325698852539, 0.8100970387458801, 0.8104106187820435, 0.8114732503890991, 0.8116997480392456, 0.8118913769721985, 0.811769425868988, 0.8125882148742676, 0.8119958639144897]}

def main():
    pass


if __name__ == "__main__":
    main()
