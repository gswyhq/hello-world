#!/usr/bin/env python
# coding=utf-8

import os
USERNAME = os.getenv("USERNAME")
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout, Input, BatchNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers

# 交叉熵损失函数
from tensorflow.python.ops.nn_ops import softmax_cross_entropy_with_logits_v2
from tensorflow.python.ops.nn_ops import sparse_softmax_cross_entropy_with_logits_v2

# softmax_cross_entropy_with_logits_v2(logits, labels)  # logits和labels维数相同，并且labels采用one-hots编码
# sparse_softmax_cross_entropy_with_logits_v2(logits, labels) # logits和labels维数不相同，labels没有采用one-hot编码，若已编码，需采用tf.argmax(y,1) 还原为原格式
# logits都是未经激活函数（sigmoid、tanh、relu）和softmax放缩后的神经网络输出值，labels为样本标签（真实值）

# 假设我们有以下的文本数据和标签

data_file = rf"D:\Users\{USERNAME}/github_project/toutiao-text-classfication-dataset/toutiao_cat_data.txt"  # 数据来源： http://github.com/skdjfla/toutiao-text-classfication-dataset
df = pd.read_csv(data_file, sep='_!_', names=['id', 'code', 'label', 'title', 'keyword'], engine='python', usecols=['title', 'label'])
df = shuffle(df)

texts = [' '.join(list(text)) for text in df['title'].values]
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

############################################### softmax_cross_entropy_with_logits_v2 ##############################################################################

# 创建模型
model = Sequential()
model.add(Embedding(8000, 32, input_length=data.shape[-1]))
model.add(LSTM(32))
model.add(Dense(num_classes, activation=None))  # 输出值的激活函数为None
# 编译模型
model.compile(loss=softmax_cross_entropy_with_logits_v2, optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)				Output Shape			  Param #
# =================================================================
#  embedding (Embedding)	   (None, 48, 32)			256000
#
#  lstm (LSTM)				 (None, 32)				8320
#
#  dense (Dense)			   (None, 15)				495
#
# =================================================================
# Total params: 264,815
# Trainable params: 264,815
# Non-trainable params: 0

# 训练模型
hist = model.fit(data, keras.utils.to_categorical(y, num_classes=num_classes), epochs=10, validation_split=0.2, batch_size=32)
print(hist)

#
# Epoch 1/10
# 9568/9568 [==============================] - 169s 18ms/step - loss: 0.9152 - accuracy: 0.7313 - val_loss: 0.6826 - val_accuracy: 0.8017
# Epoch 2/10
# 9568/9568 [==============================] - 159s 17ms/step - loss: 0.6205 - accuracy: 0.8175 - val_loss: 0.6176 - val_accuracy: 0.8175
# Epoch 3/10
# 9568/9568 [==============================] - 158s 17ms/step - loss: 0.5593 - accuracy: 0.8341 - val_loss: 0.5873 - val_accuracy: 0.8268
# Epoch 4/10
# 9568/9568 [==============================] - 178s 19ms/step - loss: 0.5223 - accuracy: 0.8441 - val_loss: 0.5755 - val_accuracy: 0.8298
# Epoch 5/10
# 9568/9568 [==============================] - 191s 20ms/step - loss: 0.4947 - accuracy: 0.8530 - val_loss: 0.5718 - val_accuracy: 0.8318
# Epoch 6/10
# 9568/9568 [==============================] - 162s 17ms/step - loss: 0.4708 - accuracy: 0.8597 - val_loss: 0.5682 - val_accuracy: 0.8334
# Epoch 7/10
# 9568/9568 [==============================] - 162s 17ms/step - loss: 0.4504 - accuracy: 0.8655 - val_loss: 0.5648 - val_accuracy: 0.8365
# Epoch 8/10
# 9568/9568 [==============================] - 159s 17ms/step - loss: 0.4316 - accuracy: 0.8710 - val_loss: 0.5676 - val_accuracy: 0.8369
# Epoch 9/10
# 9568/9568 [==============================] - 170s 18ms/step - loss: 0.4145 - accuracy: 0.8765 - val_loss: 0.5737 - val_accuracy: 0.8381
# Epoch 10/10
# 9568/9568 [==============================] - 201s 21ms/step - loss: 0.3983 - accuracy: 0.8815 - val_loss: 0.5804 - val_accuracy: 0.8352

# 在进行分类时，输出层激活函数为空时，网络输出值z并不是最终的类别，需要进行如下操作：
# #softmax压缩变换
# y_=tf.softmax(z)
# #精确度
# correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(y,1))
# accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

################################################### SparseCategoricalCrossentropy ##########################################################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 创建一个简单的模型
keras.backend.clear_session()
model = Sequential()
model.add(Embedding(8000, 32, input_length=data.shape[-1]))
model.add(LSTM(32))
model.add(Dense(num_classes, activation=None))  # 输出值的激活函数为None

# 编译模型
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
# 损失函数为`sparse_categorical_crossentropy`，并设置`from_logits`参数为True，表示我们直接传递logits到损失函数, 即激活函数为空。
print(model.summary())

# Model: "sequential"
# _________________________________________________________________
#  Layer (type)				Output Shape			  Param #
# =================================================================
#  embedding (Embedding)	   (None, 48, 32)			256000
#
#  lstm (LSTM)				 (None, 32)				8320
#
#  dense (Dense)			   (None, 15)				495
#
# =================================================================
# Total params: 264,815
# Trainable params: 264,815
# Non-trainable params: 0

# 训练模型
hist = model.fit(data, y, epochs=10, validation_split=0.2, batch_size=32)
print(hist.history)

# Epoch 1/10
# 9568/9568 [==============================] - 126s 13ms/step - loss: 0.9163 - accuracy: 0.7304 - val_loss: 0.6894 - val_accuracy: 0.7985
# Epoch 2/10
# 9568/9568 [==============================] - 159s 17ms/step - loss: 0.6276 - accuracy: 0.8146 - val_loss: 0.6131 - val_accuracy: 0.8192
# Epoch 3/10
# 9568/9568 [==============================] - 184s 19ms/step - loss: 0.5621 - accuracy: 0.8336 - val_loss: 0.5884 - val_accuracy: 0.8256
# Epoch 4/10
# 9568/9568 [==============================] - 173s 18ms/step - loss: 0.5230 - accuracy: 0.8443 - val_loss: 0.5671 - val_accuracy: 0.8334
# Epoch 5/10
# 9568/9568 [==============================] - 179s 19ms/step - loss: 0.4935 - accuracy: 0.8530 - val_loss: 0.5620 - val_accuracy: 0.8352
# Epoch 6/10
# 9568/9568 [==============================] - 170s 18ms/step - loss: 0.4681 - accuracy: 0.8607 - val_loss: 0.5608 - val_accuracy: 0.8352

#############################################################################################################################

keras.backend.clear_session()

input = Input(shape=(maxlen), name='inputs')
x = Embedding(8000, 32)(input)
x = LSTM(32)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(input, outputs)
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), metrics=['accuracy'])

print(model.summary())

# 训练模型
hist = model.fit(data, y, epochs=10, validation_split=0.2, batch_size=32)
print(hist.history)
########################################################### focal_loss ##################################################################

def multi_category_focal_loss2(gamma=2., alpha=.25):
	"""
	focal loss for multi category of multi label problem
	适用于多分类或多标签问题的focal loss
	alpha控制真值y_true为1/0时的权重
		1的权重为alpha, 0的权重为1-alpha
	当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
	当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
	当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
		尝试将alpha调大,鼓励模型进行预测出1。
	Usage:
	 model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
	"""
	epsilon = 1.e-7
	gamma = float(gamma)
	alpha = tf.constant(alpha, dtype=tf.float32)

	def multi_category_focal_loss2_fixed(y_true, y_pred):
		y_true = tf.cast(y_true, tf.float32)
		y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

		alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
		y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
		ce = -K.log(y_t)
		weight = tf.pow(tf.subtract(1., y_t), gamma)
		fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
		loss = tf.reduce_mean(fl)
		return loss

	return multi_category_focal_loss2_fixed


keras.backend.clear_session()

input = Input(shape=(maxlen), name='inputs')
x = Embedding(8000, 32)(input)
x = LSTM(32)(x)
outputs = Dense(1, activation='softmax')(x)
model = Model(input, outputs)
model.compile(optimizer='adam', loss=multi_category_focal_loss2(alpha=0.25, gamma=2), metrics=['accuracy'])

print(model.summary())

# 训练模型
hist = model.fit(data, y, epochs=10, validation_split=0.2, batch_size=32)
print(hist.history)

# Epoch 1/3
# 9568/9568 [==============================] - 141s 15ms/step - loss: 0.0121 - accuracy: 0.7308 - val_loss: 0.0092 - val_accuracy: 0.7983
# Epoch 2/3
# 9568/9568 [==============================] - 161s 17ms/step - loss: 0.0085 - accuracy: 0.8111 - val_loss: 0.0086 - val_accuracy: 0.8092
# Epoch 3/3
# 9568/9568 [==============================] - 202s 21ms/step - loss: 0.0077 - accuracy: 0.8289 - val_loss: 0.0080 - val_accuracy: 0.8220
# {'loss': [0.012094911187887192, 0.008512254804372787, 0.007663319353014231], 'accuracy': [0.7307822704315186, 0.811076283454895, 0.8289498686790466], 'val_loss': [0.009194166399538517, 0.008613640442490578, 0.007974338717758656], 'val_accuracy': [0.7983354926109314, 0.8092058897018433, 0.82203608751297]}

################################################ FocalLoss #############################################################################

# https://github.com/keras-team/keras-cv/blob/v0.8.2/keras_cv/losses/focal.py#L20
class FocalLoss(keras.losses.Loss):
	def __init__(
		self,
		alpha=0.25,
		gamma=2,
		from_logits=False,
		label_smoothing=0,
		**kwargs,
	):
		super().__init__(**kwargs)
		self.alpha = float(alpha)
		self.gamma = float(gamma)
		self.from_logits = from_logits
		self.label_smoothing = label_smoothing

	def _smooth_labels(self, y_true):
		return (
			y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
		)

	def call(self, y_true, y_pred):
		y_pred = tf.convert_to_tensor(y_pred)
		y_true = tf.cast(y_true, y_pred.dtype)

		if self.label_smoothing:
			y_true = self._smooth_labels(y_true)

		if self.from_logits:
			y_pred = tf.sigmoid(y_pred)

		cross_entropy = K.binary_crossentropy(y_true, y_pred)

		alpha = tf.where(
			tf.equal(y_true, 1.0), self.alpha, (1.0 - self.alpha)
		)
		pt = y_true * y_pred + (1.0 - y_true) * (1.0 - y_pred)
		loss = (
			alpha
			* tf.cast(tf.pow(1.0 - pt, self.gamma), alpha.dtype)
			* tf.cast(cross_entropy, alpha.dtype)
		)
		return K.sum(loss, axis=-1)

	def get_config(self):
		config = super().get_config()
		config.update(
			{
				"alpha": self.alpha,
				"gamma": self.gamma,
				"from_logits": self.from_logits,
				"label_smoothing": self.label_smoothing,
			}
		)
		return config

keras.backend.clear_session()

input = Input(shape=(maxlen), name='inputs')
x = Embedding(8000, 32)(input)
x = LSTM(32)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(input, outputs)
model.compile(optimizer='adam', loss=FocalLoss(), metrics=['accuracy'])

print(model.summary())

# 训练模型
hist = model.fit(data, keras.utils.to_categorical(y, num_classes=num_classes), epochs=10, validation_split=0.2, batch_size=32)
print(hist.history)

# Epoch 1/2
# 9568/9568 [==============================] - 137s 14ms/step - loss: 0.0762 - accuracy: 0.7222 - val_loss: 0.0658 - val_accuracy: 0.7884
# Epoch 2/2
# 9568/9568 [==============================] - 141s 15ms/step - loss: 0.0619 - accuracy: 0.8023 - val_loss: 0.0615 - val_accuracy: 0.8051
# {'loss': [0.07619865983724594, 0.061900243163108826], 'accuracy': [0.7222113609313965, 0.8023191094398499], 'val_loss': [0.06575921177864075, 0.0615016333758831], 'val_accuracy': [0.7884318828582764, 0.8051164150238037]}

################################################### ArcFace ##########################################################################


class ArcFace(Layer):
    def __init__(self, n_classes=10, s=30.0, m=0.50, regularizer=None, **kwargs):
        super(ArcFace, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.s = s
        self.m = m
        self.regularizer = regularizers.get(regularizer)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'n_classes': self.n_classes,
            's': self.s,
            'm': self.m,
            'regularizer': self.regularizer
        })
        return config

    def build(self, input_shape):
        super(ArcFace, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1], self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)

    def call(self, inputs):
        x, y = inputs
        c = K.shape(x)[-1]
        # normalize feature
        x = tf.nn.l2_normalize(x, axis=1)
        # normalize weights
        W = tf.nn.l2_normalize(self.W, axis=0)
        # dot product
        logits = x @ W
        # add margin
        # clip logits to prevent zero division when backward
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        # sin = tf.sqrt(1 - logits**2)
        # cos_m = tf.cos(logits)
        # sin_m = tf.sin(logits)
        # target_logits = logits * cos_m - sin * sin_m
        #
        logits = logits * (1 - y) + target_logits * y
        # feature re-scale
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return (None, self.n_classes)

y_train = keras.utils.to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(data, y_train, test_size=0.15)

keras.backend.clear_session()

input = Input(shape=(maxlen), name='inputs')
label = Input(shape=(num_classes,))
x = Embedding(8000, 32)(input)
x = LSTM(256)(x)
x = Dropout(0.2)(x)
x = Dense(256, kernel_initializer='he_normal')(x)
x = Dropout(0.2)(x)
x = BatchNormalization()(x)
output = ArcFace(n_classes=num_classes)([x, label])
model = Model([input, label], output)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print(model.summary())

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='results/arc_face_model.hdf5',
                                       save_best_only=True, save_weights_only=False)
# 训练模型
hist = model.fit([x_train, y_train], y_train, shuffle=True, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stopping, model_checkpoint])
print(hist.history)

# for layer in model.layers:
#     layer.trainable = False

custom_objects = {"ArcFace": ArcFace}
model = tf.keras.models.load_model('results/arc_face_model.hdf5', custom_objects=custom_objects)

x = model.layers[-3].output
x = Dense(64, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(num_classes, activation='softmax')(x)
cls_model = Model(inputs=model.input[0], outputs=outputs)
cls_model.compile(optimizer=Adam(learning_rate=5e-5), loss='categorical_crossentropy', metrics=['accuracy'])

print(cls_model.summary())

# 训练模型

model_checkpoint = ModelCheckpoint(filepath='results/cls_model.hdf5',
                                       save_best_only=True, save_weights_only=False)
hist = cls_model.fit(x_train, y_train, epochs=10, shuffle=True, validation_split=0.2, batch_size=32, callbacks=[early_stopping, model_checkpoint])
print(hist.history)

# model = Model(inputs=model.input[0], outputs=model.layers[-3].output)
# embedded_features = model.predict(x_test, verbose=1)
# embedded_features /= np.linalg.norm(embedded_features, axis=1, keepdims=True)

###################################################### CosineSimilarity #######################################################################


y_train = keras.utils.to_categorical(y, num_classes=num_classes)
x_train, x_test, y_train, y_test = train_test_split(data, y_train, test_size=0.15)

keras.backend.clear_session()

input = Input(shape=(maxlen), name='inputs')
label = Input(shape=(num_classes,))
x = Embedding(8000, 32)(input)
x = LSTM(256, dropout=0.2, recurrent_dropout=0.2)(x)
# x = Dropout(0.2)(x)
x = Dense(512, activation="relu")(x)
# x = Dropout(0.2)(x)
output = Dense(num_classes, activation="softmax", name="output")(x)
model = Model(input, output)
model.compile(optimizer='adam', loss=tf.keras.losses.CosineSimilarity(), metrics=["accuracy"])
print(model.summary())


early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model_checkpoint = ModelCheckpoint(filepath='results/CosineSimilarity_model.hdf5', save_best_only=True, save_weights_only=False)
# 训练模型
hist = model.fit(x_train, y_train, shuffle=True, epochs=10, validation_split=0.2, batch_size=32, callbacks=[early_stopping, model_checkpoint])
print(hist.history)

model = load_model('results/CosineSimilarity_model.hdf5')
# 评估模型
score, acc = model.evaluate(x_test, y_test, batch_size=64)
print("Test score:", score)
print("Test accuracy:", acc)


#############################################################################################################################


#############################################################################################################################

#############################################################################################################################

#############################################################################################################################


