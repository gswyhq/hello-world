'''
#Train a recurrent convolutional network on the IMDB sentiment classification task.

Gets to 0.8498 test accuracy after 2 epochs. 41 s/epoch on K520 GPU.
'''
from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D, MaxPooling1D
from keras.datasets import imdb
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# 实现f1_score(多分类、二分类)

# Embedding
max_features = 20000
maxlen = 100
embedding_size = 128

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 30
epochs = 2

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

class Metrics_multi(Callback):
    """
    多分类的F1
    """
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
#         val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_predict = np.argmax(np.asarray(self.model.predict(self.validation_data[0])), axis=1)
#         val_targ = self.validation_data[1]
        val_targ = np.argmax(self.validation_data[1], axis=1)
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
        self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('— val_f1: %f — val_precision: %f — val_recall %f' %(_val_f1, _val_precision, _val_recall))
        print(' — val_f1:' ,_val_f1)
        return

class Metrics(Callback):
    """二分类"""
    def on_train_begin(self, logs=None):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs=None):
        val_targ = self.validation_data[1]
        val_predict = self.model.predict(self.validation_data[0])

        # 求解最优阈值
        best_threshold = 0
        best_f1 = 0
        best_recall = 0
        best_precision = 0
        for threshold in [i * 0.01 for i in range(20, 80)]:
            y_pred = (val_predict > threshold).astype(int)
            val_recall = recall_score(val_targ, y_pred)
            val_precision = precision_score(val_targ, y_pred)
            val_f1 = f1_score(val_targ, y_pred)
            if val_f1 > best_f1:
                best_threshold = threshold
                best_f1 = val_f1
                best_recall = val_recall
                best_precision = val_precision

        self.val_f1s.append(best_f1)
        self.val_recalls.append(best_recall)
        self.val_precisions.append(best_precision)
        # print('— val_f1: %f' % (_val_f1))
        print("best_threshold: {}, best_recall: {}, best_precision: {}, best_f1: {}".format(best_threshold, best_recall, best_precision, best_f1))
        return

# 构造metrics，实现求解f1值
metrics = Metrics()  # 二分类
# metrics = Metrics_multi()  # 多分类

print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
model.add(MaxPooling1D(pool_size=pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          callbacks =[metrics])
score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
print('F1值: {}'.format(metrics.val_f1s))
