"""Train CRF and BiLSTM-CRF on CONLL2000 chunking data,
similar to https://arxiv.org/pdf/1508.01991v1.pdf.
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
from collections import Counter
import tensorflow.keras.backend as K
from keras.models import Sequential, Model
from keras.layers import Embedding, Bidirectional, LSTM, Lambda, Conv1D, Dropout, concatenate, Input
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from keras_contrib.datasets import conll2000
import tensorflow as tf
from keras.layers.normalization import BatchNormalization

EPOCHS = 10
EMBED_DIM = 200
BiRNN_UNITS = 200


def classification_report(y_true, y_pred, labels):
    '''Similar to the one in sklearn.metrics,
    reports per classs recall, precision and F1 score'''
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    corrects = Counter(yt for yt, yp in zip(y_true, y_pred) if yt == yp)
    y_true_counts = Counter(y_true)
    y_pred_counts = Counter(y_pred)
    report = ((lab,  # label
               corrects[i] / max(1, y_true_counts[i]),  # recall
               corrects[i] / max(1, y_pred_counts[i]),  # precision
               y_true_counts[i]  # support
               ) for i, lab in enumerate(labels))
    report = [(l, r, p, 2 * r * p / max(1e-9, r + p), s) for l, r, p, s in report]

    print('{:<15}{:>10}{:>10}{:>10}{:>10}\n'.format('',
                                                    '召回率（Recall）',
                                                    '精确率（Precision）',
                                                    'f1-score',
                                                    'support'))
    formatter = '{:<15}{:>10.2f}{:>10.2f}{:>10.2f}{:>10d}'.format
    for r in report:
        print(formatter(*r))
    print('')
    report2 = list(zip(*[(r * s, p * s, f1 * s) for l, r, p, f1, s in report]))
    N = len(y_true)
    print(formatter('avg / total',
                    sum(report2[0]) / N,
                    sum(report2[1]) / N,
                    sum(report2[2]) / N, N) + '\n')


# ------
# Data
# -----

# conll200 has two different targets, here will only use
# IBO like chunking as an example
train, test, voc = conll2000.load_data()
(train_x, _, train_y) = train
(test_x, _, test_y) = test
(vocab, _, class_labels) = voc

# --------------
# 1. Regular CRF
# --------------
def regular_crf(train_x, train_y, test_x, test_y):
    print('==== training CRF ====')
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()

    # The default `crf_loss` for `learn_mode='join'` is negative log likelihood.
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)

# -------------
# 2. BiLSTM-CRF
# -------------

def bilstm_crf(train_x, train_y, test_x, test_y):
    print('==== training BiLSTM-CRF ====')
    
    model = Sequential()
    model.add(Embedding(len(vocab), EMBED_DIM, mask_zero=True))  # Random embedding
    model.add(Bidirectional(LSTM(BiRNN_UNITS // 2, return_sequences=True)))
    crf = CRF(len(class_labels), sparse_target=True)
    model.add(crf)
    model.summary()
    
    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of BiLSTM-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)

# -------------
# 3. IDCNN-CRF
# -------------

class MaskedConv1D(Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)

def IDCNN(input, cnn_filters=128, cnn_kernel_size=3, cnn_blocks=4, **kwargs):
    def _dilation_conv1d(dilation_rate):
        return MaskedConv1D(filters=cnn_filters, kernel_size=cnn_kernel_size, padding="same", dilation_rate=dilation_rate)

    def _idcnn_block():
        idcnn_1 = _dilation_conv1d(1)
        idcnn_2 = _dilation_conv1d(1)
        idcnn_3 = _dilation_conv1d(2)
        return [idcnn_1, idcnn_2, idcnn_3]

    input = BatchNormalization(name='normalization')(input)

    stack_idcnn_layers = []
    for layer_idx in range(cnn_blocks):
        idcnn_block = _idcnn_block()
        cnn = idcnn_block[0](input)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[1](cnn)
        cnn = Dropout(0.02)(cnn)
        cnn = idcnn_block[2](cnn)
        cnn = Dropout(0.02)(cnn)
        stack_idcnn_layers.append(cnn)
    stack_idcnn = concatenate(stack_idcnn_layers, axis=-1)
    return stack_idcnn

def seq_padding(X, padding=0, max_len=100):
    if len(X.shape) == 2:
        return np.array([
            np.concatenate([[padding] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    elif len(X.shape) == 3:
        return np.array([
            np.concatenate([[[padding]] * (max_len - len(x)), x]) if len(x) < max_len else x for x in X
        ])
    else:
        return X

def idcnn_crf(train_x, train_y, test_x, test_y):
    test_x = seq_padding(test_x, padding=0, max_len=train_x.shape[1])
    test_y = seq_padding(test_y, padding=-1, max_len=train_y.shape[1])
    
    print('==== training IDCNN-CRF ====')
    
    # build models
    input = Input(shape=(train_x.shape[-1],))
    emb = Embedding(len(vocab), EMBED_DIM, mask_zero=True)(input)
    idcnn = IDCNN(emb)
    crf_out = CRF(len(class_labels), sparse_target=True)(idcnn)
    model = Model(input, crf_out)
    model.summary()

    model.compile('adam', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.fit(train_x, train_y, epochs=EPOCHS, validation_data=[test_x, test_y])
    
    test_y_pred = model.predict(test_x).argmax(-1)[test_x > 0]
    test_y_true = test_y[test_x > 0]
    
    print('\n---- Result of IDCNN-CRF ----\n')
    classification_report(test_y_true, test_y_pred, class_labels)


def main():
    regular_crf(train_x, train_y, test_x, test_y)
    bilstm_crf(train_x, train_y, test_x, test_y)
    idcnn_crf(train_x, train_y, test_x, test_y)
    
if __name__ == '__main__':
    main()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_1 (Embedding)      (None, None, 200)         1787200
# _________________________________________________________________
# crf_1 (CRF)                  (None, None, 23)          5198
# =================================================================
# Total params: 1,792,398
# Trainable params: 1,792,398
# Non-trainable params: 0
# _________________________________________________________________

# ---- Result of CRF ----
#
#                召回率（Recall）精确率（Precision）  f1-score   support
#
# B-ADJP               0.47      0.69      0.56       438
# B-ADVP               0.73      0.74      0.73       866
# B-CONJP              0.00      0.00      0.00         9
# B-INTJ               0.00      0.00      0.00         2
# B-LST                0.00      0.00      0.00         5
# B-NP                 0.92      0.92      0.92     12422
# B-PP                 0.97      0.92      0.95      4811
# B-PRT                0.73      0.65      0.68       106
# B-SBAR               0.61      0.88      0.72       535
# B-UCP                0.00      0.00      0.00         0
# B-VP                 0.87      0.88      0.87      4658
# I-ADJP               0.23      0.53      0.32       167
# I-ADVP               0.26      0.39      0.31        89
# I-CONJP              0.15      0.67      0.25        13
# I-INTJ               0.00      0.00      0.00         0
# I-LST                0.00      0.00      0.00         2
# I-NP                 0.92      0.91      0.91     14376
# I-PP                 0.08      0.36      0.14        48
# I-PRT                0.00      0.00      0.00         0
# I-SBAR               0.00      0.00      0.00         4
# I-UCP                0.00      0.00      0.00         0
# I-VP                 0.84      0.84      0.84      2646
# O                    0.96      0.94      0.95      6180
#
# avg / total          0.90      0.90      0.90     47377

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding_2 (Embedding)      (None, None, 200)         1787200
# _________________________________________________________________
# bidirectional_1 (Bidirection (None, None, 200)         240800
# _________________________________________________________________
# crf_2 (CRF)                  (None, None, 23)          5198
# =================================================================
# Total params: 2,033,198
# Trainable params: 2,033,198
# Non-trainable params: 0
# _________________________________________________________________
# Train on 8936 samples, validate on 2012 samples

# ---- Result of BiLSTM-CRF ----
#
#                召回率（Recall）精确率（Precision）  f1-score   support
#
# B-ADJP               0.66      0.65      0.65       438
# B-ADVP               0.82      0.78      0.80       866
# B-CONJP              0.56      0.83      0.67         9
# B-INTJ               0.50      0.50      0.50         2
# B-LST                0.00      0.00      0.00         5
# B-NP                 0.95      0.94      0.94     12422
# B-PP                 0.97      0.95      0.96      4811
# B-PRT                0.67      0.77      0.72       106
# B-SBAR               0.79      0.86      0.82       535
# B-UCP                0.00      0.00      0.00         0
# B-VP                 0.92      0.92      0.92      4658
# I-ADJP               0.51      0.57      0.54       167
# I-ADVP               0.60      0.53      0.56        89
# I-CONJP              0.69      0.82      0.75        13
# I-INTJ               0.00      0.00      0.00         0
# I-LST                0.00      0.00      0.00         2
# I-NP                 0.94      0.95      0.94     14376
# I-PP                 0.58      0.82      0.68        48
# I-PRT                0.00      0.00      0.00         0
# I-SBAR               0.75      0.23      0.35         4
# I-UCP                0.00      0.00      0.00         0
# I-VP                 0.90      0.92      0.91      2646
# O                    0.96      0.95      0.95      6180
#
# avg / total          0.93      0.93      0.93     47377

# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_1 (InputLayer)            (None, 78)           0
# __________________________________________________________________________________________________
# embedding_1 (Embedding)         (None, 78, 200)      1787200     input_1[0][0]
# __________________________________________________________________________________________________
# normalization (BatchNormalizati (None, 78, 200)      800         embedding_1[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_1 (MaskedConv1D)  (None, 78, 128)      76928       normalization[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_4 (MaskedConv1D)  (None, 78, 128)      76928       normalization[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_7 (MaskedConv1D)  (None, 78, 128)      76928       normalization[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_10 (MaskedConv1D) (None, 78, 128)      76928       normalization[0][0]
# __________________________________________________________________________________________________
# dropout_1 (Dropout)             (None, 78, 128)      0           masked_conv1d_1[0][0]
# __________________________________________________________________________________________________
# dropout_4 (Dropout)             (None, 78, 128)      0           masked_conv1d_4[0][0]
# __________________________________________________________________________________________________
# dropout_7 (Dropout)             (None, 78, 128)      0           masked_conv1d_7[0][0]
# __________________________________________________________________________________________________
# dropout_10 (Dropout)            (None, 78, 128)      0           masked_conv1d_10[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_2 (MaskedConv1D)  (None, 78, 128)      49280       dropout_1[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_5 (MaskedConv1D)  (None, 78, 128)      49280       dropout_4[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_8 (MaskedConv1D)  (None, 78, 128)      49280       dropout_7[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_11 (MaskedConv1D) (None, 78, 128)      49280       dropout_10[0][0]
# __________________________________________________________________________________________________
# dropout_2 (Dropout)             (None, 78, 128)      0           masked_conv1d_2[0][0]
# __________________________________________________________________________________________________
# dropout_5 (Dropout)             (None, 78, 128)      0           masked_conv1d_5[0][0]
# __________________________________________________________________________________________________
# dropout_8 (Dropout)             (None, 78, 128)      0           masked_conv1d_8[0][0]
# __________________________________________________________________________________________________
# dropout_11 (Dropout)            (None, 78, 128)      0           masked_conv1d_11[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_3 (MaskedConv1D)  (None, 78, 128)      49280       dropout_2[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_6 (MaskedConv1D)  (None, 78, 128)      49280       dropout_5[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_9 (MaskedConv1D)  (None, 78, 128)      49280       dropout_8[0][0]
# __________________________________________________________________________________________________
# masked_conv1d_12 (MaskedConv1D) (None, 78, 128)      49280       dropout_11[0][0]
# __________________________________________________________________________________________________
# dropout_3 (Dropout)             (None, 78, 128)      0           masked_conv1d_3[0][0]
# __________________________________________________________________________________________________
# dropout_6 (Dropout)             (None, 78, 128)      0           masked_conv1d_6[0][0]
# __________________________________________________________________________________________________
# dropout_9 (Dropout)             (None, 78, 128)      0           masked_conv1d_9[0][0]
# __________________________________________________________________________________________________
# dropout_12 (Dropout)            (None, 78, 128)      0           masked_conv1d_12[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 78, 512)      0           dropout_3[0][0]
#                                                                  dropout_6[0][0]
#                                                                  dropout_9[0][0]
#                                                                  dropout_12[0][0]
# __________________________________________________________________________________________________
# crf_1 (CRF)                     (None, 78, 23)       12374       concatenate_1[0][0]
# ==================================================================================================
# Total params: 2,502,326
# Trainable params: 2,501,926
# Non-trainable params: 400
# __________________________________________________________________________________________________
# Train on 8936 samples, validate on 2012 samples

# ---- Result of IDCNN-CRF ----
#
#                召回率（Recall）精确率（Precision）  f1-score   support
#
# B-ADJP               0.58      0.63      0.60       438
# B-ADVP               0.73      0.75      0.74       866
# B-CONJP              0.78      0.70      0.74         9
# B-INTJ               0.50      1.00      0.67         2
# B-LST                0.00      0.00      0.00         5
# B-NP                 0.92      0.92      0.92     12422
# B-PP                 0.95      0.94      0.94      4811
# B-PRT                0.69      0.71      0.70       106
# B-SBAR               0.80      0.75      0.77       535
# B-UCP                0.00      0.00      0.00         0
# B-VP                 0.90      0.88      0.89      4658
# I-ADJP               0.57      0.61      0.59       167
# I-ADVP               0.44      0.51      0.47        89
# I-CONJP              0.54      0.88      0.67        13
# I-INTJ               0.00      0.00      0.00         0
# I-LST                0.00      0.00      0.00         2
# I-NP                 0.92      0.93      0.92     14376
# I-PP                 0.65      0.69      0.67        48
# I-PRT                0.00      0.00      0.00         0
# I-SBAR               0.25      0.05      0.08         4
# I-UCP                0.00      0.00      0.00         0
# I-VP                 0.88      0.89      0.89      2646
# O                    0.94      0.93      0.93      6180
#
# avg / total          0.91      0.91      0.91     47377

# sudo pip3 install Keras==2.2.4
# sudo pip3 install git+https://www.github.com/keras-team/keras-contrib.git
# https://github.com/keras-team/keras-contrib/blob/master/examples/conll2000_chunking_crf.py
# https://github.com/liushaoweihua/keras-bert-ner/blob/master/keras_bert_ner/train/models.py

