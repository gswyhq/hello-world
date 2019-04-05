#!/usr/bin/python3
# coding: utf-8

# 安装 keras-contrib
#
# pip3 install git+https://www.github.com/keras-team/keras-contrib.git

from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Dropout
from keras_contrib.layers.crf import CRF
from keras_contrib.utils import save_load_utils
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding_1 (Embedding)      (None, 100, 128)          320000    
_________________________________________________________________
bidirectional_1 (Bidirection (None, 100, 400)          526400    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100, 400)          0         
_________________________________________________________________
bidirectional_2 (Bidirection (None, 100, 400)          961600    
_________________________________________________________________
dropout_2 (Dropout)          (None, 100, 400)          0         
_________________________________________________________________
time_distributed_1 (TimeDist (None, 100, 5)            2005      
_________________________________________________________________
crf_1 (CRF)                  (None, 100, 5)            65        
=================================================================
Total params: 1,810,070
Trainable params: 1,810,070
Non-trainable params: 0
_________________________________________________________________
None
'''

VOCAB_SIZE = 2500
EMBEDDING_OUT_DIM = 128
TIME_STAMPS = 100
HIDDEN_UNITS = 200
DROPOUT_RATE = 0.3
NUM_CLASS = 5


def build_embedding_bilstm2_crf_model():
    """
    带embedding的双向LSTM + crf
    """
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, output_dim=EMBEDDING_OUT_DIM, input_length=TIME_STAMPS))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(DROPOUT_RATE))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf_layer = CRF(NUM_CLASS)
    model.add(crf_layer)
    model.compile('rmsprop', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    return model

def save_embedding_bilstm2_crf_model(model, filename):
    save_load_utils.save_all_weights(model,filename)

def load_embedding_bilstm2_crf_model(filename):
    model = build_embedding_bilstm2_crf_model()
    save_load_utils.load_all_weights(model, filename)
    return model


def main():
    model = build_embedding_bilstm2_crf_model()
    print(model.summary())


if __name__ == '__main__':
    main()