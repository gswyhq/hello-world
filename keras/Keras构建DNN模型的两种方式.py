#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from keras.layers.core import Dense, Dropout
from keras.models import Sequential,Model
from keras.layers import Input,Add, LeakyReLU
from keras.layers.normalization.batch_normalization import BatchNormalization

def build_model(input_dim, output_dim, hidden_dim_list=None):
    '''
    :param inputdim: int type, the dim of input data.
    :param outputdim: int type, the number of class.
    '''
    input_1 = Input(shape=(input_dim,))
    hidden_0 = Dense(units=hidden_dim_list[0],activation='linear')(input_1)
    hidden_0 = LeakyReLU()(hidden_0)
    hidden_0 = BatchNormalization()(hidden_0)
    hidden_0 = Dropout(0.5)(hidden_0)
    #
    hidden_1 = Dense(hidden_dim_list[1], activation='linear')(hidden_0)
    hidden_1 = LeakyReLU()(hidden_1)
    hidden_1 = BatchNormalization()(hidden_1)
    hidden_1 = Dropout(0.5)(hidden_1)
    hidden_2 = Add()([hidden_0, hidden_1])
    hidden_3 = Dense(hidden_dim_list[2], activation='relu')(hidden_2)
    predictions = Dense(output_dim, activation='softmax')(hidden_3)
    model = Model(inputs=input_1, outputs=predictions)
    return model

def build_model2(input_dim, output_dim, hidden_dim_list=None):
    '''
    :param inputdim: int type, the dim of input data.
    :param outputdim: int type, the number of class.
    '''
    model = Sequential()
    model.add(Dense(hidden_dim_list[0], input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    for i in range(1, len(hidden_dim_list)):
        model.add(Dense(hidden_dim_list[i], activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
    model.add(Dense(output_dim, activation='softmax'))
    return model


def main():
    model = build_model(32, 2, hidden_dim_list=[128, 128, 16])
    print(model.count_params())
    model = build_model2(32, 2, hidden_dim_list=[128, 128, 16])
    print(model.count_params())

if __name__ == '__main__':
    main()
