#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Smiles2vec
# 简而言之，它是自然语言处理（NLP）领域的一项技术，可将字符串转换为矢量。 许多人用smiles字符串预测物理属性。

import os
import keras
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input, GlobalMaxPooling2D, BatchNormalization
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

print("RDKit: %s"%rdkit.__version__)
print("Keras: %s"%keras.__version__)
# RDKit: 2022.03.5
# Keras: 2.9.0

# 载入数据
# https://github.com/CHEMPHY/Chemception/blob/master/IGC50.xlsx
# IGC50是毒性药物数据集，其毒性值从0.334 - log10 mol/L到6.36 - log10 mol/L。
USERNAME = os.getenv('USERNAME')
data = pd.read_excel(rf'D:\Users\{USERNAME}\data\IGC50.xlsx')
X_train_smiles = np.array(list(data["smiles"][data["split"]==1]))
X_test_smiles = np.array(list(data["smiles"][data["split"]==0]))
print(X_train_smiles.shape) # (1434,)
print(X_test_smiles.shape) # (358,)

# 转换 Smiles 到 One-hot
assay = "Activity"
Y_train = data[assay][data["split"]==1].values.reshape(-1,1)
Y_test = data[assay][data["split"]==0].values.reshape(-1,1)
charset = set("".join(list(data.smiles))+"!E")
char_to_int = dict((c,i) for i,c in enumerate(charset))
int_to_char = dict((i,c) for i,c in enumerate(charset))
embed = max([len(smile) for smile in data.smiles]) + 5
print (str(charset)) # {'!', '/', 'O', 'P', '(', '2', 'C', '-', '[', '1', ')', 'N', '4', '\\', 'H', 'l', 'E', '+', 'F', 'r', 'I', '=', '#', 'B', '3', ']', 'S'}
print(len(charset), embed) # 27 57
print(char_to_int )
# {'!': 0, '/': 1, 'O': 2, 'P': 3, '(': 4, '2': 5, 'C': 6, '-': 7, '[': 8, '1': 9, ')': 10, 'N': 11, '4': 12, '\\': 13, 'H': 14, 'l': 15, 'E': 16, '+': 17, 'F': 18, 'r': 19, 'I': 20, '=': 21, '#': 22, 'B': 23, '3': 24, ']': 25, 'S': 26}



def vectorize(smiles):
    one_hot = np.zeros((smiles.shape[0], embed, len(charset)), dtype=np.int8)
    for i, smile in enumerate(smiles):
        # encode the startchar
        one_hot[i, 0, char_to_int["!"]] = 1
        # encode the rest of the chars
        for j, c in enumerate(smile):
            one_hot[i, j + 1, char_to_int[c]] = 1
        # Encode endchar
        one_hot[i, len(smile) + 1:, char_to_int["E"]] = 1
    # Return two, one for input and the other for output
    return one_hot[:, 0:-1, :], one_hot[:, 1:, :]


X_train, _ = vectorize(X_train_smiles)
X_test, _ = vectorize(X_test_smiles)

X_train[8].shape
# (56, 27)

mol_str_train = []
mol_str_test = []
for x in range(1434):
    mol_str_train.append("".join([int_to_char[idx] for idx in np.argmax(X_train[x, :, :], axis=1)]))

for x in range(358):
    mol_str_test.append("".join([int_to_char[idx] for idx in np.argmax(X_test[x, :, :], axis=1)]))
    vocab_size = len(charset)

# Smiles2vec层 定义
from keras.preprocessing.text import one_hot
# from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
# from keras.layers.embeddings import Embedding
from keras.layers.core.embedding import Embedding
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=embed - 1))
model.add(keras.layers.Conv1D(192, 10, activation='relu'))
model.add(BatchNormalization())
model.add(keras.layers.Conv1D(192, 5, activation='relu'))
model.add(keras.layers.Conv1D(192, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.4))
model.add(Dense(1, activation='linear'))


def coeff_determination(y_true, y_pred):
    from keras import backend as K
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr

    return lr

mol_str_train = np.asarray(mol_str_train)
mol_str_test = np.asarray(mol_str_test)

optimizer = Adam(lr=0.00025)
lr_metric = get_lr_metric(optimizer)
model.compile(loss="mse", optimizer=optimizer, metrics=[coeff_determination, lr_metric])
callbacks_list = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-15, verbose=1, mode='auto', cooldown=0),
    ModelCheckpoint(filepath="weights.best.hdf5", monitor='val_loss', save_best_only=True, verbose=1, mode='auto')

]

history = model.fit(x=np.argmax(X_train, axis=2), y=Y_train,
                    batch_size=128,
                    epochs=150,
                    validation_data=(np.argmax(X_test, axis=2), Y_test),
                    callbacks=callbacks_list)
hist = history.history

plt.figure(figsize=(10, 8))

for label in ['val_coeff_determination', 'coeff_determination']:
    plt.subplot(221)
    plt.plot(hist[label], label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("coeff_determination")

for label in ['val_loss', 'loss']:
    plt.subplot(222)
    plt.plot(hist[label], label=label)
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("loss")

plt.subplot(223)
plt.plot(hist['lr'], hist['val_coeff_determination'])
plt.legend()
plt.xlabel("lr")
plt.ylabel("val_coeff_determination")

plt.subplot(224)
plt.plot(hist['lr'], hist['val_loss'])
plt.legend()
plt.xlabel("lr")
plt.ylabel("val_loss")

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.35)

def main():
    pass


if __name__ == '__main__':
    main()
