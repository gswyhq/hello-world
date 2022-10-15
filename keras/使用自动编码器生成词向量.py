#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源：https://alvinntnu.github.io/python-notes/nlp/word-embeddings-autoencoder.html

import os
from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import pad_sequences  # keras version 2.9.0
from keras import Input, Model, optimizers
from keras.layers import Bidirectional, LSTM, Embedding, RepeatVector, Dense
import numpy as np

USERNAME = os.getenv("USERNAME")
with open(rf"D:\Users\{USERNAME}\data/sentiment/sentiment.train.data", encoding='utf-8')as f:
    sents = [t.strip() for t in f.readlines()]
len(sents)
# 57340
maxlen = max([len(s) for s in sents])

print(maxlen)
# 180

vocab = set(''.join(sents))
num_words = len(vocab)
print(num_words)
print(len(sents))
# 56057
# 57340
num_words = 10000
embed_dim = 128
batch_size = 512
maxlen = 60

# 第一步：编码和填充（Tokenizing and Padding）
sents = [' '.join(list(t)) for t in sents]
tokenizer = Tokenizer(num_words = num_words, split=' ')
tokenizer.fit_on_texts(sents)
seqs = tokenizer.texts_to_sequences(sents)
pad_seqs = pad_sequences(seqs, maxlen)

# 第二步：编码器模型（Encoder Model）
encoder_inputs = Input(shape=(maxlen,), name='Encoder-Input')
emb_layer = Embedding(num_words, embed_dim,input_length = maxlen, name='Body-Word-Embedding', mask_zero=False)
x = emb_layer(encoder_inputs)
state_h = Bidirectional(LSTM(128, activation='relu', name='Encoder-Last-LSTM'))(x)
encoder_model = Model(inputs=encoder_inputs, outputs=state_h, name='Encoder-Model')
seq2seq_encoder_out = encoder_model(encoder_inputs)

# 第三步：解码器模型（Decoder Model）
decoded = RepeatVector(maxlen)(seq2seq_encoder_out)
decoder_lstm = Bidirectional(LSTM(128, return_sequences=True, name='Decoder-LSTM-before'))
decoder_lstm_output = decoder_lstm(decoded)
decoder_dense = Dense(num_words, activation='softmax', name='Final-Output-Dense-before')
decoder_outputs = decoder_dense(decoder_lstm_output)

# 第四步：构建模型及训练(Combining Model and Training)
seq2seq_Model = Model(encoder_inputs, decoder_outputs)
seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.001), loss='sparse_categorical_crossentropy')
history = seq2seq_Model.fit(pad_seqs, np.expand_dims(pad_seqs, -1),
          batch_size=batch_size,
          epochs=10)


vecs = encoder_model.predict(pad_seqs)
sentence = '今 天 天 气 真 好'
seq = tokenizer.texts_to_sequences([sentence])
pad_seq = pad_sequences(seq, maxlen)
sentence_vec = encoder_model.predict(pad_seq)[0]

def main():
    pass


if __name__ == '__main__':
    main()
