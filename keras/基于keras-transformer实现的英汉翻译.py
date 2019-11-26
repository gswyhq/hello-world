# encoding: utf-8
from __future__ import unicode_literals

import numpy as np
import sys
import os
import pickle
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import TensorBoard

sys.path.append('.')
# 依赖包 keras_transformer 的来源：https://github.com/CyberZHG/keras-transformer
from keras_transformer import get_model, decode


source_max_len = 128
target_max_len = 128

class Translate():

    @staticmethod
    def _build_token_dict(token_list):
        token_dict = {
            '<PAD>': 0,
            '<START>': 1,
            '<END>': 2,
            '<UNKOWN>': 3,
        }
        for tokens in token_list:
            for token in tokens:
                if token not in token_dict:
                    token_dict[token] = len(token_dict)
        return token_dict

    def train(self, train_file='/home/gswyhq/data/cmn-eng/cmn.txt'):
        source_tokens = [
            'i need more power'.split(' '),
            'eat jujube and pill'.split(' '),
        ]
        target_tokens = [
            list('我要更多的抛瓦'),
            list('吃枣💊'),
        ]

        with open(train_file)as f:
            for data in f.readlines():
                if '\t' in data:
                    source, target = data.strip().split('\t', maxsplit=1)
                    source_tokens.append(source.split(' '))
                    target_tokens.append(list(target))

        # Generate dictionaries
        source_token_dict = self._build_token_dict(source_tokens)
        target_token_dict = self._build_token_dict(target_tokens)
        target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

        # Add special tokens
        encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens]
        decode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in target_tokens]
        output_tokens = [tokens + ['<END>', '<PAD>'] for tokens in target_tokens]
        # print('output_tokens: {}'.format(output_tokens))
        # Padding
        # source_max_len = max(map(len, encode_tokens))
        # target_max_len = max(map(len, decode_tokens))

        print('source_max_len: {}; target_max_len: {}'.format(source_max_len, target_max_len))  # source_max_len: 34; target_max_len: 46
        print("len(source_token_dict): {}, len(target_token_dict): {}".format(len(source_token_dict), len(target_token_dict)))  # len(source_token_dict): 10814, len(target_token_dict): 3442

        with open('./models/target_token_dict.pkl', 'wb')as f:
            pickle.dump(target_token_dict, f)

        with open('./models/source_token_dict.pkl', 'wb')as f:
            pickle.dump(source_token_dict, f)

        encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
        decode_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in decode_tokens]
        output_tokens = [tokens + ['<PAD>'] * (target_max_len - len(tokens)) for tokens in output_tokens]
        # print('output_tokens: {}'.format(output_tokens))
        encode_input = [list(map(lambda x: source_token_dict[x], tokens)) for tokens in encode_tokens]
        decode_input = [list(map(lambda x: target_token_dict[x], tokens)) for tokens in decode_tokens]
        decode_output = [list(map(lambda x: [target_token_dict[x]], tokens)) for tokens in output_tokens]
        # print("decode_output: {}".format(decode_output))
        # Build & fit model
        model = get_model(
            token_num=max(len(source_token_dict), len(target_token_dict)),
            embed_dim=32,
            encoder_num=2,
            decoder_num=2,
            head_num=4,
            hidden_dim=128,
            dropout_rate=0.05,
            use_same_embed=False,  # Use different embeddings for different languages
        )
        model.compile('adam', 'sparse_categorical_crossentropy')
        model.summary()

        early_stopping = EarlyStopping(monitor='loss', patience=3)

        model_checkpoint = ModelCheckpoint(filepath=os.path.join('./models',
                                                                 'translate-{epoch:02d}-{loss:.4f}.hdf5'),
                                           save_best_only=False, save_weights_only=False)

        model.fit(
            x=[np.array(encode_input * 1), np.array(decode_input * 1)],
            y=np.array(decode_output * 1),
            epochs=10,
            batch_size=32,
            callbacks=[early_stopping, model_checkpoint]
        )

        model.save('./models/model.h5')

        # Predict
        encode_input = encode_input[:30]
        decoded = decode(
            model,
            encode_input,
            start_token=target_token_dict['<START>'],
            end_token=target_token_dict['<END>'],
            pad_token=target_token_dict['<PAD>'],
            max_repeat=len(encode_input),
            max_repeat_block=len(encode_input)
        )

        right_count = 0
        error_count = 0

        for i in range(len(encode_input)):
            predicted = ''.join(map(lambda x: target_token_dict_inv[x], decoded[i][1:-1]))
            print("原始结果：{}，预测结果：{}".format(''.join(target_tokens[i]), predicted))

            if ''.join(target_tokens[i]) == predicted:
                right_count += 1
            else:
                error_count += 1

        print("正确： {}， 错误：{}， 正确率： {}".format(right_count, error_count, right_count/(right_count+error_count+0.001)))

def predict():
    with open('./models/target_token_dict.pkl', 'rb')as f:
        target_token_dict = pickle.load(f
                                        )
    with open('./models/source_token_dict.pkl', 'rb')as f:
        source_token_dict = pickle.load(f)

    target_token_dict_inv = {v: k for k, v in target_token_dict.items()}

    source_tokens_list = [t.split() for t in '''He lost.
    I try.
    I won!
    I runs.
    I came.
    He run.
    We lost.
    We runs in the park every day.
    He calmed down.
    See you about 8.
    He get you.
    She wears a wig.'''.split('\n') if t]

    encode_tokens = [['<START>'] + tokens + ['<END>'] for tokens in source_tokens_list]
    encode_tokens = [tokens + ['<PAD>'] * (source_max_len - len(tokens)) for tokens in encode_tokens]
    encode_input = [list(map(lambda x: source_token_dict.get(x, source_token_dict['<UNKOWN>']), tokens)) for tokens in encode_tokens]

    model = get_model(
        token_num=max(len(source_token_dict), len(target_token_dict)),
        embed_dim=32,
        encoder_num=2,
        decoder_num=2,
        head_num=4,
        hidden_dim=128,
        dropout_rate=0.05,
        use_same_embed=False,  # Use different embeddings for different languages
    )
    model.load_weights('./models/model.h5', by_name=True, reshape=True)
    # Predict
    decoded = decode(
        model,
        encode_input,
        start_token=target_token_dict['<START>'],
        end_token=target_token_dict['<END>'],
        pad_token=target_token_dict['<PAD>'],
        max_repeat=len(encode_input),
        max_repeat_block=len(encode_input)
    )
    for i, source in enumerate(source_tokens_list):
        predicted = ''.join(map(lambda x: target_token_dict_inv[x], decoded[i][1:-1]))
        print("{}，预测结果：{}".format(source, predicted))

def main():
    # 训练 英汉互译语料
    # cmn-eng.zip
    # http://www.manythings.org/anki/cmn-eng.zip
    translate = Translate()
    translate.train(train_file='/home/gswyhq/data/cmn-eng/cmn.txt')
    del translate

    # 预测
    predict()

if __name__ == '__main__':
    main()
