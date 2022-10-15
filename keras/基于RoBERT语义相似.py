#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Semantic Similarity with RoBERT
# https://keras.io/examples/nlp/semantic_similarity_with_bert/#semantic-similarity-with-bert

import os
import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
from tensorflow import keras

max_length = 32  # Maximum length of input sentence to the model.
batch_size = 8
epochs = 2

# Labels in our dataset.
labels = [0, 1]


task_name = ''
def load_data(filename):
    """加载数据（带标签）
    单条格式：(文本1, 文本2, 标签)
    """
    D = []
    df = pd.read_csv(filename, sep='\t', encoding='utf-8')
    for text_a, text_b, label in df[ ['text_a', 'text_b', 'label']].values:
        D.append((text_a, text_b, float(label)))
    return D

# 数据来源：https://github.com/xiaohai-AI/lcqmc_data
USERNAME = os.getenv("USERNAME")
data_path = rf'D:\Users\{USERNAME}\github_project\lcqmc_data'
datasets = {
    '%s-%s' % (task_name, f):
    load_data('%s/%s.txt' % (data_path, f))
    for f in ['train', 'dev', 'test']
}


train_data = datasets['-train']
dev_data = datasets['-dev']
test_data = datasets['-test']
# train_data = train_data[:100]
# dev_data = dev_data[:10]

class BertSemanticDataGenerator(tf.keras.utils.Sequence):
    """Generates batches of data.

    Args:
        sentence_pairs: Array of premise and hypothesis input sentences.
        labels: Array of labels.
        batch_size: Integer batch size.
        shuffle: boolean, whether to shuffle the data.
        include_targets: boolean, whether to incude the labels.

    Returns:
        Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
        (or just `[input_ids, attention_mask, `token_type_ids]`
         if `include_targets=False`)
    """

    def __init__(
        self,
        sentence_pairs,
        labels,
        batch_size=batch_size,
        shuffle=True,
        include_targets=True,
    ):
        self.sentence_pairs = sentence_pairs
        self.labels = labels
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.include_targets = include_targets
        # Load our BERT Tokenizer to encode the text.
        # We will use base-base-uncased pretrained model.
        # https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main
        self.tokenizer = transformers.BertTokenizer.from_pretrained( rf'D:\Users\{USERNAME}\data\chinese_roberta_L-4_H-128',
            # "bert-base-uncased",
                                                                     do_lower_case=True
        )
        self.indexes = np.arange(len(self.sentence_pairs))
        self.on_epoch_end()

    def __len__(self):
        # Denotes the number of batches per epoch.
        return len(self.sentence_pairs) // self.batch_size

    def __getitem__(self, idx):
        # Retrieves the batch of index.
        indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
        sentence_pairs = self.sentence_pairs[indexes]

        # With BERT tokenizer's batch_encode_plus batch of both the sentences are
        # encoded together and separated by [SEP] token.
        encoded = self.tokenizer.batch_encode_plus(
            sentence_pairs.tolist(),
            add_special_tokens=True,
            max_length=max_length,
            return_attention_mask=True,
            return_token_type_ids=True,
            pad_to_max_length=True,
            return_tensors="tf",
        )

        # Convert batch of encoded features to numpy array.
        input_ids = np.array(encoded["input_ids"], dtype="int32") # Bert的输入通常需要两个句子，句子A前由[CLS]开始，以[SEP]结束，后面再连接句子B
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")  # 由于不同批次的数据长度不同，因此会对数据进行补全。但补全的信息对于网络是无用的，这个主要是输入的句子可能存在填0的操作，attention模块不需要把填0的无意义的信息算进来，所以使用mask操作。
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32") # 用于标记一个input_ids序列中哪些位置是第一句话，哪些位置是第二句话。

        # Set to true if data generator is used for training/validation.
        if self.include_targets:
            labels = np.array(self.labels[indexes], dtype="int32")
            return [input_ids, attention_masks, token_type_ids], labels
        else:
            return [input_ids, attention_masks, token_type_ids]

    def on_epoch_end(self):
        # Shuffle indexes after each epoch if shuffle is set to True.
        if self.shuffle:
            np.random.RandomState(42).shuffle(self.indexes)



# Create the model under a distribution strategy scope.
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Encoded token ids from BERT tokenizer.
    input_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="input_ids"
    )
    # Attention masks indicates to the model which tokens should be attended to.
    attention_masks = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="attention_masks"
    )
    # Token type ids are binary masks identifying different sequences in the model.
    token_type_ids = tf.keras.layers.Input(
        shape=(max_length,), dtype=tf.int32, name="token_type_ids"
    )
    # Loading pretrained BERT model.
    bert_model = transformers.TFBertModel.from_pretrained(rf'D:\Users\{USERNAME}\data\chinese_roberta_L-4_H-128',
        # "bert-base-uncased"
                                                          )
    # 冻结 BERT 模型以重用预训练的特征而不修改它们。
    bert_model.trainable = False

    bert_output = bert_model.bert(
        input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
    )
    sequence_output = bert_output.last_hidden_state
    pooled_output = bert_output.pooler_output
    # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
    bi_lstm = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(64, return_sequences=True)
    )(sequence_output)
    # Applying hybrid pooling approach to bi_lstm sequence output.
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(bi_lstm)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(bi_lstm)
    concat = tf.keras.layers.concatenate([avg_pool, max_pool])
    dropout = tf.keras.layers.Dropout(0.3)(concat)
    output = tf.keras.layers.Dense(2, activation="softmax")(dropout)
    model = tf.keras.models.Model(
        inputs=[input_ids, attention_masks, token_type_ids], outputs=output
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["acc"],
    )


print(f"Strategy: {strategy}")

keras.backend.clear_session()
model.summary()

# Model: "model_1"
# __________________________________________________________________________________________________
#  Layer (type)                   Output Shape         Param #     Connected to
# ==================================================================================================
#  input_ids (InputLayer)         [(None, 32)]         0           []
#
#  attention_masks (InputLayer)   [(None, 32)]         0           []
#
#  token_type_ids (InputLayer)    [(None, 32)]         0           []
#
#  bert (TFBertMainLayer)         TFBaseModelOutputWi  3580032     ['input_ids[0][0]',
#                                 thPoolingAndCrossAt               'attention_masks[0][0]',
#                                 tentions(last_hidde               'token_type_ids[0][0]']
#                                 n_state=(None, 32,
#                                 128),
#                                  pooler_output=(Non
#                                 e, 128),
#                                  past_key_values=No
#                                 ne, hidden_states=N
#                                 one, attentions=Non
#                                 e, cross_attentions
#                                 =None)
#
#  bidirectional_1 (Bidirectional  (None, 32, 128)     98816       ['bert[0][0]']
#  )
#
#  global_average_pooling1d_1 (Gl  (None, 128)         0           ['bidirectional_1[0][0]']
#  obalAveragePooling1D)
#
#  global_max_pooling1d_1 (Global  (None, 128)         0           ['bidirectional_1[0][0]']
#  MaxPooling1D)
#
#  concatenate_1 (Concatenate)    (None, 256)          0           ['global_average_pooling1d_1[0][0
#                                                                  ]',
#                                                                   'global_max_pooling1d_1[0][0]']
#
#  dropout_40 (Dropout)           (None, 256)          0           ['concatenate_1[0][0]']
#
#  dense_1 (Dense)                (None, 2)            514         ['dropout_40[0][0]']
#
# ==================================================================================================
# Total params: 3,679,362
# Trainable params: 99,330
# Non-trainable params: 3,580,032
# __________________________________________________________________________________________________

train_sentence_pairs = np.array([(t[0], t[1]) for t in train_data])
y_train = tf.keras.utils.to_categorical([int(t[-1]) for t in train_data], num_classes=2)

valid_sentence_pairs = np.array([(t[0], t[1]) for t in dev_data])
y_val = tf.keras.utils.to_categorical([int(t[-1]) for t in dev_data], num_classes=2)

train_data = BertSemanticDataGenerator(
    train_sentence_pairs,
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_sentence_pairs,
    y_val,
    batch_size=batch_size,
    shuffle=False,
)

# 训练模型
# 仅对顶层进行训练以执行“特征提取”，这将允许模型使用预训练模型的表示。
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
# 29845/29845 [==============================] - 1369s 46ms/step - loss: 0.5388 - acc: 0.7291 - val_loss: 0.7196 - val_acc: 0.6166
# history.history
# Out[15]:
# {'loss': [0.5748350620269775, 0.538760781288147],
#  'acc': [0.6992461085319519, 0.7290710210800171],
#  'val_loss': [0.7768730521202087, 0.7196312546730042],
#  'val_acc': [0.5847727060317993, 0.6165909171104431]}

# 微调
# 此步骤只能在特征提取模型经过训练以收敛于新数据后执行。
# 这是可选的最后一步，bert_model 以非常低的学习率解冻和重新训练。这可以通过将预训练的特征逐步适应新数据来提供有意义的改进。

# # Unfreeze the bert_model.
# bert_model.trainable = True
# for l in model.layers:
#     l.trainable = True
# # Recompile the model to make the change effective.
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(1e-5),
#     loss="categorical_crossentropy",
#     metrics=["accuracy"],
# )
# model.summary()
# ==================================================================================================
# Total params: 3,679,362
# Trainable params: 3,679,362
# Non-trainable params: 0
# __________________________________________________________________________________________________
# history = model.fit(
#     train_data,
#     validation_data=valid_data,
#     epochs=epochs,
#     use_multiprocessing=True,
#     workers=-1,
# )
# 29845/29845 [==============================] - 3486s 117ms/step - loss: 0.3818 - accuracy: 0.8305 - val_loss: 0.5923 - val_accuracy: 0.7397

# 在测试集上评估模型
# test_sentence_pairs = np.array([(t[0], t[1]) for t in test_data])
# y_test = tf.keras.utils.to_categorical([int(t[-1]) for t in test_data], num_classes=2)
# test_data = BertSemanticDataGenerator(
#     test_sentence_pairs,
#     y_test,
#     batch_size=batch_size,
#     shuffle=False,
# )
# model.evaluate(test_data, verbose=1)
# 微调前：测试集评估结果：loss: 0.6796 - acc: 0.6401
# 微调前：测试集评估结果：loss: 0.5295 - accuracy: 0.7629


# 对自定义句子的推断
# def check_similarity(sentence1, sentence2):
#     sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
#     test_data = BertSemanticDataGenerator(
#         sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
#     )
#
#     proba = model.predict(test_data[0])[0]
#     idx = np.argmax(proba)
#     proba = f"{proba[idx]: .2f}%"
#     pred = labels[idx]
#     return pred, proba
# 检查一些例句对的结果。

# sentence1 = "谁有高清图？"
# sentence2 = "这张图高清的，谁有"
# check_similarity(sentence1, sentence2)
# (1, ' 0.92%')

def main():
    pass


if __name__ == '__main__':
    main()
