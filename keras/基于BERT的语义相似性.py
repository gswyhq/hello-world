#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 来源： https://keras.io/examples/nlp/semantic_similarity_with_bert/#finetuning

# 介绍
# 语义相似性是确定两个句子的相似程度的任务，就它们的意思而言。这个例子演示了使用 SNLI（斯坦福自然语言推理）语料库来预测与 Transformers 的句子语义相似度。我们将微调一个以两个句子作为输入并输出这两个句子的相似度分数的 BERT 模型。

# 设置
# 注意：transformers通过pip install transformers（版本 >= 2.11.0）安装 HuggingFace。

import numpy as np
import pandas as pd
import tensorflow as tf
import transformers
import os
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, average_precision_score, precision_recall_curve
from sklearn.utils import shuffle
# 配置
max_length = 128  # Maximum length of input sentence to the model.
batch_size = 32
epochs = 2

# # 加载数据
# 数据来源：https://github.com/IAdmireu/ChineseSTS.git
data_df = pd.read_csv(r"D:\Users\{}\github_project\ChineseSTS\simtrain_to05sts.txt".format(os.getenv('USERNAME')), header=None, sep='\t', names=['id1', 'sentence1', 'id2', 'sentence2', 'label'])
data_df = data_df[data_df['label'].isin([5.0, 0.0])][['sentence1', 'sentence2', 'label']]
data_df['label'] = [1 if t>0 else 0 for t in data_df['label'].values]
data_df = shuffle(data_df)
train_df, test_df = train_test_split(data_df, test_size=0.1)
train_df, valid_df = train_test_split(train_df, test_size=0.1)
y_train = tf.keras.utils.to_categorical(train_df.label, num_classes=2)
y_val = tf.keras.utils.to_categorical(valid_df.label, num_classes=2)
y_test = tf.keras.utils.to_categorical(test_df.label, num_classes=2)

# 创建自定义数据生成器
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
        # self.tokenizer = transformers.BertTokenizer.from_pretrained(
        #     "bert-base-uncased", do_lower_case=True
        # )
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            r"D:\Users\{}\data\chinese_roberta_L-4_H-128".format(os.getenv('USERNAME')), do_lower_case=True
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
        input_ids = np.array(encoded["input_ids"], dtype="int32")
        attention_masks = np.array(encoded["attention_mask"], dtype="int32")
        token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

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

# 建立模型
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
    # bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
    # https://huggingface.co/uer/chinese_roberta_L-4_H-128/tree/main
    bert_model = transformers.TFBertModel.from_pretrained(r"D:\Users\{}\data\chinese_roberta_L-4_H-128".format(os.getenv('USERNAME')))
    # Freeze the BERT model to reuse the pretrained features without modifying them.
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
model.summary()
# Model: "model"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_ids (InputLayer)          [(None, 128)]        0
# __________________________________________________________________________________________________
# attention_masks (InputLayer)    [(None, 128)]        0
# __________________________________________________________________________________________________
# token_type_ids (InputLayer)     [(None, 128)]        0
# __________________________________________________________________________________________________
# bert (TFBertMainLayer)          TFBaseModelOutputWit 3580032     input_ids[0][0]
#                                                                  attention_masks[0][0]
#                                                                  token_type_ids[0][0]
# __________________________________________________________________________________________________
# bidirectional (Bidirectional)   (None, 128, 128)     98816       bert[0][0]
# __________________________________________________________________________________________________
# global_average_pooling1d (Globa (None, 128)          0           bidirectional[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d (GlobalMax (None, 128)          0           bidirectional[0][0]
# __________________________________________________________________________________________________
# concatenate (Concatenate)       (None, 256)          0           global_average_pooling1d[0][0]
#                                                                  global_max_pooling1d[0][0]
# __________________________________________________________________________________________________
# dropout_52 (Dropout)            (None, 256)          0           concatenate[0][0]
# __________________________________________________________________________________________________
# dense (Dense)                   (None, 3)            771         dropout_52[0][0]
# ==================================================================================================
# Total params: 3,679,619
# Trainable params: 99,587
# Non-trainable params: 3,580,032
# __________________________________________________________________________________________________

# 创建训练和验证数据生成器

train_data = BertSemanticDataGenerator(
    train_df[["sentence1", "sentence2"]].values.astype("str"),
    y_train,
    batch_size=batch_size,
    shuffle=True,
)
valid_data = BertSemanticDataGenerator(
    valid_df[["sentence1", "sentence2"]].values.astype("str"),
    y_val,
    batch_size=batch_size,
    shuffle=False,
)
# HBox(children=(FloatProgress(value=0.0, description='Downloading', max=231508.0, style=ProgressStyle(descripti…
# 训练模型
# 仅对顶层进行训练以执行“特征提取”，这将允许模型使用预训练模型的表示。

history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
# Epoch 1/2
# 319/319 [==============================] - 188s 555ms/step - loss: 0.2509 - acc: 0.8969 - val_loss: 0.1698 - val_acc: 0.9375
# Epoch 2/2
#

# 微调
# 此步骤只能在特征提取模型经过训练以收敛于新数据后执行。
#
# 这是可选的最后一步，bert_model以非常低的学习率解冻和重新训练。这可以通过将预训练的特征逐步适应新数据来提供有意义的改进。

# Unfreeze the bert_model.
bert_model.trainable = True
# Recompile the model to make the change effective.
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

# Model: "model_1"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to
# ==================================================================================================
# input_ids (InputLayer)          [(None, 128)]        0
# __________________________________________________________________________________________________
# attention_masks (InputLayer)    [(None, 128)]        0
# __________________________________________________________________________________________________
# token_type_ids (InputLayer)     [(None, 128)]        0
# __________________________________________________________________________________________________
# bert (TFBertMainLayer)          TFBaseModelOutputWit 3580032     input_ids[0][0]
#                                                                  attention_masks[0][0]
#                                                                  token_type_ids[0][0]
# __________________________________________________________________________________________________
# bidirectional_1 (Bidirectional) (None, 128, 128)     98816       bert[0][0]
# __________________________________________________________________________________________________
# global_average_pooling1d_1 (Glo (None, 128)          0           bidirectional_1[0][0]
# __________________________________________________________________________________________________
# global_max_pooling1d_1 (GlobalM (None, 128)          0           bidirectional_1[0][0]
# __________________________________________________________________________________________________
# concatenate_1 (Concatenate)     (None, 256)          0           global_average_pooling1d_1[0][0]
#                                                                  global_max_pooling1d_1[0][0]
# __________________________________________________________________________________________________
# dropout_66 (Dropout)            (None, 256)          0           concatenate_1[0][0]
# __________________________________________________________________________________________________
# dense_1 (Dense)                 (None, 2)            514         dropout_66[0][0]
# ==================================================================================================
# Total params: 3,679,362
# Trainable params: 3,679,362
# Non-trainable params: 0
# __________________________________________________________________________________________________

# 端到端训练整个模型
history = model.fit(
    train_data,
    validation_data=valid_data,
    epochs=epochs,
    use_multiprocessing=True,
    workers=-1,
)
# Epoch 1/2
# 319/319 [==============================] - 405s 1s/step - loss: 0.1160 - accuracy: 0.9557 - val_loss: 0.0980 - val_accuracy: 0.9652
# Epoch 2/2
# 319/319 [==============================] - 432s 1s/step - loss: 0.0980 - accuracy: 0.9641 - val_loss: 0.0920 - val_accuracy: 0.9652


# 在测试集上评估模型
test_data = BertSemanticDataGenerator(
    test_df[["sentence1", "sentence2"]].values.astype("str"),
    y_test,
    batch_size=batch_size,
    shuffle=False,
)
model.evaluate(test_data, verbose=1)
# 312/312 [==============================] - 55s 177ms/step - loss: 0.3697 - accuracy: 0.8629

# [0.10339359194040298, 0.9591346383094788]
# 对自定义句子的推断
labels = ["不相似", "相似"]
def check_similarity(sentence1, sentence2):
    sentence_pairs = np.array([[str(sentence1), str(sentence2)]])
    test_data = BertSemanticDataGenerator(
        sentence_pairs, labels=None, batch_size=1, shuffle=False, include_targets=False,
    )

    proba = model.predict(test_data[0])[0]
    idx = np.argmax(proba)
    proba = f"{proba[idx]: .4f}"
    pred = labels[idx]
    return pred, proba
# 检查一些例句对的结果。

sentence1 = "我给了她一只笔。"
sentence2 = "我给一支笔她了"
check_similarity(sentence1, sentence2)
# ('相似', ' 0.9861')

# 检查一些例句对的结果。

sentence1 = "他犯了一点错误。"
sentence2 = "他一点错误没犯。"
check_similarity(sentence1, sentence2)
# ('相似', ' 0.9194')

# 检查一些例句对的结果

sentence1 = "不是每家都有汽车。"
sentence2 = "汽车不是每家都有。"
check_similarity(sentence1, sentence2)
# ('相似', ' 0.9781')

sentence1 = "这个计划没有一点问题。"
sentence2 = "这个计划有问题。"
check_similarity(sentence1, sentence2)
# ('相似', ' 0.8408')

# 最终训练的模型存在问题，好多不相似的，相似分数却很高；
# 可能是因为样本不平衡，及数据集问题；

def main():
    pass


if __name__ == '__main__':
    main()