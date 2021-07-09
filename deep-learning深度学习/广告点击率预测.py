#!/usr/bin/python3
# coding: utf-8

# 数据集及资料来源
# wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
# gzip -d reviews_Electronics_5.json.gz
# wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
# gzip -d meta_Electronics.json.gz
# https://zhuanlan.zhihu.com/p/144153291
# https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master/DIN
# https://mp.weixin.qq.com/s/uIs_FpeowSEpP5fkVDq1Nw


import tensorflow as tf
from time import time
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import AUC

import pandas as pd
import numpy as np
import pickle
import random
from tqdm import tqdm
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, Dense, BatchNormalization, Input, PReLU, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, BatchNormalization, Dense

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



# 其中reviews_Electronics_5.json为用户的行为数据，meta_Electronics为广告的元数据。

# reviews某单个样本如下：

# {
#   "reviewerID": "A2SUAM1J3GNN3B",
#   "asin": "0000013714",
#   "reviewerName": "J. McDonald",
#   "helpful": [2, 3],
#   "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
#   "overall": 5.0,
#   "summary": "Heavenly Highway Hymns",
#   "unixReviewTime": 1252800000,
#   "reviewTime": "09 13, 2009"
# }
# 各字段分别为：

# reviewerID：用户ID；
# asin： 物品ID；
# reviewerName：用户姓名；
# helpful ：评论帮助程度，例如上述为2/3；
# reviewText ：文本信息；
# overall ：物品评分；
# summary：评论总结
# unixReviewTime ：时间戳
# reviewTime ：时间
# meta某样本如下：

# {
#   "asin": "0000031852",
#   "title": "Girls Ballet Tutu Zebra Hot Pink",
#   "price": 3.17,
#   "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
#   "related":
#   {
#     "also_bought": ["B00JHONN1S", "B002BZX8Z6", "B00D2K1M3O", "0000031909", "B00613WDTQ", "B00D0WDS9A", "B00D0GCI8S", "0000031895", "B003AVKOP2", "B003AVEU6G", "B003IEDM9Q", "B002R0FA24", "B00D23MC6W", "B00D2K0PA0", "B00538F5OK", "B00CEV86I6", "B002R0FABA", "B00D10CLVW", "B003AVNY6I", "B002GZGI4E", "B001T9NUFS", "B002R0F7FE", "B00E1YRI4C", "B008UBQZKU", "B00D103F8U", "B007R2RM8W"],
#     "also_viewed": ["B002BZX8Z6", "B00JHONN1S", "B008F0SU0Y", "B00D23MC6W", "B00AFDOPDA", "B00E1YRI4C", "B002GZGI4E", "B003AVKOP2", "B00D9C1WBM", "B00CEV8366", "B00CEUX0D8", "B0079ME3KU", "B00CEUWY8K", "B004FOEEHC", "0000031895", "B00BC4GY9Y", "B003XRKA7A", "B00K18LKX2", "B00EM7KAG6", "B00AMQ17JA", "B00D9C32NI", "B002C3Y6WG", "B00JLL4L5Y", "B003AVNY6I", "B008UBQZKU", "B00D0WDS9A", "B00613WDTQ", "B00538F5OK", "B005C4Y4F6", "B004LHZ1NY", "B00CPHX76U", "B00CEUWUZC", "B00IJVASUE", "B00GOR07RE", "B00J2GTM0W", "B00JHNSNSM", "B003IEDM9Q", "B00CYBU84G", "B008VV8NSQ", "B00CYBULSO", "B00I2UHSZA", "B005F50FXC", "B007LCQI3S", "B00DP68AVW", "B009RXWNSI", "B003AVEU6G", "B00HSOJB9M", "B00EHAGZNA", "B0046W9T8C", "B00E79VW6Q", "B00D10CLVW", "B00B0AVO54", "B00E95LC8Q", "B00GOR92SO", "B007ZN5Y56", "B00AL2569W", "B00B608000", "B008F0SMUC", "B00BFXLZ8M"],
#     "bought_together": ["B002BZX8Z6"]
#   },
#   "salesRank": {"Toys & Games": 211836},
#   "brand": "Coxlures",
#   "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
# }
# 各字段分别为：

# asin ：物品ID；
# title ：物品名称；
# price ：物品价格；
# imUrl ：物品图片的URL；
# related ：相关产品(也买，也看，一起买，看后再买)；
# salesRank： 销售排名信息；
# brand ：品牌名称；
# categories ：该物品属于的种类列表；
# 2、首先将原生数据存储的json格式转化为pickle数据流格式，方便读取：

def to_df(file_path):
    """
    转化为DataFrame结构
    :param file_path: 文件路径
    :return:
    """
    with open(file_path, 'r') as fin:
        df = {}
        i = 0
        for line in fin:
            df[i] = eval(line)
            i += 1
        df = pd.DataFrame.from_dict(df, orient='index')
        return df

reviews_df = to_df('../raw_data/reviews_Electronics_5.json')

# 可以直接调用pandas的read_json方法，但会改变列的顺序
# reviews2_df = pd.read_json('../raw_data/reviews_Electronics_5.json', lines=True)

# 序列化保存
with open('../raw_data/reviews.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)

meta_df = to_df('../raw_data/meta_Electronics.json')
# 只保留review_df出现过的广告
meta_df = meta_df[meta_df['asin'].isin(reviews_df['asin'].unique())]
meta_df = meta_df.reset_index(drop=True)

with open('../raw_data/meta.pkl', 'wb') as f:
    pickle.dump(meta_df, f, pickle.HIGHEST_PROTOCOL)

# 3、对reaviews和meta数据进行处理：
#
# reviews选取'reviewerID', 'asin', 'unixReviewTime'列，并将用户ID、物品ID【通过meta】映射为数值；
# meta选取'asin', 'categories'列，物品种类只选取列表最后一个，并将物品ID、种类ID进行映射；
# 统计用户人数user_count、物品总数item_count，总样本数sample_count；
# 保存reviews数据、物品种类列表、各个数值数据以及映射字典；
def build_map(df, col_name):
    """
    制作一个映射，键为列名，值为序列数字
    :param df: reviews_df / meta_df
    :param col_name: 列名
    :return: 字典，键
    """
    key = sorted(df[col_name].unique().tolist())
    m = dict(zip(key, range(len(key))))
    df[col_name] = df[col_name].map(lambda x: m[x])
    return m, key


# reviews
reviews_df = pd.read_pickle('../raw_data/reviews.pkl')
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# meta
meta_df = pd.read_pickle('../raw_data/meta.pkl')
meta_df = meta_df[['asin', 'categories']]
# 类别只保留最后一个
meta_df['categories'] = meta_df['categories'].map(lambda x: x[-1][-1])

# meta_df文件的物品ID映射
asin_map, asin_key = build_map(meta_df, 'asin')
# meta_df文件物品种类映射
cate_map, cate_key = build_map(meta_df, 'categories')
# reviews_df文件的用户ID映射
revi_map, revi_key = build_map(reviews_df, 'reviewerID')

# user_count: 192403 item_count: 63001 cate_count: 801 example_count: 1689188
user_count, item_count, cate_count, example_count = \
    len(revi_map), len(asin_map), len(cate_map), reviews_df.shape[0]
# print('user_count: %d\titem_count: %d\tcate_count: %d\texample_count: %d' %
#       (user_count, item_count, cate_count, example_count))

# 按物品id排序，并重置索引
meta_df = meta_df.sort_values('asin')
meta_df = meta_df.reset_index(drop=True)

# reviews_df文件物品id进行映射，并按照用户id、浏览时间进行排序，重置索引
reviews_df['asin'] = reviews_df['asin'].map(lambda x: asin_map[x])
reviews_df = reviews_df.sort_values(['reviewerID', 'unixReviewTime'])
reviews_df = reviews_df.reset_index(drop=True)
reviews_df = reviews_df[['reviewerID', 'asin', 'unixReviewTime']]

# 各个物品对应的类别
cate_list = np.array(meta_df['categories'], dtype='int32')

# 保存所需数据为pkl文件
with open('../raw_data/remap.pkl', 'wb') as f:
    pickle.dump(reviews_df, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump(cate_list, f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((user_count, item_count, cate_count, example_count),
                f, pickle.HIGHEST_PROTOCOL)
    pickle.dump((asin_key, cate_key, revi_key), f, pickle.HIGHEST_PROTOCOL)


class Attention_Layer(Layer):
    def __init__(self, att_hidden_units, activation='sigmoid'):
        """
        """
        super(Attention_Layer, self).__init__()
        self.att_dense = [Dense(unit, activation=activation) for unit in att_hidden_units]
        self.att_final_dense = Dense(1)

    def call(self, inputs):
        # query: candidate item  (None, d * 2), d is the dimension of embedding
        # key: hist items  (None, seq_len, d * 2)
        # value: hist items  (None, seq_len, d * 2)
        # mask: (None, seq_len)
        q, k, v, mask = inputs
        q = tf.tile(q, multiples=[1, k.shape[1]])  # (None, seq_len * d * 2)
        q = tf.reshape(q, shape=[-1, k.shape[1], k.shape[2]])  # (None, seq_len, d * 2)

        # q, k, out product should concat
        info = tf.concat([q, k, q - k, q * k], axis=-1)

        # dense
        for dense in self.att_dense:
            info = dense(info)

        outputs = self.att_final_dense(info)  # (None, seq_len, 1)
        outputs = tf.squeeze(outputs, axis=-1)  # (None, seq_len)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)  # (None, seq_len)
        outputs = tf.where(tf.equal(mask, 0), paddings, outputs)  # (None, seq_len)

        # softmax
        outputs = tf.nn.softmax(logits=outputs)  # (None, seq_len)
        outputs = tf.expand_dims(outputs, axis=1)  # None, 1, seq_len)

        outputs = tf.matmul(outputs, v)  # (None, 1, d * 2)
        outputs = tf.squeeze(outputs, axis=1)

        return outputs


class Dice(Layer):
    def __init__(self):
        super(Dice, self).__init__()
        self.bn = BatchNormalization(center=False, scale=False)
        self.alpha = self.add_weight(shape=(), dtype=tf.float32, name='alpha')

    def call(self, x):
        x_normed = self.bn(x)
        x_p = tf.sigmoid(x_normed)

        return self.alpha * (1.0 - x_p) * x + x_p * x


class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40),
                 ffn_hidden_units=(80, 40), att_activation='sigmoid', ffn_activation='prelu', maxlen=40, dnn_dropout=0.,
                 embed_reg=1e-4):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
        super(DIN, self).__init__()
        self.maxlen = maxlen

        self.dense_feature_columns, self.sparse_feature_columns = feature_columns

        # len
        self.other_sparse_len = len(self.sparse_feature_columns) - len(behavior_feature_list)
        self.dense_len = len(self.dense_feature_columns)
        self.behavior_num = len(behavior_feature_list)

        # other embedding layers
        self.embed_sparse_layers = [Embedding(input_dim=feat['feat_num'],
                                              input_length=1,
                                              output_dim=feat['embed_dim'],
                                              embeddings_initializer='random_uniform',
                                              embeddings_regularizer=l2(embed_reg))
                                    for feat in self.sparse_feature_columns
                                    if feat['feat'] not in behavior_feature_list]
        # behavior embedding layers, item id and category id
        self.embed_seq_layers = [Embedding(input_dim=feat['feat_num'],
                                           input_length=1,
                                           output_dim=feat['embed_dim'],
                                           embeddings_initializer='random_uniform',
                                           embeddings_regularizer=l2(embed_reg))
                                 for feat in self.sparse_feature_columns
                                 if feat['feat'] in behavior_feature_list]

        # attention layer
        self.attention_layer = Attention_Layer(att_hidden_units, att_activation)

        self.bn = BatchNormalization(trainable=True)
        # ffn
        self.ffn = [Dense(unit, activation=PReLU() if ffn_activation == 'prelu' else Dice()) \
                    for unit in ffn_hidden_units]
        self.dropout = Dropout(dnn_dropout)
        self.dense_final = Dense(1)

    def call(self, inputs):
        # dense_inputs and sparse_inputs is empty
        # seq_inputs (None, maxlen, behavior_num)
        # item_inputs (None, behavior_num)
        dense_inputs, sparse_inputs, seq_inputs, item_inputs = inputs
        # attention ---> mask, if the element of seq_inputs is equal 0, it must be filled in.
        mask = tf.cast(tf.not_equal(seq_inputs[:, :, 0], 0), dtype=tf.float32)  # (None, maxlen)
        # other
        other_info = dense_inputs
        for i in range(self.other_sparse_len):
            other_info = tf.concat([other_info, self.embed_sparse_layers[i](sparse_inputs[:, i])], axis=-1)

        # seq, item embedding and category embedding should concatenate
        seq_embed = tf.concat([self.embed_seq_layers[i](seq_inputs[:, :, i]) for i in range(self.behavior_num)],
                              axis=-1)
        item_embed = tf.concat([self.embed_seq_layers[i](item_inputs[:, i]) for i in range(self.behavior_num)], axis=-1)

        # att
        user_info = self.attention_layer([item_embed, seq_embed, seq_embed, mask])  # (None, d * 2)

        # concat user_info(att hist), cadidate item embedding, other features
        if self.dense_len > 0 or self.other_sparse_len > 0:
            info_all = tf.concat([user_info, item_embed, other_info], axis=-1)
        else:
            info_all = tf.concat([user_info, item_embed], axis=-1)

        info_all = self.bn(info_all)

        # ffn
        for dense in self.ffn:
            info_all = dense(info_all)

        info_all = self.dropout(info_all)
        outputs = tf.nn.sigmoid(self.dense_final(info_all))
        return outputs

    def summary(self):
        dense_inputs = Input(shape=(self.dense_len,), dtype=tf.float32)
        sparse_inputs = Input(shape=(self.other_sparse_len,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen, self.behavior_num), dtype=tf.int32)
        item_inputs = Input(shape=(self.behavior_num,), dtype=tf.int32)
        tf.keras.Model(inputs=[dense_inputs, sparse_inputs, seq_inputs, item_inputs],
                       outputs=self.call([dense_inputs, sparse_inputs, seq_inputs, item_inputs])).summary()


def sparseFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name
    :param feat_num: the total number of sparse features that do not repeat
    :param embed_dim: embedding dimension
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}


def denseFeature(feat):
    """
    create dictionary for dense feature
    :param feat: dense feature name
    :return:
    """
    return {'feat': feat}


def create_amazon_electronic_dataset(remap_file, embed_dim=8, maxlen=40):
    """
    :param file: dataset path
    :param embed_dim: latent factor
    :param maxlen:
    :return: user_num, item_num, train_df, test_df
    """
    print('==========Data Preprocess Start============')
    with open(remap_file, 'rb') as f:
        reviews_df = pickle.load(f)
        cate_list = pickle.load(f)
        user_count, item_count, cate_count, example_count = pickle.load(f)

    reviews_df = reviews_df
    reviews_df.columns = ['user_id', 'item_id', 'time']

    train_data, val_data, test_data = [], [], []

    for user_id, hist in tqdm(reviews_df.groupby('user_id')):
        pos_list = hist['item_id'].tolist()

        def gen_neg():
            neg = pos_list[0]
            while neg in pos_list:
                neg = random.randint(0, item_count - 1)
            return neg

        neg_list = [gen_neg() for i in range(len(pos_list))]
        hist = []
        for i in range(1, len(pos_list)):
            # hist.append([pos_list[i-1], cate_list[pos_list[i-1]]])
            hist.append([pos_list[i - 1]])
            if i == len(pos_list) - 1:
                # test_data.append([hist, [pos_list[i], cate_list[pos_list[i]]], 1])
                # test_data.append([hist, [neg_list[i], cate_list[neg_list[i]]], 0])
                test_data.append([hist, [pos_list[i]], 1])
                test_data.append([hist, [neg_list[i]], 0])
            elif i == len(pos_list) - 2:
                # val_data.append([hist, [pos_list[i], cate_list[pos_list[i]]], 1])
                # val_data.append([hist, [neg_list[i], cate_list[neg_list[i]]], 0])
                val_data.append([hist, [pos_list[i]], 1])
                val_data.append([hist, [neg_list[i]], 0])
            else:
                # train_data.append([hist, [pos_list[i], cate_list[pos_list[i]]], 1])
                # train_data.append([hist, [neg_list[i], cate_list[neg_list[i]]], 0])
                train_data.append([hist, [pos_list[i]], 1])
                train_data.append([hist, [neg_list[i]], 0])

                # feature columns
    feature_columns = [[],
                       [sparseFeature('item_id', item_count, embed_dim),
                        ]]  # sparseFeature('cate_id', cate_count, embed_dim)

    # behavior
    behavior_list = ['item_id']  # , 'cate_id'

    # shuffle
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # create dataframe
    train = pd.DataFrame(train_data, columns=['hist', 'target_item', 'label'])
    val = pd.DataFrame(val_data, columns=['hist', 'target_item', 'label'])
    test = pd.DataFrame(test_data, columns=['hist', 'target_item', 'label'])

    # if no dense or sparse features, can fill with 0
    print('==================Padding===================')
    train_X = [np.array([0.] * len(train)), np.array([0] * len(train)),
               pad_sequences(train['hist'], maxlen=maxlen),
               np.array(train['target_item'].tolist())]
    train_y = train['label'].values
    val_X = [np.array([0] * len(val)), np.array([0] * len(val)),
             pad_sequences(val['hist'], maxlen=maxlen),
             np.array(val['target_item'].tolist())]
    val_y = val['label'].values
    test_X = [np.array([0] * len(test)), np.array([0] * len(test)),
              pad_sequences(test['hist'], maxlen=maxlen),
              np.array(test['target_item'].tolist())]
    test_y = test['label'].values
    print('============Data Preprocess End=============')
    return feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)


def main():
    # ========================= Hyper Parameters =======================
    remap_file = '../raw_data/remap.pkl'
    maxlen = 20

    embed_dim = 8
    att_hidden_units = [80, 40]
    ffn_hidden_units = [256, 128, 64]
    dnn_dropout = 0.5
    att_activation = 'sigmoid'
    ffn_activation = 'prelu'

    learning_rate = 0.001
    batch_size = 4096
    epochs = 5
    # ========================== Create dataset =======================
    feature_columns, behavior_list, train, val, test = create_amazon_electronic_dataset(remap_file, embed_dim, maxlen)
    train_X, train_y = train
    val_X, val_y = val
    test_X, test_y = test
    # ============================Build Model==========================
    model = DIN(feature_columns, behavior_list, att_hidden_units, ffn_hidden_units, att_activation,
                ffn_activation, maxlen, dnn_dropout)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/din_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)],  # checkpoint
        validation_data=(val_X, val_y),
        batch_size=batch_size,
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])


if __name__ == '__main__':
    main()
    