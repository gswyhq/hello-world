#! -*- coding:utf-8 -*-
# 情感分析例子，利用MLM做 Zero-Shot/Few-Shot/Semi-Supervised Learning

# 给任务一个文本描述，然后转换为完形填空问题即可。举个例子，假如给定句子“这趟北京之旅我感觉很不错。”，那么我们补充个描述，构建如下的完形填空：
# ______满意。这趟北京之旅我感觉很不错。
# 进一步地，我们限制空位处只能填一个“很”或“不”，问题就很清晰了，就是要我们根据上下文一致性判断是否满意，如果“很”的概率大于“不”的概率，说明是正面情感倾向，否则就是负面的，这样我们就将情感分类问题转换为一个完形填空问题了，它可以用 MLM 模型给出预测结果，而 MLM 模型的训练可以不需要监督数据，因此理论上这能够实现零样本学习了。
#
# 多分类问题也可以做类似转换，比如新闻主题分类，输入句子为“八个月了，终于又能在赛场上看到女排姑娘们了。”，那么就可以构建：
# 下面播报一则______新闻。八个月了，终于又能在赛场上看到女排姑娘们了。
# 这样我们就将新闻主题分类也转换为完形填空问题了，一个好的 MLM 模型应当能预测出“体育”二字来。
# 来源：https://mp.weixin.qq.com/s?__biz=MzIwMTc4ODE0Mw==&mid=2247512167&idx=1&sn=cc7695d92362e3b18a6e8969fb14dc27&chksm=96ea6fe7a19de6f1be86b965e268df1b9c6320810cf32b6d64ddd3d238bf9088be41fb36adfe#rd

import os
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open
from keras.layers import Lambda, Dense

USERNAME = os.getenv('USERNAME')
MODEL_PATH = rf'D:\Users\{USERNAME}\data\chinese_roberta_wwm_ext_L-12_H-768_A-12'
DATA_PATH = rf'D:\Users\{USERNAME}\data\sentiment'
num_classes = 2
maxlen = 128
batch_size = 32
# config_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_roberta_wwm_ext_L-12_H-768_A-12/vocab.txt'
config_path = os.path.join(MODEL_PATH, 'bert_config.json')
checkpoint_path = os.path.join(MODEL_PATH, 'bert_model.ckpt')
dict_path = os.path.join(MODEL_PATH, 'vocab.txt')

def load_data(filename):
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            text, label = l.strip().split('\t')
            D.append((text, int(label)))
    return D


# 加载数据集
train_data = load_data(os.path.join(DATA_PATH, 'sentiment.train.data'))
valid_data = load_data(os.path.join(DATA_PATH, 'sentiment.valid.data'))
test_data = load_data(os.path.join(DATA_PATH, 'sentiment.test.data'))

# 模拟标注和非标注数据
train_frac = 0.01  # 标注数据的比例
num_labeled = int(len(train_data) * train_frac)
unlabeled_data = [(t, 2) for t, l in train_data[num_labeled:]]
train_data = train_data[:num_labeled]
train_data = train_data + unlabeled_data

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# 对应的任务描述
prefix = u'很满意。'
mask_idx = 1
pos_id = tokenizer.token_to_id(u'很')
neg_id = tokenizer.token_to_id(u'不')


def random_masking(token_ids):
    """对输入进行随机mask
    """
    rands = np.random.random(len(token_ids))
    source, target = [], []
    for r, t in zip(rands, token_ids):
        if r < 0.15 * 0.8:
            source.append(tokenizer._token_mask_id)
            target.append(t)
        elif r < 0.15 * 0.9:
            source.append(t)
            target.append(t)
        elif r < 0.15:
            source.append(np.random.choice(tokenizer._vocab_size - 1) + 1)
            target.append(t)
        else:
            source.append(t)
            target.append(0)
    return source, target


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []
        for is_end, (text, label) in self.sample(random):
            if label != 2:
                text = prefix + text
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            if random:
                source_ids, target_ids = random_masking(token_ids)
            else:
                source_ids, target_ids = token_ids[:], token_ids[:]
            if label == 0:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = neg_id
            elif label == 1:
                source_ids[mask_idx] = tokenizer._token_mask_id
                target_ids[mask_idx] = pos_id
            batch_token_ids.append(source_ids)
            batch_segment_ids.append(segment_ids)
            batch_output_ids.append(target_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_output_ids = sequence_padding(batch_output_ids)
                yield [
                    batch_token_ids, batch_segment_ids, batch_output_ids
                ], None
                batch_token_ids, batch_segment_ids, batch_output_ids = [], [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
        accuracy = keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
        accuracy = K.sum(accuracy * y_mask) / K.sum(y_mask)
        self.add_metric(accuracy, name='accuracy')
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


# 加载预训练模型
model = build_transformer_model(
    config_path=config_path, checkpoint_path=checkpoint_path, with_mlm=True
)

# 训练用模型
y_in = keras.layers.Input(shape=(None,))
outputs = CrossEntropy(1)([y_in, model.output])

train_model = keras.models.Model(model.inputs + [y_in], outputs)
train_model.compile(optimizer=Adam(1e-5))
train_model.summary()

# 转换数据集
train_generator = data_generator(train_data, batch_size)
valid_generator = data_generator(valid_data, batch_size)
test_generator = data_generator(test_data, batch_size)


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        model.save_weights('mlm_model.weights')
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


def evaluate(data):
    total, right = 0., 0.
    for x_true, _ in data:
        x_true, y_true = x_true[:2], x_true[2]
        y_pred = model.predict(x_true)
        y_pred = y_pred[:, mask_idx, [neg_id, pos_id]].argmax(axis=1)
        y_true = (y_true[:, mask_idx] == pos_id).astype(int)
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


if __name__ == '__main__':

    evaluator = Evaluator()

    train_model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=1000,
        callbacks=[evaluator]
    )

else:

    model.load_weights('best_model.weights')


# 数据来源：https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip
# 代码来源：https://github.com/bojone/Pattern-Exploiting-Training/blob/master/sentiment.py


