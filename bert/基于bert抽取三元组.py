#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 三元组抽取任务，基于“半指针-半标注”结构
# 文章介绍：https://kexue.fm/archives/7161
# 来源： https://github.com/bojone/bert4keras/tree/master/examples
# 数据集：http://ai.baidu.com/broad/download?dataset=sked
# 最优f1=0.82198
# 换用RoBERTa Large可以达到f1=0.829+
# 说明：由于使用了EMA，需要跑足够多的步数(5000步以上）才生效，如果
#      你的数据总量比较少，那么请务必跑足够多的epoch数，或者去掉EMA。

import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import json
import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.layers import Loss
from bert4keras.layers import LayerNormalization
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_exponential_moving_average
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from keras.layers import Input, Dense, Lambda, Reshape
from keras.models import Model
from tqdm import tqdm
import tensorflow as tf

maxlen = 128
batch_size = 64
# config_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = rf'D:/Users/{USERNAME}/data/chinese_L-12_H-768_A-12/vocab.txt'

# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：{'text': text, 'spo_list': [(s, p, o)]}
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        for l in f:
            l = json.loads(l)
            D.append({
                'text': l['text'],
                'spo_list': [(spo['subject'], spo['predicate'], spo['object'])
                             for spo in l['spo_list']]
            })
    return D


# 加载数据集
train_data = load_data(rf'D:/Users/{USERNAME}/data/KnowledgeExtraction/train_data.json')
valid_data = load_data(rf'D:/Users/{USERNAME}/data/KnowledgeExtraction/dev_data.json')
predicate2id, id2predicate = {}, {}

with open(rf'D:/Users/{USERNAME}/data/KnowledgeExtraction/all_50_schemas', encoding='utf-8') as f:
    for l in f:
        l = json.loads(l)
        if l['predicate'] not in predicate2id:
            id2predicate[len(predicate2id)] = l['predicate']
            predicate2id[l['predicate']] = len(predicate2id)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


def search(pattern, sequence):
    """从sequence中寻找子串pattern
    如果找到，返回第一个下标；否则返回-1。
    """
    n = len(pattern)
    for i in range(len(sequence)):
        if sequence[i:i + n] == pattern:
            return i
    return -1


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []
        for is_end, d in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(d['text'], maxlen=maxlen)
            # 整理三元组 {s: [(o, p)]}
            spoes = {}
            for s, p, o in d['spo_list']:
                s = tokenizer.encode(s)[0][1:-1]
                p = predicate2id[p]
                o = tokenizer.encode(o)[0][1:-1]
                s_idx = search(s, token_ids)
                o_idx = search(o, token_ids)
                if s_idx != -1 and o_idx != -1:
                    s = (s_idx, s_idx + len(s) - 1)
                    o = (o_idx, o_idx + len(o) - 1, p)
                    if s not in spoes:
                        spoes[s] = []
                    spoes[s].append(o)
            if spoes:
                # subject标签
                subject_labels = np.zeros((len(token_ids), 2))
                for s in spoes:
                    subject_labels[s[0], 0] = 1
                    subject_labels[s[1], 1] = 1
                # 随机选一个subject（这里没有实现错误！这就是想要的效果！！）
                # 比如有两个subject 坐标 (10, 24) (3, 5), start 随机选了3， end 随机选了24，得到的 subject 就是(3, 24) ,即为负样本；
                start, end = np.array(list(spoes.keys())).T
                start = np.random.choice(start)
                end = np.random.choice(end[end >= start])
                subject_ids = (start, end)
                # 对应的object标签
                object_labels = np.zeros((len(token_ids), len(predicate2id), 2))
                for o in spoes.get(subject_ids, []):
                    object_labels[o[0], o[2], 0] = 1
                    object_labels[o[1], o[2], 1] = 1
                # 构建batch
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
                batch_subject_labels.append(subject_labels)
                batch_subject_ids.append(subject_ids)
                batch_object_labels.append(object_labels)
                if len(batch_token_ids) == self.batch_size or is_end:
                    batch_token_ids = sequence_padding(batch_token_ids)
                    batch_segment_ids = sequence_padding(batch_segment_ids)
                    batch_subject_labels = sequence_padding(
                        batch_subject_labels
                    )
                    batch_subject_ids = np.array(batch_subject_ids)
                    batch_object_labels = sequence_padding(batch_object_labels)
                    yield [
                        batch_token_ids, batch_segment_ids,
                        batch_subject_labels, batch_subject_ids,
                        batch_object_labels
                    ], None
                    batch_token_ids, batch_segment_ids = [], []
                    batch_subject_labels, batch_subject_ids, batch_object_labels = [], [], []


def extract_subject(inputs):
    """根据subject_ids从output中取出subject的向量表征
    """
    output, subject_ids = inputs
    start = batch_gather(output, subject_ids[:, :1])
    end = batch_gather(output, subject_ids[:, 1:])
    subject = K.concatenate([start, end], 2)
    return subject[:, 0]

'''
模型是基于“半指针-半标注”的方式来做抽取，顺序是先抽取s，然后传入s来抽取o、p：
1、原始序列转id后，传入bert的编码器，得到编码序列；
2、编码序列接两个二分类器，预测s；
3、根据传入的s，从编码序列中抽取出s的首和尾对应的编码向量；
4、以s的编码向量作为条件，对编码序列做一次条件Layer Norm；
5、条件Layer Norm后的序列来预测该s对应的o、p。

我们可以先预测s，然后传入s来预测该s对应的o，然后传入s、o来预测所传入的s、o的关系p，实际应用中，我们还可以把o、p的预测合并为一步，
所以总的步骤只需要两步：先预测s，然后传入s来预测该s所对应的o及p。

理论上，上述模型只能抽取单一一个三元组，而为了处理可能由多个s、多个o甚至多个p的情况，将softmx换成sigmoid，并且在关系分类的时候也使用sigmoid而不是softmax激活。

注1：为什么不先预测o然后再预测s及对应的p？
那是因为引入在第二步预测的时候要采样传入第一步的结果（而且只采样一个），而前面已经分析了，多数样本的o的数目比s的数目要多，
所以我们先预测s，然后传入s再预测o、p的时候，对s的采样就很容易充分了（因为s少），反过来如果要对o进行采样就不那么容易充分（因为o可能很多）。

'''


# 补充输入
subject_labels = Input(shape=(None, 2), name='Subject-Labels')
subject_ids = Input(shape=(2,), name='Subject-Ids')
object_labels = Input(shape=(None, len(predicate2id), 2), name='Object-Labels')

# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    return_keras_model=False,
)

# 预测subject
# 用两个sigmoid激活的标注方式来分别标注实体的首和尾
output = Dense(
    units=2, activation='sigmoid', kernel_initializer=bert.initializer
)(bert.model.output)
subject_preds = Lambda(lambda x: x**2)(output)  # 输出为概率的平方，后面筛选阈值若为0.6,那么这里的得需为0.7746, 0.7746*0.7746=0.6

subject_model = Model(bert.model.inputs, subject_preds)

# 对 subject model 进行微调；
subject_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

def generator_subject_model_data(train_data, batch_size):
    while True:
        train_generator = data_generator(train_data, batch_size)
        for (batch_token_ids, batch_segment_ids,
                            batch_subject_labels, batch_subject_ids,
                            batch_object_labels), _ in train_generator:
            yield (batch_token_ids, batch_segment_ids), batch_subject_labels

# 微调时候，先不训练预训练模型原有参数；
for layer in subject_model.layers[:-2]:
    layer.trainable = False

subject_model.fit(
    generator_subject_model_data(train_data, batch_size),
    steps_per_epoch=len(train_data)//batch_size,
    epochs=10,
)

# 传入subject，预测object
# 通过Conditional Layer Normalization将subject融入到object的预测中
output = bert.model.layers[-2].get_output_at(-1)  # 这里-2是为了取Transformer-11-FeedForward-Add做Conditional Layer Normalization吗，做Conditional Layer Normalization就不需要Transformer-11-FeedForward-Norm了
subject = Lambda(extract_subject)([output, subject_ids])
output = LayerNormalization(conditional=True)([output, subject])
output = Dense(
    units=len(predicate2id) * 2,
    activation='sigmoid',
    kernel_initializer=bert.initializer
)(output)
output = Lambda(lambda x: x**4)(output)  # 输出为概率的4次方
object_preds = Reshape((-1, len(predicate2id), 2))(output)

object_model = Model(bert.model.inputs + [subject_ids], object_preds)

# 对object model 进行微调
object_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

def generator_object_model_data(train_data, batch_size):
    while True:
        train_generator = data_generator(train_data, batch_size)
        for (batch_token_ids, batch_segment_ids,
                            batch_subject_labels, batch_subject_ids,
                            batch_object_labels), _ in train_generator:
            yield (batch_token_ids, batch_segment_ids, batch_subject_ids), batch_object_labels

# 微调时候，先不训练预训练模型原有参数；
for layer in object_model.layers[:-3]:
    layer.trainable = False

object_model.fit(
    generator_object_model_data(train_data, batch_size),
    steps_per_epoch=len(train_data)//batch_size,
    epochs=10,
)

'''
用“半指针-半标注”结构做实体抽取时，会面临类别不均衡的问题，因为通常来说目标实体词比非目标词要少得多，所以标签1会比标签0少得多。
这里将概率值做n次方。
具体来说，原来输出一个概率值p，代表类别1的概率是p，我现在将它变为p^n，也就是认为类别1的概率是p^n，除此之外不变，loss还是用正常的二分类交叉熵loss。
由于原来就有0≤p≤1，所以p^n整体会更接近于0，因此初始状态就符合目标分布了，所以最终能加速收敛。

'''
class TotalLoss(Loss):
    """subject_loss与object_loss之和，都是二分类交叉熵
    """
    def compute_loss(self, inputs, mask=None):
        subject_labels, object_labels = inputs[:2]
        subject_preds, object_preds, _ = inputs[2:]
        if mask[4] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[4], K.floatx())
        # subject部分loss
        subject_loss = K.binary_crossentropy(subject_labels, subject_preds)
        subject_loss = K.mean(subject_loss, 2)
        subject_loss = K.sum(subject_loss * mask) / K.sum(mask)
        # object部分loss
        object_loss = K.binary_crossentropy(object_labels, object_preds)
        object_loss = K.sum(K.mean(object_loss, 3), 2)
        object_loss = K.sum(object_loss * mask) / K.sum(mask)
        # 总的loss
        return subject_loss + object_loss


subject_preds, object_preds = TotalLoss([2, 3])([
    subject_labels, object_labels, subject_preds, object_preds,
    bert.model.output
])

# 训练模型
train_model = Model(
    bert.model.inputs + [subject_labels, subject_ids, object_labels],
    [subject_preds, object_preds]
)

AdamEMA = extend_with_exponential_moving_average(Adam, name='AdamEMA')
optimizer = AdamEMA(learning_rate=1e-5)
train_model.compile(optimizer=optimizer)


def extract_spoes(text):
    """抽取输入text所包含的三元组
    """
    tokens = tokenizer.tokenize(text, maxlen=maxlen)
    mapping = tokenizer.rematch(text, tokens)
    token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    # 抽取subject
    subject_preds = subject_model.predict([token_ids, segment_ids])
    subject_preds[:, [0, -1]] *= 0  # 首尾[CLS], [SEP]位置的概率置为0
    # 用两个sigmoid激活的标注方式来分别标注实体的首和尾，在模型中，发现将“首”的阈值设为0.6，将“尾”的阈值设为0.5，即超过阈值就认为此处出现了实体。
    start = np.where(subject_preds[0, :, 0] > 0.6)[0]
    end = np.where(subject_preds[0, :, 1] > 0.5)[0]
    subjects = []
    for i in start:
        j = end[end >= i]
        if len(j) > 0:
            j = j[0]
            subjects.append((i, j))
    if subjects:
        spoes = []
        token_ids = np.repeat(token_ids, len(subjects), 0)
        segment_ids = np.repeat(segment_ids, len(subjects), 0)
        subjects = np.array(subjects)
        # 传入subject，抽取object和predicate
        object_preds = object_model.predict([token_ids, segment_ids, subjects])
        object_preds[:, [0, -1]] *= 0
        for subject, object_pred in zip(subjects, object_preds):
            start = np.where(object_pred[:, :, 0] > 0.6)
            end = np.where(object_pred[:, :, 1] > 0.5)
            for _start, predicate1 in zip(*start):
                for _end, predicate2 in zip(*end):
                    if _start <= _end and predicate1 == predicate2:
                        spoes.append(
                            ((mapping[subject[0]][0],
                              mapping[subject[1]][-1]), predicate1,
                             (mapping[_start][0], mapping[_end][-1]))
                        )
                        break
        return [(text[s[0]:s[1] + 1], id2predicate[p], text[o[0]:o[1] + 1])
                for s, p, o, in spoes]
    else:
        return []


class SPO(tuple):
    """用来存三元组的类
    表现跟tuple基本一致，只是重写了 __hash__ 和 __eq__ 方法，
    使得在判断两个三元组是否等价时容错性更好。
    """
    def __init__(self, spo):
        self.spox = (
            tuple(tokenizer.tokenize(spo[0])),
            spo[1],
            tuple(tokenizer.tokenize(spo[2])),
        )

    def __hash__(self):
        return self.spox.__hash__()

    def __eq__(self, spo):
        return self.spox == spo.spox


def evaluate(data):
    """评估函数，计算f1、precision、recall
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    f = open('dev_pred.json', 'w', encoding='utf-8')
    pbar = tqdm()
    for d in data:
        R = set([SPO(spo) for spo in extract_spoes(d['text'])])
        T = set([SPO(spo) for spo in d['spo_list']])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
        f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
        pbar.update()
        pbar.set_description(
            'f1: %.5f, precision: %.5f, recall: %.5f' % (f1, precision, recall)
        )
        s = json.dumps({
            'text': d['text'],
            'spo_list': list(T),
            'spo_list_pred': list(R),
            'new': list(R - T),
            'lack': list(T - R),
        },
                       ensure_ascii=False,
                       indent=4)
        f.write(s + '\n')
    pbar.close()
    f.close()
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0.

    def on_epoch_end(self, epoch, logs=None):
        optimizer.apply_ema_weights()
        f1, precision, recall = evaluate(valid_data)
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            train_model.save_weights('best_model.weights')
        optimizer.reset_old_weights()
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )



def main():
    train_generator = data_generator(train_data, batch_size)
    evaluator = Evaluator()

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=20,
        callbacks=[evaluator]
    )

# train_model.load_weights('best_model.weights')

if __name__ == '__main__':
    main()
