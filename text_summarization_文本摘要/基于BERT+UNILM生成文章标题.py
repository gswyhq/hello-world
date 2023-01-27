#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# UNILM直接将Seq2Seq当成句子补全来做。假如输入是“你想吃啥”，目标句子是“白切鸡”，那UNILM将这两个句子拼成一个：[CLS] 你 想 吃 啥 [SEP] 白 切 鸡 [SEP]。
# 经过这样转化之后，最简单的方案就是训练一个语言模型，然后输入“[CLS] 你 想 吃 啥 [SEP]”来逐字预测“白 切 鸡”，直到出现“[SEP]”为止
# 来源： https://kexue.fm/archives/6933
# 来源：https://github.com/bojone/bert4keras/tree/master/examples

import os
os.environ['TF_KERAS'] = '1'
USERNAME = os.getenv("USERNAME")
import glob
import numpy as np
from bert4keras.backend import keras, K
from bert4keras.layers import Loss
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer, load_vocab
from bert4keras.optimizers import Adam
from bert4keras.snippets import sequence_padding, open
from bert4keras.snippets import DataGenerator, AutoRegressiveDecoder
from keras.models import Model

# 基本参数
maxlen = 256
batch_size = 16
steps_per_epoch = 1000
epochs = 10000

# # bert配置
# config_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_config.json'
# checkpoint_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/bert_model.ckpt'
# dict_path = '/root/kg/bert/chinese_wwm_L-12_H-768_A-12/vocab.txt'

# Robert配置
# https://github.com/ZhuiyiTechnology/pretrained-models
# https://open.zhuiyi.ai/releases/nlp/models/zhuiyi/chinese_roberta_L-4_H-312_A-12.zip
config_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_config.json'
checkpoint_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/bert_model.ckpt'
dict_path = rf'D:/Users/{USERNAME}/data/chinese_roberta_L-4_H-312_A-12/vocab.txt'


# 训练样本。THUCNews数据集，每个样本保存为一个txt。
# 数据集转自：http://thuctc.thunlp.org/
# THUCNews是根据新浪新闻RSS订阅频道2005~2011年间的历史数据筛选过滤生成，包含74万篇新闻文档（2.19 GB），均为UTF-8纯文本格式。
# 在原始新浪新闻分类体系的基础上，重新整合划分出14个候选分类类别：财经、彩票、房产、股票、家居、教育、科技、社会、时尚、时政、体育、星座、游戏、娱乐。
# 处理流程；
# 1、分类提取，已“财经”文件夹为例：首先读取这个文件夹中的所有txt文档，取每个文档第一行，然后写入一个新的txt文档，写入格式为“财经+Tab符+第一行标题”
# 2、14个类别中分别挑出10%的数据作为验证集，剩下的10%作为训练集数据；
# 3、14个类别提取完毕之后，训练集合并为一个文档，验证集合并为一个文档，待用
# 4、详细代码请见我的github：https://github.com/happyAmanda/cnewsExtractTitle
# 5、剔除无效数据：比如有些文档是空的
# 以下是下载地址：
# 链接：https://pan.baidu.com/s/1zR5ymSBZ5wF0KJVpYVSb5g
# 提取码：5e76
txts = glob.glob('/root/thuctc/THUCNews/*/*.txt')

# 加载并精简词表，建立分词器
token_dict, keep_tokens = load_vocab(
    dict_path=dict_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]'],
)
tokenizer = Tokenizer(token_dict, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids = [], []
        for is_end, txt in self.sample(random):
            text = open(txt, encoding='utf-8').read()
            text = text.split('\n')
            if len(text) > 1:
                title = text[0]
                content = '\n'.join(text[1:])
                token_ids, segment_ids = tokenizer.encode(
                    content, title, maxlen=maxlen
                )
                batch_token_ids.append(token_ids)
                batch_segment_ids.append(segment_ids)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                yield [batch_token_ids, batch_segment_ids], None
                batch_token_ids, batch_segment_ids = [], []


class CrossEntropy(Loss):
    """交叉熵作为loss，并mask掉输入部分
    """
    def compute_loss(self, inputs, mask=None):
        y_true, y_mask, y_pred = inputs
        y_true = y_true[:, 1:]  # 目标token_ids
        y_mask = y_mask[:, 1:]  # segment_ids，刚好指示了要预测的部分
        y_pred = y_pred[:, :-1]  # 预测序列，错开一位
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * y_mask) / K.sum(y_mask)
        return loss


model = build_transformer_model(
    config_path,
    checkpoint_path,
    application='unilm',
    keep_tokens=keep_tokens,  # 只保留keep_tokens中的字，精简原字表
)

output = CrossEntropy(2)(model.inputs + model.outputs)

model = Model(model.inputs, output)
model.compile(optimizer=Adam(1e-5))
model.summary()


class AutoTitle(AutoRegressiveDecoder):
    """seq2seq解码器
    """
    @AutoRegressiveDecoder.wraps(default_rtype='probas')
    def predict(self, inputs, output_ids, states):
        token_ids, segment_ids = inputs
        token_ids = np.concatenate([token_ids, output_ids], 1)
        segment_ids = np.concatenate([segment_ids, np.ones_like(output_ids)], 1)
        return self.last_token(model).predict([token_ids, segment_ids])

    def generate(self, text, topk=1):
        max_c_len = maxlen - self.maxlen
        token_ids, segment_ids = tokenizer.encode(text, maxlen=max_c_len)
        output_ids = self.beam_search([token_ids, segment_ids],
                                      topk=topk)  # 基于beam search
        return tokenizer.decode(output_ids)


autotitle = AutoTitle(start_id=None, end_id=tokenizer._token_end_id, maxlen=32)


def just_show():
    s1 = u'夏天来临，皮肤在强烈紫外线的照射下，晒伤不可避免，因此，晒后及时修复显得尤为重要，否则可能会造成长期伤害。专家表示，选择晒后护肤品要慎重，芦荟凝胶是最安全，有效的一种选择，晒伤严重者，还请及 时 就医 。'
    s2 = u'8月28日，网络爆料称，华住集团旗下连锁酒店用户数据疑似发生泄露。从卖家发布的内容看，数据包含华住旗下汉庭、禧玥、桔子、宜必思等10余个品牌酒店的住客信息。泄露的信息包括华住官网注册资料、酒店入住登记的身份信息及酒店开房记录，住客姓名、手机号、邮箱、身份证号、登录账号密码等。卖家对这个约5亿条数据打包出售。第三方安全平台威胁猎人对信息出售者提供的三万条数据进行验证，认为数据真实性非常高。当天下午 ，华 住集 团发声明称，已在内部迅速开展核查，并第一时间报警。当晚，上海警方消息称，接到华住集团报案，警方已经介入调查。'
    for s in [s1, s2]:
        print(u'生成标题:', autotitle.generate(s))
    print()


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        # 保存最优
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save_weights('./best_model.weights')
        # 演示效果
        just_show()


def main():

    evaluator = Evaluator()
    train_generator = data_generator(txts, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        callbacks=[evaluator]
    )


if __name__ == '__main__':
    main()
