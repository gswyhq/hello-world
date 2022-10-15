#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# 来源： https://blog.csdn.net/zhangdongren/article/details/117354437

# 导入库
import os
import torch
import torch.nn as nn
# 处理数据
from torchtext.legacy import data
import torch.optim as optim
import numpy as np
# 中文分词处理工具
import jieba
import re
from tqdm import tqdm
import torch.nn.functional as F

BATCH_SIZE = 32

# 产生同样的结果
SEED = 2019
torch.manual_seed(SEED)
# Cuda 算法
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def x_tokenize(x):
    str1 = re.sub('[^\u4e00-\u9fa5]', "", x)
    # return jieba.lcut(str1)
    return list(str1)

# fix_length：将序列填充至指定长度
TEXT = data.Field(sequential=True, tokenize=x_tokenize, fix_length=32, include_lengths=True, use_vocab=True, batch_first=True)

LABEL = data.Field(sequential=False, use_vocab=False, is_target=True)

# 使用torchtext.data.Tabulardataset.splits读取文件
# train, dev, test = data.TabularDataset.splits(path='dataset', train='csv_train.csv', validation='csv_dev.csv',
#                                               test='csv_test.csv', format='csv', skip_header=True,
#                                               csv_reader_params={'delimiter': ','},
#                                               fields=[('Sentence', TEXT), ('Label', LABEL)])

# 数据来源：
# 情感分类 https://github.com/bojone/bert4keras/blob/master/examples/datasets/sentiment.zip

data_path = r"D:\Users\{}\data/sentiment".format(os.getenv("USERNAME"))
train, dev, test = data.TabularDataset.splits(path=data_path, train='sentiment.train.data', validation='sentiment.valid.data',
                                              test='sentiment.test.data', format='csv', skip_header=False,
                                              csv_reader_params={'delimiter': '\t'},
                                              fields=[('Sentence', TEXT), ('Label', LABEL)])
# 查看数据
print(next(train.Sentence), next(train.Label))

# 构建词表，即需要给每个单词编码，也就是用数字表示每个单词，这样才能传入模型。
TEXT.build_vocab(train)

# 构建迭代器BucketIterator
# sort_within_batch设为True的话，一个batch内的数据就会按sort_key的排列规则降序排列，sort_key是排列的规则，这里未排序，也可以使用text的长度, sort_key = lambda x:len(x.text),。
train_iter, dev_iter, test_iter = data.BucketIterator.splits((train, dev, test), batch_size=BATCH_SIZE, shuffle=True,
                                                             sort=False, sort_within_batch=False, repeat=False)
# train_iter = data.BucketIterator(train, batch_size=BATCH_SIZE, shuffle=True, train=True)
# dev_iter = data.BucketIterator(dev, batch_size=BATCH_SIZE, shuffle=True, train=True)
# test_iter = data.BucketIterator(test, batch_size=BATCH_SIZE, shuffle=False, train=False)

for batch in train_iter:
    text, pos = batch.Sentence
    label = batch.Label
    print("text.shape:", text.shape)
    print("pos.shape:", pos.shape)
    print("label.shape:", label.shape)
    print("第一句话前10个字", text[0][:10])
    break

# ****************************************************
class TextRNN(nn.Module):
    # 定义所有层
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout, num_classes=2):
        super().__init__()
        # embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # lstm 层
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.num_classes = num_classes

    def forward(self, text):
        # text = [batch size, sent_length]
        embedded = self.embedding(text)
        # embedded = [batch size, sent_len, emb dim]
        out, (hidden, cell) = self.lstm(embedded)
        # 句子最后时刻的 hidden state
        out = self.fc(out[:, -1, :])
        # out.view(out.size(0), self.num_classes)
        return out

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional,
                            dropout=dropout, batch_first=True)
        # batch_first:输入和输出的第一个维度总是批处理大小;
        # batch_first:如果为真，则输入和输出张量以(batch, seq, feature)的形式提供。

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        tag_space = self.hidden2tag(lstm_out[:, -1, :])
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

# ****************************************************
# 定义超参数
size_of_vocab = len(TEXT.vocab)
embedding_dim = 300
num_hidden = 128
num_layers = 2
num_output = 2
dropout = 0.2

# 实例化模型
# model = TextRNN(size_of_vocab, embedding_dim, num_hidden, num_output, num_layers, bidirectional=True, dropout=dropout)
model = LSTMModel(size_of_vocab, embedding_dim, num_hidden, num_output, num_layers, bidirectional = True, dropout = dropout)
# model.load_state_dict(torch.load('dataset/myjd.pt'))

# ****************************************************
# 定义优化器和损失
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 如果cuda可用
model = model.to(device)
criterion = criterion.to(device)

# ****************************************************
# 训练模型
for epoch in range(2):
    batch_loss = []
    for batch in tqdm(train_iter):
        # 在每一个batch后设置0梯度
        optimizer.zero_grad()

        text = batch.Sentence[0].to(device)
        label = batch.Label.to(device)

        # 转换成一维张量
        predictions = model(text).squeeze()

        # 计算损失
        loss = criterion(predictions, label)
        batch_loss.append(loss.item())

        # 反向传播损耗并计算梯度
        loss.backward()

        # 更新权重
        optimizer.step()

    avg_loss = np.array(batch_loss).mean()
    print("==epoch===>%s===avg_loss===>%s" % (epoch, avg_loss))

torch.save(model.state_dict(), 'dataset/myjd.pt')


def main():
    # 测试代码
    # rnn = TextRNN(size_of_vocab, embedding_dim, num_hidden, num_output, num_layers, bidirectional=True, dropout=dropout)
    model = LSTMModel(size_of_vocab, embedding_dim, num_hidden, num_output, num_layers, bidirectional = True, dropout = dropout)
    model.load_state_dict(torch.load('dataset/myjd.pt'))


    def predict():
        sent_list =[ '不推荐买', '蒙牛真果粒、美丽有新意。', '书本质量不错，但是感觉布局不是很合理，打开后感觉很乱，密密麻麻的，孩子也不喜欢它。']
        demo = [data.Example.fromlist(data=[sent1, 0], fields=[('Sentence', TEXT), ('Label', LABEL)]) for sent1 in sent_list]
        demo_iter = data.BucketIterator(dataset=data.Dataset(demo, fields=[('Sentence', TEXT), ('Label', LABEL)]),
                                        batch_size=BATCH_SIZE, shuffle=True, sort_key=lambda x: len(x.text),
                                        sort_within_batch=False, repeat=False)
        for batch in demo_iter:
            text = batch.Sentence[0]
            # text = torch.t(text)
            out = model(text)
            for sent, label in zip(sent_list, out.argmax(dim=1)):
                # if torch.argmax(out, dim=1).item() == 0:
                #     print('差评')
                # elif torch.argmax(out, dim=1).item() == 1:
                #     print('好评')
                if label == 0:
                    print("{}\t{}".format(sent, '差评'))
                else:
                    print("{}\t{}".format(sent, '好评'))

    predict()

if __name__ == '__main__':
    main()