#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import print_function
import jieba
import pickle
import numpy as np
import os
import struct
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
# from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

texts = [] # list of text samples
labels_index = {} # dictionary mapping label name to numeric id
labels = [] # list of label ids

# 数据集来源：
# https://github.com/BenDerPan/toutiao-text-classfication-dataset.git
USERNAME = os.getenv('USERNAME')
TEXT_DATA_DIR=rf'D:\Users\{USERNAME}\github_project\toutiao-text-classfication-dataset\toutiao_cat_data.txt'
TEXT_DATA_DIR = '../data/toutiao_cat_data.txt'

if os.path.isfile('texts_labels.pkl'):
    with open('texts_labels.pkl', 'rb')as f:
        pkl_data = pickle.load(f)
        # {"texts": texts, "labels": labels, "labels_index": labels_index}
        texts = pkl_data['texts']
        labels = pkl_data['labels']
        labels_index = pkl_data['labels_index']
else:
    code_label_names = [['100', '民生', '故事', 'news_story'],
                     ['101', '文化', '文化', 'news_culture'],
                     ['102', '娱乐', '娱乐', 'news_entertainment'],
                     ['103', '体育', '体育', 'news_sports'],
                     ['104', '财经', '财经', 'news_finance'],
                     ['106', '房产', '房产', 'news_house'],
                     ['107', '汽车', '汽车', 'news_car'],
                     ['108', '教育', '教育', 'news_edu'],
                     ['109', '科技', '科技', 'news_tech'],
                     ['110', '军事', '军事', 'news_military'],
                     ['112', '旅游', '旅游', 'news_travel'],
                     ['113', '国际', '国际', 'news_world'],
                     ['114', '证券', '股票', 'stock'],
                     ['115', '农业', '三农', 'news_agriculture'],
                     ['116', '电竞', '游戏', 'news_game']]

    code_name_dict = {code:name for code, name, _, _ in code_label_names}

    with open(TEXT_DATA_DIR, encoding='utf-8')as f:
        train_datas = f.readlines()
        random.shuffle(train_datas)
        for line in train_datas:
            text_split = line.split('_!_')
            if len(text_split) != 5:
                continue
            else:
                code = text_split[1]
                text = text_split[3]
                if len(text) < 2:
                    continue
                name = code_name_dict[code]
                labels_index.setdefault(name, len(labels_index))
                label_id = labels_index[name]
                texts.append(' '.join(jieba.lcut(text)))
                labels.append(label_id)

    with open('texts_labels.pkl', 'wb')as f:
        pickle.dump({"texts": texts, "labels": labels, "labels_index": labels_index}, f)

print('Found %s texts.' % len(texts))
print('labels length %s .' % len(labels))

MAX_NB_WORDS=20000
EMBEDDING_DIM=100
HIDDEN_DIM=100
MAX_SEQUENCE_LENGTH=128
epochs = 5
batch_size = 32

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = pad_sequences(tokenizer.texts_to_sequences(texts), MAX_SEQUENCE_LENGTH)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
# print('word_index:',word_index)
data=[]

for i in range(len(sequences)):
    data.append(np.asarray(sequences[i]))

labels =np.asarray(labels)
print("data len:",len(data))
#print("data sample:",data[0])

print('Shape of data tensor:', len(data))
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
VALIDATION_SPLIT=0.2
nb_validation_samples = int(VALIDATION_SPLIT * len(data))

x_train = data[:-nb_validation_samples]
y_train = labels[:-nb_validation_samples]
x_test = data[-nb_validation_samples:]
y_test = labels[-nb_validation_samples:]
print("train length:",len(x_train))
print("test length:",len(x_test))

target_size= len(labels_index)
num_samples=len(x_train)

'''
build torch model
'''
class LSTMNet(nn.Module):
    def __init__(self):
        super(LSTMNet, self).__init__()
        self.hidden_dim = HIDDEN_DIM
        self.embedding_dim = EMBEDDING_DIM
        self.word_embeddings = nn.Embedding(MAX_NB_WORDS, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim,2,batch_first =True)
        self.hidden2tag = nn.Linear(self.hidden_dim, target_size)
        self.hidden = self.init_hidden()
        self.drop = nn.Dropout(p=0.2)

    def init_hidden(self):
        return (Variable(torch.zeros(2, batch_size, self.hidden_dim)),
                Variable(torch.zeros(2, batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        embeds = self.drop(embeds)
        #lstm_out, self.hidden = self.lstm(embeds,self.hidden)
        lstm_out= self.lstm(embeds)
        out = lstm_out[0][:,-1,:]
        flat = out.view(-1, HIDDEN_DIM)
        tag_space = self.hidden2tag(flat)
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

model = LSTMNet()
#if os.path.exists('torch_lstm.pkl'):
# model = torch.load('torch_lstm.pkl')
print(model)

'''
trainning
'''
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
#loss=torch.nn.CrossEntropyLoss(size_average=True)

def train(epoch,x_train,y_train):
    num_batchs = num_samples// batch_size
    model.train()
    model.hidden = model.init_hidden()
    for k in range(num_batchs):
        start,end = k*batch_size,(k+1)*batch_size
        data=Variable( torch.Tensor(x_train[start:end]).long())
        target = Variable(torch.Tensor(y_train[start:end]).long(),requires_grad=False)
        #embeds = word_embeddings( Variable(t)) #,requires_grad=False)) #,requires_grad=#data, target = Variable(x_train[start:end],requires_grad=False), Variable(y_#data, target = Variable(x_train[start:end]), Variable(y_train[start:end])
        optimizer.zero_grad()
        #print("train data size:",data.size())
        output = model(data)
        #print("output :",output.size())
        #print("target :",target.size())
        loss = F.nll_loss(output,target) #criterion(output,target)
        loss.backward()
        optimizer.step()
        if k % 10 == 0:
            print('Train Epoch: {} {:.4f}%\tLoss: {:.6f}'.format(
                epoch, 100*k/num_batchs, loss))
    torch.save(model, 'torch_lstm.pkl')

'''
evaludate
'''
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    size=400
    batch_size = 20
    print("x_test size:",len(x_test))
    acc_num = 0
    for i in range(size//batch_size):
        data, target = Variable(torch.Tensor(x_test[i*batch_size:(i+1)*batch_size]).long()), \
                       Variable(torch.Tensor(y_test[i*batch_size:(i+1)*batch_size]).long())
        output = model(data)
        acc_num += len([k for k, v in zip(target, [t.argmax() for t in output]) if k==v])
        test_loss += F.nll_loss(output, target)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        test_loss = test_loss
        test_loss /= len(x_test) # loss function already averages over batch size
        if i % 10 == 0:
            print("single loss:",test_loss,",right counts:",correct)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(x_test),
        100. * correct / size))
    print('正确率：', acc_num/size)  # 0.6875

def main():
    for epoch in range(1, epochs):
        train(epoch, x_train, y_train)
        # test(epoch)


if __name__ == '__main__':
    main()