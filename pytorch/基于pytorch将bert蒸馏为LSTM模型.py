# -*- coding: utf-8 -*-

# 代码及数据来源： https://github.com/qiangsiwei/bert_distill.git

import os
import torch
import jieba
from transformers import BertTokenizer
import torch, numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from keras.preprocessing import sequence
import os, csv, random, torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score

from keras.preprocessing.text import Tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor

USERNAME = os.getenv('USERNAME')
BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'


class InputExample(object):
    def __init__(self, guid, text, label=None):
        self.guid = guid
        self.text = text
        self.label = label


class InputFeatures(object):
    def __init__(self, input_ids, input_mask, label_id=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.label_id = label_id

class Processor(object):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'train.txt'), 'train')

    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'test.txt'), 'test')

    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dev.txt'), 'dev')

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, data_path, set_type):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                label, text = line.strip().split('\t', 1)
                guid = "{0}-{1}-{2}".format(set_type, label, i)
                examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples


def convert_examples_to_features(examples, label_list, max_seq, tokenizer):
    label_map = {label: i for i, label in enumerate(label_list)}
    features = []
    for ex_index, example in enumerate(examples):
        tokens = tokenizer.tokenize(example.text)
        tokens = ["[CLS]"] + tokens[:max_seq - 2] + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (max_seq - len(input_ids))
        label_id = label_map[example.label]
        features.append(InputFeatures(
            input_ids=input_ids + padding,
            input_mask=input_mask + padding,
            label_id=label_id))
    return features


class BertClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels=2):
        super(BertClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.init_weights()

    def forward(self, input_ids, input_mask, label_ids):
        _, pooled_output = self.bert(input_ids, None, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits


class BertTextCNN(BertPreTrainedModel):
    def __init__(self, config, hidden_size=128, num_labels=2):
        super(BertTextCNN, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.conv1 = nn.Conv2d(1, hidden_size, (3, config.hidden_size))
        self.conv2 = nn.Conv2d(1, hidden_size, (4, config.hidden_size))
        self.conv3 = nn.Conv2d(1, hidden_size, (5, config.hidden_size))
        self.classifier = nn.Linear(hidden_size * 3, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, input_mask, label_ids):
        sequence_output, _ = self.bert(input_ids, None, input_mask, output_all_encoded_layers=False)
        out = self.dropout(sequence_output).unsqueeze(1)
        c1 = torch.relu(self.conv1(out).squeeze(3))
        p1 = F.max_pool1d(c1, c1.size(2)).squeeze(2)
        c2 = torch.relu(self.conv2(out).squeeze(3))
        p2 = F.max_pool1d(c2, c2.size(2)).squeeze(2)
        c3 = torch.relu(self.conv3(out).squeeze(3))
        p3 = F.max_pool1d(c3, c3.size(2)).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        logits = self.classifier(pool)
        if label_ids is not None:
            loss_fct = CrossEntropyLoss()
            return loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
        return logits


def compute_metrics(preds, labels):
    return {'ac': (preds == labels).mean(), 'f1': f1_score(y_true=labels, y_pred=preds)}


def load_data(name):
    def get_w2v():
        for line in open('data/cache/word2vec', encoding="utf-8").read().strip().split('\n'):
            line = line.strip().split()
            if not line: continue
            yield line[0], np.array(list(map(float, line[1:])))

    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    texts = [' '.join(jieba.cut(line.split('\t', 1)[1].strip())) \
             for line in open('data/{}/{}.txt'.format(name, name), encoding="utf-8",
                              ).read().strip().split('\n')]
    tokenizer.fit_on_texts(texts)
    # with open('word2vec','w') as out:
    # 	for line in fileinput.input('sgns.sogou.word'):
    # 		word = line.strip().split()[0]
    # 		if word in tokenizer.word_index:
    # 			out.write(line+'\n')
    # 	fileinput.close()
    x_train, y_train = [], [];
    text_train = []
    for line in open('data/{}/train.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_train.append(text.strip())
        x_train.append(' '.join(jieba.cut(text.strip())))
        y_train.append(int(label))
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev, y_dev = [], []
    text_dev = []
    for line in open('data/{}/dev.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_dev.append(text.strip())
        x_dev.append(' '.join(jieba.cut(text.strip())))
        y_dev.append(int(label))
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test, y_test = [], []
    text_test = []
    for line in open('data/{}/test.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_test.append(text.strip())
        x_test.append(' '.join(jieba.cut(text.strip())))
        y_test.append(int(label))
    x_test = tokenizer.texts_to_sequences(x_test)
    v_size = len(tokenizer.word_index) + 1
    embs, w2v = np.zeros((v_size, 300)), dict(get_w2v())
    for word, index in tokenizer.word_index.items():
        if word in w2v: embs[index] = w2v[word]
    return (x_train, y_train, text_train), \
           (x_dev, y_dev, text_dev), \
           (x_test, y_test, text_test), \
           v_size, embs

def load_data_aug(name, n_iter=20, p_mask=0.1, p_ng=0.25, ngram_range=(3,6)):
    def get_w2v():
        for line in open('data/cache/word2vec', encoding="utf-8").read().strip().split('\n'):
            line = line.strip().split()
            if not line: continue
            yield line[0], np.array(list(map(float, line[1:])))

    tokenizer = Tokenizer(filters='', lower=True, split=' ', oov_token=1)
    texts = [' '.join(jieba.cut(line.split('\t', 1)[1].strip())) \
             for line in open('data/{}/{}.txt'.format(name, name), encoding="utf-8",
                              ).read().strip().split('\n')]
    tokenizer.fit_on_texts(texts)

    x_train, y_train = [], [];
    text_train = []
    for line in open('data/{}/train.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text = text.strip()
        # preserve one original sample first
        text_train.append(text)
        x_train.append(' '.join(jieba.cut(text)))
        y_train.append(int(label))
        # data augmentation
        used_texts = {text}
        for i in range(n_iter):
            words = jieba.lcut(text)
            # word masking
            words = [x if np.random.rand() < p_mask else "[MASK]" for x in words]
            # n-gram sampling
            if np.random.rand() < p_ng:
                n_gram_len = np.random.randint(ngram_range[0], ngram_range[1]+1)
                n_gram_len = min(n_gram_len, len(words))
                n_gram_start = np.random.randint(0, len(words)-n_gram_len+1)
                words = words[n_gram_start:n_gram_start+n_gram_len]
            new_text = "".join(words)
            if new_text not in used_texts:
                text_train.append(new_text)
                x_train.append(' '.join(words))
                y_train.append(int(label))
                used_texts.add(new_text)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_dev, y_dev = [], []
    text_dev = []
    for line in open('data/{}/dev.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_dev.append(text.strip())
        x_dev.append(' '.join(jieba.cut(text.strip())))
        y_dev.append(int(label))
    x_dev = tokenizer.texts_to_sequences(x_dev)
    x_test, y_test = [], []
    text_test = []
    for line in open('data/{}/test.txt'.format(name), encoding="utf-8").read().strip().split('\n'):
        label, text = line.split('\t', 1)
        text_test.append(text.strip())
        x_test.append(' '.join(jieba.cut(text.strip())))
        y_test.append(int(label))
    x_test = tokenizer.texts_to_sequences(x_test)
    v_size = len(tokenizer.word_index) + 1
    embs, w2v = np.zeros((v_size, 300)), dict(get_w2v())
    for word, index in tokenizer.word_index.items():
        if word in w2v: embs[index] = w2v[word]
    return (x_train, y_train, text_train), \
           (x_dev, y_dev, text_dev), \
           (x_test, y_test, text_test), \
           v_size, embs

class RNN(nn.Module):
    def __init__(self, x_dim, e_dim, h_dim, o_dim):
        super(RNN, self).__init__()
        self.h_dim = h_dim
        self.dropout = nn.Dropout(0.2)
        self.emb = nn.Embedding(x_dim, e_dim, padding_idx=0)
        self.lstm = nn.LSTM(e_dim, h_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(h_dim * 2, o_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):
        embed = self.dropout(self.emb(x))
        out, _ = self.lstm(embed)
        hidden = self.fc(out[:, -1, :])
        return self.softmax(hidden), self.log_softmax(hidden)


class CNN(nn.Module):
    def __init__(self, x_dim, e_dim, h_dim, o_dim):
        super(CNN, self).__init__()
        self.emb = nn.Embedding(x_dim, e_dim, padding_idx=0)
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv2d(1, h_dim, (3, e_dim))
        self.conv2 = nn.Conv2d(1, h_dim, (4, e_dim))
        self.conv3 = nn.Conv2d(1, h_dim, (5, e_dim))
        self.fc = nn.Linear(h_dim * 3, o_dim)
        self.softmax = nn.Softmax(dim=1)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, lens):
        embed = self.dropout(self.emb(x)).unsqueeze(1)
        c1 = torch.relu(self.conv1(embed).squeeze(3))
        p1 = torch.max_pool1d(c1, c1.size()[2]).squeeze(2)
        c2 = torch.relu(self.conv2(embed).squeeze(3))
        p2 = torch.max_pool1d(c2, c2.size()[2]).squeeze(2)
        c3 = torch.relu(self.conv3(embed).squeeze(3))
        p3 = torch.max_pool1d(c3, c3.size()[2]).squeeze(2)
        pool = self.dropout(torch.cat((p1, p2, p3), 1))
        hidden = self.fc(pool)
        return self.softmax(hidden), self.log_softmax(hidden)


class Model(object):
    def __init__(self, v_size):
        self.model = None
        self.b_size = 64
        self.lr = 0.001
        self.model = RNN(v_size, 256, 256, 2)

    # self.model = CNN(v_size,256,128,2)
    def train(self, x_tr, y_tr, l_tr, x_te, y_te, l_te, epochs=15):
        assert self.model is not None
        if USE_CUDA: self.model = self.model.cuda()
        loss_func = nn.NLLLoss()
        opt = optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(epochs):
            losses = []
            accu = []
            self.model.train()
            for i in range(0, len(x_tr), self.b_size):
                self.model.zero_grad()
                bx = Variable(LTensor(x_tr[i:i + self.b_size]))
                by = Variable(LTensor(y_tr[i:i + self.b_size]))
                bl = Variable(LTensor(l_tr[i:i + self.b_size]))
                _, py = self.model(bx, bl)
                loss = loss_func(py, by)
                loss.backward()
                opt.step()
                losses.append(loss.item())
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(x_te), self.b_size):
                    bx = Variable(LTensor(x_te[i:i + self.b_size]))
                    by = Variable(LTensor(y_te[i:i + self.b_size]))
                    bl = Variable(LTensor(l_te[i:i + self.b_size]))
                    _, py = torch.max(self.model(Variable(LTensor(bx)), bl)[1], 1)
                    accu.append((py == by).float().mean().item())
            print(np.mean(losses), np.mean(accu))


def test_cnn_rnn_mode():
    x_len = 50
    # ----- ----- ----- ----- -----
    # from keras.datasets import imdb
    # v_size = 10000
    # (x_tr,y_tr),(x_te,y_te) = imdb.load_data(num_words=v_size)
    # ----- ----- ----- ----- -----
    name = 'hotel'  # clothing, fruit, hotel, pda, shampoo
    (x_tr, y_tr, _), _, (x_te, y_te, _), v_size, _ = load_data(name)
    l_tr = list(map(lambda x: min(len(x), x_len), x_tr))
    l_te = list(map(lambda x: min(len(x), x_len), x_te))
    x_tr = sequence.pad_sequences(x_tr, maxlen=x_len)
    x_te = sequence.pad_sequences(x_te, maxlen=x_len)
    clf = Model(v_size)
    clf.train(x_tr, y_tr, l_tr, x_te, y_te, l_te)

class Teacher(object):
    def __init__(self, bert_model=BERT_BASE_CHINESE_PATH, max_seq=128):
        self.max_seq = max_seq
        self.tokenizer = BertTokenizer.from_pretrained(
            bert_model, do_lower_case=True)
        self.model = torch.load('./data/cache/model')
        self.model.eval()

    def predict(self, text):
        tokens = self.tokenizer.tokenize(text)[:self.max_seq]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        padding = [0] * (self.max_seq - len(input_ids))
        input_ids = torch.tensor([input_ids + padding], dtype=torch.long).to(device)
        input_mask = torch.tensor([input_mask + padding], dtype=torch.long).to(device)
        logits = self.model(input_ids, input_mask, None)
        return F.softmax(logits, dim=1).detach().cpu().numpy()


def train_teacher_model(bert_model='bert-base-chinese', cache_dir=None,
         max_seq=128, batch_size=16, num_epochs=10, lr=2e-5):
    '''
    训练教师模型；
    :param bert_model:
    :param cache_dir:
    :param max_seq:
    :param batch_size:
    :param num_epochs:
    :param lr:
    :return:
    '''
    processor = Processor()
    train_examples = processor.get_train_examples('data/hotel')
    label_list = processor.get_labels()
    tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
    model = BertClassification.from_pretrained(bert_model,
                                               cache_dir=cache_dir, num_labels=len(label_list))
    # model = BertTextCNN.from_pretrained(bert_model,\
    # 	cache_dir=cache_dir,num_labels=len(label_list))
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not \
            any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if \
                    any(nd in n for nd in no_decay)], 'weight_decay': 0.00}]
    print('train...')
    num_train_steps = int(len(train_examples) / batch_size * num_epochs)
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    train_features = convert_examples_to_features(train_examples, label_list, max_seq, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    model.train()
    for _ in trange(num_epochs, desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            loss = model(input_ids, input_mask, label_ids)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
        print('tr_loss', tr_loss)
    print('eval...')
    eval_examples = processor.get_dev_examples('data/hotel')
    eval_features = convert_examples_to_features(eval_examples, label_list, max_seq, tokenizer)
    eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
    eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
    model.eval()
    preds = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(input_ids, input_mask, None)
            preds.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.vstack(preds), axis=1)
    print(compute_metrics(preds, eval_label_ids.numpy()))
    torch.save(model, 'data/cache/model')

def train_student_model():
    # BERT 作为 Teacher 模型将知识蒸馏到 LSTM Student 模型中
    teacher = Teacher()
    print(teacher.predict('还不错！这个价位算是物有所值了！'))

    import pickle
    from tqdm import tqdm

    x_len = 50
    b_size = 64
    lr = 0.002
    epochs = 10
    name = 'hotel'  # clothing, fruit, hotel, pda, shampoo
    alpha = 0.5     # portion of the original one-hot CE loss
    use_aug = False  # whether to use data augmentation
    n_iter = 5
    p_mask = 0.1
    p_ng = 0.25
    ngram_range = (3, 6)
    teach_on_dev = True
    if not use_aug:
        (x_train, y_train, text_train), (x_dev, y_dev, text_dev), (x_test, y_test, text_test), v_size, embs = load_data(name)
    else:
        # will introduce randomness, thus can't be loaded below
        (x_train, y_train, text_train), (x_dev, y_dev, text_dev), (x_test, y_test, text_test), v_size, embs = \
        load_data_aug(name, n_iter, p_mask, p_ng, ngram_range)
    l_tr = list(map(lambda x: min(len(x), x_len), x_train))
    l_de = list(map(lambda x: min(len(x), x_len), x_dev))
    l_te = list(map(lambda x: min(len(x), x_len), x_test))
    x_train = sequence.pad_sequences(x_train, maxlen=x_len)
    x_dev = sequence.pad_sequences(x_dev, maxlen=x_len)
    x_test = sequence.pad_sequences(x_test, maxlen=x_len)
    if os.path.isfile('./data/cache/text_train') and os.path.isfile('./data/cache/text_dev'):
        with open('./data/cache/text_train', 'rb') as fin:
            text_train = pickle.load(fin)
        with open('./data/cache/text_dev', 'rb') as fin:
            text_dev = pickle.load(fin)
    else:
        with torch.no_grad():
            text_train = np.vstack([teacher.predict(text) for text in tqdm(text_train)])
            text_dev = np.vstack([teacher.predict(text) for text in tqdm(text_dev)])
        with open('./data/cache/text_train','wb') as fout: pickle.dump(text_train,fout)
        with open('./data/cache/text_dev','wb') as fout: pickle.dump(text_dev,fout)

    model = RNN(v_size, 256, 256, 2)
    # model = CNN(v_size,256,128,2)
    if USE_CUDA: model = model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    ce_loss = nn.NLLLoss()
    mse_loss = nn.MSELoss()
    for epoch in range(epochs):
        losses = []
        accu = []
        model.train()
        for i in range(0, len(x_train), b_size):
            model.zero_grad()
            bx = Variable(LTensor(x_train[i:i + b_size]))
            by = Variable(LTensor(y_train[i:i + b_size]))
            bl = Variable(LTensor(l_tr[i:i + b_size]))
            bt = Variable(FTensor(text_train[i:i + b_size]))
            py1, py2 = model(bx, bl)
            loss = alpha * ce_loss(py2, by) + (1-alpha) * mse_loss(py1, bt)  # in paper, only mse is used
            if i % 10 ==0:
                print('loss', loss)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        for i in range(0, len(x_dev), b_size):
            model.zero_grad()
            bx = Variable(LTensor(x_dev[i:i + b_size]))
            bl = Variable(LTensor(l_de[i:i + b_size]))
            bt = Variable(FTensor(text_dev[i:i + b_size]))
            py1, py2 = model(bx, bl)
            loss = mse_loss(py1, bt)
            if teach_on_dev:
                loss.backward()             
                opt.step()                       # train only with teacher on dev set
            losses.append(loss.item())
        model.eval()
        with torch.no_grad():
            for i in range(0, len(x_dev), b_size):
                bx = Variable(LTensor(x_dev[i:i + b_size]))
                by = Variable(LTensor(y_dev[i:i + b_size]))
                bl = Variable(LTensor(l_de[i:i + b_size]))
                _, py = torch.max(model(bx, bl)[1], 1)
                accu.append((py == by).float().mean().item())
        print(np.mean(losses), np.mean(accu))

if __name__ == '__main__':
    USERNAME = os.getenv('USERNAME')
    BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'
    train_teacher_model(bert_model=BERT_BASE_CHINESE_PATH)
    train_student_model()

# 0
# loss tensor(0.4687, grad_fn=<AddBackward0>)
# loss tensor(0.4239, grad_fn=<AddBackward0>)
# loss tensor(0.4407, grad_fn=<AddBackward0>)
# loss tensor(0.4236, grad_fn=<AddBackward0>)
# 0.1720391446394278 0.812
# 1
# loss tensor(0.3362, grad_fn=<AddBackward0>)
# loss tensor(0.3015, grad_fn=<AddBackward0>)
# loss tensor(0.2981, grad_fn=<AddBackward0>)
# loss tensor(0.3085, grad_fn=<AddBackward0>)
# 0.13185456104523746 0.8385
# 2
# loss tensor(0.2569, grad_fn=<AddBackward0>)
# loss tensor(0.2117, grad_fn=<AddBackward0>)
# loss tensor(0.1947, grad_fn=<AddBackward0>)
# loss tensor(0.1256, grad_fn=<AddBackward0>)
# 0.10212433639358967 0.852875
# 3
# loss tensor(0.2125, grad_fn=<AddBackward0>)
# loss tensor(0.1873, grad_fn=<AddBackward0>)
# loss tensor(0.1755, grad_fn=<AddBackward0>)
# loss tensor(0.0420, grad_fn=<AddBackward0>)
# 0.08027397851783333 0.869125
# 4
# loss tensor(0.1612, grad_fn=<AddBackward0>)
# loss tensor(0.1135, grad_fn=<AddBackward0>)
# loss tensor(0.1050, grad_fn=<AddBackward0>)
# loss tensor(0.0252, grad_fn=<AddBackward0>)
# 0.059840057322636565 0.872125