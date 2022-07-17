#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import torch  # 1.2.0
from transformers import BertTokenizer  # 2.5.1
import pickle
import torch.nn as nn
import numpy as np  # 1.18.4
from torch.nn import CrossEntropyLoss
from transformers import BertConfig
from keras.utils import to_categorical  # 2.2.4
import os, csv, random
import torch.nn.functional as F
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
from transformers import BertModel, BertPreTrainedModel
from transformers import BertTokenizer
from transformers import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm, trange
from sklearn.metrics import f1_score  # 0.19.1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_CUDA = torch.cuda.is_available()
if USE_CUDA: torch.cuda.set_device(0)
LTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
FTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor

USERNAME = os.getenv('USERNAME') or os.getenv('USER')
BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'
BERT_BASE_CHINESE_PATH = f'/appcom/apps-data/tmp/{USERNAME}/data/bert_base_pytorch/bert-base-chinese'

# 数据集来源： https://github.com/qiangsiwei/bert_distill.git

max_seq = 128
batch_size = 16
num_epochs = 10
lr = 2e-5  # 学习率若是 2e-3 时效果不是很好，几乎不收敛；
LABEL_LIST = ['0', '1']

tokenizer = BertTokenizer.from_pretrained(BERT_BASE_CHINESE_PATH, do_lower_case=True)

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
        return LABEL_LIST

    def _create_examples(self, data_path, set_type):
        examples = []
        with open(data_path, encoding="utf-8") as f:
            for i, line in enumerate(f):
                label, text = line.strip().split('\t', 1)
                guid = "{0}-{1}-{2}".format(set_type, label, i)
                examples.append(InputExample(guid=guid, text=text, label=label))
        random.shuffle(examples)
        return examples

    def create_texts_examples(self, texts, set_type='test'):
        examples = []
        for i, text in enumerate(texts):
            label = None
            guid = "{0}-{1}-{2}".format(set_type, label, i)
            examples.append(InputExample(guid=guid, text=text, label=label))
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
        label_id = label_map.get(example.label)
        features.append(InputFeatures(
            input_ids=input_ids + padding,
            input_mask=input_mask + padding,
            label_id=label_id))
    return features

loss_fun = CrossEntropyLoss()
criterion  = nn.KLDivLoss()#KL散度

ce_loss = nn.NLLLoss()
mse_loss = nn.MSELoss()

def kd_ce_loss(logits_S, logits_T, temperature=1):
    if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
        temperature = temperature.unsqueeze(-1)
    beta_logits_T = logits_T / temperature
    beta_logits_S = logits_S / temperature
    p_T = F.softmax(beta_logits_T, dim=-1)
    loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
    return loss

def lower_hidden_size_max_pool(weight, kernel_size, stride=None, mode='1d'):
    '''

    :param weight:
    :param kernel_size:
    :param stride:
    :return:
    '''
    dim = weight.dim()
    for i in range(3-dim):
        weight = weight.unsqueeze(i)
    if mode == '1d':
        max_pool = torch.nn.AvgPool1d(kernel_size, stride=stride)
    elif mode == '2d':
        max_pool = torch.nn.AvgPool2d(kernel_size, stride=stride)
    else:
        raise ValueError(f"不支持：{mode}")
    # weight.shape
    # Out[110]: torch.Size([1, 768, 768])
    # nn.MaxPool2d(2, stride=2)(weight).shape
    # Out[111]: torch.Size([1, 384, 384])
    # nn.MaxPool1d(2, stride=2)(weight).shape
    # Out[113]: torch.Size([1, 768, 384])
    weight = max_pool(weight)
    for _ in range(3-dim):
        weight = weight.squeeze(0)
    return weight

def layers_map(key='bert.encoder.layer.0.output.dense.weight'):
    layers_map_dict = {
        ".layer.0.": ".layer.0.",
        ".layer.1.": ".layer.4.",
        ".layer.2.": ".layer.8.",
        ".layer.3.": ".layer.11."
    }
    for low_layer, high_layer in layers_map_dict.items():
        key = key.replace(low_layer, high_layer)
    return key

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

def compute_metrics(preds, labels):
    return {'ac': (preds == labels).mean(), 'f1': f1_score(y_true=labels, y_pred=preds)}

def generator_dataset(dataloader_path='data/dataloader.pkl'):
    # 第一步生成数据集
    if os.path.exists(dataloader_path):
        with open(dataloader_path, 'rb')as f:
            pkl_data = pickle.load(f)
            train_dataloader = pkl_data['train_dataloader']
            eval_dataloader = pkl_data['eval_dataloader']
    else:
        processor = Processor()
        # 数据集来源： https://github.com/qiangsiwei/bert_distill.git
        train_examples = processor.get_train_examples('data/hotel')
        label_list = processor.get_labels()

        train_features = convert_examples_to_features(train_examples, label_list, max_seq, tokenizer)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_label_ids)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

        eval_examples = processor.get_dev_examples('data/hotel')
        eval_features = convert_examples_to_features(eval_examples, label_list, max_seq, tokenizer)
        eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        eval_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(eval_input_ids, eval_input_mask, eval_label_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=batch_size)
        with open(dataloader_path, 'wb')as f:
            pickle.dump({"train_dataloader": train_dataloader, "eval_dataloader": eval_dataloader}, f)
    return train_dataloader, eval_dataloader

def train_teacher_model(train_dataloader, eval_dataloader, bert_model=BERT_BASE_CHINESE_PATH, teacher_model_path = 'data/teacher_model'):
    # 第二步训练教师模型
    if os.path.exists(teacher_model_path):
        print(f'从 {teacher_model_path} 加载教师模型；')
        return
    teacher_model = BertClassification.from_pretrained(bert_model, cache_dir=None, num_labels=len(LABEL_LIST))
    teacher_model.to(device)
    optimizer = AdamW(teacher_model.parameters(), lr=lr)
    teacher_model.train()
    for _ in trange(num_epochs, desc='Epoch'):
        tr_loss = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            loss = teacher_model(input_ids, input_mask, label_ids)  # input_ids.shape=[batch_size, max_seq]; label_ids.shape [batch_size]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if step % 10 == 0:
                print(f"loss: {loss}")
            tr_loss += loss.item()
        print('tr_loss', tr_loss)
    print('eval...')
    teacher_model.eval()
    preds = []
    eval_label_ids = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
        eval_label_ids.extend(label_ids)
        with torch.no_grad():
            logits = teacher_model(input_ids, input_mask, None)
            preds.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.vstack(preds), axis=1)
    eval_label_ids = [item.cpu().detach().numpy() for item in eval_label_ids]
    print("教师模型的评估结果：", compute_metrics(preds, eval_label_ids))  # {'ac': 0.908125, 'f1': }
    torch.save(teacher_model, teacher_model_path)

def train_student_model(train_dataloader, eval_dataloader, bert_model=BERT_BASE_CHINESE_PATH, teacher_model_path = 'data/teacher_model', student_model_path = 'data/student_model',
                        teacher_predict_path = './data/teacher_predict.pkl'):
    if os.path.exists(student_model_path):
        student_model = torch.load(student_model_path)
        print(f'从{student_model_path}加载学生模型')
        return
    # 第三步训练学生模型
    model_config = BertConfig.from_pretrained(bert_model)
    model_config.num_hidden_layers = 4  # 仅仅是层数减少，以教师模型权重初始化，acc: 0.8945;
    model_config.hidden_size = 384 # 312  # 隐藏层数减少， 隐藏层单元个数也减半，用教师模型的权重初始化，结果波动较大： acc: 0.8538, 0.825, 0.80375, 0.8246, 0.8368, 0.8466, 0.837
    # 更改loss计算 acc 0.82575, 0.8538, 0.823， 0.846；  0.853625，0.851375

    student_model = BertClassification(model_config, num_labels=len(LABEL_LIST)) # 随机初始化，acc : 0.628
    teacher_model = torch.load(teacher_model_path)

    teacher_state_dict = teacher_model.state_dict()

    student_state_dict = student_model.state_dict()
    # for key in student_state_dict.keys():
    #     student_state_dict[key] = teacher_state_dict[key]

    for key in student_state_dict.keys():
        high_key = layers_map(key)
        if key in ['bert.encoder.layer.0.intermediate.dense.bias', 'bert.encoder.layer.1.intermediate.dense.bias',
                   'bert.encoder.layer.2.intermediate.dense.bias', 'bert.encoder.layer.3.intermediate.dense.bias',
                   'classifier.bias']:
            student_state_dict[key] = teacher_state_dict[high_key]
        elif key in ['bert.embeddings.LayerNorm.weight', 'bert.embeddings.LayerNorm.bias',
                     'bert.encoder.layer.0.attention.self.query.bias', 'bert.encoder.layer.0.attention.self.key.bias',
                     'bert.encoder.layer.0.attention.self.value.bias',
                     'bert.encoder.layer.0.attention.output.dense.bias',
                     'bert.encoder.layer.0.attention.output.LayerNorm.weight',
                     'bert.encoder.layer.0.attention.output.LayerNorm.bias', 'bert.encoder.layer.0.output.dense.bias',
                     'bert.encoder.layer.0.output.LayerNorm.weight', 'bert.encoder.layer.0.output.LayerNorm.bias',
                     'bert.encoder.layer.1.attention.self.query.bias', 'bert.encoder.layer.1.attention.self.key.bias',
                     'bert.encoder.layer.1.attention.self.value.bias',
                     'bert.encoder.layer.1.attention.output.dense.bias',
                     'bert.encoder.layer.1.attention.output.LayerNorm.weight',
                     'bert.encoder.layer.1.attention.output.LayerNorm.bias', 'bert.encoder.layer.1.output.dense.bias',
                     'bert.encoder.layer.1.output.LayerNorm.weight', 'bert.encoder.layer.1.output.LayerNorm.bias',
                     'bert.encoder.layer.2.attention.self.query.bias', 'bert.encoder.layer.2.attention.self.key.bias',
                     'bert.encoder.layer.2.attention.self.value.bias',
                     'bert.encoder.layer.2.attention.output.dense.bias',
                     'bert.encoder.layer.2.attention.output.LayerNorm.weight',
                     'bert.encoder.layer.2.attention.output.LayerNorm.bias', 'bert.encoder.layer.2.output.dense.bias',
                     'bert.encoder.layer.2.output.LayerNorm.weight', 'bert.encoder.layer.2.output.LayerNorm.bias',
                     'bert.encoder.layer.3.attention.self.query.bias', 'bert.encoder.layer.3.attention.self.key.bias',
                     'bert.encoder.layer.3.attention.self.value.bias',
                     'bert.encoder.layer.3.attention.output.dense.bias',
                     'bert.encoder.layer.3.attention.output.LayerNorm.weight',
                     'bert.encoder.layer.3.attention.output.LayerNorm.bias', 'bert.encoder.layer.3.output.dense.bias',
                     'bert.encoder.layer.3.output.LayerNorm.weight', 'bert.encoder.layer.3.output.LayerNorm.bias',
                     'bert.pooler.dense.bias']:
            student_state_dict[key] = lower_hidden_size_max_pool(teacher_state_dict[high_key], 2, 2, '1d')
        elif key in ['bert.embeddings.word_embeddings.weight', 'bert.embeddings.position_embeddings.weight',
                     'bert.embeddings.token_type_embeddings.weight', 'bert.encoder.layer.0.intermediate.dense.weight',
                     'bert.encoder.layer.1.intermediate.dense.weight', 'bert.encoder.layer.2.intermediate.dense.weight',
                     'bert.encoder.layer.3.intermediate.dense.weight', 'classifier.weight']:
            student_state_dict[key] = lower_hidden_size_max_pool(teacher_state_dict[high_key], 2, 2, '1d')
        elif key in ['bert.encoder.layer.0.attention.self.query.weight',
                     'bert.encoder.layer.0.attention.self.key.weight',
                     'bert.encoder.layer.0.attention.self.value.weight',
                     'bert.encoder.layer.0.attention.output.dense.weight',
                     'bert.encoder.layer.1.attention.self.query.weight',
                     'bert.encoder.layer.1.attention.self.key.weight',
                     'bert.encoder.layer.1.attention.self.value.weight',
                     'bert.encoder.layer.1.attention.output.dense.weight',
                     'bert.encoder.layer.2.attention.self.query.weight',
                     'bert.encoder.layer.2.attention.self.key.weight',
                     'bert.encoder.layer.2.attention.self.value.weight',
                     'bert.encoder.layer.2.attention.output.dense.weight',
                     'bert.encoder.layer.3.attention.self.query.weight',
                     'bert.encoder.layer.3.attention.self.key.weight',
                     'bert.encoder.layer.3.attention.self.value.weight',
                     'bert.encoder.layer.3.attention.output.dense.weight', 'bert.pooler.dense.weight']:
            #  [[312, 312], [768, 768]]
            student_state_dict[key] = lower_hidden_size_max_pool(teacher_state_dict[high_key], 2, 2, '2d')
        elif key in ['bert.encoder.layer.0.output.dense.weight', 'bert.encoder.layer.1.output.dense.weight',
                     'bert.encoder.layer.2.output.dense.weight', 'bert.encoder.layer.3.output.dense.weight']:
            # [[312, 3072], [768, 3072]]
            student_state_dict[key] = lower_hidden_size_max_pool(teacher_state_dict[high_key].T, 2, 2, '1d').T
        else:
            raise ValueError(f"不支持{key}")

    student_model.load_state_dict(student_state_dict)

    student_model.to(device)
    optimizer = AdamW(student_model.parameters(), lr=lr)
    student_model.train()

    # 加载教师模型预测结果：
    if os.path.isfile(teacher_predict_path):
        with open(teacher_predict_path, 'rb') as fin:
            teacher_predict_result = pickle.load(fin)
    else:
        teacher_model.eval()
        teacher_predict_result = []
        for step, batch in enumerate(tqdm(train_dataloader, desc='Iteration')):
            input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
            # 预先把教师模型结果计算好
            with torch.no_grad():
                output_teacher = teacher_model(input_ids, input_mask, None)
            teacher_predict_result.append({"output_teacher": output_teacher, "input_ids": input_ids, "input_mask": input_mask, "label_ids": label_ids})
        with open(teacher_predict_path, 'wb') as fout:
            pickle.dump(teacher_predict_result, fout)
        print('教师模型的预测结果保存到：', teacher_predict_path)
        # del teacher_model

    for _ in trange(num_epochs, desc='Epoch'):
        tr_loss = 0
        with tqdm(teacher_predict_result, desc='Iteration') as pbar:
            for step, batch_data in enumerate(pbar):
                # optimizer.zero_grad()  # 意思是把梯度置零，也就是把loss关于weight的导数变成0. 即将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）

                input_ids = batch_data["input_ids"].to(device)
                input_mask = batch_data["input_mask"].to(device)
                label_ids = batch_data["label_ids"].to(device)

                output_teacher = batch_data["output_teacher"]
                # input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
                logits = student_model(input_ids, input_mask, None)  # input_ids.shape=[batch_size, max_seq]; label_ids.shape [batch_size]
                loss_hard = loss_fun(logits.view(-1, len(LABEL_LIST)), label_ids.view(-1))

                # 计算学生模型预测结果和教师模型预测结果之间的KL散度
                # loss_soft = criterion(torch.as_tensor(output_student), torch.as_tensor(output_teacher))  # tensor(-0.0763, grad_fn=<KlDivBackward>)
                loss_soft = kd_ce_loss(torch.as_tensor(logits), torch.as_tensor(output_teacher), temperature=1)
                loss = 0.9 * loss_soft + 0.1 * loss_hard
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                pbar.set_postfix({'loss': '{0:1.5f}'.format(loss)})  # 输入一个字典，显示实验指标
                pbar.update(1)

                tr_loss += loss.item()
                if step % 10 == 0:
                    print(step, loss)
        print('tr_loss', tr_loss)
    print('eval...')
    student_model.eval()
    preds = []
    eval_label_ids = []
    for batch in tqdm(eval_dataloader, desc='Evaluating'):
        input_ids, input_mask, label_ids = tuple(t.to(device) for t in batch)
        eval_label_ids.extend(label_ids)
        with torch.no_grad():
            logits = student_model(input_ids, input_mask, None)
            preds.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.vstack(preds), axis=1)
    eval_label_ids = [item.cpu().detach().numpy() for item in eval_label_ids]
    print("学生模型的评估结果：", compute_metrics(preds, eval_label_ids))  # {'ac': 0.86475}
    torch.save(student_model, student_model_path)

def student_model_predict(student_model_path = 'data/student_model'):
    # 第四步使用学生模型进行预测
    student_model = torch.load(student_model_path)
    student_model.to(device)
    student_model.eval()
    processor = Processor()
    # 数据集来源： https://github.com/qiangsiwei/bert_distill.git

    with open(r'./data/hotel/hotel.txt', encoding='utf-8')as f:
        all_texts = [t.strip().split('\t') for t in f.readlines()]
        random.shuffle(all_texts)
    label_texts = all_texts[:10]
    predict_examples = processor.create_texts_examples([t[-1] for t in label_texts])
    label_list = processor.get_labels()

    predict_features = convert_examples_to_features(predict_examples, label_list, max_seq, tokenizer)
    input_ids = torch.tensor([f.input_ids for f in predict_features], dtype=torch.long).to(device)
    input_mask = torch.tensor([f.input_mask for f in predict_features], dtype=torch.long).to(device)
    # label_ids = torch.tensor([f.label_id for f in predict_features], dtype=torch.long).to(device)
    preds = []
    with torch.no_grad():
        logits = student_model(input_ids, input_mask, None)
        preds.append(logits.detach().cpu().numpy())
    preds = np.argmax(np.vstack(preds), axis=1)
    result = [{"text": text, "pred": pred, "label": label } for (label, text), pred in zip(label_texts, preds)]
    print('预测结果：', result)
    return result

def main():
    train_dataloader, eval_dataloader = generator_dataset() # 第一步生成数据集
    train_teacher_model(train_dataloader, eval_dataloader, teacher_model_path='data/teacher_model')  # 第二步训练教师模型
    train_student_model(train_dataloader, eval_dataloader, teacher_model_path='data/teacher_model', student_model_path = 'data/student_model')  # 第三步训练学生模型
    student_model_predict(student_model_path='data/student_model')  # 第四步使用学生模型进行预测


if __name__ == '__main__':
    main()

# 隐藏层由12 -> 4, acc变化不明显， 0.90 -> 0.89
# 在隐藏层变为4后，隐藏层单元数减半后，acc变化明显， 0.90 -> 0.8~0.85之间；
# loss 的计算方式，对结果影响不大；不论是通过教师模型loss_soft + loss_hard，还是单独loss_hard 计算，影响不明显；
