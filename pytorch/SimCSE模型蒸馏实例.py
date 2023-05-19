#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import torch
from sklearn.metrics.pairwise import cosine_similarity
from glob import glob
from tqdm import tqdm
import time, os
import torch.nn as nn
import torch.nn.functional as F
from torchkeras import summary
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
from transformers import BertForMaskedLM, PretrainedConfig, BertConfig, BertTokenizer
import torchvision
from torchvision import transforms
from datasets import Dataset, IterableDataset
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchinfo import summary
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses

# For reproducibility
torch.manual_seed(0)
torch.backends.cudnn.benchmark = True

# 更多参考：
# https://github.com/zejunwang1/simbert_distill/blob/main/task_distill_sentence_pairs.py

#############################################################################################################################
# 第一步：定义教师模型
# 模型来源：https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese
# model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
# tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
USERNAME = os.getenv('USERNAME')
teacher_model = AutoModelForMaskedLM.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')
teacher_tokenizer = AutoTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')

#############################################################################################################################
# 第二步：定义学生模型

# 词表裁剪
# 可以按照词频对词表进行裁剪，去除出现频次较低的词，这样能够精度得到最大限度保持的同时，减少分词后[UNK]的出现。
student_vocab_file = r'D:\Users\{}\data\RoBERTa-tiny-clue\vocab.txt'.format(os.getenv("USERNAME"))

student_model_config = {
  # "architectures": ["BertModel"],
  # "gradient_checkpointing": False,
  # "position_embedding_type": "absolute",
  # "transformers_version": "4.2.1",
  # "use_cache": True,
    'bos_token_id': 0,
    'pad_token_id': 1,
    'eos_token_id': 2,
  'attention_probs_dropout_prob': 0.1,
 'directionality': 'bidi',
 'hidden_act': 'gelu',
 'hidden_dropout_prob': 0.1,
 'hidden_size': 312,
 'initializer_range': 0.02,
 'intermediate_size': 1248,
 'layer_norm_eps': 1e-12,
 'max_position_embeddings': 512,
 'model_type': 'roberta',
 'num_attention_heads': 12,
 'num_hidden_layers': 3,
 'pooler_fc_size': 768,
 'pooler_num_attention_heads': 12,
 'pooler_num_fc_layers': 3,
 'pooler_size_per_head': 128,
 'pooler_type': 'first_token_transform',
 'type_vocab_size': 2,
 'vocab_size': 8021}

student_model = BertForMaskedLM(PretrainedConfig(**student_model_config))
student_tokenizer = BertTokenizer(student_vocab_file)
student_tokenizer.bos_token_id = 0
student_tokenizer.eos_token_id = 2

#############################################################################################################################
# 第三步：读取数据，方法1，全量读取到内存
def read_data_dict(data_type='train', shuff=True):
    columns_sep_dict = {
        # file_name, header, index_col, names, sep, quoting, engine
        'ATEC': ('ATEC.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'BQ': ('BQ.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'LCQMC': ('LCQMC.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'PAWSX': ('PAWSX.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'STS-B': ('STS-B.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
    }
    test_file_path = rf'D:\Users\{USERNAME}\data/similarity/senteval_cn/senteval_cn'
    text_a_list = []
    text_b_list = []
    label_list = []
    for task_name, (_, header, index_col, names, sep, quoting, engine) in columns_sep_dict.items():
        file_name = f"{task_name}.{data_type}.data"
        test_file = os.path.join(test_file_path, task_name, file_name)
        df = pd.read_csv(test_file, header=header, index_col=index_col, names=names, sep=sep, quoting=quoting, engine=engine)
        df['label'] = [int(t) for t in df['label'].values]
        if len(df['label'].unique()) != 2:
            df = df[df['label'].isin([0, 5])]
            df['label'] = [1 if t == 5 else 0 for t in df['label'].values]
        df = df[df['label'].isin({0, 1})]
        df = df[(~df['text_a'].isna()) & (~df['text_b'].isna())]
        print(task_name, df.shape)
        # for text_a, text_b, label in df[['text_a', 'text_b', 'label']].values:
        #     yield {"text_a": text_a, "text_b": text_b, "label": label}
        text_a_list.extend(list(df['text_a'].values))
        text_b_list.extend(list(df['text_b'].values))
        label_list.extend(list(df['label'].values))
        if shuff:
            text_a_list, text_b_list, label_list = shuffle(text_a_list, text_b_list, label_list)
    label_list = [(t-0.5)/0.5 for t in label_list]
    data_dict = {"text_a": text_a_list, "text_b": text_b_list, "label": label_list}
    return data_dict



def read_train_pred_data_dict(train_file=rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/不同任务测评结果.txt', shuff=True):
    df = pd.read_csv(train_file, sep='\t', dtype={"label": int, 'pred': float})

    df = df[((df['name'] == 'STS-B') & (df['label'].isin([0, 5]))) | (df['name']!='STS-B')]
    df['label'] = [1 if ((name=='STS-B' and t==5) or (name!='STS-B' and t==1)) else 0 for name, t in df[['name', 'label']].values]
    text_a_list = list(df['text_a'].values)
    text_b_list = list(df['text_b'].values)
    label_list = list(df['label'].values)
    pred_list = list(df['hist_norm'].values) # pred: 原始模型得分，hist_norm: 直方图均衡化得分

    if shuff:
            text_a_list, text_b_list, label_list, pred_list = shuffle(text_a_list, text_b_list, label_list, pred_list)

    label_list = [(t - 0.5) / 0.5 for t in label_list]
    pred_list = [(t - 0.5) / 0.5 for t in pred_list]
    data_dict = {"text_a": text_a_list, "text_b": text_b_list, "label": label_list, 'pred': pred_list}
    return data_dict

def tokenize_function(examples, tokenizer=None, max_length=32, student_tokenizer=None):
    encode_a = tokenizer(examples["text_a"], padding="max_length", truncation=True, max_length=max_length)
    encode_b = tokenizer(examples["text_b"], padding="max_length", truncation=True, max_length=max_length)
    if student_tokenizer is None:
        return {'input_ids_a': encode_a['input_ids'], 'token_type_ids_a': encode_a['token_type_ids'], 'attention_mask_a': encode_a['attention_mask'],
                'input_ids_b': encode_b['input_ids'], 'token_type_ids_b': encode_b['token_type_ids'], 'attention_mask_b': encode_b['attention_mask'],}
    else:
        encode_a2 = student_tokenizer(examples["text_a"], padding="max_length", truncation=True, max_length=max_length)
        encode_b2 = student_tokenizer(examples["text_b"], padding="max_length", truncation=True, max_length=max_length)
        return {'input_ids_a': encode_a['input_ids'], 'token_type_ids_a': encode_a['token_type_ids'], 'attention_mask_a': encode_a['attention_mask'],
                'input_ids_b': encode_b['input_ids'], 'token_type_ids_b': encode_b['token_type_ids'], 'attention_mask_b': encode_b['attention_mask'],
                'input_ids_a2': encode_a2['input_ids'], 'token_type_ids_a2': encode_a2['token_type_ids'],
                'attention_mask_a2': encode_a2['attention_mask'],
                'input_ids_b2': encode_b2['input_ids'], 'token_type_ids_b2': encode_b2['token_type_ids'],
                'attention_mask_b2': encode_b2['attention_mask'],
                }

train_dataset = Dataset.from_dict(read_data_dict(data_type='train'))
valid_dataset = Dataset.from_dict(read_data_dict(data_type='valid'))
test_dataset = Dataset.from_dict(read_data_dict(data_type='test'))

train_tokenized_datasets = train_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})
valid_tokenized_datasets = valid_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})
test_tokenized_datasets = test_dataset.map(tokenize_function, batched=True, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})

def collate(batch):
    elem = batch[0]
    elem_type = type(elem)
    return elem_type({key: [d[key] for d in batch] for key in elem})

train_loader = DataLoader(train_tokenized_datasets, shuffle=True, batch_size=32, collate_fn=collate)
valid_loader = DataLoader(valid_tokenized_datasets, shuffle=True, batch_size=32, collate_fn=collate)
test_loader = DataLoader(test_tokenized_datasets, shuffle=True, batch_size=32, collate_fn=collate)

#############################################################################################################################
# 第三步：读取数据, 方法2，生成器
def generator_data_dict(data_type='train'):
    columns_sep_dict = {
        # file_name, header, index_col, names, sep, quoting, engine
        'ATEC': ('ATEC.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'BQ': ('BQ.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'LCQMC': ('LCQMC.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'PAWSX': ('PAWSX.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
        'STS-B': ('STS-B.test.data', 'infer', None, ['text_a', 'text_b', 'label'], '\t', 0, 'c'),
    }
    test_file_path = rf'D:\Users\{USERNAME}\data/similarity/senteval_cn/senteval_cn'

    for task_name, (_, header, index_col, names, sep, quoting, engine) in columns_sep_dict.items():
        file_name = f"{task_name}.{data_type}.data"
        test_file = os.path.join(test_file_path, task_name, file_name)
        df = pd.read_csv(test_file, header=header, index_col=index_col, names=names, sep=sep, quoting=quoting, engine=engine)
        df['label'] = [int(t) for t in df['label'].values]
        if len(df['label'].unique()) != 2:
            df = df[df['label'].isin([0, 5])]
            df['label'] = [1 if t == 5 else 0 for t in df['label'].values]
        df = df[df['label'].isin({0, 1})]
        print(task_name, df.shape)
        for text_a, text_b, label in df[['text_a', 'text_b', 'label']].values:
            if text_a is np.nan or text_b is np.nan:
                continue
            yield {"text_a": text_a, "text_b": text_b, "label": label}

train_dataset = Dataset.from_generator(generator_data_dict, gen_kwargs={"data_type": 'train'})
valid_dataset = Dataset.from_generator(generator_data_dict, gen_kwargs={"data_type": 'valid'})
test_dataset = Dataset.from_generator(generator_data_dict, gen_kwargs={"data_type": 'test'})
#
train_dataset = train_dataset.shuffle(seed=42)  # shuffles the shards order + uses a shuffle buffer
valid_dataset = valid_dataset.shuffle(seed=42)  # shuffles the shards order + uses a shuffle buffer
test_dataset = test_dataset.shuffle(seed=42)  # shuffles the shards order + uses a shuffle buffer

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})
valid_tokenized_dataset = valid_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": teacher_tokenizer, "max_length":32})
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])

train_loader = DataLoader(train_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
valid_loader = DataLoader(valid_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
test_loader = DataLoader(test_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)

#############################################################################################################################
# 第四步：训练我们的教师模型 n 个 epoch。

# similarity = torch.cosine_similarity(x, y, dim=0)

def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    losses = []
    model.eval()  # 在设置 model.eval() 之后，pytorch模型的性能会很差，原因暂不明

    with torch.no_grad():
        for item in loader:
            inputs_a = {'input_ids': torch.as_tensor(item['input_ids_a']), 'token_type_ids': torch.as_tensor(item['token_type_ids_a']), 'attention_mask': torch.as_tensor(item['attention_mask_a'])}
            inputs_b = {'input_ids': torch.as_tensor(item['input_ids_b']), 'token_type_ids': torch.as_tensor(item['token_type_ids_b']),
                       'attention_mask': torch.as_tensor(item['attention_mask_b'])}
            y = item['label']

            # inputs_a = inputs_a.to(device)
            # inputs_b = inputs_b.to(device)
            # y = y.to(device)

            outputs_a = model(**inputs_a, output_hidden_states=True)
            texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

            outputs_b = model(**inputs_b, output_hidden_states=True)
            textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

            silimarity_soce = torch.cosine_similarity(texta_embedding, textb_embedding, dim=-1)
            predictions = torch.where(silimarity_soce>=0.0, 1, -1)
            num_correct += (predictions == y).sum()
            num_samples += len(y)

            loss = cosine_corr_loss(silimarity_soce, y)
            losses.append(loss.item())

    acc = (num_correct/num_samples).item()
    loss = sum(losses) / len(losses)

    model.train()
    return acc, loss # item方法是得到只有一个元素张量里面的元素值。

def cosine_corr_loss(scores, targets, loss_fct = torch.nn.MSELoss()):
    '''求解相关性,越相关性越大，则得分越低'''
    # if len(targets) == 1:
    #     # scores = torch.as_tensor([0, scores, 1])
    #     # targets = torch.as_tensor([0, targets, 1])
    #     loss_soce = (targets - scores.abs()).abs().mean()
    # elif torch.all(targets == 1) or torch.all(targets == 0):
    #     loss_soce = (targets-scores.abs()).abs().mean()
    # else:
    #     loss_soce = 1-torch.stack([scores.abs(), torch.as_tensor(targets)], dim=0).corrcoef()[0, 1]

    loss_soce = loss_fct(scores, targets)
    return loss_soce

def train_model(model, epochs=100, train_loader=None, valid_loader=None, save_model_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/teacher_model.bin', lr=1e-4):
    '''训练教师模型，或训练学生模型'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss() # 交叉熵损失
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_loss = 100
    for epoch in range(epochs):
        model.train()
        losses = []
        num_correct = 0
        num_samples = 0
        pbar = tqdm(train_loader, total=len(train_loader), position=0, leave=True, desc=f"Epoch {epoch}")

        for item in pbar:
            inputs_a = {'input_ids': torch.as_tensor(item['input_ids_a']), 'token_type_ids': torch.as_tensor(item['token_type_ids_a']), 'attention_mask': torch.as_tensor(item['attention_mask_a'])}
            inputs_b = {'input_ids': torch.as_tensor(item['input_ids_b']), 'token_type_ids': torch.as_tensor(item['token_type_ids_b']),
                       'attention_mask': torch.as_tensor(item['attention_mask_b'])}
            targets = item['label']

            outputs_a = model(**inputs_a, output_hidden_states=True)
            texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

            outputs_b = model(**inputs_b, output_hidden_states=True)
            textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

            silimarity_soce = torch.cosine_similarity(texta_embedding, textb_embedding, dim=-1)
            loss = cosine_corr_loss(silimarity_soce, targets)
            losses.append(loss.item())

            predictions = torch.where(silimarity_soce>=0.0, 1, -1)
            num_correct += (predictions == targets).sum()
            num_samples += len(targets)

            # backward
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

        avg_loss = sum(losses) / len(losses)
        acc = (num_correct / num_samples).item()
        valid_acc, valid_loss = check_accuracy(valid_loader, model, device)
        print(f"Loss:{avg_loss:.4f}\taccuracy:{acc:.4f}\tval_loss:{valid_loss:.4f}\tval_accuracy:{valid_acc:.4f}")

        if valid_loss < best_loss:
            torch.save(model, save_model_path)
            best_loss = valid_loss
    return model


# teacher_model2 = train_model(teacher_model, 3, train_loader, valid_loader, save_model_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/teacher_model.bin', lr=1e-5)
# ################################ pearson
# STS-B 0.7813574052371914
# ################################ kendall
# STS-B 0.6229166436471536
# ################################ spearman
# STS-B 0.7729485401203372

#############################################################################################################################
# 第五步：训练相同的epoch 蒸馏模型。

def train_step(
        teacher_model,
        student_model,
        optimizer,
        student_loss_fn,
        divergence_loss_fn,
        temp,
        alpha,
        epoch,
        device
):
    losses = []
    num_correct = 0
    num_samples = 0
    pbar = tqdm(train_loader, total=len(train_loader), position=0, leave=True, desc=f"Epoch {epoch}")

    for item in pbar:
        inputs_a = {'input_ids': torch.as_tensor(item['input_ids_a']),
                    'token_type_ids': torch.as_tensor(item['token_type_ids_a']),
                    'attention_mask': torch.as_tensor(item['attention_mask_a'])}
        inputs_b = {'input_ids': torch.as_tensor(item['input_ids_b']),
                    'token_type_ids': torch.as_tensor(item['token_type_ids_b']),
                    'attention_mask': torch.as_tensor(item['attention_mask_b'])}

        targets = item['label']
        if teacher_model is None:
            # 直接读取教师模型离线预测结果
            teacher_preds = item['pred']
            inputs_a2 = inputs_a
            inputs_b2 = inputs_b
        else:
            inputs_a2 = {'input_ids': torch.as_tensor(item['input_ids_a2']),
                        'token_type_ids': torch.as_tensor(item['token_type_ids_a2']),
                        'attention_mask': torch.as_tensor(item['attention_mask_a2'])}
            inputs_b2 = {'input_ids': torch.as_tensor(item['input_ids_b2']),
                        'token_type_ids': torch.as_tensor(item['token_type_ids_b2']),
                        'attention_mask': torch.as_tensor(item['attention_mask_b2'])}


            with torch.no_grad():
                outputs_a = teacher_model(**inputs_a, output_hidden_states=True)
                texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

                outputs_b = teacher_model(**inputs_b, output_hidden_states=True)
                textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

                teacher_preds = torch.cosine_similarity(texta_embedding, textb_embedding, dim=-1)

        outputs_a = student_model(**inputs_a2, output_hidden_states=True)
        texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

        outputs_b = student_model(**inputs_b2, output_hidden_states=True)
        textb_embedding = outputs_b.hidden_states[-1][:, 0, :].squeeze()

        student_preds = torch.cosine_similarity(texta_embedding, textb_embedding, dim=-1)

        student_loss = student_loss_fn(student_preds, targets)

        # ditillation_loss = divergence_loss_fn(
        #     F.softmax(student_preds / temp, dim=0).log(),
        #     F.softmax(teacher_preds / temp, dim=0)
        # )
        ditillation_loss = student_loss_fn(student_preds, teacher_preds)
        loss = alpha * student_loss + (1 - alpha) * ditillation_loss
        losses.append(loss.item())

        predictions = torch.where(student_preds >= 0.0, 1, -1)
        num_correct += (predictions == targets).sum()
        num_samples += len(targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    avg_loss = sum(losses) / len(losses)
    acc = (num_correct / num_samples).item()
    return student_model, avg_loss, acc


def distillation(epochs, teacher, student, temp=7, alpha=0.3, lr=1e-5, save_model_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/student_model.bin', patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if teacher is not None:
        teacher = teacher.to(device)
        teacher.eval()
    student = student.to(device)
    # student_loss_fn = nn.CrossEntropyLoss()
    student_loss_fn = cosine_corr_loss
    divergence_loss_fn = nn.KLDivLoss(reduction="batchmean")  # KL散度，又叫相对熵，用于衡量两个分布之间的距离。
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    best_loss = np.inf
    epochs_without_improvement = 0

    student.train()
    for epoch in range(epochs):
        student, loss, acc = train_step(
            teacher,
            student,
            optimizer,
            student_loss_fn,
            divergence_loss_fn,
            temp,
            alpha,
            epoch,
            device
        )
        valid_acc, valid_loss = check_accuracy(valid_loader, student, device)
        print(f"Loss:{loss:.4f}\taccuracy:{acc:.4f}\tval_loss:{valid_loss:.4f}\tval_accuracy:{valid_acc:.4f}")
        if valid_loss < best_loss:
            torch.save(student, save_model_path)
            best_loss = valid_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        # 如果验证集上的损失连续patience个epoch没有提高，则停止训练
        if epochs_without_improvement >= patience:
            print('早停 epoch {}...'.format(epoch + 1))
            break
    return student_model


train_dataset = Dataset.from_dict(read_train_pred_data_dict(train_file=rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/不同任务训练集测评结果4.txt', shuff=True))
valid_dataset = Dataset.from_dict(read_data_dict(data_type='valid'))
test_dataset = Dataset.from_dict(read_data_dict(data_type='test'))

train_tokenized_dataset = train_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
valid_tokenized_dataset = valid_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])

train_loader = DataLoader(train_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
valid_loader = DataLoader(valid_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
test_loader = DataLoader(test_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)

distillation(100, teacher_model, student_model, temp=7, alpha=0.5, lr=1e-3, save_model_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/student_model.bin')
# ATEC (20000, 3)
# BQ (10000, 3)
# LCQMC (8802, 3)
# PAWSX (2000, 3)
# STS-B (385, 3)
# 100%|██████████| 448477/448477 [08:05<00:00, 924.29ex/s]
# 100%|██████████| 41187/41187 [00:42<00:00, 973.45ex/s]
# Epoch 0: 100%|██████████| 14015/14015 [4:31:03<00:00,  1.16s/it]
# Loss:-0.1893	accuracy:0.0006	val_loss:0.8359	val_accuracy:0.3398
# Epoch 1: 100%|██████████| 14015/14015 [4:27:18<00:00,  1.14s/it]
# Loss:-0.1904	accuracy:0.0006	val_loss:0.8382	val_accuracy:0.3398
# Epoch 2: 100%|██████████| 14015/14015 [4:25:51<00:00,  1.14s/it]
# Loss:-0.1909	accuracy:0.0006	val_loss:0.8417	val_accuracy:0.3398
# Epoch 3: 100%|██████████| 14015/14015 [4:30:59<00:00,  1.16s/it]
# Loss:-0.1912	accuracy:0.0006	val_loss:0.8457	val_accuracy:0.3398
# Epoch 4: 100%|██████████| 14015/14015 [4:39:03<00:00,  1.19s/it]
# Loss:-0.1915	accuracy:0.0006	val_loss:0.8481	val_accuracy:0.3398
# Epoch 5: 100%|██████████| 14015/14015 [4:40:42<00:00,  1.20s/it]
# Loss:-0.1917	accuracy:0.0006	val_loss:0.8501	val_accuracy:0.3398
# 早停 epoch 6...

# 100%|██████████| 448477/448477 [07:28<00:00, 1001.01ex/s]
# 100%|██████████| 41187/41187 [00:37<00:00, 1112.00ex/s]
# 100%|██████████| 44838/44838 [00:37<00:00, 1205.76ex/s]
# Epoch 0: 100%|██████████| 14015/14015 [4:49:20<00:00,  1.24s/it]
# Loss:0.0157	accuracy:0.9942	val_loss:0.8319	val_accuracy:0.3398
# Epoch 1: 100%|██████████| 14015/14015 [4:38:44<00:00,  1.19s/it]
# Loss:0.0135	accuracy:0.9994	val_loss:0.8374	val_accuracy:0.3398
# Epoch 2:  90%|█████████ | 12618/14015 [4:11:25<30:01,  1.29s/it]

# Epoch 0: 100%|██████████| 14015/14015 [5:30:58<00:00,  1.42s/it]
# Loss:0.6778	accuracy:0.4973	val_loss:1.0258	val_accuracy:0.4363
# Epoch 1: 100%|██████████| 14015/14015 [6:57:43<00:00,  1.79s/it]
# Loss:0.6728	accuracy:0.4979	val_loss:1.0180	val_accuracy:0.4615
#############################################################################################################################
# 第六步(非必须)，为了对比效果，我们将在没有知识蒸馏的情况下训练我们的学生模型。
train_tokenized_dataset = train_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
valid_tokenized_dataset = valid_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
test_tokenized_dataset = test_dataset.map(tokenize_function, batched=False, fn_kwargs={"tokenizer": student_tokenizer, "max_length":32})
# tokenized_datasets = tokenized_datasets.remove_columns(["text"])

train_loader = DataLoader(train_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
valid_loader = DataLoader(valid_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)
test_loader = DataLoader(test_tokenized_dataset.with_format("torch"), num_workers=4, batch_size=32)

student_model2 = train_model(student_model, epochs=3, train_loader=train_loader, valid_loader=valid_loader,
                             save_model_path=rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/student_model.bin', lr=1e-3)
# Epoch 0: 100%|██████████| 34/34 [01:22<00:00,  2.41s/it]
# Loss:0.2804	accuracy:0.4449	val_loss:0.7481	val_accuracy:0.1584
# Epoch 1: 100%|██████████| 34/34 [01:04<00:00,  1.89s/it]
# Loss:0.2026	accuracy:0.8428	val_loss:0.7167	val_accuracy:0.2883
# Epoch 2: 100%|██████████| 34/34 [00:59<00:00,  1.76s/it]
# Loss:0.1604	accuracy:0.9421	val_loss:0.6959	val_accuracy:0.4571
# Epoch 3: 100%|██████████| 34/34 [01:03<00:00,  1.88s/it]
# Loss:0.1343	accuracy:0.9522	val_loss:0.6905	val_accuracy:0.4909
# Epoch 4: 100%|██████████| 34/34 [00:58<00:00,  1.71s/it]
# Loss:0.1156	accuracy:0.9632	val_loss:0.6846	val_accuracy:0.5039
# Epoch 5: 100%|██████████| 34/34 [01:09<00:00,  2.04s/it]
# Loss:0.0931	accuracy:0.9678	val_loss:0.6834	val_accuracy:0.5221
# Epoch 6: 100%|██████████| 34/34 [01:01<00:00,  1.81s/it]
# Loss:0.0911	accuracy:0.9660	val_loss:0.6761	val_accuracy:0.5273
# Epoch 7: 100%|██████████| 34/34 [01:10<00:00,  2.08s/it]
# Loss:0.0775	accuracy:0.9752	val_loss:0.6972	val_accuracy:0.5377
# Epoch 8: 100%|██████████| 34/34 [00:57<00:00,  1.69s/it]
# Loss:0.0664	accuracy:0.9798	val_loss:0.6660	val_accuracy:0.5532
# Epoch 9: 100%|██████████| 34/34 [01:17<00:00,  2.29s/it]
# Loss:0.0541	accuracy:0.9835	val_loss:0.6720	val_accuracy:0.5532
# Epoch 10: 100%|██████████| 34/34 [01:15<00:00,  2.22s/it]
# Loss:0.0492	accuracy:0.9853	val_loss:0.7252	val_accuracy:0.4831
# ################################ pearson
# STS-B 0.2606548870011991
# ################################ kendall
# STS-B 0.2085463177626542
# ################################ spearman
# STS-B 0.28569762559713774

def main():
    # 对教师模型进行微调
    teacher_model2 = train_model(teacher_model, epochs=20, train_loader=train_loader, valid_loader=valid_loader, save_model_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/teacher_model.bin', lr=1e-5)



if __name__ == '__main__':
    main()
