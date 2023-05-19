#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch
import os

USERNAME = os.getenv("USERNAME")

# 模型来源：https://huggingface.co/IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment
model_dir = rf"D:/Users/{USERNAME}/data/Erlangshen-Roberta-110M-Sentiment"

tokenizer=BertTokenizer.from_pretrained(model_dir)
model=BertForSequenceClassification.from_pretrained(model_dir)

text='今天心情不好'

output=model(torch.tensor([tokenizer.encode(text)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
# tensor([[0.9551, 0.0449]], grad_fn=<SoftmaxBackward0>)

############################################################################################################################
# 使用 Trainer API 在 PyTorch 中进行微调
# 由于 PyTorch 不提供封装好的训练循环，🤗 Transformers 库写了了一个transformers.Trainer API，它是一个简单但功能完整的 PyTorch 训练和评估循环，
# 针对 🤗 Transformers 进行了优化，有很多的训练选项和内置功能，同时也支持多GPU/TPU分布式训练和混合精度。
# 也支持混合精度训练，可以在训练的配置 TrainingArguments 中，设置 fp16 = True。
# 即Trainer API是一个封装好的训练器（Transformers库内置的小框架，如果是Tensorflow，则是TFTrainer）。
# ————————————————

from transformers import TrainingArguments
from datasets import load_metric
from transformers import Trainer
import numpy as np

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, matthews_corrcoef, accuracy_score, precision_recall_fscore_support

def simple_accuracy(preds, labels):
    return float((preds == labels).mean())


def acc_and_f1(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1 = float(f1_score(y_true=labels, y_pred=preds))
    return {
        "accuracy": acc,
        "f1": f1,
    }


def pearson_and_spearman(preds, labels):
    pearson_corr = float(pearsonr(preds, labels)[0])
    spearman_corr = float(spearmanr(preds, labels)[0])
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
    }

def compute_metrics(eval_preds):
    '''计算评价指标'''
    logits, labels = eval_preds
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

from datasets import load_dataset
raw_datasets = load_dataset('text', data_files={'train': rf"D:/Users/{USERNAME}/data/sentiment/sentiment.train.txt",
                                           'test': rf"D:/Users/{USERNAME}/data/sentiment/sentiment.test.txt",
                                           "valid": rf"D:/Users/{USERNAME}/data/sentiment/sentiment.valid.txt"})

def tokenize_function(example):
    text_list = []
    labels = []
    for raw_text in example['text']:
        if not raw_text or not raw_text.strip():
            continue
        text, label = raw_text.split('\t')
        text_list.append(text)
        labels.append(int(label))
    tokenized = tokenizer(text_list, padding='max_length', truncation=True, max_length=128)
    tokenized['labels'] = labels
    return tokenized

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

# from transformers import DataCollatorWithPadding
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# data_collator的作用是自动将同一个batch的句子padding成同一长度，而不是一次性padding整个数据集。

# 定义模型训练参数，这里未修改预训练模型结构，仅仅使用极低学习率微调模型
training_args = TrainingArguments("test-trainer",
                                  evaluation_strategy="epoch", # 有三个选项: “no”：训练时不做任何评估; “step”：每个 eval_steps 完成（并记录）评估; “epoch”：在每个 epoch 结束时进行评估。
                                  learning_rate=1e-5,
                                  load_best_model_at_end=True, # 训练结束时加载在训练期间找到的最佳模型
                                  save_strategy='epoch',  # 在训练期间采用的检查点保存策略。可能的值为：“no”：训练期间不保存；“epoch”：在每个epoch结束时进行保存；“steps”：每个step保存一次。
                                  per_device_train_batch_size=32,
                                  num_train_epochs=5,  # 训练轮数
                                  )

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
    # data_collator=data_collator,
    # tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

def main():
    pass


if __name__ == '__main__':
    main()
