# !/usr/bin/env python3
"""

文本匹配(Sentence Transformer)训练模型。
资料来源：https://github.com/HarderThenHarder/transformers_tasks/tree/main/text_matching/supervised
"""

import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler

from typing import List
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class iSummaryWriter(object):

    def __init__(self, log_path: str, log_name: str, params=None, extention='.png', max_columns=2,
                 log_title=None, figsize=None):
        """
        初始化函数，创建日志类。

        Args:
            log_path (str): 日志存放文件夹
            log_name (str): 日志文件名
            parmas (list): 要记录的参数名字列表，e.g. -> ["loss", "reward", ...]
            extension (str): 图片存储格式
            max_columns (int): 一行中排列几张图，默认为一行2张（2个变量）的图。
        """
        if params is None:
            params = []
        self.log_path = log_path
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log_name = log_name
        self.extention = extention
        self.max_param_index = -1
        self.max_columns_threshold = max_columns
        self.figsize = figsize
        self.params_dict = self.create_params_dict(params)
        self.log_title = log_title
        self.init_plt()
        self.update_ax_list()

    def init_plt(self) -> None:
        plt.style.use('seaborn-darkgrid')

    def create_params_dict(self, params: list) -> dict:
        """
        根据传入需要记录的变量名列表，创建监控变量字典。

        Args:
            params (list): 监控变量名列表

        Returns:
            dict: 监控变量名字典 -> {
                'loss': {'values': [0.44, 0.32, ...], 'epochs': [10, 20, ...], 'index': 0},
                'reward': {'values': [10.2, 13.2, ...], 'epochs': [10, 20, ...], 'index': 1},
                ...
            }
        """
        params_dict = {}
        for i, param in enumerate(params):
            params_dict[param] = {'values': [], 'epochs': [], 'index': i}
            self.max_param_index = i
        return params_dict

    def update_ax_list(self) -> None:
        """
        根据当前的监控变量字典，为每一个变量分配一个图区。
        """
        # * 重新计算每一个变量对应的图幅索引
        params_num = self.max_param_index + 1
        if params_num <= 0:
            return

        self.max_columns = params_num if params_num < self.max_columns_threshold else self.max_columns_threshold
        max_rows = (params_num - 1) // self.max_columns + 1   # * 所有变量最多几行
        figsize = self.figsize if self.figsize else (self.max_columns * 6,max_rows * 3)    # 根据图个数计算整个图的figsize
        self.fig, self.axes = plt.subplots(max_rows, self.max_columns, figsize=figsize)

        # * 如果只有一行但又不止一个图，需要手动reshape成(1, n)的形式
        if params_num > 1 and len(self.axes.shape) == 1:
            self.axes = np.expand_dims(self.axes, axis=0)

        # * 重新设置log标题
        log_title = self.log_title if self.log_title else '[Training Log] {}'.format(
            self.log_name)
        self.fig.suptitle(log_title, fontsize=15)

    def add_scalar(self, param: str, value: float, epoch: int) -> None:
        """
        添加一条新的变量值记录。

        Args:
            param (str): 变量名，e.g. -> 'loss'
            value (float): 此时的值。
            epoch (int): 此时的epoch数。
        """
        # * 如果该参数是第一次加入，则将该参数加入到监控变量字典中
        if param not in self.params_dict:
            self.max_param_index += 1
            self.params_dict[param] = {'values': [],
                                       'epochs': [], 'index': self.max_param_index}
            self.update_ax_list()

        self.params_dict[param]['values'].append(value)
        self.params_dict[param]['epochs'].append(epoch)

    def record(self, dpi=200) -> None:
        """
        调用该接口，对该类中目前所有监控的变量状态进行一次记录，将结果保存到本地文件中。
        """
        for param, param_elements in self.params_dict.items():
            param_index = param_elements["index"]
            param_row, param_column = param_index // self.max_columns, param_index % self.max_columns
            ax = self.axes[param_row, param_column] if self.max_param_index > 0 else self.axes
            # ax.set_title(param)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(param)
            ax.plot(self.params_dict[param]['epochs'],
                    self.params_dict[param]['values'],
                    color='darkorange')

        plt.savefig(os.path.join(self.log_path,
                    self.log_name + self.extention), dpi=dpi)


def test_iSummaryWriter():
    '''对iSummaryWriter工具测试'''
    import random
    import time

    n_epochs = 10
    log_path, log_name = './', 'test'
    writer = iSummaryWriter(log_path=log_path, log_name=log_name)
    for i in range(n_epochs):
        loss, reward = 100 - random.random() * i, random.random() * i
        writer.add_scalar('loss', loss, i)
        writer.add_scalar('reward', reward, i)
        writer.add_scalar('random', reward, i)
        writer.record()
        print("Log has been saved at: {}".format(
            os.path.join(log_path, log_name)))
        time.sleep(3)


def convert_dssm_example(examples: dict, tokenizer, max_seq_len: int):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '今天天气好吗	今天天气怎样	1',
                                                            '今天天气好吗	胡歌结婚了吗	0',
                                                            ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'query_input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'query_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'query_attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'doc_input_ids': [[101, 2648, ...], [101, 3342, ...]],
                            'doc_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'doc_attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'query_input_ids': [],
        'query_token_type_ids': [],
        'query_attention_mask': [],
        'doc_input_ids': [],
        'doc_token_type_ids': [],
        'doc_attention_mask': [],
        'labels': []
    }

    for example in examples['text']:
        try:
            query, doc, label = example.split('\t')
            query_encoded_inputs = tokenizer(
                text=query,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
            doc_encoded_inputs = tokenizer(
                text=doc,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'"{example}" -> {traceback.format_exc()}')
            exit()

        tokenized_output['query_input_ids'].append(query_encoded_inputs["input_ids"])
        tokenized_output['query_token_type_ids'].append(query_encoded_inputs["token_type_ids"])
        tokenized_output['query_attention_mask'].append(query_encoded_inputs["attention_mask"])
        tokenized_output['doc_input_ids'].append(doc_encoded_inputs["input_ids"])
        tokenized_output['doc_token_type_ids'].append(doc_encoded_inputs["token_type_ids"])
        tokenized_output['doc_attention_mask'].append(doc_encoded_inputs["attention_mask"])
        tokenized_output['labels'].append(int(label))

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


class SentenceTransformer(nn.Module):
    """
    Sentence Transomer实现, 双塔网络, 精度适中, 计算速度快。
    Paper Reference: https://arxiv.org/pdf/1908.10084.pdf
    Code Reference: https://github.com/PaddlePaddle/PaddleNLP/blob/develop/examples/text_matching/sentence_transformers/model.py

    Args:
        nn (_type_): _description_
    """

    def __init__(self, encoder, dropout=0.1):
        """
        init func.

        Args:
            encoder (transformers.PretrainedModel): backbone, 默认使用 ernie 3.0
            dropout (float): dropout.
        """
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.classifier = nn.Linear(768 * 3, 2)  # concat(u, v, u - v) -> 2, 相似/不相似

    def forward(
            self,
            query_input_ids: torch.tensor,
            query_token_type_ids: torch.tensor,
            query_attention_mask: torch.tensor,
            doc_embeddings: torch.tensor,
    ) -> torch.tensor:
        """
        forward 函数，输入query句子和doc_embedding向量，将query句子过一遍模型得到
        query embedding再和doc_embedding做二分类。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)
            doc_embedding (torch.LongTensor): 所有需要匹配的doc_embedding -> (batch, doc_embedding_numbers, hidden_size)

        Returns:
            torch.tensor: embedding_match_logits -> (batch, doc_embedding_numbers, 2)
        """
        query_embedding = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )["last_hidden_state"]  # (batch, seq_len, hidden_size)

        query_attention_mask = torch.unsqueeze(query_attention_mask, dim=-1)  # (batch, seq_len, 1)
        query_embedding = query_embedding * query_attention_mask  # (batch, seq_len, hidden_size)
        query_sum_embedding = torch.sum(query_embedding, dim=1)  # (batch, hidden_size)
        query_sum_mask = torch.sum(query_attention_mask, dim=1)  # (batch, 1)
        query_mean = query_sum_embedding / query_sum_mask  # (batch, hidden_size)

        query_mean = query_mean.unsqueeze(dim=1).repeat(1, doc_embeddings.size()[1],
                                                        1)  # (batch, doc_embedding_numbers, hidden_size)
        sub = torch.abs(torch.subtract(query_mean, doc_embeddings))  # (batch, doc_embedding_numbers, hidden_size)
        concat = torch.cat([query_mean, doc_embeddings, sub], dim=-1)  # (batch, doc_embedding_numbers, hidden_size * 3)
        logits = self.classifier(concat)  # (batch, doc_embedding_numbers, 2)
        return logits

    def get_embedding(
            self,
            input_ids: torch.tensor,
            token_type_ids: torch.tensor,
            attention_mask: torch.tensor,
    ) -> torch.tensor:
        """
        输入句子，返回这个句子的embedding，用于事先计算doc embedding并存储。

        Args:
            input_ids (torch.LongTensor): (batch, seq_len)
            token_type_ids (torch.LongTensor): (batch, seq_len)
            attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: embedding向量 -> (batch, hidden_size)
        """
        embedding = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )["last_hidden_state"]  # (batch, seq_len, hidden_size)

        attention_mask = torch.unsqueeze(attention_mask, dim=-1)  # (batch, seq_len, 1)
        embedding = embedding * attention_mask  # (batch, seq_len, hidden_size)
        sum_embedding = torch.sum(embedding, dim=1)  # (batch, hidden_size)
        sum_mask = torch.sum(attention_mask, dim=1)  # (batch, 1)
        mean = sum_embedding / sum_mask  # (batch, hidden_size)
        return mean

    def get_similarity_label(
            self,
            query_input_ids: torch.tensor,
            query_token_type_ids: torch.tensor,
            query_attention_mask: torch.tensor,
            doc_input_ids: torch.tensor,
            doc_token_type_ids: torch.tensor,
            doc_attention_mask: torch.tensor
    ) -> torch.tensor:
        """
        forward 函数，输入query和doc的向量，返回两个向量相似/不相似的二维向量。

        Args:
            query_input_ids (torch.LongTensor): (batch, seq_len)
            query_token_type_ids (torch.LongTensor): (batch, seq_len)
            query_attention_mask (torch.LongTensor): (batch, seq_len)
            doc_input_ids (torch.LongTensor): (batch, seq_len)
            doc_token_type_ids (torch.LongTensor): (batch, seq_len)
            doc_attention_mask (torch.LongTensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, 2)
        """
        query_embedding = self.encoder(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )["last_hidden_state"]  # (batch, seq_len, hidden_size)
        query_embedding = self.dropout(query_embedding)  # (batch, seq_len, hidden_size)
        query_attention_mask = torch.unsqueeze(query_attention_mask, dim=-1)  # (batch, seq_len, 1)
        query_embedding = query_embedding * query_attention_mask  # (batch, seq_len, hidden_size)
        query_sum_embedding = torch.sum(query_embedding, dim=1)  # (batch, hidden_size)
        query_sum_mask = torch.sum(query_attention_mask, dim=1)  # (batch, 1)
        query_mean = query_sum_embedding / query_sum_mask  # (batch, hidden_size)

        doc_embedding = self.encoder(
            input_ids=doc_input_ids,
            token_type_ids=doc_token_type_ids,
            attention_mask=doc_attention_mask
        )["last_hidden_state"]  # (batch, seq_len, hidden_size)
        doc_embedding = self.dropout(doc_embedding)  # (batch, seq_len, hidden_size)
        doc_attention_mask = torch.unsqueeze(doc_attention_mask, dim=-1)  # (batch, seq_len, 1)
        doc_embedding = doc_embedding * doc_attention_mask  # (batch, seq_len, hidden_size)
        doc_sum_embdding = torch.sum(doc_embedding, dim=1)  # (batch, hidden_size)
        doc_sum_mask = torch.sum(doc_attention_mask, dim=1)  # (batch, 1)
        doc_mean = doc_sum_embdding / doc_sum_mask  # (batch, hidden_size)

        sub = torch.abs(torch.subtract(query_mean, doc_mean))  # (batch, hidden_size)
        concat = torch.cat([query_mean, doc_mean, sub], dim=-1)  # (batch, hidden_size * 3)
        logits = self.classifier(concat)  # (batch, 2)

        return logits


def test_model_loss():
    from rich import print
    from transformers import AutoTokenizer, AutoModel

    device = 'cpu'
    tokenizer = AutoTokenizer.from_pretrained('nghuyong/ernie-3.0-base-zh')
    encoder = AutoModel.from_pretrained('nghuyong/ernie-3.0-base-zh')
    model = SentenceTransformer(encoder).to(device)

    example = {
        "text": [
            '今天天气好吗	今天天气怎样	1'
        ]
    }
    batch = convert_dssm_example(example, tokenizer, 10)
    print(batch)

    # * 测试sentence bert训练输出logits
    output = model.get_similarity_label(query_input_ids=torch.LongTensor(batch['query_input_ids']),
                            query_token_type_ids=torch.LongTensor(batch['query_token_type_ids']),
                            query_attention_mask=torch.LongTensor(batch['query_attention_mask']),
                            doc_input_ids=torch.LongTensor(batch['doc_input_ids']),
                            doc_token_type_ids=torch.LongTensor(batch['doc_token_type_ids']),
                            doc_attention_mask=torch.LongTensor(batch['doc_attention_mask']))
    print(output)

    # * 测试sentence bert的inference功能
    output = model(query_input_ids=torch.LongTensor(batch['query_input_ids']).to(device),
                    query_token_type_ids=torch.LongTensor(batch['query_token_type_ids']).to(device),
                    query_attention_mask=torch.LongTensor(batch['query_attention_mask']).to(device),
                    doc_embeddings=torch.randn(1, 10, 768).to(device))

    print(output)

    # * 测试sentence bert获取sentence embedding功能
    output = model.get_embedding(input_ids=torch.LongTensor(batch['query_input_ids']).to(device),
                                 token_type_ids=torch.LongTensor(batch['query_token_type_ids']).to(device),
                                 attention_mask=torch.LongTensor(batch['query_attention_mask']).to(device))
    print(output, output.size())


parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
args = parser.parse_args()

# 类似于SummaryWriter功能, iSummaryWriter工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
# https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/iTrainingLogger.py
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader, global_step):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            logits = model.get_similarity_label(query_input_ids=batch['query_input_ids'].to(args.device),
                            query_token_type_ids=batch['query_token_type_ids'].to(args.device),
                            query_attention_mask=batch['query_attention_mask'].to(args.device),
                            doc_input_ids=batch['doc_input_ids'].to(args.device),
                            doc_token_type_ids=batch['doc_token_type_ids'].to(args.device),
                            doc_attention_mask=batch['doc_attention_mask'].to(args.device))
            predictions = logits.argmax(dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
    eval_metric = metric.compute()
    model.train()
    return eval_metric['accuracy'], eval_metric['precision'], eval_metric['recall'], eval_metric['f1']


def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = SentenceTransformer(encoder)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(convert_dssm_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=5e-5)
    model.to(args.device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    metric = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    criterion = torch.nn.CrossEntropyLoss()
    tic_train = time.time()
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            logits = model.get_similarity_label(query_input_ids=batch['query_input_ids'].to(args.device),
                            query_token_type_ids=batch['query_token_type_ids'].to(args.device),
                            query_attention_mask=batch['query_attention_mask'].to(args.device),
                            doc_input_ids=batch['doc_input_ids'].to(args.device),
                            doc_token_type_ids=batch['doc_token_type_ids'].to(args.device),
                            doc_attention_mask=batch['doc_attention_mask'].to(args.device))
            labels = batch['labels'].to(args.device)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)

                acc, precision, recall, f1 = evaluate_model(model, metric, eval_dataloader, global_step)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()
                
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                    tokenizer.save_pretrained(cur_save_dir)
                tic_train = time.time()
            
            global_step += 1


if __name__ == '__main__':
    train()

#########################################################################################################################
# 模型训练命令：
# python train_sentence_transformer.py \
#     --model "nghuyong/ernie-3.0-base-zh" \
#     --train_path "data/comment_classify/train.txt" \
#     --dev_path "data/comment_classify/dev.txt" \
#     --save_dir "checkpoints/comment_classify/sentence_transformer" \
#     --img_log_dir "logs/comment_classify" \
#     --img_log_name "Sentence-Ernie" \
#     --batch_size 8 \
#     --max_seq_len 256 \
#     --valid_steps 50 \
#     --logging_steps 10 \
#     --num_train_epochs 10 \
#     --device "cuda:0"
# ```
