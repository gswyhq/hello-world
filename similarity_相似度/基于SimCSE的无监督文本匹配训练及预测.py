# !/usr/bin/env python3

"""
文本匹配多用于计算两个文本之间的相似度，该示例会基于 ESimCSE 实现一个无监督的文本匹配模型的训练流程。
资料来源：https://github.com/HarderThenHarder/transformers_tasks/tree/main/text_matching/unsupervised/simcse
"""

import os
import time
import argparse
from functools import partial

import torch
import evaluate
from scipy import stats
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, default_data_collator, get_scheduler
import torch.nn as nn
import torch.nn.functional as F
import random
import traceback
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

def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        mode='train'
):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 数据样本（不同mode下的数据集不一样）, e.g. -> {
                                                "text": '蛋黄吃多了有什么坏处',                               # train mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处	0',  # evaluate mode
                                                        or '蛋黄吃多了有什么坏处	吃鸡蛋白过多有什么坏处',     # inference mode
                                            }
        mode (bool): 数据集格式 -> 'train': （无监督）训练集模式，一行只有一句话；
                                'evaluate': 验证集训练集模式，两句话 + 标签
                                'inference': 推理集模式，两句话。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'query_input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'query_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'doc_input_ids': [[101, 2648, ...], [101, 3342, ...]],
                            'doc_token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'labels': [1, 0, ...]
                        }
    """
    tokenized_output = {
        'query_input_ids': [],
        'query_token_type_ids': [],
        'doc_input_ids': [],
        'doc_token_type_ids': []
    }

    for example in examples['text']:
        try:
            if mode == 'train':
                query = doc = example.strip()
            elif mode == 'evaluate':
                query, doc, label = example.strip().split('\t')
            elif mode == 'inference':
                query, doc = example.strip().split('\t')
            else:
                raise ValueError(f'No mode called {mode}, expected in ["train", "evaluate", "inference"].')

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
            print(f'{examples["text"]} -> {traceback.format_exc()}')
            exit()

        tokenized_output['query_input_ids'].append(query_encoded_inputs["input_ids"])
        tokenized_output['query_token_type_ids'].append(query_encoded_inputs["token_type_ids"])
        tokenized_output['doc_input_ids'].append(doc_encoded_inputs["input_ids"])
        tokenized_output['doc_token_type_ids'].append(doc_encoded_inputs["token_type_ids"])
        if mode == 'evaluate':
            if 'labels' not in tokenized_output:
                tokenized_output['labels'] = []
            tokenized_output['labels'].append(int(label))

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v)

    return tokenized_output


def word_repetition(
        input_ids,
        token_type_ids,
        dup_rate=0.32,
        min_dup_sentence_len_threshold=5,
        device='cpu'
) -> torch.tensor:
    """
    随机重复单词策略，用于在正例样本中添加噪声。

    Args:
        input_ids (_type_): y
        token_type_ids (_type_): _description_
        dup_rate (float, optional): 重复字数占总句子长度的比例. Defaults to 0.32.
        min_dup_sentence_len_threshold (int): 触发随机重复的最小句子长度
        device (str): 使用设备

    Returns:
        _type_: 随机重复后的 input_ids 和 token_type_ids.

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a337ced850ee9f4fabc8c3f304a2f3bf9055013e/examples/text_matching/simcse/data.py#L97
    """
    input_ids = input_ids.numpy().tolist()
    token_type_ids = token_type_ids.numpy().tolist()

    batch_size, seq_len = len(input_ids), len(input_ids[0])
    repetitied_input_ids = []
    repetitied_token_type_ids = []
    rep_seq_len = seq_len

    for batch_id in range(batch_size):
        cur_input_id = input_ids[batch_id]
        actual_len = np.count_nonzero(cur_input_id)  # 去掉padding token，求句子真实长度
        dup_word_index = []

        if actual_len > min_dup_sentence_len_threshold:  # 句子太短则不进行随机重复
            dup_len = random.randint(a=0, b=max(2, int(dup_rate * actual_len)))
            dup_word_index = random.sample(list(range(1, actual_len - 1)), k=dup_len)  # 不重复[CLS]和[SEP]

        r_input_id = []
        r_token_type_id = []
        for idx, word_id in enumerate(cur_input_id):  # 「今天很开心」 -> 「今今天很开开心」
            if idx in dup_word_index:
                r_input_id.append(word_id)
                r_token_type_id.append(token_type_ids[batch_id][idx])
            r_input_id.append(word_id)
            r_token_type_id.append(token_type_ids[batch_id][idx])
        after_dup_len = len(r_input_id)
        repetitied_input_ids.append(r_input_id)
        repetitied_token_type_ids.append(r_token_type_id)

        if after_dup_len > rep_seq_len:
            rep_seq_len = after_dup_len

    for batch_id in range(batch_size):  # padding到最大长度
        after_dup_len = len(repetitied_input_ids[batch_id])
        pad_len = rep_seq_len - after_dup_len
        repetitied_input_ids[batch_id] += [0] * pad_len
        repetitied_token_type_ids[batch_id] += [0] * pad_len

    return torch.tensor(repetitied_input_ids).to(device), torch.tensor(repetitied_token_type_ids).to(device)


class SimCSE(nn.Module):
    """
    SimCSE模型，采用ESimCSE方式实现。

    Args:
        nn (_type_): _description_
    """

    def __init__(
            self,
            encoder,
            dropout=None,
            margin=0.0,
            scale=20,
            output_embedding_dim=256):
        """
        Init func.

        Args:
            encoder (_type_): pretrained model, 默认使用 ernie3.0。
            dropout (_type_, optional): hidden_state 的 dropout 比例。
            margin (float, optional): 为所有正例的余弦相似度降低的值. Defaults to 0.0.
            scale (int, optional): 缩放余弦相似度的值便于模型收敛. Defaults to 20.
            output_embedding_dim (_type_, optional): 输出维度（是否将默认的768维度压缩到更小的维度）. Defaults to 256.
        """
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)
        self.output_embedding_dim = output_embedding_dim
        if output_embedding_dim > 0:
            self.embedding_reduce_linear = nn.Linear(768, output_embedding_dim)
        self.margin = margin
        self.scale = scale
        self.criterion = torch.nn.CrossEntropyLoss()

    def get_pooled_embedding(
            self,
            input_ids,
            token_type_ids=None,
            attention_mask=None
    ) -> torch.tensor:
        """
        获得句子的embedding，如果有压缩，则返回压缩后的embedding。

        Args:
            input_ids (_type_): _description_
            token_type_ids (_type_, optional): _description_. Defaults to None.
            attention_mask (_type_, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: (batch, self.output_embedding_dim)
        """
        pooled_embedding = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )["pooler_output"]

        if self.output_embedding_dim > 0:
            pooled_embedding = self.embedding_reduce_linear(pooled_embedding)  # 维度压缩
        pooled_embedding = self.dropout(pooled_embedding)  # dropout
        pooled_embedding = F.normalize(pooled_embedding, p=2, dim=-1)

        return pooled_embedding

    def forward(
            self,
            query_input_ids: torch.tensor,
            query_token_type_ids: torch.tensor,
            query_attention_mask: torch.tensor,
            doc_input_ids: torch.tensor,
            doc_token_type_ids: torch.tensor,
            doc_attention_mask: torch.tensor,
            device='cpu'
    ) -> torch.tensor:
        """
        传入query/doc对，构建正/负例并计算contrastive loss。

        Args:
            query_input_ids (torch.LongTensor): (batch, seq_len)
            query_token_type_ids (torch.LongTensor): (batch, seq_len)
            doc_input_ids (torch.LongTensor): (batch, seq_len)
            doc_token_type_ids (torch.LongTensor): (batch, seq_len)
            device (str): 使用设备

        Returns:
            torch.tensor: (1)
        """
        query_embedding = self.get_pooled_embedding(
            input_ids=query_input_ids,
            token_type_ids=query_token_type_ids,
            attention_mask=query_attention_mask
        )  # (batch, self.output_embedding_dim)

        doc_embedding = self.get_pooled_embedding(
            input_ids=doc_input_ids,
            token_type_ids=doc_token_type_ids,
            attention_mask=doc_attention_mask,
        )  # (batch, self.output_embedding_dim)

        cos_sim = torch.matmul(query_embedding, doc_embedding.T)  # (batch, batch)
        margin_diag = torch.diag(torch.full(  # (batch, batch), 只有对角线等于margin值的对角矩阵
            size=[query_embedding.size()[0]],
            fill_value=self.margin
        )).to(device)
        cos_sim = cos_sim - margin_diag  # 主对角线（正例）的余弦相似度都减掉 margin
        cos_sim *= self.scale  # 缩放相似度，便于收敛

        labels = torch.arange(  # 只有对角上为正例，其余全是负例，所以这个batch样本标签为 -> [0, 1, 2, ...]
            0,
            query_embedding.size()[0],
            dtype=torch.int64
        ).to(device)
        loss = self.criterion(cos_sim, labels)

        return loss


def test_model_loss():
    """测试SimCSE训练输出loss"""
    from rich import print
    from transformers import AutoTokenizer, AutoModel
    USERNAME = os.getenv("USERNAME")

    device = 'cpu'
    model_dir = f"D:\\Users\\{USERNAME}\\data/bert-base-chinese" # https://huggingface.co/bert-base-chinese
    # model_dir = f"D:\\Users\\{USERNAME}\\data/ernie-3.0-base-zh" # https://huggingface.co/nghuyong/ernie-3.0-base-zh
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    encoder = AutoModel.from_pretrained(model_dir)
    model = SimCSE(encoder).to(device)
    model.eval()
    sentences = ['一个男孩在打篮球', '他是蔡徐坤吗', '他怎么这么帅呢']
    query_inputs = tokenizer(
        sentences,
        return_tensors='pt',
        max_length=20,
        padding='max_length'
    )
    doc_inputs = tokenizer(
        sentences,
        return_tensors='pt',
        max_length=20,
        padding='max_length'
    )

    # * 测试SimCSE训练输出loss
    loss = model(query_input_ids=query_inputs['input_ids'],
                 query_token_type_ids=query_inputs['token_type_ids'],
                 query_attention_mask=query_inputs['attention_mask'],
                 doc_input_ids=doc_inputs['input_ids'],
                 doc_token_type_ids=doc_inputs['token_type_ids'],
                 doc_attention_mask=doc_inputs['attention_mask'])
    print('loss: ', loss)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--dropout", default=0.1, type=float, help="Dropout for pretrained model encoder.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.0, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--duplicate_ratio", default=0.32, type=float, help="random duplicate text ratio.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()

# 类似于SummaryWriter功能, iSummaryWriter工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
# https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/iTrainingLogger.py
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader, cosine_similarity_threshold=0.5):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        cosine_similarity_threshold (float): 余弦相似度阈值，大于阈值则算作匹配样本（1），否则算作不相似样本（0）
    """
    model.eval()
    sims, labels = [], []
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            query_input_ids, query_token_type_ids, \
                doc_input_ids, doc_token_type_ids = batch["query_input_ids"], batch["query_token_type_ids"], \
                                                    batch["doc_input_ids"], batch["doc_token_type_ids"]
            
            query_embedding = model.get_pooled_embedding(
                query_input_ids.to(args.device), 
                query_token_type_ids.to(args.device)
            )                                                                                   # (batch, hidden_dim)
            doc_embedding = model.get_pooled_embedding(
                doc_input_ids.to(args.device), 
                doc_token_type_ids.to(args.device)
            )                                                                                   # (batch, hidden_dim)
            cos_sim = torch.nn.functional.cosine_similarity(query_embedding, doc_embedding)     # (batch)
            predictions = [1 if p else 0 for p in cos_sim > cosine_similarity_threshold]        # (batch)
            metric.add_batch(predictions=predictions, references=batch["labels"].cpu().tolist())
            sims.extend(cos_sim.tolist())
            labels.extend(batch["labels"].cpu().tolist())
    eval_metric = metric.compute()
    spearman_corr = stats.spearmanr(labels, sims).correlation
    model.train()
    return eval_metric['accuracy'], eval_metric['precision'], eval_metric['recall'], eval_metric['f1'], spearman_corr


def train():
    encoder = AutoModel.from_pretrained(args.model)
    model = SimCSE(encoder, dropout=args.dropout)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    
    train_dataset = dataset["train"]
    train_convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_seq_len=args.max_seq_len,
        mode='train'
    )
    train_dataset = train_dataset.map(train_convert_func, batched=True)
    train_dataloader = DataLoader(
        train_dataset, 
        shuffle=True, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )

    eval_dataset = dataset["dev"]
    eval_convert_func = partial(
        convert_example, 
        tokenizer=tokenizer, 
        max_seq_len=args.max_seq_len,
        mode='evaluate'
    )
    eval_dataset = eval_dataset.map(eval_convert_func, batched=True)
    eval_dataloader = DataLoader(eval_dataset, 
        shuffle=False, 
        collate_fn=default_data_collator, 
        batch_size=args.batch_size
    )

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
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
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
    tic_train = time.time()
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            query_input_ids, query_token_type_ids, \
                doc_input_ids, doc_token_type_ids = batch["query_input_ids"], batch["query_token_type_ids"], \
                                                    batch["doc_input_ids"], batch["doc_token_type_ids"]
            
            if args.duplicate_ratio > 0:
                query_input_ids, query_token_type_ids = word_repetition(
                    query_input_ids,
                    query_token_type_ids,
                    device=args.device
                )
                doc_input_ids, doc_token_type_ids = word_repetition(
                    doc_input_ids,
                    doc_token_type_ids,
                    device=args.device
                )
            
            loss = model(
                query_input_ids=query_input_ids,
                query_token_type_ids=query_token_type_ids,
                doc_input_ids=doc_input_ids,
                doc_token_type_ids=doc_token_type_ids,
                device=args.device
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
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

                acc, precision, recall, f1, spearman_corr = evaluate_model(model, metric, eval_dataloader)
                writer.add_scalar('eval/accuracy', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.add_scalar('eval/spearman_corr', spearman_corr, global_step)
                writer.record()
                
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f, spearman_corr: %.5f" % (precision, recall, f1, spearman_corr))
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


if __name__ == '__main__':
    from rich import print
    train()

##############################################################################################################################
# 训练模型命令：
# python train.py \
#     --model "nghuyong/ernie-3.0-base-zh" \
#     --train_path "data/LCQMC/train.txt" \
#     --dev_path "data/LCQMC/dev.tsv" \
#     --save_dir "checkpoints/LCQMC" \
#     --img_log_dir "logs/LCQMC" \
#     --img_log_name "ERNIE-ESimCSE" \
#     --learning_rate 1e-5 \
#     --dropout 0.3 \
#     --batch_size 64 \
#     --max_seq_len 64 \
#     --valid_steps 400 \
#     --logging_steps 50 \
#     --num_train_epochs 8 \
#     --device "cuda:2"
# 注意：训练是无监督训练，故而训练集train_path是每一行单个句子；但验证集dev_path每一行是两个句子及其对应的label
# 若想快速复现，相关数据集，可在 https://github.com/HarderThenHarder/transformers_tasks/tree/main/text_matching/unsupervised/simcse/data/LCQMC 下载；
