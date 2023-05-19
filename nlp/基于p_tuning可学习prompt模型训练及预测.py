# !/usr/bin/env python3
"""

PTuning v1 基于 transformers 实现。
PTuning 是一种自动生成 prompt 模板的算法，属于 few-shot 领域的一个分支，
其优势在于不用人工手动构建 prompt 模板，可以通过模型的自我学习找到最优的 prompt 模板。

Code Reference:
    https://github.com/PaddlePaddle/PaddleNLP/tree/develop/examples/few_shot/p-tuning
    https://github.com/HarderThenHarder/transformers_tasks/tree/main/prompt_tasks/p-tuning

Paper Reference:
    https://arxiv.org/pdf/2103.10385.pdf

"""
import os
import time
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator, get_scheduler

import torch.nn.functional as F
from rich import print
from rich.table import Table
from rich.align import Align
from rich.console import Console

import json
import traceback

import torch
import numpy as np

from typing import List, Union

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix


##############################################################################################################################


def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        p_embedding_num=6,
        train_mode=True,
        return_tensor=False
) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '娱乐	嗨放派怎么停播了',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        p_embedding_num (int): p-tuning token 的个数
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'mask_positions': [[5, 6, ...], [3, 4, ...]],
                            'mask_labels': [[183, 234], [298, 322], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'attention_mask': [],
        'mask_positions': [],  # 记录label的位置（即MASK Token的位置）
        'mask_labels': []  # 记录MASK Token的原始值（即Label值）
    }

    for i, example in enumerate(examples['text']):
        try:
            start_mask_position = 1  # 将 prompt token(s) 插在 [CLS] 之后

            if train_mode:
                label, content = example.strip().split('\t')
            else:
                content = example.strip()

            encoded_inputs = tokenizer(
                text=content,
                truncation=True,
                max_length=max_seq_len,
                padding='max_length')
        except:
            print(f'Error Line {i + 1}: "{example}" -> {traceback.format_exc()}')
            continue

        input_ids = encoded_inputs['input_ids']
        mask_tokens = ['[MASK]'] * max_label_len  # 1.生成 MASK Tokens, 和label长度一致
        mask_ids = tokenizer.convert_tokens_to_ids(mask_tokens)  # token 转 id

        p_tokens = ["[unused{}]".format(i + 1) for i in range(p_embedding_num)]  # 2.构建 prompt token(s)
        p_tokens_ids = tokenizer.convert_tokens_to_ids(p_tokens)  # token 转 id

        tmp_input_ids = input_ids[:-1]
        tmp_input_ids = tmp_input_ids[
                        :max_seq_len - len(mask_ids) - len(p_tokens_ids) - 1]  # 根据最大长度-p_token长度-label长度，裁剪content的长度
        tmp_input_ids = tmp_input_ids[:start_mask_position] + mask_ids + tmp_input_ids[
                                                                         # 3.插入 MASK -> [CLS][MASK][MASK]世界杯...[SEP]
                                                                         start_mask_position:]
        input_ids = tmp_input_ids + [input_ids[-1]]  # 补上[SEP]
        input_ids = p_tokens_ids + input_ids  # 4.插入 prompt -> [unused1][unused2]...[CLS][MASK]...[SEP]
        mask_positions = [len(p_tokens_ids) + start_mask_position + i for  # 将 Mask Tokens 的位置记录下来
                          i in range(max_label_len)]

        tokenized_output['input_ids'].append(input_ids)
        if 'token_type_ids' in tokenized_output:  # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
            if 'token_type_ids' not in tokenized_output:
                tokenized_output['token_type_ids'] = [encoded_inputs['token_type_ids']]
            else:
                tokenized_output['token_type_ids'].append(encoded_inputs['token_type_ids'])
        tokenized_output['attention_mask'].append(encoded_inputs['attention_mask'])
        tokenized_output['mask_positions'].append(mask_positions)

        if train_mode:
            mask_labels = tokenizer(text=label)  # label token 转 id
            mask_labels = mask_labels['input_ids'][1:-1]  # 丢掉[CLS]和[SEP]
            mask_labels = mask_labels[:max_label_len]
            mask_labels += [tokenizer.pad_token_id] * (max_label_len - len(mask_labels))  # 将 label 补到最长
            tokenized_output['mask_labels'].append(mask_labels)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


def mlm_loss2(
        logits: torch.tensor,
        mask_positions: torch.tensor,
        mask_labels: torch.tensor,
        cross_entropy_criterion: torch.nn.CrossEntropyLoss,
        masked_lm_scale=1.0,
        device='cpu'
) -> torch.tensor:
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        mask_labels (torch.tensor): mask token的label -> (batch, mask_label_num)
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        masked_lm_scale (float): scale 参数
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    mask_positions_after_reshaped = []
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)

    logits = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]  # (batch * label_num, vocab_size)
    mask_labels = mask_labels.reshape(-1, 1).squeeze()  # (batch * label_num)
    loss = cross_entropy_criterion(mask_logits, mask_labels)

    return loss / masked_lm_scale


def mlm_loss(
        logits: torch.tensor,
        mask_positions: torch.tensor,
        sub_mask_labels: list,
        cross_entropy_criterion: torch.nn.CrossEntropyLoss,
        masked_lm_scale=1.0,
        device='cpu'
) -> torch.tensor:
    """
    计算指定位置的mask token的output与label之间的cross entropy loss。

    Args:
        logits (torch.tensor): 模型原始输出 -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置  -> (batch, mask_label_num)
        sub_mask_labels (list): mask token的sub label, 由于每个label的sub_label数目不同，所以这里是个变长的list,
                                    e.g. -> [
                                        [[2398, 3352]],
                                        [[2398, 3352], [3819, 3861]]
                                    ]
        cross_entropy_criterion (CrossEntropyLoss): CE Loss计算器
        masked_lm_scale (float): scale 参数
        device (str): cpu还是gpu

    Returns:
        torch.tensor: CE Loss
    """
    batch_size, seq_len, vocab_size = logits.size()
    loss = None
    for single_logits, single_sub_mask_labels, single_mask_positions in zip(logits, sub_mask_labels, mask_positions):
        single_mask_logits = single_logits[single_mask_positions]  # (mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.repeat(len(single_sub_mask_labels), 1,
                                                       1)  # (sub_label_num, mask_label_num, vocab_size)
        single_mask_logits = single_mask_logits.reshape(-1, vocab_size)  # (sub_label_num * mask_label_num, vocab_size)
        single_sub_mask_labels = torch.LongTensor(single_sub_mask_labels).to(device)  # (sub_label_num, mask_label_num)
        single_sub_mask_labels = single_sub_mask_labels.reshape(-1, 1).squeeze()  # (sub_label_num * mask_label_num)
        cur_loss = cross_entropy_criterion(single_mask_logits, single_sub_mask_labels)
        cur_loss = cur_loss / len(single_sub_mask_labels)
        if not loss:
            loss = cur_loss
        else:
            loss += cur_loss
    loss = loss / batch_size  # (1,)
    return loss / masked_lm_scale


def convert_logits_to_ids(
        logits: torch.tensor,
        mask_positions: torch.tensor
) -> torch.LongTensor:
    """
    输入Language Model的词表概率分布（LMModel的logits），将mask_position位置的
    token logits转换为token的id。

    Args:
        logits (torch.tensor): model output -> (batch, seq_len, vocab_size)
        mask_positions (torch.tensor): mask token的位置 -> (batch, mask_label_num)

    Returns:
        torch.LongTensor: 对应mask position上最大概率的推理token -> (batch, mask_label_num)
    """
    label_length = mask_positions.size()[1]  # 标签长度
    batch_size, seq_len, vocab_size = logits.size()
    mask_positions_after_reshaped = []
    for batch, mask_pos in enumerate(mask_positions.detach().cpu().numpy().tolist()):
        for pos in mask_pos:
            mask_positions_after_reshaped.append(batch * seq_len + pos)

    logits = logits.reshape(batch_size * seq_len, -1)  # (batch_size * seq_len, vocab_size)
    mask_logits = logits[mask_positions_after_reshaped]  # (batch * label_num, vocab_size)
    predicate_tokens = mask_logits.argmax(dim=-1)  # (batch * label_num)
    predicate_tokens = predicate_tokens.reshape(-1, label_length)  # (batch, label_num)

    return predicate_tokens


def test_convert():
    from rich import print

    logits = torch.randn(1, 20, 21193)
    mask_positions = torch.LongTensor([
        [3, 4]
    ])
    predicate_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predicate_tokens)

##############################################################################################################################

class Verbalizer(object):
    """
    Verbalizer类，用于将一个Label对应到其子Label的映射。

    Args:
        object (_type_): _description_
    """

    def __init__(self, verbalizer_file: str, tokenizer, max_label_len: int):
        """
        init func.

        Args:
            verbalizer_file (str): verbalizer文件存放地址。
            tokenizer: 用于文本和id之间的转换。
            max_label_len (int): 标签长度，若大于则截断，若小于则补齐
        """
        self.tokenizer = tokenizer
        self.label_dict = self.load_label_dict(verbalizer_file)
        self.max_label_len = max_label_len

    def load_label_dict(self, verbalizer_file: str) -> dict:
        """
        读取本地文件，构建verbalizer字典。

        Args:
            verbalizer_file (str): verbalizer文件存放地址。

        Returns:
            dict -> {
                '体育': ['足球', '篮球', '排球', '乒乓', ...],
                '酒店': ['旅店', '旅馆', '宾馆', '酒店', ...],
                ...
            }
        """
        assert os.path.exists(verbalizer_file), f'Verbalizer File: {verbalizer_file} not exists.'

        label_dict = {}
        with open(verbalizer_file, 'r', encoding='utf8') as f:
            for line in f.readlines():
                label, sub_labels = line.strip().split('\t')
                label_dict[label] = list(set(sub_labels.split(',')))
        return label_dict

    def find_sub_labels(self, label: Union[list, str]) -> dict:
        """
        通过标签找到所有的子标签。

        Args:
            label (Union[list, str]): 标签, 文本型 或 id_list, e.g. -> '体育' or [860, 5509]

        Returns:
            dict -> {
                'sub_labels': ['笔记本', '电脑'],
                'token_ids': [[5011, 6381, 3315], [4510, 5554]]
            }
        """
        if type(label) == list:  # 如果传入为id_list, 则通过tokenizer转回来
            while self.tokenizer.pad_token_id in label:
                label.remove(self.tokenizer.pad_token_id)
            label = ''.join(self.tokenizer.convert_ids_to_tokens(label))

        if label not in self.label_dict:
            raise ValueError(f'Lable Error: "{label}" not in label_dict {list(self.label_dict)}.')

        sub_labels = self.label_dict[label]
        ret = {'sub_labels': sub_labels}
        token_ids = [_id[1:-1] for _id in self.tokenizer(sub_labels)['input_ids']]
        for i in range(len(token_ids)):
            token_ids[i] = token_ids[i][:self.max_label_len]  # 对标签进行截断与补齐
            if len(token_ids[i]) < self.max_label_len:
                token_ids[i] = token_ids[i] + [self.tokenizer.pad_token_id] * (self.max_label_len - len(token_ids[i]))
        ret['token_ids'] = token_ids
        return ret

    def batch_find_sub_labels(self, label: List[Union[list, str]]) -> list:
        """
        批量找到子标签。

        Args:
            label (List[list, str]): 标签列表, [[4510, 5554], [860, 5509]] or ['体育', '电脑']

        Returns:
            list -> [
                        {
                            'sub_labels': ['笔记本', '电脑'],
                            'token_ids': [[5011, 6381, 3315], [4510, 5554]]
                        },
                        ...
                    ]
        """
        return [self.find_sub_labels(l) for l in label]

    def get_common_sub_str(self, str1: str, str2: str):
        """
        寻找最大公共子串。

        Args:
            str1 (_type_): _description_
            str2 (_type_): _description_

        Returns:
            _type_: _description_
        """
        lstr1, lstr2 = len(str1), len(str2)
        record = [[0 for i in range(lstr2 + 1)] for j in range(lstr1 + 1)]
        p, maxNum = 0, 0

        for i in range(lstr1):
            for j in range(lstr2):
                if str1[i] == str2[j]:
                    record[i + 1][j + 1] = record[i][j] + 1
                    if record[i + 1][j + 1] > maxNum:
                        maxNum = record[i + 1][j + 1]
                        p = i + 1

        return str1[p - maxNum:p], maxNum

    def hard_mapping(self, sub_label: str) -> str:
        """
        强匹配函数，当模型生成的子label不存在时，通过最大公共子串找到重合度最高的主label。

        Args:
            sub_label (str): 子label。

        Returns:
            str: 主label。
        """
        label, max_overlap_str = '', 0
        for main_label, sub_labels in self.label_dict.items():
            overlap_num = 0
            for s_label in sub_labels:  # 求所有子label与当前推理label之间的最长公共子串长度
                overlap_num += self.get_common_sub_str(sub_label, s_label)[1]
            if overlap_num >= max_overlap_str:
                max_overlap_str = overlap_num
                label = main_label
        return label

    def find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True) -> dict:
        """
        通过子标签找到父标签。

        Args:
            sub_label (List[Union[list, str]]): 子标签, 文本型 或 id_list, e.g. -> '苹果' or [5741, 3362]
            hard_mapping (bool): 当生成的词语不存在时，是否一定要匹配到一个最相似的label。

        Returns:
            dict -> {
                'label': '水果',
                'token_ids': [3717, 3362]
            }
        """
        if type(sub_label) == list:  # 如果传入为id_list, 则通过tokenizer转回来
            pad_token_id = self.tokenizer.pad_token_id
            while pad_token_id in sub_label:  # 移除[PAD]token
                sub_label.remove(pad_token_id)
            sub_label = ''.join(self.tokenizer.convert_ids_to_tokens(sub_label))

        main_label = '无'
        for label, s_labels in self.label_dict.items():
            if sub_label in s_labels:
                main_label = label
                break

        if main_label == '无' and hard_mapping:
            main_label = self.hard_mapping(sub_label)

        ret = {
            'label': main_label,
            'token_ids': self.tokenizer(main_label)['input_ids'][1:-1]
        }
        return ret

    def batch_find_main_label(self, sub_label: List[Union[list, str]], hard_mapping=True) -> list:
        """
        批量通过子标签找父标签。

        Args:
            sub_label (List[Union[list, str]]): 子标签列表, ['苹果', ...] or [[5741, 3362], ...]

        Returns:
            list: [
                    {
                    'label': '水果',
                    'token_ids': [3717, 3362]
                    },
                    ...
            ]
        """
        return [self.find_main_label(l, hard_mapping) for l in sub_label]


def test_verbalizer():
    from rich import print
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    verbalizer = Verbalizer(
        verbalizer_file='data/comment_classify/verbalizer.txt',
        tokenizer=tokenizer,
        max_label_len=2
    )
    # print(verbalizer.label_dict)
    # label = ['电脑', '衣服']
    # label = [[4510, 5554], [4510, 5554]]
    # ret = verbalizer.batch_find_sub_labels(label)
    # print(ret)

    # sub_label = ['苹果', '牛奶']
    sub_label = [[2506, 2506]]
    ret = verbalizer.batch_find_main_label(sub_label, hard_mapping=True)
    print(ret)

##############################################################################################################################


class RDropLoss(object):
    """
    RDrop Loss 类。

    Args:
        object (_type_): _description_
    """

    def __init__(self, reduction='none'):
        """
        init func.

        Args:
            reduction (str, optional): kl-divergence param. Defaults to 'none'.
        """
        super().__init__()
        if reduction not in ['sum', 'mean', 'none', 'batchmean']:
            raise ValueError(
                "@param reduction must in ['sum', 'mean', 'batchmean', 'none'], "
                "while received {}.".format(reduction))
        self.reduction = reduction

    def compute_kl_loss(
            self,
            logits: torch.tensor,
            logtis2: torch.tensor,
            pad_mask=None,
            device='cpu'
    ) -> torch.tensor:
        """
        输入同一个样本经过两次backbone后的结果，计算KL-Divergence。

        Args:
            logits (torch.tensor): 第一次logits
            logtis2 (torch.tensor): 第二次logits
            pad_mask (torch.tensor): mask向量，用于去掉padding token的影响
            device (str): cpu or gpu

        Returns:
            torch.tensor: _description_
        """
        loss1 = F.kl_div(F.log_softmax(logits, dim=-1),
                         F.softmax(logtis2, dim=-1),
                         reduction=self.reduction)
        loss2 = F.kl_div(F.log_softmax(logtis2, dim=-1),
                         F.softmax(logits, dim=-1),
                         reduction=self.reduction)

        if pad_mask is not None:
            pad_mask = self.generate_mask_tensor(loss1, pad_mask).to(device)
            loss1 = torch.masked_select(loss1, pad_mask)
            loss2 = torch.masked_select(loss2, pad_mask)

        loss = (loss1.sum() + loss2.sum()) / 2
        return loss

    def generate_mask_tensor(
            self,
            loss1: torch.tensor,
            pad_mask: torch.tensor
    ) -> torch.tensor:
        """
        根据二维的attention_mask生成三维的mask矩阵，用于过滤掉loss中
        的padding token的值。

        Args:
            loss1 (torch.tensor): (batch, seq_len, vocab_size)
            pad_mask (torch.tensor): (batch, seq_len)

        Returns:
            torch.tensor: (batch, seq_len, vocab_size)
        """
        mask_tensor = []
        batch, seq_len, vocab_size = loss1.size()
        for batch_idx in range(batch):
            for seq_idx in range(seq_len):
                if pad_mask[batch_idx][seq_idx]:
                    mask_tensor.append([True] * vocab_size)
                else:
                    mask_tensor.append([False] * vocab_size)
        mask_tensor = torch.tensor(mask_tensor).reshape(batch, seq_len, vocab_size)
        return mask_tensor


def test_loss():
    rdrop = RDropLoss()
    loss = torch.randn(2, 5, 3)  # (2, 5, 3)
    pad_mask = torch.LongTensor([
        [1, 1, 0, 0, 0],
        [1, 1, 1, 1, 0]
    ])  # (2, 5)
    pad_mask = rdrop.generate_mask_tensor(loss, pad_mask)
    print(torch.masked_select(loss, pad_mask))


##############################################################################################################################

class ClassEvaluator(object):

    def __init__(self):
        """
        init func.
        """
        self.goldens = []
        self.predictions = []

    def add_batch(self, pred_batch: List[List], gold_batch: List[List]):
        """
        添加一个batch中的prediction和gold列表，用于后续统一计算。

        Args:
            pred_batch (list): 模型预测标签列表, e.g. -> [0, 0, 1, 2, 0, ...] or [['体', '育'], ['财', '经'], ...]
            gold_batch (list): 真实标签标签列表, e.g. -> [1, 0, 1, 2, 0, ...] or [['体', '育'], ['财', '经'], ...]
        """
        assert len(pred_batch) == len(gold_batch), \
            f"@params pred_spans_batch(len: {len(pred_batch)}) does not match @param gold_spans_batch(len: {len(gold_batch)})"

        if type(gold_batch[0]) in [list, tuple]:  # 若遇到多个子标签构成一个标签的情况
            pred_batch = [','.join([str(e) for e in ele]) for ele in
                          pred_batch]  # 将所有的label拼接为一个整label: ['体', '育'] -> '体育'
            gold_batch = [','.join([str(e) for e in ele]) for ele in gold_batch]
        self.goldens.extend(gold_batch)
        self.predictions.extend(pred_batch)

    def compute(self, round_num=2) -> dict:
        """
        根据当前类中累积的变量值，计算当前的P, R, F1。

        Args:
            round_num (int): 计算结果保留小数点后几位, 默认小数点后2位。

        Returns:
            dict -> {
                'accuracy': 准确率,
                'precision': 精准率,
                'recall': 召回率,
                'f1': f1值,
                'class_metrics': {
                    '0': {
                            'precision': 该类别下的precision,
                            'recall': 该类别下的recall,
                            'f1': 该类别下的f1
                        },
                    ...
                }
            }
        """
        classes, class_metrics, res = sorted(list(set(self.goldens) | set(self.predictions))), {}, {}
        res['accuracy'] = round(accuracy_score(self.goldens, self.predictions), round_num)  # 构建全局指标
        res['precision'] = round(precision_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['recall'] = round(recall_score(self.goldens, self.predictions, average='weighted'), round_num)
        res['f1'] = round(f1_score(self.goldens, self.predictions, average='weighted'), round_num)

        try:
            conf_matrix = np.array(confusion_matrix(self.goldens, self.predictions))  # (n_class, n_class)
            assert conf_matrix.shape[0] == len(
                classes), f"confusion_matrix shape ({conf_matrix.shape[0]}) doesn't match labels number ({len(classes)})!"
            for i in range(conf_matrix.shape[0]):  # 构建每个class的指标
                precision = 0 if sum(conf_matrix[:, i]) == 0 else conf_matrix[i, i] / sum(conf_matrix[:, i])
                recall = 0 if sum(conf_matrix[i, :]) == 0 else conf_matrix[i, i] / sum(conf_matrix[i, :])
                f1 = 0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
                class_metrics[classes[i]] = {
                    'precision': round(precision, round_num),
                    'recall': round(recall, round_num),
                    'f1': round(f1, round_num)
                }
            res['class_metrics'] = class_metrics
        except Exception as e:
            print(f'[Warning] Something wrong when calculate class_metrics: {e}')
            print(f'-> goldens: {set(self.goldens)}')
            print(f'-> predictions: {set(self.predictions)}')
            print(f'-> diff elements: {set(self.predictions) - set(self.goldens)}')
            res['class_metrics'] = {}

        return res

    def reset(self):
        """
        重置积累的数值。
        """
        self.goldens = []
        self.predictions = []


def test_metrics():
    from rich import print

    metric = ClassEvaluator()
    metric.add_batch(
        [['财', '经'], ['财', '经'], ['体', '育'], ['体', '育'], ['计', '算', '机']],
        [['体', '育'], ['财', '经'], ['体', '育'], ['计', '算', '机'], ['计', '算', '机']],
    )
    # metric.add_batch(
    #     [0, 0, 1, 1, 0],
    #     [1, 1, 1, 0, 0]
    # )
    print(metric.compute())

##############################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument("--model", default='bert-base-chinese', type=str, help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoints", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=512, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--p_embedding_num", type=int, default=6, help="number of p-embedding")
parser.add_argument("--max_label_len", type=int, default=6, help="max length of label")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--verbalizer", default='Verbalizer File', required=True, type=str, help="verbalizer file.")
args = parser.parse_args()

# 类似于SummaryWriter功能, iSummaryWriter工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
# https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/iTrainingLogger.py
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader, global_step, tokenizer, verbalizer, device):
    """
    在测试集上评估当前模型的训练效果。

    Args:
        model: 当前模型
        metric: 评估指标类(metric)
        data_loader: 测试集的dataloader
        global_step: 当前训练步数
    """
    model.eval()
    metric.reset()

    with torch.no_grad():
        for step, batch in enumerate(data_loader):
            if 'token_type_ids' in batch:
                logits = model(input_ids=batch['input_ids'].to(device),
                                token_type_ids=batch['token_type_ids'].to(device)).logits
            else:                                                                                        # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
                logits = model(input_ids=batch['input_ids'].to(device)).logits
            mask_labels = batch['mask_labels'].numpy().tolist()                                          # (batch, label_num)
            for i in range(len(mask_labels)):                                                            # 去掉label中的[PAD] token
                while tokenizer.pad_token_id in mask_labels[i]:
                    mask_labels[i].remove(tokenizer.pad_token_id)
            mask_labels = [''.join(tokenizer.convert_ids_to_tokens(t)) for t in mask_labels]             # id转文字
            predictions = convert_logits_to_ids(logits, batch['mask_positions']).cpu().numpy().tolist()  # (batch, label_num)
            predictions = verbalizer.batch_find_main_label(predictions)                                  # 找到子label属于的主label
            predictions = [ele['label'] for ele in predictions]
            metric.add_batch(pred_batch=predictions, gold_batch=mask_labels)
    eval_metric = metric.compute()
    model.train()
    return eval_metric['accuracy'], eval_metric['precision'], \
            eval_metric['recall'], eval_metric['f1'], \
            eval_metric['class_metrics']


def train():
    model = AutoModelForMaskedLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    verbalizer = Verbalizer(
        verbalizer_file=args.verbalizer,
        tokenizer=tokenizer,
        max_label_len=args.max_label_len
    )

    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(convert_example, 
                            tokenizer=tokenizer, 
                            max_seq_len=args.max_seq_len,
                            max_label_len=args.max_label_len,
                            p_embedding_num=args.p_embedding_num
                            )
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
    tic_train = time.time()
    metric = ClassEvaluator()
    criterion = torch.nn.CrossEntropyLoss()
    rdrop_loss = RDropLoss()
    global_step, best_f1 = 0, 0
    for epoch in range(args.num_train_epochs):
        for batch in train_dataloader:
            if 'token_type_ids' in batch:
                logits = model(input_ids=batch['input_ids'].to(args.device),
                                token_type_ids=batch['token_type_ids'].to(args.device)).logits
            else:                                                                                        # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
                logits = model(input_ids=batch['input_ids'].to(args.device)).logits
            mask_labels = batch['mask_labels'].numpy().tolist()
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            sub_labels = [ele['token_ids'] for ele in sub_labels]
    
            if args.rdrop_coef > 0:
                logits2 = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device)).logits
                ce_loss = (mlm_loss(logits, batch['mask_positions'].to(args.device), sub_labels, criterion, 1.0, args.device) + \
                            mlm_loss(logits, batch['mask_positions'].to(args.device), sub_labels, criterion, 1.0, args.device)) / 2
                kl_loss = rdrop_loss.compute_kl_loss(logits, logits2, device=args.device)
                loss = ce_loss + kl_loss * args.rdrop_coef
            else:
                loss = mlm_loss(logits, batch['mask_positions'].to(args.device), sub_labels, criterion, 1.0, args.device)
            
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
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))

                acc, precision, recall, f1, class_metrics = evaluate_model(model, 
                                                                        metric, 
                                                                        eval_dataloader, 
                                                                        global_step,
                                                                        tokenizer,
                                                                        verbalizer,
                                                                           args.device)
                writer.add_scalar('eval/acc', acc, global_step)
                writer.add_scalar('eval/precision', precision, global_step)
                writer.add_scalar('eval/recall', recall, global_step)
                writer.add_scalar('eval/f1', f1, global_step)
                writer.record()
                
                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(args.save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()

USERNAME = os.getenv("USERNAME")

def train2(model_dir = f"D:\\Users\\{USERNAME}\\data\\bert-base-chinese",
    train_path= f"D:\\Users\\{USERNAME}\\github_project/transformers_tasks/prompt_tasks/p-tuning/data/comment_classify/train.txt",
    dev_path= f"D:\\Users\\{USERNAME}\\github_project/transformers_tasks/prompt_tasks/p-tuning/data/comment_classify/dev.txt",
    verbalizer_dir= f"D:\\Users\\{USERNAME}\\github_project/transformers_tasks/prompt_tasks/p-tuning/data/comment_classify/verbalizer.txt",
    save_dir = "checkpoints/comment_classify/",
    learning_rate = 5e-5,
    weight_decay= 0.0,
    warmup_ratio=0.06,
    rdrop_coef=0.0,
    batch_size = 8,
    max_seq_len = 128,
    valid_steps = 20 ,
    logging_steps = 5,
    num_train_epochs = 20,
    max_label_len = 2,
    p_embedding_num = 15,
    device= "cpu"):
    print("save_dir: ", os.path.abspath(save_dir))
    model = AutoModelForMaskedLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    verbalizer = Verbalizer(
        verbalizer_file=verbalizer_dir,
        tokenizer=tokenizer,
        max_label_len=max_label_len
    )

    dataset = load_dataset('text', data_files={'train': train_path,
                                               'dev': dev_path})
    print(dataset)
    convert_func = partial(convert_example,
                           tokenizer=tokenizer,
                           max_seq_len=max_seq_len,
                           max_label_len=max_label_len,
                           p_embedding_num=p_embedding_num
                           )
    dataset = dataset.map(convert_func, batched=True)

    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator,
                                  batch_size=batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=batch_size)

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)
    model.to(device)

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )

    loss_list = []
    tic_train = time.time()
    metric = ClassEvaluator()
    criterion = torch.nn.CrossEntropyLoss()
    rdrop_loss = RDropLoss()
    global_step, best_f1 = 0, 0
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            if 'token_type_ids' in batch:
                logits = model(input_ids=batch['input_ids'].to(device),
                               token_type_ids=batch['token_type_ids'].to(device)).logits
            else:  # 兼容不需要 token_type_id 的模型, e.g. Roberta-Base
                logits = model(input_ids=batch['input_ids'].to(device)).logits
            mask_labels = batch['mask_labels'].numpy().tolist()
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            sub_labels = [ele['token_ids'] for ele in sub_labels]

            if rdrop_coef > 0:
                logits2 = model(input_ids=batch['input_ids'].to(device),
                                token_type_ids=batch['token_type_ids'].to(device)).logits
                ce_loss = (mlm_loss(logits, batch['mask_positions'].to(device), sub_labels, criterion, 1.0,
                                    device) + \
                           mlm_loss(logits, batch['mask_positions'].to(device), sub_labels, criterion, 1.0,
                                    device)) / 2
                kl_loss = rdrop_loss.compute_kl_loss(logits, logits2, device=device)
                loss = ce_loss + kl_loss * rdrop_coef
            else:
                loss = mlm_loss(logits, batch['mask_positions'].to(device), sub_labels, criterion, 1.0,
                                device)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print('train/train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                      % (global_step, epoch, loss_avg, logging_steps / time_diff))
                tic_train = time.time()

            if global_step % valid_steps == 0:
                cur_save_dir = os.path.join(save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))

                acc, precision, recall, f1, class_metrics = evaluate_model(model,
                                                                           metric,
                                                                           eval_dataloader,
                                                                           global_step,
                                                                           tokenizer,
                                                                           verbalizer,
                                                                           device)
                print('eval/acc', acc, global_step)
                print('eval/precision', precision, global_step)
                print('eval/recall', recall, global_step)
                print('eval/f1', f1, global_step)

                print("Evaluation precision: %.5f, recall: %.5f, F1: %.5f" % (precision, recall, f1))
                if f1 > best_f1:
                    print(
                        f"best F1 performence has been updated: {best_f1:.5f} --> {f1:.5f}"
                    )
                    print(f'Each Class Metrics are: {class_metrics}')
                    best_f1 = f1
                    cur_save_dir = os.path.join(save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__ == '__main__':
    train()

# python p_tuning.py \
#     --model "bert-base-chinese" \
#     --train_path "data/comment_classify/train.txt" \
#     --dev_path "data/comment_classify/dev.txt" \
#     --verbalizer "data/comment_classify/verbalizer.txt" \
#     --save_dir "checkpoints/comment_classify/" \
#     --img_log_dir "logs/comment_classify" \
#     --img_log_name "BERT" \
#     --batch_size 8 \
#     --max_seq_len 128 \
#     --valid_steps 20  \
#     --logging_steps 5 \
#     --num_train_epochs 20 \
#     --max_label_len 2 \
#     --p_embedding_num 15 \
#     --device "cuda:0"


