# !/usr/bin/env python3
"""
基于人工模板的prompt模型。
来源：https://github.com/HarderThenHarder/transformers_tasks/tree/main/prompt_tasks/PET
"""
import os
import time
import argparse
from functools import partial
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, default_data_collator, get_scheduler

from typing import List, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score, confusion_matrix

import json
import traceback


##############################################################################################################################

class HardTemplate(object):
    """
    硬模板，人工定义句子和[MASK]之间的位置关系。
    模板定义类，按照人为定义的模板进行encoding。
    """

    def __init__(
            self,
            prompt: str
    ):
        """
        init func.

        Args:
            prompt (str): prompt格式定义字符串, e.g. -> "这是一条{MASK}评论：{textA}。"
        """
        self.prompt = prompt
        self.inputs_list = []  # 根据文字prompt拆分为各part的列表
        self.custom_tokens = set(['MASK'])  # 从prompt中解析出的自定义token集合
        self.parse_prompt()  # 解析prompt模板

    def parse_prompt(self):
        """
        将prompt文字模板拆解为可映射的数据结构。

        Examples:
            prompt -> "这是一条{MASK}评论：{textA}。"
            inputs_list -> ['这', '是', '一', '条', 'MASK', '评', '论', '：', 'textA', '。']
            custom_tokens -> {'textA', 'MASK'}
        """
        idx = 0
        while idx < len(self.prompt):
            part = ''
            if self.prompt[idx] not in ['{', '}']:
                self.inputs_list.append(self.prompt[idx])
            if self.prompt[idx] == '{':  # 进入自定义字段
                idx += 1
                while self.prompt[idx] != '}':
                    part += self.prompt[idx]  # 拼接该自定义字段的值
                    idx += 1
            elif self.prompt[idx] == '}':
                raise ValueError("Unmatched bracket '}', check your prompt.")
            if part:
                self.inputs_list.append(part)
                self.custom_tokens.add(part)  # 将所有自定义字段存储，后续会检测输入信息是否完整
            idx += 1

    def __call__(
            self,
            inputs_dict: dict,
            tokenizer,
            mask_length,
            max_seq_len=512,
    ) -> dict:
        """
        输入一个样本，转换为符合模板的格式。

        Args:
            inputs_dict (dict): prompt中的参数字典, e.g. -> {
                                                            "textA": "这个手机也太卡了",
                                                            "MASK": "[MASK]"
                                                        }
            tokenizer: 用于encoding文本
            mask_length (int): MASK token 的长度

        Returns:
            dict -> {
                'text': '[CLS]这是一条[MASK]评论：这个手机也太卡了。[SEP]',
                'input_ids': [1, 47, 10, 7, 304, 3, 480, 279, 74, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2],
                'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                'mask_position': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            }
        """
        assert self.custom_tokens == set(inputs_dict), \
            f"@params inputs_dict doesn't match @param prompt, @prompt needs: {self.custom_tokens}, while @inputs_dict keys are: {set(inputs_dict)}."

        outputs = {
            'text': '',
            'input_ids': [],
            'token_type_ids': [],
            'attention_mask': [],
            'mask_position': []
        }

        formated_str = ''
        for ele in self.inputs_list:
            if ele in self.custom_tokens:
                if ele == 'MASK':
                    formated_str += inputs_dict[ele] * mask_length
                else:
                    formated_str += inputs_dict[ele]
            else:
                formated_str += ele

        encoded = tokenizer(
            text=formated_str,
            truncation=True,
            max_length=max_seq_len,
            padding='max_length')
        outputs['input_ids'] = encoded['input_ids']
        outputs['token_type_ids'] = encoded['token_type_ids']
        outputs['attention_mask'] = encoded['attention_mask']
        outputs['text'] = ''.join(tokenizer.convert_ids_to_tokens(encoded['input_ids']))
        mask_token_id = tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
        mask_position = np.where(np.array(outputs['input_ids']) == mask_token_id)[0].tolist()
        outputs['mask_position'] = mask_position
        return outputs


def test_template():
    from rich import print
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    template = HardTemplate(
        prompt='这是一条{MASK}评论：{textA}'
    )
    tep = template(
        inputs_dict={'textA': '包装不错，苹果挺甜的，个头也大。', 'MASK': '[MASK]'},
        tokenizer=tokenizer,
        max_seq_len=30,
        mask_length=2
    )
    print(tep)

    # print(tokenizer.convert_ids_to_tokens([3819, 3352, 3819, 3352]))
    # print(tokenizer.convert_tokens_to_ids(['水', '果']))

##############################################################################################################################

class Verbalizer(object):
    """
    Verbalizer类，用于将一个Label对应到其子Label的映射。
    verbalizer对象，实现对从Type到Type词（一对一/一对多）之间的转换。
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

def convert_example(
        examples: dict,
        tokenizer,
        max_seq_len: int,
        max_label_len: int,
        template: HardTemplate,
        train_mode=True,
        return_tensor=False
) -> dict:
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            '手机	这个手机也太卡了。',
                                                            '体育	世界杯为何迟迟不见宣传',
                                                            ...
                                                ]
                                            }
        max_seq_len (int): 句子的最大长度，若没有达到最大长度，则padding为最大长度
        max_label_len (int): 最大label长度，若没有达到最大长度，则padding为最大长度
        template (HardTemplate): 模板类。
        train_mode (bool): 训练阶段 or 推理阶段。
        return_tensor (bool): 是否返回tensor类型，如不是，则返回numpy类型。

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[1, 47, 10, 7, 304, 3, 3, 3, 3, 47, 27, 247, 98, 105, 512, 777, 15, 12043, 2], ...],
                            'token_type_ids': [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], ...],
                            'attention_mask': [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], ...],
                            'mask_positions': [[5, 6, 7, 8], ...],
                            'mask_labels': [[2372, 3442, 0, 0], [2643, 4434, 2334, 0], ...]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'mask_positions': [],
        'mask_labels': []
    }

    for i, example in enumerate(examples['text']):
        try:
            if train_mode:
                label, content = example.strip().split('\t')
            else:
                content = example.strip()

            inputs_dict = {
                'textA': content,
                'MASK': '[MASK]'
            }
            encoded_inputs = template(
                inputs_dict=inputs_dict,
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                mask_length=max_label_len
            )
        except:
            print(f'Error Line {i + 1}: "{example}" -> {traceback.format_exc()}')
            exit()
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['mask_positions'].append(encoded_inputs["mask_position"])

        if train_mode:
            label_encoded = tokenizer(text=[label])  # 将label补到最大长度
            label_encoded = label_encoded['input_ids'][0][1:-1]
            label_encoded = label_encoded[:max_label_len]
            label_encoded = label_encoded + [tokenizer.pad_token_id] * (max_label_len - len(label_encoded))
            tokenized_output['mask_labels'].append(label_encoded)

    for k, v in tokenized_output.items():
        if return_tensor:
            tokenized_output[k] = torch.LongTensor(v)
        else:
            tokenized_output[k] = np.array(v)

    return tokenized_output


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
        if not single_sub_mask_labels.size():  # 处理单token维度下维度缺失的问题
            single_sub_mask_labels = single_sub_mask_labels.unsqueeze(dim=0)
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
    predict_tokens = mask_logits.argmax(dim=-1)  # (batch * label_num)
    predict_tokens = predict_tokens.reshape(-1, label_length)  # (batch, label_num)

    return predict_tokens


def test_convert():
    from rich import print

    logits = torch.randn(1, 20, 21193)
    mask_positions = torch.LongTensor([
        [3, 4]
    ])
    predict_tokens = convert_logits_to_ids(logits, mask_positions)
    print(predict_tokens)

##############################################################################################################################

class RDropLoss(object):
    """
    RDrop Loss 类。
    R-Drop Loss, 由于Backbone中通常存在Dropout，因此可以通过减小同一个样本
    经过两次backbone之后的logits分布差异，来增强模型的鲁棒性。

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
    '''对自定义loss函数进行测试'''
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
    '''（多）分类问题下的指标评估（acc, precision, recall, f1）。'''
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
    '''对评估函数进行测试'''
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
parser.add_argument("--max_label_len", type=int, default=6, help="max length of label")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
parser.add_argument("--verbalizer", default='Verbalizer File', required=True, type=str, help="verbalizer file.")
parser.add_argument("--prompt_file", default='Prompt File', required=True, type=str, help="prompt file.")
args = parser.parse_args()


# 类似于SummaryWriter功能, iSummaryWriter工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
# https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/iTrainingLogger.py
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate_model(model, metric, data_loader, global_step, tokenizer, verbalizer):
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
            logits = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device)).logits
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
    prompt = open(args.prompt_file, 'r', encoding='utf8').readlines()[0].strip()    # prompt定义
    template = HardTemplate(prompt=prompt)                                          # 模板转换器定义
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    print(f'Prompt is -> {prompt}')
    convert_func = partial(convert_example, 
                            tokenizer=tokenizer, 
                            template=template,
                            max_seq_len=args.max_seq_len,
                            max_label_len=args.max_label_len)
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
            logits = model(input_ids=batch['input_ids'].to(args.device),
                            token_type_ids=batch['token_type_ids'].to(args.device),
                            attention_mask=batch['attention_mask'].to(args.device)).logits
            mask_labels = batch['mask_labels'].numpy().tolist()
            sub_labels = verbalizer.batch_find_sub_labels(mask_labels)
            sub_labels = [ele['token_ids'] for ele in sub_labels]
            loss = mlm_loss(logits, 
                            batch['mask_positions'].to(args.device), 
                            sub_labels, 
                            criterion,
                            1.0,
                            args.device)
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
                                                                        verbalizer)
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


if __name__ == '__main__':
    from rich import print
    train()

## 模型训练命令：
# python pet.py \
#     --model "bert-base-chinese" \
#     --train_path "data/comment_classify/train.txt" \
#     --dev_path "data/comment_classify/dev.txt" \
#     --save_dir "checkpoints/comment_classify/" \
#     --img_log_dir "logs/comment_classify" \
#     --img_log_name "BERT" \
#     --verbalizer "data/comment_classify/verbalizer.txt" \
#     --prompt_file "data/comment_classify/prompt.txt" \
#     --batch_size 8 \
#     --max_seq_len 256 \
#     --valid_steps 40  \
#     --logging_steps 5 \
#     --num_train_epochs 200 \
#     --max_label_len 2 \
#     --rdrop_coef 5e-2 \
#     --device "cuda:1"

# 训练流程步骤：
# 1、标签数据准备
# 2、Verbalizer准备；Verbalizer用于定义「真实标签」到「标签预测词」之间的映射。
# 3、Prompt设定
# 4、模型训练
# 5、模型预测
