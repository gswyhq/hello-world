# !/usr/bin/env python3
"""
torch 版本 UIE fintuning 脚本。
资料来源：
https://github.com/HarderThenHarder/transformers_tasks/tree/main/UIE
https://github.com/PaddlePaddle/PaddleNLP/tree/develop/model_zoo/uie#数据标注

 信息抽取、事件抽取数据增强（DA）策略（提升 recall）
 信息抽取、事件抽取自分析负例生成（Auto Neg）策略（提升 precision）

"""
import os
import time
import json
import random
import argparse
from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, default_data_collator, get_scheduler

from rich.table import Table
from rich.align import Align
from rich.console import Console
from rich import print


import json
from typing import List

import torch
import torch.nn as nn
import numpy as np

import copy
import json
import random
import traceback
from typing import List, Union

import jieba
from tqdm import tqdm


class Augmenter(object):
    """
    数据增强类。

    Args:
        object (_type_): _description_
    """

    @staticmethod
    def augment(text: str, methods=None, **args) -> List[str]:
        """
        对一段文本进行数据增强，包含：随机删词、随机替换、随机重复几种方法。

        Args:
            text (str): 原文本, e.g. -> '王宝强是一个演员。'
            methods (list, optional): 数据增强方法，包含 -> ['delete', 'replace', 'repeat']. Defaults to [].
            del_ratio: _description_
            len_threshold (float, Optional): _description_
            delete_aug_counts (int, Optional): _description_
            replace_ratio (float, Optional): _description_
            replace_aug_counts (int, Optional): _description_
            similarity_threhold (float, Optional): _description_
            repeat_ratio (float, Optional): _description_
            repeat_aug_counts (int, Optional): _description_

        Returns:
            List[str]: 增强后的文本, -> ['刘德华是一个演员。', '王宝强强是一个个演员。', ...]
        """
        if methods is None:
            methods = []
        results = [text]
        for method in methods:
            if method == 'delete':
                del_ratio = 0.2 if 'del_ratio' not in args else args['del_ratio']  # 随机删除指定比例的文字
                len_threshold = 5 if 'len_threshold' not in args else args['len_threshold']  # 执行删除策略的最小长度
                aug_counts = 1 if 'delete_aug_counts' not in args else args['delete_aug_counts']  # 数据增强次数
                del_words_count = int(len(text) * del_ratio)
                if len(text) > len_threshold:
                    for _ in range(aug_counts):
                        temp_res = ''
                        random_del_index = random.sample([i for i in range(len(text))], k=del_words_count)
                        for i, t in enumerate(text):
                            if i not in random_del_index:
                                temp_res += t
                        if temp_res not in results:
                            results.append(temp_res)
            elif method == 'replace':
                import synonyms
                words = jieba.lcut(text)
                replace_ratio = 0.4 if 'replace_ratio' not in args else args['replace_ratio']  # 替换词个数占总数的比例
                aug_counts = 1 if 'replace_aug_counts' not in args else args['replace_aug_counts']  # 数据增强次数
                similarity_threhold = 0.7 if 'similarity_threhold' not in args else args[
                    'similarity_threhold']  # 同义词替换时的最低相似度
                replace_words_count = int(replace_ratio * len(words))
                for _ in range(aug_counts):
                    temp_res = []
                    replace_words_index = random.sample([i for i in range(len(words))], k=replace_words_count)
                    for i, w in enumerate(words):
                        if i in replace_words_index:
                            replaced_res = synonyms.nearby(w)
                            candidate = [w for w, p in zip(*replaced_res) if p > similarity_threhold]  # 找到所有大于相似度阈值的替换词
                            if len(candidate) < 2:  # 没有符合要求的同义词则使用原词
                                temp_res.append(w)
                            else:
                                temp_res.append(random.choice(candidate))
                        else:
                            temp_res.append(w)
                    if ''.join(temp_res) not in results:
                        results.append(''.join(temp_res))
            elif method == 'repeat':
                repeat_ratio = 0.32 if 'repeat_ratio' not in args else args['repeat_ratio']  # 随机重复字个数占总数的比例
                aug_counts = 1 if 'repeat_aug_counts' not in args else args['repeat_aug_counts']  # 数据增强次数
                repeat_words_count = int(repeat_ratio * len(text))
                for _ in range(aug_counts):
                    temp_res = ''
                    random_repeat_index = random.sample([i for i in range(len(text))], k=repeat_words_count)
                    for i, w in enumerate(text):
                        if i in random_repeat_index:
                            temp_res += w * 2
                        else:
                            temp_res += w
                    if temp_res not in results:
                        results.append(temp_res)
            else:
                raise ValueError(f'no method called {method}, must in ["add", "delete", "replace", "repeat"].')
        return results

    @staticmethod
    def batch_augment(texts: List[str], methods=None, **args) -> List[str]:
        """
        批量数据增强，用于对一个文本列表里的所有句子做增强。

        Args:
            texts (List[str]): 原文本列表, e.g. -> ['王宝强是一个演员。', ...]
            methods (list, optional): _description_. Defaults to [].

        methods (list, optional): 数据增强方法，包含 -> [
                'delete',
                'replace',
                'repeat'
            ]. Defaults to [].

        Returns:
            List[str]: 增强后的文本, -> ['刘德华是一个演员。', '王宝强强是一个个演员。', ...]
        """
        if methods is None:
            methods = []
        res = []
        for text in texts:
            res.extend(Augmenter.augment(text, methods, args))
        return res

    @staticmethod
    def add_uie_relation_negative_samples(
            sample: dict,
            negative_predicates: List[str]
    ) -> List[dict]:
        """
        为UIE添加关系抽取的负例数据。

        Args:
            sample (dict): UIE训练数据样本, e.g. -> {"content": "大明是小明的父亲", "result_list": [{"text": "大明", "start": 0, "end": 2}], "prompt": "小明的父亲"}
            negative_predicates (List[str]): 负例 p 列表, e.g. -> ['母亲', '叔叔', '二姨']

        Returns:
            List[dict]: [
                {"content": "大明是小明的父亲", "result_list": [], "prompt": "小明的母亲"},
                {"content": "大明是小明的父亲", "result_list": [], "prompt": "小明的叔叔"},
                {"content": "大明是小明的父亲", "result_list": [], "prompt": "小明的二姨"},
                ...
            ]
        """
        assert 'prompt' in sample and '的' in sample['prompt'], \
            "key:'prompt' must in @param:sample and sample['prompt'] must contain '的'."
        res = []
        elements = sample['prompt'].split('的')
        subject = '的'.join(elements[:-1])

        for negtive_predicate in negative_predicates:
            res.append({
                'content': sample['content'],
                'result_list': [],
                'prompt': f'{subject}的{negtive_predicate}'
            })
        return res

    @staticmethod
    def auto_add_uie_relation_negative_samples(
            model,
            tokenizer,
            samples: Union[List[str], List[dict]],
            inference_func,
            negative_samples_file=None,
            details_file=None,
            device='cpu',
            max_seq_len=256,
            batch_size=64
    ):
        """
        自动为UIE添加关系抽取的负例数据。

        Args:
            model (_type_): fine-tuning 好的 UIE 模型
            tokenizer (_type_): tokenizer
            samples (Union(List[str], List[dict])): 数据集文件名列表（自动读取），或样本列表
            inference_func (callable): 模型推理函数
            negative_samples_file (str): 负例文件存放地址
            details_file (str): 详细信息文件存放地址，默认为'details.log'
        """
        predicate_error_dict, summary_dict = Augmenter.auto_find_uie_negative_predicates(
            model,
            tokenizer,
            samples=samples,
            inference_func=inference_func,
            device=device,
            max_seq_len=max_seq_len,
            batch_size=batch_size
        )

        if details_file:
            print(predicate_error_dict, file=open(details_file, 'w', encoding='utf8'))
            print('\n-- Error Count of Predicates --\n', file=open(details_file, 'a', encoding='utf8'))
            error_count_dict = dict([(k, v['total_error']) for k, v in predicate_error_dict.items()])
            error_count_dict = dict(sorted(error_count_dict.items(), key=lambda x: x[1], reverse=True))
            print(f'Total Error: {sum(list(error_count_dict.values()))}', file=open(details_file, 'a', encoding='utf8'))
            print(f'{error_count_dict}', file=open(details_file, 'a', encoding='utf8'))
            print('\n-- Summary of Confused Predicates --\n', file=open(details_file, 'a', encoding='utf8'))
            print(summary_dict, file=open(details_file, 'a', encoding='utf8'))
            print(f'[Done] Model Performance details have saved at "{details_file}".')

        if type(samples[0]) == str:  # 若传入的是文件路径，则读取全部的文件路径
            parse_samples = []
            for sample_file in samples:
                with open(sample_file, 'r', encoding='utf8') as f:
                    for i, line in enumerate(f.readlines()):
                        try:
                            sample = json.loads(line)
                            parse_samples.append(sample)
                        except:
                            print(f'[Error Line {i}] {line}')
                            print(traceback.format_exc())
                            exit()
            samples = parse_samples

        negative_samples = []
        for sample in samples:
            if not sample['result_list'] or '的' not in sample['prompt']:
                continue

            elements = sample['prompt'].split('的')  # 解析文件宾语
            predicate = elements[-1]
            if predicate in summary_dict:  # 添加宾语负例
                res = Augmenter.add_uie_relation_negative_samples(sample, summary_dict[predicate])
                negative_samples.extend(res)

        negative_samples = [json.dumps(sample, ensure_ascii=False) for sample in negative_samples]
        negative_samples = list(set(negative_samples))
        if negative_samples_file:
            with open(negative_samples_file, 'w', encoding='utf8') as f:
                for sample in negative_samples:
                    f.write(f'{sample}\n')
            print(f'[Done] Negative Samples have saved at "{negative_samples_file}".')

        return predicate_error_dict, summary_dict, negative_samples

    @staticmethod
    def add_positive_samples_by_swap_spo(samples: List[dict]):
        """
        通过交换同Predicate的Subject和Object的方式，实现数据增强。

        Args:
            samples (List[dict]): 原始数据中的样本

        Returns:
            positive_samples: 交换SO之后的正例
            error_num: 交换失败的样本数
            predicates_sentence_dict: 同P的句子样本
        """
        predicates_sentence_dict = {}  # 将句子按照「predicate」为key的方式存储
        for sample in samples:
            if len(sample['result_list']) == 1 and '的' in sample['prompt']:  # 只处理宾语只有一个答案的样本
                predicate = sample['prompt'].split('的')[-1]
                if predicate not in predicates_sentence_dict:
                    predicates_sentence_dict[predicate] = [sample]
                else:
                    predicates_sentence_dict[predicate].append(sample)

        positive_samples, error_num = [], 0
        for _, samples in predicates_sentence_dict.items():
            if len(samples) < 2:
                continue
            for sample in samples:
                candidates = copy.deepcopy(samples)
                candidates.remove(sample)
                candidate = copy.deepcopy(random.choice(candidates))  # 从同predicate的句子中随机选则一条，将当前的s和o替换过去

                elements = sample['prompt'].split('的')
                cur_sub = '的'.join(elements[:-1])
                cur_obj = sample['result_list'][0]['text']

                candidate_new_prompt = sample['prompt']
                candidate_text = candidate['content']
                elements = candidate['prompt'].split('的')
                candidate_sub = '的'.join(elements[:-1])
                candidate_obj = candidate['result_list'][0]['text']

                new_candidate_text = candidate_text.replace(candidate_sub, cur_sub)  # 主语替换
                new_candidate_text = new_candidate_text.replace(candidate_obj, cur_obj)  # 宾语替换

                if new_candidate_text.find(cur_obj) != -1:
                    result_list = [{
                        "text": cur_obj,
                        "start": new_candidate_text.find(cur_obj),
                        "end": new_candidate_text.find(cur_obj) + len(cur_obj)
                    }]
                    positive_samples.append({
                        "content": new_candidate_text,
                        "prompt": candidate_new_prompt,
                        "result_list": result_list
                    })
                else:
                    error_num += 1
        return positive_samples, error_num, predicates_sentence_dict

    @staticmethod
    def add_positive_samples_by_mask_then_fill(
            samples: List[dict],
            filling_model,
            filling_tokenizer,
            batch_size,
            max_seq_len,
            device,
            aug_num
    ):
        """
        通过[MASK]非关键片段，再用filling模型还原掩码片段，实现数据增强。

        Args:
            samples (List[dict]): 原数据集中的样本(doccano导出数据), dict e.g -> {
                "text": "Google was founded on September 4, 1998, by Larry Page and Sergey Brin.",
                "entities": [
                    {
                        "id": 0,
                        "start_offset": 0,
                        "end_offset": 6,
                        "label": "ORG"
                    },
                    ...
                ],
                "relations": [
                    {
                        "id": 0,
                        "from_id": 0,
                        "to_id": 1,
                        "type": "foundedAt"
                    },
                    ...
                ]
            }
            filling_model (_type_): 掩码还原模型
            filling_tokenizer (_type_): 掩码还原模型tokenizer
            aug_num (int): 一个样本增强几次

        Raises:
            ValueError: _description_
            ValueError: _description_

        Returns:
            _type_: _description_
        """
        MAX_SEQ_LEN = 512  # Filling模型最大输入长度
        SUFFIX = '中[MASK]位置的文本是：'  # 添加在每句话后的后缀
        MAX_SENTENCE_LEN = MAX_SEQ_LEN - 2 - len(SUFFIX)  # 原始输入句子的最大长度（除掉引号和后缀提示句）

        positive_samples = []
        samples = [sample for sample in samples if len(sample['entities']) > 0]  # 只增强正例

        for _ in range(aug_num):
            for i in range(0, len(samples), batch_size):
                batch_sample_texts, batch_samples = [], []
                for sample in samples[i:i + batch_size]:
                    text = list(sample['text'])
                    key_spans = [[ele['start_offset'], ele['end_offset']] for ele in sample['entities']]  # 宾语span
                    key_spans.sort(key=lambda x: x[0])  # 带有关键信息的span（不能被MASK）

                    merged_key_spans = [key_spans[0]]  # 合并重叠的词区间
                    for i in range(1, len(key_spans)):
                        if key_spans[i][0] <= merged_key_spans[-1][1]:
                            if key_spans[i][1] > merged_key_spans[-1][1]:
                                merged_key_spans[-1][1] = key_spans[i][1]
                        else:
                            merged_key_spans.append(key_spans[i])

                    masked_span_candidates = []
                    for i in range(len(merged_key_spans)):
                        if i == 0:  # 首区间处理
                            if merged_key_spans[i][0] > 0:
                                start = 0
                                end = merged_key_spans[i][0]
                                masked_span_candidates.append((start, end))
                        else:
                            start = merged_key_spans[i - 1][1]
                            end = merged_key_spans[i][0]
                            if start < end:
                                masked_span_candidates.append((start, end))
                        if i == len(merged_key_spans) - 1:  # 尾区间处理
                            if merged_key_spans[i][1] < len(text):
                                start = merged_key_spans[i][1]
                                end = len(text)
                                masked_span_candidates.append((start, end))
                    masked_span = random.choice(masked_span_candidates)
                    masked_text = text[:masked_span[0]] + ['[MASK]'] + text[masked_span[1]:]
                    masked_text = ''.join(masked_text)
                    masked_text = masked_text[:MAX_SENTENCE_LEN]
                    masked_text = f'"{masked_text}"{SUFFIX}'
                    batch_sample_texts.append(masked_text)
                    batch_samples.append({
                        'masked_span_start': masked_span[0],
                        'masked_span_end': masked_span[1],
                        'origin_text': sample['text'],
                        'entities': sample['entities'],
                        'relations': sample['relations']
                    })

                inputs = filling_tokenizer(
                    text=batch_sample_texts,
                    truncation=True,
                    max_length=max_seq_len,
                    padding='max_length',
                    return_tensors='pt'
                )
                outputs = filling_model.generate(input_ids=inputs["input_ids"].to(device))  # 将[MASK]的部分通过filling模型还原
                outputs = [filling_tokenizer.decode(output.cpu().numpy(), skip_special_tokens=True).replace(" ", "") \
                           for output in outputs]

                for output, origin_sample in zip(outputs, batch_samples):
                    origin_text = origin_sample['origin_text']  # 融合filling结果
                    new_text = origin_text[:origin_sample['masked_span_start']] + \
                               output + \
                               origin_text[origin_sample['masked_span_end']:]
                    new_entities_list = []
                    for entity in origin_sample['entities']:  # 重算每一个key span在新文本中的开始/结束位置
                        entity_text = origin_sample['origin_text'][entity['start_offset']:entity['end_offset']]
                        start = new_text.find(entity_text)
                        if start == -1:
                            raise ValueError(
                                f'Can not find span:{entity_text} in new_text: {new_text}, which origin sample is {origin_sample}.')
                        end = start + len(entity_text)
                        new_entities_list.append({
                            'id': entity['id'],
                            'start_offset': start,
                            'end_offset': end,
                            'label': entity['label'],
                            'text': entity_text
                        })
                    positive_samples.append({
                        'text': new_text,
                        'entities': new_entities_list,
                        'relations': origin_sample['relations']
                    })
        return positive_samples

    @staticmethod
    def auto_add_uie_relation_positive_samples(
            samples: Union[List[str], List[dict]],
            positive_samples_file=None,
            mode='rule',
            **args
    ):
        """
        自动为UIE添加关系抽取的正例数据。

        Args:
            samples (Union(List[str], List[dict])): 数据集文件名列表（自动读取），或样本（字典）列表
            positive_samples_file (str): 正例文件存放地址, 若为空则不存到本地
            mode (str): 正例增强的方式，
                        'rule': 基于规则，同P的SO互换。
                        'mask-then-fill': [MASK]非关键片段，再通过生成模型还原掩码片段。
        """
        assert type(samples) == list, '@params:samples must be [list] type.'
        if type(samples[0]) == str:  # 若传入的是文件路径，则读取全部的文件路径
            parse_samples = []
            for sample_file in samples:
                with open(sample_file, 'r', encoding='utf8') as f:
                    for i, line in enumerate(f.readlines()):
                        try:
                            sample = json.loads(line)
                            parse_samples.append(sample)
                        except:
                            print(f'[Error Line {i}] {line}')
                            print(traceback.format_exc())
                            exit()
            samples = parse_samples

        if mode == 'rule':
            positive_samples, \
            error_num, \
            predicates_sentence_dict = Augmenter.add_positive_samples_by_swap_spo(samples)
            print('error samples in positive augment: ', error_num)
        elif mode == 'mask_then_fill':
            if 'filling_model' not in args or 'filling_tokenizer' not in args:
                raise ValueError('@param filling_model and @param filling_tokenizer must be specified.')
            positive_samples = Augmenter.add_positive_samples_by_mask_then_fill(
                samples,
                args['filling_model'],
                args['filling_tokenizer'],
                batch_size=args['batch_size'] if 'batch_size' in args else 16,
                max_seq_len=args['max_seq_len'] if 'max_seq_len' in args else 128,
                device=args['device'] if 'device' in args else 'cpu',
                aug_num=args['aug_num'] if 'aug_num' in args else 1
            )
        else:
            raise ValueError(f'Invalid @param mode={mode}, check it again.')

        positive_samples = [json.dumps(sample, ensure_ascii=False) for sample in positive_samples]
        positive_samples = list(set(positive_samples))
        if positive_samples_file:
            with open(positive_samples_file, 'w', encoding='utf8') as f:
                for sample in positive_samples:
                    f.write(f'{sample}\n')
            print(f'[Done] Positive Samples have saved at {positive_samples_file}.')

        if mode == 'rule':
            return positive_samples, predicates_sentence_dict
        elif mode == 'mask_then_fill':
            return positive_samples

    @staticmethod
    def auto_find_uie_negative_predicates(
            model: str,
            tokenizer: str,
            samples: Union[List[str], List[dict]],
            inference_func,
            device='cpu',
            max_seq_len=256,
            batch_size=64
    ) -> tuple:
        """
        根据标注数据集自动找出易混淆的，需要添加负例的predicates。

        Args:
            model (_type_): fine-tuning 好的 UIE 模型
            tokenizer (_type_): tokenizer 存放地址
            samples (List[str], List[dict]): 数据集文件名列表（自动读取），或样本列表
            inference_func (callabel): 模型推理函数

        Returns:
            predicate_error_dict (混淆负例P的详细信息) -> {
                                                        "上级行政区": {
                                                                        "total_error": 3,
                                                                        "confused_predicates": {
                                                                            "地理位置": {
                                                                                "count": 1,
                                                                                "error_samples": [
                                                                                    "content:  邓台村属于哪个镇？大名县大街镇 prompt:邓台村的地理位置 answer:['大名县大街镇']"
                                                                                ]
                                                                            },
                                                                            "所属机构": {
                                                                                "count": 1,
                                                                                "error_samples": [
                                                                                    "content:  邓台村属于哪个镇？大名县大街镇 prompt:邓台村的所属机构 answer:['大名县大街镇']"
                                                                                ]
                                                                            },
                                                                            ...
                                                                        }
                                                                    },
                                                                    ...
                                                    }
            summary_dict (混淆负例P的简要信息) -> {
                                                '上级行政区': ['地理位置', '所属机构', '行政区等级'],
                                                '重量': ['使用材料', '标准'],
                                                '品牌类型': ['档次', '品牌', '企业类型'],
                                                ...
                                            }
        """
        if type(samples[0]) == str:  # 若传入的是文件路径，则读取全部的文件路径
            parse_samples = []
            for sample_file in samples:
                with open(sample_file, 'r', encoding='utf8') as f:
                    for i, line in enumerate(f.readlines()):
                        try:
                            sample = json.loads(line)
                            parse_samples.append(sample)
                        except:
                            print(f'[Error Line {i}] {line}')
                            print(traceback.format_exc())
                            exit()
            samples = parse_samples

        all_predicates = []  # 统计所有的谓语列表
        predicates_of_each_sample = {}  # 通过整个数据集，计算每个句子中包含的全部p
        for sample in samples:
            if '的' in sample['prompt']:  # 提取prompt中的predicate
                try:
                    elements = sample['prompt'].split('的')
                    predicate = elements[-1]
                except:
                    print(f'[Error Prompt] {sample}')
                    exit()
                if predicate not in all_predicates:
                    all_predicates.append(predicate)

                if sample['result_list'] != []:  # 记录每一个句子都有哪些predicate
                    if sample['content'] not in predicates_of_each_sample:
                        predicates_of_each_sample[sample['content']] = [predicate]
                    else:
                        if predicate not in predicates_of_each_sample[sample['content']]:
                            predicates_of_each_sample[sample['content']].append(predicate)

        predicate_error_dict = {}
        for sample in tqdm(samples):
            if not sample['result_list'] or '的' not in sample['prompt']:
                continue

            elements = sample['prompt'].split('的')
            subject = '的'.join(elements[:-1])
            predicate = elements[-1]

            sample_predicates = predicates_of_each_sample[sample['content']]  # 当前样本包含的p
            negative_predictes = [p for p in all_predicates if p not in sample_predicates]  # 当前样本不包含的p

            for i in range(0, len(negative_predictes), batch_size):
                new_prompts = [f'{subject}的{p}' for p in negative_predictes[i:i + batch_size]]
                contents = [sample['content']] * len(new_prompts)
                res_list = inference_func(
                    model,
                    tokenizer,
                    device,
                    contents=contents,
                    prompts=new_prompts,
                    max_length=max_seq_len
                )

                for new_prompt, res, p in zip(new_prompts, res_list, negative_predictes[i:i + batch_size]):
                    origin_answers = [ele['text'] for ele in sample['result_list']]
                    if len(res) and res[0] in origin_answers:  # 如果模型通过其余的p抽出了结果，且结果与原始结果相同
                        if predicate not in predicate_error_dict:
                            predicate_error_dict[predicate] = {
                                'total_error': 0,
                                'confused_predicates': {}
                            }
                        predicate_error_dict[predicate]['total_error'] += 1
                        if p not in predicate_error_dict[predicate]['confused_predicates']:  # 记录（p-负例p）的映射关系
                            predicate_error_dict[predicate]['confused_predicates'][p] = {
                                'count': 1,
                                'error_samples': [f"content: {sample['content']} prompt:{new_prompt} answer:{res}"]
                            }  # 记录（p-负例p）的出错次数、出错样本
                        else:
                            predicate_error_dict[predicate]['confused_predicates'][p]['count'] += 1
                            predicate_error_dict[predicate]['confused_predicates'][p]['error_samples'].append(
                                f"content: {sample['content']} prompt:{new_prompt} answer:{res}"
                            )

        predicate_error_sorted_tuple = sorted(predicate_error_dict.items(), key=lambda x: x[1]['total_error'])
        summary_dict = dict(
            [(ele[0], list(ele[1]['confused_predicates'].keys())) for ele in predicate_error_sorted_tuple])

        return predicate_error_dict, summary_dict

class UIE(nn.Module):

    def __init__(self, encoder):
        """
        init func.

        Args:
            encoder (transformers.AutoModel): backbone, 默认使用 ernie 3.0

        Reference:
            https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/model.py
        """
        super().__init__()
        self.encoder = encoder
        hidden_size = 768
        self.linear_start = nn.Linear(hidden_size, 1)
        self.linear_end = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            input_ids: torch.tensor,
            token_type_ids: torch.tensor,
            attention_mask=None,
            pos_ids=None,
    ) -> tuple:
        """
        forward 函数，返回开始/结束概率向量。

        Args:
            input_ids (torch.tensor): (batch, seq_len)
            token_type_ids (torch.tensor): (batch, seq_len)
            attention_mask (torch.tensor): (batch, seq_len)
            pos_ids (torch.tensor): (batch, seq_len)

        Returns:
            tuple:  start_prob -> (batch, seq_len)
                    end_prob -> (batch, seq_len)
        """
        sequence_output = self.encoder(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=pos_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"]
        start_logits = self.linear_start(sequence_output)  # (batch, seq_len, 1)
        start_logits = torch.squeeze(start_logits, -1)  # (batch, seq_len)
        start_prob = self.sigmoid(start_logits)  # (batch, seq_len)
        end_logits = self.linear_end(sequence_output)  # (batch, seq_len, 1)
        end_logits = torch.squeeze(end_logits, -1)  # (batch, seq_len)
        end_prob = self.sigmoid(end_logits)  # (batch, seq_len)
        return start_prob, end_prob


def get_bool_ids_greater_than(probs: list, limit=0.5, return_prob=False) -> list:
    """
    筛选出大于概率阈值的token_ids。

    Args:
        probs (_type_):
        limit (float, optional): _description_. Defaults to 0.5.
        return_prob (bool, optional): _description_. Defaults to False.

    Returns:
        list: [1, 3, 5, ...] (return_prob=False)
                or
            [(1, 0.56), (3, 0.78), ...] (return_prob=True)

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/paddlenlp/taskflow/utils.py#L810
    """
    probs = np.array(probs)
    dim_len = len(probs.shape)
    if dim_len > 1:
        result = []
        for p in probs:
            result.append(get_bool_ids_greater_than(p, limit, return_prob))
        return result
    else:
        result = []
        for i, p in enumerate(probs):
            if p > limit:
                if return_prob:
                    result.append((i, p))
                else:
                    result.append(i)
        return result


def model_get_span(start_ids: list, end_ids: list, with_prob=False) -> set:
    """
    输入start_ids和end_ids，计算answer span列表。

    Args:
        start_ids (list): [1, 2, 10]
        end_ids (list):  [4, 12]
        with_prob (bool, optional): _description_. Defaults to False.

    Returns:
        set: set((2, 4), (10, 12))

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/paddlenlp/taskflow/utils.py#L835
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            if start_ids[start_pointer][0] == end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer][0] < end_ids[end_pointer][0]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer][0] > end_ids[end_pointer][0]:
                end_pointer += 1
                continue
        else:
            if start_ids[start_pointer] == end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                end_pointer += 1
                continue
            if start_ids[start_pointer] < end_ids[end_pointer]:
                couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
                start_pointer += 1
                continue
            if start_ids[start_pointer] > end_ids[end_pointer]:
                end_pointer += 1
                continue
    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result


def convert_inputs(tokenizer, prompts: List[str], contents: List[str], max_length=512) -> dict:
    """
    处理输入样本，包括prompt/content的拼接和offset的计算。

    Args:
        tokenizer (tokenizer): tokenizer
        prompt (List[str]): prompt文本列表
        content (List[str]): content文本列表
        max_length (int): 句子最大长度

    Returns:
        dict -> {
                    'input_ids': tensor([[1, 57, 405, ...]]),
                    'token_type_ids': tensor([[0, 0, 0, ...]]),
                    'attention_mask': tensor([[1, 1, 1, ...]]),
                    'pos_ids': tensor([[0, 1, 2, 3, 4, 5,...]])
                    'offset_mapping': tensor([[[0, 0], [0, 1], [1, 2], [0, 0], [3, 4], ...]])
            }

    Reference:
        https://github.com/PaddlePaddle/PaddleNLP/blob/a12481fc3039fb45ea2dfac3ea43365a07fc4921/model_zoo/uie/utils.py#L150
    """
    inputs = tokenizer(text=prompts,  # [SEP]前内容
                       text_pair=contents,  # [SEP]后内容
                       truncation=True,  # 是否截断
                       max_length=max_length,  # 句子最大长度
                       padding="max_length",  # padding类型
                       return_offsets_mapping=True,  # 返回offsets用于计算token_id到原文的映射
                       )
    pos_ids = []
    for i in range(len(contents)):
        pos_ids += [[j for j in range(len(inputs['input_ids'][i]))]]
    pos_ids = torch.tensor(pos_ids)
    inputs['pos_ids'] = pos_ids

    offset_mappings = [[list(x) for x in offset] for offset in inputs["offset_mapping"]]

    # * Desc:
    # *    经过text & text_pair后，生成的offset_mapping会将prompt和content的offset独立计算，
    # *    这里将content的offset位置补回去。
    # *
    # * Example:
    # *    offset_mapping(before):[[0, 0], [0, 1], [1, 2], [0, 0], [0, 1], [1, 2], [2, 3], ...]
    # *    offset_mapping(after):[[0, 0], [0, 1], [1, 2], [0, 0], [2, 3], [4, 5], [5, 6], ...]
    # *
    for i in range(len(offset_mappings)):  # offset 重计算
        bias = 0
        for index in range(1, len(offset_mappings[i])):
            mapping = offset_mappings[i][index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = offset_mappings[i][index - 1][1]
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mappings[i][index][0] += bias
            offset_mappings[i][index][1] += bias

    inputs['offset_mapping'] = offset_mappings

    for k, v in inputs.items():  # list转tensor
        inputs[k] = torch.LongTensor(v)

    return inputs


def map_offset(ori_offset, offset_mapping):
    """
    map ori offset to token offset
    """
    for index, span in enumerate(offset_mapping):
        if span[0] <= ori_offset < span[1]:
            return index
    return -1


def convert_example(examples, tokenizer, max_seq_len):
    """
    将样本数据转换为模型接收的输入数据。

    Args:
        examples (dict): 训练数据样本, e.g. -> {
                                                "text": [
                                                            {
                                                                "content": "北京是中国的首都",
                                                                "prompt": "城市",
                                                                "result_list": [{"text": "北京", "start": 0, "end": 2}]
                                                            },
                                                        ...
                                                ]
                                            }

    Returns:
        dict (str: np.array) -> tokenized_output = {
                            'input_ids': [[101, 3928, ...], [101, 4395, ...]],
                            'token_type_ids': [[0, 0, ...], [0, 0, ...]],
                            'attention_mask': [[1, 1, ...], [1, 1, ...]],
                            'pos_ids': [[0, 1, 2, ...], [0, 1, 2, ...], ...],
                            'start_ids': [[0, 1, 0, ...], [0, 0, ..., 1, ...]],
                            'end_ids': [[0, 0, 1, ...], [0, 0, ...]]
                        }
    """
    tokenized_output = {
        'input_ids': [],
        'token_type_ids': [],
        'attention_mask': [],
        'pos_ids': [],
        'start_ids': [],
        'end_ids': []
    }

    for example in examples['text']:
        example = json.loads(example)
        try:
            encoded_inputs = tokenizer(
                text=example['prompt'],
                text_pair=example['content'],
                stride=len(example['prompt']),
                truncation=True,
                max_length=max_seq_len,
                padding='max_length',
                return_offsets_mapping=True)
        except:
            print('[Warning] ERROR Sample: ', example)
            exit()
        offset_mapping = [list(x) for x in encoded_inputs["offset_mapping"]]

        """
        经过text & text_pair后，生成的offset_mapping会将prompt和content的offset独立计算，
        这里将content的offset位置补回去。

        offset_mapping(before):[[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 0], [0, 1], [1, 2], [2, 3], [3, 4], ...]
        offset_mapping(after):[[0, 0], [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [0, 0], [8, 9], [9, 10], [10, 11], [11, 12], ...]
        """
        bias = 0
        for index in range(len(offset_mapping)):
            if index == 0:
                continue
            mapping = offset_mapping[index]
            if mapping[0] == 0 and mapping[1] == 0 and bias == 0:
                bias = index
            if mapping[0] == 0 and mapping[1] == 0:
                continue
            offset_mapping[index][0] += bias
            offset_mapping[index][1] += bias

        start_ids = [0 for x in range(max_seq_len)]
        end_ids = [0 for x in range(max_seq_len)]

        for item in example["result_list"]:
            start = map_offset(item["start"] + bias, offset_mapping)  # 计算真实的start token的id
            end = map_offset(item["end"] - 1 + bias, offset_mapping)  # 计算真实的end token的id
            start_ids[start] = 1.0  # one-hot vector
            end_ids[end] = 1.0  # one-hot vector

        pos_ids = [i for i in range(len(encoded_inputs['input_ids']))]
        tokenized_output['input_ids'].append(encoded_inputs["input_ids"])
        tokenized_output['token_type_ids'].append(encoded_inputs["token_type_ids"])
        tokenized_output['attention_mask'].append(encoded_inputs["attention_mask"])
        tokenized_output['pos_ids'].append(pos_ids)
        tokenized_output['start_ids'].append(start_ids)
        tokenized_output['end_ids'].append(end_ids)

    for k, v in tokenized_output.items():
        tokenized_output[k] = np.array(v, dtype='int64')

    return tokenized_output



def get_span(start_ids, end_ids, with_prob=False):
    """
    Get span set from position start and end list.
    Args:
        start_ids (List[int]/List[tuple]): The start index list.
        end_ids (List[int]/List[tuple]): The end index list.
        with_prob (bool): If True, each element for start_ids and end_ids is a tuple aslike: (index, probability).
    Returns:
        set: The span set without overlapping, every id can only be used once.
    """
    if with_prob:
        start_ids = sorted(start_ids, key=lambda x: x[0])
        end_ids = sorted(end_ids, key=lambda x: x[0])
    else:
        start_ids = sorted(start_ids)
        end_ids = sorted(end_ids)

    start_pointer = 0
    end_pointer = 0
    len_start = len(start_ids)
    len_end = len(end_ids)
    couple_dict = {}

    # 将每一个span的首/尾token的id进行配对（就近匹配，默认没有overlap的情况）
    while start_pointer < len_start and end_pointer < len_end:
        if with_prob:
            start_id = start_ids[start_pointer][0]
            end_id = end_ids[end_pointer][0]
        else:
            start_id = start_ids[start_pointer]
            end_id = end_ids[end_pointer]

        if start_id == end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            end_pointer += 1
            continue

        if start_id < end_id:
            couple_dict[end_ids[end_pointer]] = start_ids[start_pointer]
            start_pointer += 1
            continue

        if start_id > end_id:
            end_pointer += 1
            continue

    result = [(couple_dict[end], end) for end in couple_dict]
    result = set(result)
    return result

class SpanEvaluator(object):
    """
    SpanEvaluator computes the precision, recall and F1-score for span detection.
    """

    def __init__(self):
        super().__init__()
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def compute(self, start_probs, end_probs, gold_start_ids, gold_end_ids):
        """
        Computes the precision, recall and F1-score for span detection.
        """
        pred_start_ids = get_bool_ids_greater_than(start_probs)
        pred_end_ids = get_bool_ids_greater_than(end_probs)
        gold_start_ids = get_bool_ids_greater_than(gold_start_ids.tolist())
        gold_end_ids = get_bool_ids_greater_than(gold_end_ids.tolist())
        num_correct_spans = 0
        num_infer_spans = 0
        num_label_spans = 0

        for predict_start_ids, predict_end_ids, label_start_ids, label_end_ids in zip(
                pred_start_ids, pred_end_ids, gold_start_ids, gold_end_ids):
            [_correct, _infer, _label] = self.eval_span(predict_start_ids, predict_end_ids,
                                                            label_start_ids, label_end_ids)
            num_correct_spans += _correct
            num_infer_spans += _infer
            num_label_spans += _label

        return num_correct_spans, num_infer_spans, num_label_spans

    def update(self, num_correct_spans, num_infer_spans, num_label_spans):
        """
        This function takes (num_infer_spans, num_label_spans, num_correct_spans) as input,
        to accumulate and update the corresponding status of the SpanEvaluator object.
        """
        self.num_infer_spans += num_infer_spans
        self.num_label_spans += num_label_spans
        self.num_correct_spans += num_correct_spans

    def eval_span(self, predict_start_ids, predict_end_ids, label_start_ids, label_end_ids):
        """
        evaluate position extraction (start, end)
        return num_correct, num_infer, num_label
        input: [1, 2, 10] [4, 12] [2, 10] [4, 11]
        output: (1, 2, 2)
        """
        pred_set = get_span(predict_start_ids, predict_end_ids)     # 得到模型输出的span集合(set), e.g. {(1, 3), (4, 5)}
        label_set = get_span(label_start_ids, label_end_ids)        # 得到标签中正确的span集合(set), e.g. {(1, 3), (4, 5), (8, 9)}
        num_correct = len(pred_set & label_set)                     # 计算正确预测的span集合(两个集合求交集), e.g. {(1, 3), {4, 5}}
        num_infer = len(pred_set)
        num_label = len(label_set)
        return (num_correct, num_infer, num_label)

    def accumulate(self):
        """
        This function returns the mean precision, recall and f1 score for all accumulated minibatches.
        Returns:
            tuple: Returns tuple (`precision, recall, f1 score`).
        """
        precision = float(self.num_correct_spans / self.num_infer_spans) if self.num_infer_spans else 0.
        recall = float(self.num_correct_spans / self.num_label_spans) if self.num_label_spans else 0.
        f1_score = float(2 * precision * recall / (precision + recall)) if self.num_correct_spans else 0.
        return precision, recall, f1_score

    def reset(self):
        """
        Reset function empties the evaluation memory for previous mini-batches.
        """
        self.num_infer_spans = 0
        self.num_label_spans = 0
        self.num_correct_spans = 0

    def name(self):
        """
        Return name of metric instance.
        """
        return "precision", "recall", "f1"



def inference(
        model,
        tokenizer,
        device: str,
        contents: List[str],
        prompts: List[str],
        max_length=512,
        prob_threshold=0.5
) -> List[str]:
    """
    输入 promot 和 content 列表，返回模型提取结果。

    Args:
        contents (List[str]): 待提取文本列表, e.g. -> [
                                                    '《琅琊榜》是胡歌主演的一部电视剧。',
                                                    '《笑傲江湖》是一部金庸的著名小说。',
                                                    ...
                                                ]
        prompts (List[str]): prompt列表，用于告知模型提取内容, e.g. -> [
                                                                    '主语',
                                                                    '类型',
                                                                    ...
                                                                ]
        max_length (int): 句子最大长度，小于最大长度则padding，大于最大长度则截断。
        prob_threshold (float): sigmoid概率阈值，大于该阈值则二值化为True。

    Returns:
        List: 模型识别结果, e.g. -> [['琅琊榜'], ['电视剧']]
    """
    inputs = convert_inputs(tokenizer, prompts, contents, max_length=max_length)
    model_inputs = {
        'input_ids': inputs['input_ids'].to(device),
        'token_type_ids': inputs['token_type_ids'].to(device),
        'attention_mask': inputs['attention_mask'].to(device),
    }
    output_sp, output_ep = model(**model_inputs)
    output_sp, output_ep = output_sp.detach().cpu().tolist(), output_ep.detach().cpu().tolist()
    start_ids_list = get_bool_ids_greater_than(output_sp, prob_threshold)
    end_ids_list = get_bool_ids_greater_than(output_ep, prob_threshold)

    res = []  # decode模型输出，将token id转换为span text
    offset_mapping = inputs['offset_mapping'].tolist()
    for start_ids, end_ids, prompt, content, offset_map in zip(start_ids_list,
                                                               end_ids_list,
                                                               prompts,
                                                               contents,
                                                               offset_mapping):
        span_set = model_get_span(start_ids, end_ids)  # e.g. {(5, 7), (9, 10)}
        current_span_list = []
        for span in span_set:
            if span[0] < len(prompt) + 2:  # 若答案出现在promot区域，过滤
                continue
            span_text = ''  # 答案span
            input_content = prompt + content  # 对齐token_ids
            for s in range(span[0], span[1] + 1):  # 将 offset map 里 token 对应的文本切回来
                span_text += input_content[offset_map[s][0]: offset_map[s][1]]
            current_span_list.append(span_text)
        res.append(current_span_list)
    return res


def event_extract_example(
        model,
        tokenizer,
        device: str,
        sentence: str,
        schema: dict,
        prob_threshold=0.6,
        max_seq_len=128,
) -> dict:
    """
    UIE事件抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (dict): 事件定义字典, e.g. -> {
                                            '加班触发词': ['时间','地点'],
                                            '出行触发词': ['时间', '出发地', '目的地', '花费']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。

    Returns:
        dict -> {
                '触发词1': {},
                '触发词2': {
                    '事件属性1': [属性值1, 属性值2, ...],
                    '事件属性2': [属性值1, 属性值2, ...],
                    '事件属性3': [属性值1, 属性值2, ...],
                    ...
                }
            }
    """
    rsp = {}
    trigger_prompts = list(schema.keys())

    for trigger_prompt in trigger_prompts:
        rsp[trigger_prompt] = {}
        triggers = inference(
            model,
            tokenizer,
            device,
            [sentence],
            [trigger_prompt],
            max_length=128,
            prob_threshold=prob_threshold)[0]

        for trigger in triggers:
            if trigger:
                arguments = schema.get(trigger_prompt)
                contents = [sentence] * len(arguments)
                prompts = [f"{trigger}的{a}" for a in arguments]
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents,
                    prompts,
                    max_length=max_seq_len,
                    prob_threshold=prob_threshold)
                for a, r in zip(arguments, res):
                    rsp[trigger_prompt][a] = r
    print('[+] Event-Extraction Results: ', rsp)


def information_extract_example(
        model,
        tokenizer,
        device: str,
        sentence: str,
        schema: dict,
        prob_threshold=0.6,
        max_seq_len=128
) -> dict:
    """
    UIE信息抽取示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '麻雀是几级保护动物？国家二级保护动物'
        schema (dict): 事件定义字典, e.g. -> {
                                            '主语': ['保护等级']
                                        }
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。

    Returns:
        dict -> {
                '麻雀': {
                        '保护等级': ['国家二级']
                    },
                ...
            }
    """
    rsp = {}
    subject_prompts = list(schema.keys())

    for subject_prompt in subject_prompts:
        subjects = inference(
            model,
            tokenizer,
            device,
            [sentence],
            [subject_prompt],
            max_length=128,
            prob_threshold=prob_threshold)[0]

        for subject in subjects:
            if subject:
                rsp[subject] = {}
                predicates = schema.get(subject_prompt)
                contents = [sentence] * len(predicates)
                prompts = [f"{subject}的{p}" for p in predicates]
                res = inference(
                    model,
                    tokenizer,
                    device,
                    contents,
                    prompts,
                    max_length=max_seq_len,
                    prob_threshold=prob_threshold
                )
                for p, r in zip(predicates, res):
                    rsp[subject][p] = r
    print('[+] Information-Extraction Results: ', rsp)


def ner_example(
        model,
        tokenizer,
        device: str,
        sentence: str,
        schema: list,
        prob_threshold=0.6
) -> dict:
    """
    UIE做NER任务示例。

    Args:
        sentence (str): 待抽取句子, e.g. -> '5月17号晚上10点35分加班打车回家，36块五。'
        schema (list): 待抽取的实体列表, e.g. -> ['出发地', '目的地', '时间']
        prob_threshold (float, optional): 置信度阈值（0~1），置信度越高则召回结果越少，越准确。

    Returns:
        dict -> {
                实体1: [实体值1, 实体值2, 实体值3...],
                实体2: [实体值1, 实体值2, 实体值3...],
                ...
            }
    """
    rsp = {}
    sentences = [sentence] * len(schema)  # 一个prompt需要对应一个句子，所以要复制n遍句子
    res = inference(
        model,
        tokenizer,
        device,
        sentences,
        schema,
        max_length=128,
        prob_threshold=prob_threshold)
    for s, r in zip(schema, res):
        rsp[s] = r
    print('[+] NER Results: ', rsp)


def test_model():
    from rich import print

    device = 'cuda:0'  # 指定GPU设备
    saved_model_path = './checkpoints/DuIE/model_best/'  # 训练模型存放地址
    tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
    model = torch.load(os.path.join(saved_model_path, 'model.pt'))
    model.to(device).eval()

    sentences = [
        '谭孝曾是谭元寿的长子，也是谭派第六代传人。'
    ]

    # NER 示例
    for sentence in sentences:
        ner_example(
            model,
            tokenizer,
            device,
            sentence=sentence,
            schema=['人物']
        )

    # SPO抽取示例
    for sentence in sentences:
        information_extract_example(
            model,
            tokenizer,
            device,
            sentence=sentence,
            schema={
                '人物': ['父亲'],
            }
        )

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
parser.add_argument("--pretrained_model", default='uie-base-zh', type=str, choices=['uie-base-zh'], help="backbone of encoder.")
parser.add_argument("--train_path", default=None, type=str, help="The path of train set.")
parser.add_argument("--dev_path", default=None, type=str, help="The path of dev set.")
parser.add_argument("--save_dir", default="./checkpoint", type=str, required=False, help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument("--max_seq_len", default=300, type=int,help="The maximum total input sequence length after tokenization. Sequences longer "
    "than this will be truncated, sequences shorter will be padded.", )
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.", )
parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--auto_neg_rate", default=0.5, type=float, help="Auto negative samples generated ratio.")
parser.add_argument("--auto_pos_rate", default=0.5, type=float, help="Auto positive samples generated ratio.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--num_train_epochs", default=10, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Linear warmup over warmup_ratio * total_steps.")
parser.add_argument("--valid_steps", default=200, type=int, required=False, help="evaluate frequecny.")
parser.add_argument("--auto_da_epoch", default=0, type=int, required=False, help="auto add positive/negative samples policy frequency.")
parser.add_argument("--logging_steps", default=10, type=int, help="log interval.")
parser.add_argument('--device', default="cuda:0", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--img_log_dir", default='logs', type=str, help="Logging image path.")
parser.add_argument("--img_log_name", default='Model Performance', type=str, help="Logging image file name.")
args = parser.parse_args()



# 类似于SummaryWriter功能, iSummaryWriter工具依赖matplotlib采用静态本地图片存储的形式，便于服务器快速查看训练结果。
# https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_matching/unsupervised/simcse/iTrainingLogger.py
writer = iSummaryWriter(log_path=args.img_log_dir, log_name=args.img_log_name)


def evaluate(model, metric, data_loader, global_step):
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
        for batch in data_loader:
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                            token_type_ids=batch['token_type_ids'].to(args.device),
                                            attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).detach().numpy()
            end_ids = batch['end_ids'].to(torch.float32).detach().numpy()
            num_correct, num_infer, num_label = metric.compute(start_prob.cpu().detach().numpy(), 
                                                                end_prob.cpu().detach().numpy(), 
                                                                start_ids, 
                                                                end_ids)
            metric.update(num_correct, num_infer, num_label)
        
        precision, recall, f1 = metric.accumulate()
        writer.add_scalar('eval-precision', precision, global_step)
        writer.add_scalar('eval-recall', recall, global_step)
        writer.add_scalar('eval-f1', f1, global_step)
        writer.record()
    model.train()
    return precision, recall, f1

def auto_add_samples(
    model, 
    tokenizer,
    epoch: int,
    convert_func,
    ) -> DataLoader:
    """
    根据模型当前学习的效果，自动添加正例/负例。

    Args:
        model (_type_): _description_
        tokenizer (_type_): _description_
        epoch (int): 当前步数
        convert_func (_type_): 数据集map函数
    
    Returns:
        加入了负例后的 train_dataloader
    """
    model.eval()
    with torch.no_grad():
        dataset_path = os.path.dirname(args.train_path)
        train_dataset_with_new_samples_added = os.path.join(dataset_path, 'new_train.txt')      # 加入正/负例后的训练数据集存放地址
        da_sample_details_path = os.path.join(dataset_path, 'auto_da_details')                  # 自动增加负例的详细信息文件存放地址
        if not os.path.exists(da_sample_details_path):
            os.makedirs(da_sample_details_path)
        
        neg_sample_file = os.path.join(da_sample_details_path, f'neg_samples_{epoch}.txt')      # 生成的负例数据存放位置
        neg_details_file = os.path.join(da_sample_details_path, f'neg_details_{epoch}.log')     # 生成的详情存放位置
        pos_sample_file = os.path.join(da_sample_details_path, f'pos_samples_{epoch}.txt')      # 生成的正例数据存放位置

        Augmenter.auto_add_uie_relation_negative_samples(
            model,
            tokenizer,
            samples=[args.train_path],
            inference_func=inference,
            negative_samples_file=neg_sample_file,
            details_file=neg_details_file,
            device=args.device,
            max_seq_len=args.max_seq_len       
        )

        Augmenter.auto_add_uie_relation_positive_samples(
            samples=[args.train_path],
            positive_samples_file=pos_sample_file
        )

        generated_negative_samples = [line.strip() for line in open(neg_sample_file, 'r', encoding='utf8').readlines()]
        generated_positive_samples = [line.strip() for line in open(pos_sample_file, 'r', encoding='utf8').readlines()]
        train_samples = [eval(line.strip()) for line in open(args.train_path, 'r', encoding='utf8').readlines()]
        train_positive_samples, train_negaitive_samples = [], []
        for train_sample in train_samples:
            if train_sample['result_list']:
                train_positive_samples.append(json.dumps(train_sample, ensure_ascii=False))                   # 保留训练数据集中的正例
            else:
                train_negaitive_samples.append(json.dumps(train_sample, ensure_ascii=False))                  # 保存训练数据集中的负例
        
        # * 添加正/负例
        negaitve_samples_generated_sample_num = int(len(generated_negative_samples) * args.auto_neg_rate)     # 随机采样等比例的新生成负例数据
        positive_samples_generated_sample_num = int(len(generated_positive_samples) * args.auto_pos_rate)     # 随机采样等比例的新生成正例数据

        # * 新数据集混合
        new_train_samples = train_positive_samples + train_negaitive_samples + \
                random.sample(generated_positive_samples, k=positive_samples_generated_sample_num) + \
                random.sample(generated_negative_samples, k=negaitve_samples_generated_sample_num)

        with open(train_dataset_with_new_samples_added, 'w', encoding='utf8') as f:                            # 保存新的训练数据集
            for line in new_train_samples:
                f.write(f'{line}\n')
        args.train_path = train_dataset_with_new_samples_added                                                 # 替换args中训练数据集为最新的训练集

        train_dataset = load_dataset('text', data_files={'train': args.train_path})["train"]
        train_dataset = train_dataset.map(convert_func, batched=True)
        train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    
    model.train()
    return train_dataloader


def get_optimizer_and_scheler(model, train_dataloader):
    """
    刷新optimizer和lr衰减器。
    如果设置了auto_da，则每做一次自动数据增强都需要重置学习率。

    Args:
        model (_type_): _description_
        train_dataloader (_type_): _description_

    Returns:
        _type_: _description_
    """
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

    # 根据训练轮数计算最大训练步数，以便于scheduler动态调整lr
    num_update_steps_per_epoch = len(train_dataloader)
    num_train_epochs = args.auto_da_epoch if args.auto_da_epoch > 0 else args.num_train_epochs
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    warm_steps = int(args.warmup_ratio * max_train_steps)
    lr_scheduler = get_scheduler(
        name='linear',
        optimizer=optimizer,
        num_warmup_steps=warm_steps,
        num_training_steps=max_train_steps,
    )
    return optimizer, lr_scheduler


def train():

    # 预训练的模型来源
    # https://huggingface.co/Pky/uie-base-zh/tree/main
    model = torch.load(os.path.join(args.pretrained_model, 'pytorch_model.bin'))        # 加载预训练好的UIE模型，模型结构见：model.UIE()
    model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)                    # 加载tokenizer，ERNIE 3.0
    dataset = load_dataset('text', data_files={'train': args.train_path,
                                                'dev': args.dev_path})    
    print(dataset)
    convert_func = partial(convert_example, tokenizer=tokenizer, max_seq_len=args.max_seq_len)
    dataset = dataset.map(convert_func, batched=True)
    
    train_dataset = dataset["train"]
    eval_dataset = dataset["dev"]
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=default_data_collator, batch_size=args.batch_size)
    optimizer, lr_scheduler = get_optimizer_and_scheler(model, train_dataloader)

    loss_list = []
    tic_train = time.time()
    metric = SpanEvaluator()
    criterion = torch.nn.BCELoss()
    global_step, best_f1 = 0, 0
    for epoch in range(1, args.num_train_epochs+1):
        for batch in train_dataloader:
            start_prob, end_prob = model(input_ids=batch['input_ids'].to(args.device),
                                        token_type_ids=batch['token_type_ids'].to(args.device),
                                        attention_mask=batch['attention_mask'].to(args.device))
            start_ids = batch['start_ids'].to(torch.float32).to(args.device)
            end_ids = batch['end_ids'].to(torch.float32).to(args.device)
            loss_start = criterion(start_prob, start_ids)
            loss_end = criterion(end_prob, end_ids)
            loss = (loss_start + loss_end) / 2.0
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))
            
            global_step += 1
            if global_step % args.logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                writer.add_scalar('train_loss', loss_avg, global_step)
                print("global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (global_step, epoch, loss_avg, args.logging_steps / time_diff))
                tic_train = time.time()

            if global_step % args.valid_steps == 0:
                cur_save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                torch.save(model, os.path.join(cur_save_dir, 'model.pt'))
                tokenizer.save_pretrained(cur_save_dir)

                precision, recall, f1 = evaluate(model, metric, eval_dataloader, global_step)
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
        
        if args.auto_da_epoch > 0 and epoch % args.auto_da_epoch == 0:
            train_dataloader = auto_add_samples(
                model, 
                tokenizer,
                epoch,
                convert_func
            )
            model = torch.load(os.path.join(args.pretrained_model, 'pytorch_model.bin'))        # 重新加载预训练模型
            model.to(args.device)
            optimizer, lr_scheduler = get_optimizer_and_scheler(model, train_dataloader)


if __name__ == '__main__':
    train()

# python train.py \
#     --pretrained_model "Pky/uie-base-zh" \
#     --save_dir "checkpoints/DuIE" \
#     --train_path "data/DuIE/train.txt" \
#     --dev_path "data/DuIE/dev.txt" \
#     --img_log_dir "logs/" \
#     --img_log_name "UIE Base" \
#     --batch_size 32 \
#     --max_seq_len 256 \
#     --learning_rate 5e-5 \
#     --num_train_epochs 20 \
#     --logging_steps 10 \
#     --valid_steps 100 \
#     --device cuda:0