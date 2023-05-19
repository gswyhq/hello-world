#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import torch
import os
import numpy as np
from transformers import AutoTokenizer

USERNAME = os.getenv("USERNAME")

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import torch
import transformers
from transformers import pipeline

print("torch: ", torch.__version__)
print("transformers: ", transformers.__version__)

# torch:  1.11.0+cpu
# transformers:  4.9.1

mask_model_dir = rf'D:\Users\{USERNAME}\data\bert-base-chinese'
# 模型来自：https://huggingface.co/bert-base-chinese

# https://huggingface.co/uer/chinese_roberta_L-4_H-128
# mask_model_dir = rf'D:\Users\{USERNAME}\data\chinese_roberta_L-4_H-128'

mask_model = BertForMaskedLM.from_pretrained(mask_model_dir)
mask_model.eval()
mask_tokenizer = BertTokenizer.from_pretrained(mask_model_dir)


def fill_mask(mask_model, mask_tokenizer, text = '[CLS] 我 是 [MASK] 国 人 [SEP]'):

    tokenized_text = mask_tokenizer.tokenize(text)
    indexed_tokens = mask_tokenizer.convert_tokens_to_ids(tokenized_text)

    # Create the segments tensors.
    segments_ids = [0] * len(tokenized_text)
    attention_mask = [1] * len(tokenized_text)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    attention_mask = torch.tensor([attention_mask])


    # masked_index = tokenized_text.index('[MASK]')
    masked_index = torch.nonzero(torch.tensor(indexed_tokens) == mask_tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    # Predict all tokens
    with torch.no_grad():
        # 注意此处的参数顺序，否则会导致预测结果异常；
        predictions = mask_model(tokens_tensor, attention_mask, segments_tensors)

    predicted_index = torch.argmax(predictions[0][0][masked_index], dim=-1)
    predicted_token = mask_tokenizer.convert_ids_to_tokens(predicted_index.tolist())
    print(predicted_token)

    print('-' * 50)
    for masked_index in range(predictions[0][0].shape[0]):
        predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
        predicted_token = mask_tokenizer.convert_ids_to_tokens([predicted_index])[0]
        print(predicted_token)
    return predicted_token

def example2():

    from transformers import pipeline
    # 方法1：
    classifier = pipeline("fill-mask", model=rf'D:\Users\{USERNAME}\data\bert-base-chinese',
                          config=rf'D:\Users\{USERNAME}\data\bert-base-chinese\config.json',
                          tokenizer=mask_tokenizer)
    ret_list = classifier("我是[MASK]国人")
    print(ret_list)

    # 方法2：
    classifier = pipeline("fill-mask", model=mask_model, tokenizer=mask_tokenizer)
    ret_list = classifier("巴黎是[MASK]国的首都。")
    print(ret_list)

    # 方法3：
    from transformers.pipelines.fill_mask import FillMaskPipeline
    classifier = FillMaskPipeline(model=mask_model, framework='pt', task="fill-mask", tokenizer=mask_tokenizer)
    ret_list = classifier("巴黎是[MASK]国的首都。")
    print(ret_list)

    # 方法4：
    model_inputs = mask_tokenizer("巴黎是[MASK]国的首都。", return_tensors='pt')
    input_ids = model_inputs["input_ids"][0]
    model_outputs = mask_model(**model_inputs)
    outputs = model_outputs["logits"]
    masked_index = torch.nonzero(input_ids == mask_tokenizer.mask_token_id, as_tuple=False).squeeze(-1)
    # Fill mask pipeline supports only one ${mask_token} per sample

    logits = outputs[0, masked_index, :]
    probs = logits.softmax(dim=-1)

    values, predictions = probs.topk(5)
    result = []
    single_mask = values.shape[0] == 1
    for i, (_values, _predictions) in enumerate(zip(values.tolist(), predictions.tolist())):
        row = []
        for v, p in zip(_values, _predictions):
            # Copy is important since we're going to modify this array in place
            tokens = input_ids.numpy().copy()
            tokens[masked_index[i]] = p
            # Filter padding out:
            tokens = tokens[np.where(tokens != mask_tokenizer.pad_token_id)]
            sequence = mask_tokenizer.decode(tokens, skip_special_tokens=single_mask)
            proposition = {"score": v, "token": p, "token_str": mask_tokenizer.decode([p]), "sequence": sequence}
            row.append(proposition)
        result.append(row)


def main():
    text_list = ['我 的 台 北 好 友 老 蔡 ， 在 大 陆 开 办 奶 牛 养 [MASK] 场 和 葡 萄 园 ， 孩 子 考 上 了 在 武 汉 的 大 学 。',
                 '祖 国 大 陆 始 终 [MASK] 开 胸 怀 期 待 游 子 ， 相 信 血 浓 于 水 的 亲 情 定 能 跨 越 浅 浅 的 海 峡 。',
                 '从 党 的 二 十 大 报 告 ， 到 中 央 经 济 工 作 会 议 ， 再 到 政 府 工 作 报 告 ， 都 在 [MASK] 示 着 这 样 一 个 事 实 ： 以 习 近 平 同 志 为 核 心 的 党 中 央 始 终 坚 持 党 对 经 济 工 作 的 全 面 领 导 ， 坚 持 稳 中 求 进 工 作 总 基 调 ， 坚 持 实 事 求 是 、 尊 重 规 律 、 系 统 观 念 、 底 线 思 维 ， 正 确 认 识 困 难 挑 战 ， 驾 [MASK] 经 济 工 作 的 能 力 不 断 加 强 ， 做 好 经 济 工 作 的 信 心 一 以 贯 之 。',
                 '新 增 1 0 个 社 区 养 老 服 务 [MASK] 站 ， 就 近 为 有 需 求 的 居 家 老 年 人 提 供 生 活 照 料 、 陪 伴 护 理 等 多 样 化 服 务 ， 提 升 老 年 人 生 活 质 量 。',
                 '这 些 成 就 是 中 国 人 民 团 结 一 心 、 砥 [MASK] 奋 进 的 结 果 , 也 与 外 国 友 人 的 关 心 和 支 持 密 不 可 分 。',
                 '智 能 [MASK] 备 的 普 遍 应 用 ， 让 业 务 办 理 由 人 工 转 变 为 客 户 自 助 与 半 自 助 ， 实 现 了 操 作 风 险 的 部 分 转 移 ， 使 柜 面 操 作 风 险 有 效 降 [MASK] 。 但 从 银 行 声 誉 风 险 角 度 来 讲 ， 由 于 客 户 自 助 操 作 而 引 起 的 风 险 ， 更 容 易 引 起 声 誉 风 险 。']
    for text in text_list:
        fill_mask(mask_model, mask_tokenizer, text=f'[CLS] {text} [SEP]')
        print('-'*80)
        unmasker = pipeline('fill-mask', model=mask_model, tokenizer=mask_tokenizer)
        print(unmasker(text.replace(' ', '')))
        print('#' * 80)

    # 通过几个例子抽测，bert-base-chinese的效果要好于：chinese_roberta_L-4_H-128
    fill_mask(mask_model, mask_tokenizer, text = '[CLS] 我 是 [MASK] 国 人 [SEP]')

    # 方法2：
    example2()

if __name__ == '__main__':
    main()
