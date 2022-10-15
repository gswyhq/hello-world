#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from transformers import BertTokenizer, BartForConditionalGeneration, Text2TextGenerationPipeline
USERNAME = os.getenv("USERNAME")

# 模型来自：https://huggingface.co/fnlp/bart-base-chinese/tree/main
pretrained_model_name_or_path = rf"D:\Users\{USERNAME}\data\bart-base-chinese"

tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path)
text2text_generator = Text2TextGenerationPipeline(model, tokenizer)
result = text2text_generator("北京是[MASK]的首都", max_length=50, do_sample=True)
print(result)

# 上面仅输出一个结果，若要多个结果，可参考：bert/用bert进行mask预测.py

def main():
    pass


if __name__ == '__main__':
    main()
