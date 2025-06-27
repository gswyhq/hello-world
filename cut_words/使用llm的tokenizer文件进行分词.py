#!/usr/bin/env python
# coding=utf-8

from transformers import AutoTokenizer
import os
USERNAME = os.getenv("USERNAME")
# 指定分词器文件(包括：tokenizer.json和tokenizer_config.json)所在的目录
tokenizer_dir = rf"D:\Users\{USERNAME}\Downloads\test3"

# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

# 进行分词，获取token IDs
tokens = tokenizer.encode("<think>让我想想怎么回答</think>")
# 将token IDs转换为对应的token字符串
tokens_str = tokenizer.convert_ids_to_tokens(tokens)
print("tokens:", tokens)
print("原始句子：", tokenizer.decode(tokens))
# 输出每个token的字符串表示
print("每个token的字符串表示:", tokens_str)

for token in tokens:
    decoded_sentence = tokenizer.decode([token], skip_special_tokens=True)
    # if decoded_sentence in (tokenizer.bos_token, tokenizer.eos_token, tokenizer.unk_token, tokenizer.sep_token, tokenizer.pad_token, tokenizer.cls_token, tokenizer.mask_token):
    #     continue
    print(f"token: {token}, 分词结果：{decoded_sentence}")

# skip_special_tokens=True 表示跳过特殊 token（如 [CLS] 和 [SEP]）
# tokenizer.decode：用于将 token IDs 转换为完整的、可读的文本句子。
# tokenizer.convert_ids_to_tokens：用于将 token IDs 转换为每个 token 的字符串表示，可能会包含子词和特殊字符，因此看起来像是“乱码”。
# tokens: [151646, 151648, 104029, 105839, 99494, 102104, 151649]
# 原始句子： <｜begin▁of▁sentence｜><think>让我想想怎么回答</think>
# 每个token的字符串表示: ['<｜begin▁of▁sentence｜>', '<think>', 'è®©æĪĳ', 'æĥ³æĥ³', 'æĢİä¹Ī', 'åĽŀçŃĶ', '</think>']
# token: 151646, 分词结果：
# token: 151648, 分词结果：<think>
# token: 104029, 分词结果：让我
# token: 105839, 分词结果：想想
# token: 99494, 分词结果：怎么
# token: 102104, 分词结果：回答
# token: 151649, 分词结果：</think>

def main():
    pass


if __name__ == "__main__":
    main()

