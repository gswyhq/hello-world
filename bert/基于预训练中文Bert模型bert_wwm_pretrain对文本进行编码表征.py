#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 先下载并解压线上环境中的模型压缩包：

# !wget -nc "https://labfile.oss.aliyuncs.com/courses/3382/bert_wwm_pretrain.zip"
# md5sum bert_wwm_pretrain.zip
# 78876ab4579c10ac25dc9524f82b93e3  bert_wwm_pretrain.zip
# !unzip "bert_wwm_pretrain.zip"+
# 解压后目录下有三个文件：md5sum *
# 677977a2f51e09f740b911866423eaa5  config.json
# e98d02abc2124d2f348edc8fadf0df7a  pytorch_model.bin
# 3b5b76c4aef48ecf8cb3abaafe960f09  vocab.txt

import os
USERNAME = os.getenv('USERNAME')

BERT_WWM_PRETRAIN_PATH = rf'D:\Users\{USERNAME}\Downloads\bert_wwm_pretrain'
# 首先将待表征文本进行分词：


import torch
from transformers import BertTokenizer
from transformers import BertModel
from transformers import BertForNextSentencePrediction
from transformers import BertForMaskedLM, BertConfig


# 加载中文预训练 Bert 模型的 tokenizer
tokenizer_bert = BertTokenizer.from_pretrained(BERT_WWM_PRETRAIN_PATH)

# 对输入进行 tokenize
text = "[CLS] 我是谁？ [SEP] 我在哪里？ [SEP]"
# 特殊的标记
# BERT可以接受一到两句话作为输入，并希望每句话的开头和结尾都有特殊的标记：
# 2个句子的输入:
# [CLS] 白日依山尽 [SEP] 黄河入海流 [SEP]
# 1个句子的输入:
# [CLS] 白日依山尽 [SEP]
tokenized_text = tokenizer_bert.tokenize(text)
print(tokenized_text)  # 查看分词后的句子

# ['[CLS]', '我', '是', '谁', '？', '[SEP]', '我', '在', '哪', '里', '？', '[SEP]']

# 词汇表中包含的一些token示例。以两个#号开头的标记是子单词或单个字符。
list(tokenizer_bert.vocab.keys())[5000:5020]

# 接下来将文本转换为 id 并且转化成 tensor 的形式。同时，由于 bert 还带有 Next Sentence Prediction 任务，还需要准备句子类型相关的向量 segments_ids。

# 将文本中的单词转化为对应的 id
indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)
# 调用tokenizer来查看tokens在tokenizer词汇表中的索引：
for tup in zip(tokenized_text, indexed_tokens):
    print (tup)
# 句子类型相关的 id，第一个句子中的单词对应的 segment id 为 0（包括第一个 [SEP]），第二个的为 1
segments_ids = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]

# 转换为 tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])
print(tokens_tensor)
print(segments_tensors)

# 加载模型：
# 加载中文预训练 Bert 模型
model = BertModel.from_pretrained(BERT_WWM_PRETRAIN_PATH)  # 'bert_wwm_pretrain'


# 将模型设置为预测模式
# eval()将我们的模型置于评估模式，而不是训练模式。在这种情况下，评估模式关闭了训练中使用的dropout正则化。
model.eval()

# 如果有 GPU, 将数据与模型转换到 cuda
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 运行环境
tokens_tensor = tokens_tensor.to(DEVICE)
segments_tensors = segments_tensors.to(DEVICE)
model.to(DEVICE)
print(model)

# 最后调用模型，进行文本编码：

# 对输入进行编码
# 接下来，让我们获取网络的隐藏状态。
# torch.no_grad禁用梯度计算，节省内存，并加快计算速度(我们不需要梯度或反向传播，因为我们只是运行向前传播)。
with torch.no_grad():
    outputs = model(tokens_tensor, token_type_ids=segments_tensors)
    # 取最后一层的输出
    encoded_layers = outputs[0]
print(encoded_layers.shape)
# 句子编码后，大小为： (batch size, sequence length, model hidden dimension)
assert tuple(encoded_layers.shape) == (
    1, len(indexed_tokens), model.config.hidden_size)
# 编码后的向量可看作文本语义表征，再可以根据具体任务接相应的下游层（如文本分类），也可以直接用于比较文本间的相似度（如文本匹配）。
# 前面提到，Bert 在预训练阶段还带有 Next Sentence Prediction 任务，因此可以天然地做输入文本是否相关的任务。对于以上输入"[CLS] 我是谁？ [SEP] 我在哪里？ [SEP]"，可直接预测两句子是否相关。


# 加载中文预训练 Bert 模型
model_nsp = BertForNextSentencePrediction.from_pretrained(BERT_WWM_PRETRAIN_PATH)  # 'bert_wwm_pretrain'

model_nsp.eval()
model_nsp.to(DEVICE)
print(model_nsp)


# 对输入进行相关性预测：

with torch.no_grad():
    outputs = model_nsp(tokens_tensor, token_type_ids=segments_tensors)
print(outputs)
# 是否相关为二分类问题，因此输出为二维向量。
# 除了 NSP，Bert 的另一大任务是基于上下文的单词预测，也就是所谓的“完型填空”，也可以将输入文本进行遮盖，让模型预测被遮盖的单词。

# Mask 文本中的某一词汇，之后进行预测
masked_index = 3
tokenized_text[masked_index] = '[MASK]'
print('掩码后的句子为：', tokenized_text)


# 加载模型：
# 加载中文预训练 Bert 语言模型
model_lm = BertForMaskedLM.from_pretrained(BERT_WWM_PRETRAIN_PATH)  # 'bert_wwm_pretrain'
model_lm.eval()
model_lm.to(DEVICE)
print(model_lm)
# 预测被 MASK 的单词：

# 单词预测
with torch.no_grad():
    outputs = model_lm(tokens_tensor, token_type_ids=segments_tensors)
    predictions = outputs[0]

# 预测被 MASK 的单词
predicted_index = torch.argmax(predictions[0, masked_index]).item()
predicted_token = tokenizer_bert.convert_ids_to_tokens([predicted_index])[0]
print("预测的字符为：", predicted_token)
# 基于 transformers, 研究者可以共享训练过的模型，而不用总是重新训练；而实践者可以减少计算时间和制作成本，这是 NLP 工程师需要熟练使用的库。
# 更多复杂用法可参考 transformers 教程(https://github.com/huggingface/transformers)。


# 输出隐藏层
# 导入配置文件
BERT_BASE_CHINESE_PATH = rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese'
model_config = BertConfig.from_pretrained(BERT_BASE_CHINESE_PATH)
# 修改配置
model_config.output_hidden_states = True  # 默认为False, 若不设置，则仅仅输出最后一层；
model_config.output_attentions = True  # 默认为False

# 通过配置和路径导入模型
bert_model = BertModel.from_pretrained(BERT_BASE_CHINESE_PATH, config = model_config)
# 加载中文预训练 Bert 模型的 tokenizer
tokenizer_bert = BertTokenizer.from_pretrained(BERT_BASE_CHINESE_PATH)
text = '白日依山尽'
marked_text = "[CLS] " + text + " [SEP]"
tokenized_text = tokenizer_bert.tokenize(marked_text)
indexed_tokens = tokenizer_bert.convert_tokens_to_ids(tokenized_text)

segments_ids = [0] * len(tokenized_text)
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

bert_model.eval()
# 接下来，让我们获取网络的隐藏状态。
# torch.no_grad禁用梯度计算，节省内存，并加快计算速度(我们不需要梯度或反向传播，因为我们只是运行向前传播)。
with torch.no_grad():
    outputs = bert_model(tokens_tensor, segments_tensors)
    sequence_output, pooled_output, hidden_states, attentions = outputs
    # [1, 7, 768]

# sequence_output：torch.Size([1, 7, 768])               输出序列, 其中，7是单词/token数(在我们的句子中有7个token) 即 len(tokenized_text) = 7; 768: 特征个数
# pooled_output：torch.Size([1, 768])                          对输出序列进行pool操作的结果
# (hidden_states)：tuple, 13*torch.Size([1, 7, 768])   隐藏层状态（包括Embedding层），取决于model_config中的output_hidden_states
# (attentions)：tuple, 12*torch.Size([1, 12, 7, 7])      注意力层，取决于model_config中的output_attentions



def main():
    pass


if __name__ == '__main__':
    main()