#! encoding=utf-8

# 导入模型和分词器
import os
from transformers import BertTokenizer,BertModel,RobertaTokenizer, RobertaModel
import torch

USERNAME = os.getenv('USERNAME')
tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese') # Bert的分词器
bertmodel = BertModel.from_pretrained(rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese',from_tf=False)

# 分词器的三大核心操作：tokenize, encode, decode
# 分词器的核心操作只有三个：tokenize, encode, decode。
# tokenize负责分词，encode将分词token转换成id，decode将id转换为文本。
# encode实际上是tokenize和convert_tokens_to_ids两个操作的组合，相当于：
# self.convert_tokens_to_ids(self.tokenize(text))
# decode也是两个操作的组合，分别为convert_ids_to_tokens和convert_tokens_to_string。相当于：
# self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))

# 如果模型原有和词表不够大，我们希望增加新token，这当然是可以的。
# 做法分两步：第一步，通过add_tokens函数添加新token；第二步，使用resize_token_embeddings函数通知模型更新词表大小。
# 我们来看个例子：
# num_added_toks = tokenizer.add_tokens(['new_tok1', 'my_new-tok2'])
# print('We have added', num_added_toks, 'tokens')
# model.resize_token_embeddings(len(tokenizer))

# 除了普通token，还可以增加特殊token。
# 与普通token唯一不同的是，添加特殊token的函数add_special_tokens需要提供的是字典，因为要指定是修改哪一个特殊项。第二步的resize_token_embeddings函数还是一样的。
# 我们看个例子：
# special_tokens_dict = {'cls_token': '<CLS>'}
# num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
# print('We have added', num_added_toks, 'tokens')
# model.resize_token_embeddings(len(tokenizer))
# print(tokenizer.cls_token)

# token的保存
# 我们添加了新token之后，就需要把我们添加之后的结果保存到持久存储上。
# 这是通过save_pretrained函数来实现的。保存之后，我们可以通过from_pretrained函数加载回来。
# 例：
# tokenizer.save_pretrained("./save/")
# 保存之后会生成下面一些文件：
# added_tokens.json: 保存了新增加的token和对应的id:
# {"new_tok1": 50257, "my_new-tok2": 50258, "<CLS>": 50259}
# special_tokens_map.json：保存了特殊token列表
# {"bos_token": "<|endoftext|>", "eos_token": "<|endoftext|>", "unk_token": "<|endoftext|>", "cls_token": "<CLS>"}
# tokenizer_config.json: 保存了一些分词器的配置信息
# {"max_len": 1024, "init_inputs": []}
# vocab.json: 这个是真正的词表，保存了所有的token和对应的id值
# {"!": 0, "\"": 1, "#": 2, "$": 3, "%": 4, "&": 5, "'": 6}
# merges.txt: 存放一份对应表
# #version: 0.2
# Ġ t

# 另外，tokenizer还有一个save_vocabulary函数，不保存新增的token，所以只有vocab.json和merges.txt.
# 例：
# tokenizer.save_vocabulary('./save2')

# vocab.txt 里头“[unused*]”保留字符存在的意义是为了，当下游finetune任务需要引入先验知识时，预先提供占位符。
# 这个时候，可以保证词向量的维度不变；
# bert-uncased-base 模型就有994个此类 tokens（[unused0]to[unused993]），使用的时候，用自定义的token，替换掉这些token即可，而不需要作其他的改变。

text = " I love <e> ! "
# 对于一个句子，首尾分别加[CLS]和[SEP]。
text = "[CLS] " + text + " [SEP]"
# 然后进行分词
tokenized_text1 = tokenizer.tokenize(text)
print(tokenized_text1)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
# 分词结束后获取BERT模型需要的tensor
segments_ids1 = [1] * len(tokenized_text1)
tokens_tensor1 = torch.tensor([indexed_tokens1]) # 将list转为tensor
segments_tensors1 = torch.tensor([segments_ids1])
# 获取所有词向量的embedding
word_vectors1 = bertmodel(tokens_tensor1, segments_tensors1)[0]
# 获取句子的embedding
sentenc_vector1 = bertmodel(tokens_tensor1, segments_tensors1)[1]


tokenizer.add_special_tokens({'additional_special_tokens':["<e>"]})
print(tokenizer.additional_special_tokens) # 查看此类特殊token有哪些
print(tokenizer.additional_special_tokens_ids) # 查看其id
tokenized_text1 = tokenizer.tokenize(text)
print(tokenized_text1)
indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
# 分词结束后获取BERT模型需要的tensor
segments_ids1 = [1] * len(tokenized_text1)
tokens_tensor1 = torch.tensor([indexed_tokens1]) # 将list转为tensor
segments_tensors1 = torch.tensor([segments_ids1])
# 获取所有词向量的embedding
bertmodel.resize_token_embeddings(len(tokenizer))  # 添加了自定义的token, 则需要添加该该句 调整模型嵌入矩阵的大小，否则报错：IndexError: index out of range in self
word_vectors2 = bertmodel(tokens_tensor1, segments_tensors1)[0]
# 获取句子的embedding
sentenc_vector2 = bertmodel(tokens_tensor1, segments_tensors1)[1]

# 添加自定义的token
tokenizer.add_tokens(['尊享惠康'])
# Out[12]: 1
tokenizer.tokenize('尊享惠康的等待期是多久？')
# Out[13]: ['尊享惠康', '的', '等', '待', '期', '是', '多', '久', '？']
tokenizer.convert_tokens_to_ids(tokenizer.tokenize('尊享惠康的等待期是多久？'))
# Out[14]: [21128, 4638, 5023, 2521, 3309, 3221, 1914, 719, 8043]
#  wc -l vocab.txt
# 21127 vocab.txt, 新增的词，对应的toekn_id，是在原有的vocab.txt上面递增；
tokenizer.tokenize('[CLS] 尊享惠康的等待期是多久？ [SEP]')
# Out[15]: ['[CLS]', '尊享惠康', '的', '等', '待', '期', '是', '多', '久', '？', '[SEP]']
tokenizer.encode('尊享惠康的等待期是多久?')
# Out[19]: [101, 21128, 4638, 5023, 2521, 3309, 3221, 1914, 719, 136, 102]

# 报错与解决方案
# 在添加特殊字符后用Roberta或者Bert模型获取embedding报错：Cuda error during evaluation - CUBLAS_STATUS_NOT_INITIALIZED

# 这是因为在将模型放到eval模式后添加了新token，必须在添加新token完毕后运行以下代码：

# robertamodel.resize_token_embeddings(len(tokenizer))

