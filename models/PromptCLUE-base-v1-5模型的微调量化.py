
# 引入相应的依赖
import os,json,time
import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler

# 导入 T5 modules，chatYuan本质是T5模型
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# rich: 让终端显示的更符合观感，可以不使用
# from rich.table import Column, Table
# from rich import box
# from rich.console import Console
print("end....")

!nvidia-smi
Sun Nov  5 17:45:20 2023       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10          Off  | 00000000:00:08.0 Off |                    0 |
|  0%   26C    P8     8W / 150W |      0MiB / 22731MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

import json
#准备用于微调的句子对 这里准备一个问答句子用于演示
#注意 我们这里使用最直观的jsonl结构 我们会反复多次演示数据集的构造过程以加深印象

s={"question": "[问]你是谁？[答]", "answer": "我是微调机器人3768!"}
raw_data = [s for _ in range(10)]

#fpath,fmode,fencoding
with open('cxk.json', 'w', encoding='utf-8') as file:
  for item in raw_data:
   json_line = json.dumps(item,ensure_ascii=False) #注意这里的ensure_ascii=False 不然文件里面内容都是unicode 极其不可读
   file.write(json_line + "\n")

print('数据集文件构造完成 注意查看其中格式进行学习')

#数据加载 针对本身就存在train和test的数据集，可以这样加载
dataset = load_dataset(“json” ,data_files={‘train’:“output1.json”,‘test’:“output2.json”} )

如果json是嵌套结构，可以指定json文件中具体的key值对应的value作为加载对象,比如，field=‘data’
dataset = load_dataset(“json” ,field=‘data’,data_files={‘train’:“train_data.json” ,“test”:‘test_data.json’} )
如果本身数据集较小，只有一个文件，可以先加载再切分成train和test
from datasets import load_dataset , load_metric 注意 split=‘train’不要忘记写
raw_datasets = load_dataset(‘json’, data_files=’./data_samples.json’,split=‘train’)
my_datasets = raw_datasets.train_test_split(test_size=0.2)
print(my_datasets)

#采用第三种方式
raw_datasets = load_dataset('json', data_files='./cxk.json',split='train')
my_datasets = raw_datasets.train_test_split(test_size=0.2) #80%比例train 20% test
print(my_datasets)

# remote_model = 'ClueAI/PromptCLUE-base-v1-5'
remote_model = "/mnt/workspace/demos/lora_bnb_int8/PromptCLUE-base-v1-5"
model_id = remote_model
tokenizer = T5Tokenizer.from_pretrained(remote_model)
model = T5ForConditionalGeneration.from_pretrained(remote_model, device_map="auto")

#注意max_source_length 是token级别的 一个token可能对应多个词  所以别看max source length17好像对应不上原句长度 实际是能对应的
#所有的句子在tokenize之后 都会被处理成固定长度，超过这个长度的会截断，不足这个长度的会padding
#注意 remove_columns的用法，map函数会增加3列（由tokenizer函数实现）,而原有的question和answer列被移除 当然这只是为了选择出合适的长度，并不影响后面的流程
from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
tokenized_inputs = concatenate_datasets([my_datasets["train"], my_datasets["test"]]).map(lambda x: tokenizer(x["question"], truncation=True), batched=True, remove_columns=["question", "answer"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 100 percentile of max length for better utilization
max_source_length =int(np.percentile(input_lenghts, 100))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([my_datasets["train"], my_datasets["test"]]).map(lambda x: tokenizer(x["answer"], truncation=True), batched=True, remove_columns=["question", "answer"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 100 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 100)) #max(target_lenghts)
print(f"Max target length: {max_target_length}")
print(my_datasets['test'][0] ,'\n',len(my_datasets['test'][0])) #注意len(my_datasets['test'][0]) 是打印一个dict的len(其实就是kv对的数量) 并不是dict中value的长度

x=tokenized_targets[0]['input_ids']
print(x ,'\ntoken_len=',len(x))
xx= tokenizer.decode(x)
print(xx,"\nchar_len=",len(xx))
#应该是一个token可以对应多个汉字 所以长度不同 这是符合设计的
#</s>为eos_token 是tokenizer引入的表示结尾的特殊字符 类似的还有bos_token,padding_token

#定义数据集tokenize处理函数
def preprocess_function(sample,padding="max_length"):
    # add prefix to the input for t5
    #这里对输入做一个修饰 相当于提示这种格式才用这个风格回答 也就是prompt
    #inputs = ["[问]" + item  +"[答]" for item in sample["question"]  ]
    #句子里面已经有问和答的标记了 这里就不加了
    inputs = [item for item in sample["question"]  ]
    # tokenize inputs
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding=padding, truncation=True)

    # Tokenize targets with the `text_target` keyword argument
    labels = tokenizer(text_target=sample["answer"], max_length=max_target_length, padding=padding, truncation=True)

    # 把padding的tokenizer.pad_token_id设为-100 后面计算loss的时候根据这个设置的值 忽略补齐字符padding token

    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs



tokenized_dataset = my_datasets.map(preprocess_function, batched=True, remove_columns=['question' ,'answer'])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

# 保存datasets 后续备用
# 训练文件现在是arrow格式
tokenized_dataset["train"].save_to_disk("data/train")
tokenized_dataset["test"].save_to_disk("data/eval")

from transformers import DataCollatorForSeq2Seq

# 计算loss时忽略pad token，即label_pad_token_id=-100的token不参与loss计算
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8 #网上都说这个值比较合适 按照8的倍数进行padding
)

# 模型训练：
#使用了auto_find_batch_size=True  也可以自己定batch_size
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

log_output_dir="Log_finetune_chatyuan_v1_0416" #训练日志位置
model_output_dir = 'trained_chatyuan_v1_0419'
# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=model_output_dir,
		auto_find_batch_size=True,
    learning_rate=1e-4, # higher learning rate
    num_train_epochs=100,
    logging_dir=f"{log_output_dir}",
    logging_strategy="steps",
    logging_steps=5,
    save_strategy="no",
    report_to="tensorboard",
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"], #只用到了train的部分 当然也可以全用
)
model.config.use_cache = False  # silence the warnings. Re-enable for inference

# train model
trainer.train() #这个输出的train_loss 貌似有bug 首次train的最终loss很大 需要训练两次才能正确？

#将训练好的模型保存
trainer.save_model('0419_trained_chatyuan_v1')

#一同保存tokenizer
tokenizer.save_pretrained('0419_trained_chatyuan_v1')



import torch
from transformers import T5Tokenizer,T5ForConditionalGeneration
from torch import cuda

# 原始的fp32精度加载训练好的模型
#model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
device = 'cuda' if cuda.is_available() else 'cpu'
trained_model = '0419_trained_chatyuan_v1'

model_trained = T5ForConditionalGeneration.from_pretrained(trained_model,  load_in_8bit=False, device_map='auto' )#device_map={"":0}
tokenizer_trained = AutoTokenizer.from_pretrained(trained_model)

model_trained.eval()
print("trained model loaded")


# 训练后的模型infer
from datasets import load_dataset
from random import randrange

model_trained.config.use_cache = True

# Load dataset from the hub and get a sample
#test_dataset = load_dataset("json" ,data_files={"test":'test_data.json'} ,field='data' )
sample = my_datasets['test'][randrange(len(my_datasets["test"]))]['question']

#s1=sample # sample =[问]'篮球+背带裤，告诉我他是谁？'[答]

#试试不要提示词
s1='你是谁？' 
s2 ="苹果是什么颜色?"
s3 ="请介绍一下深度学习有哪些模型?"
s4 ="请写一个判断数字n奇数还是偶数的python程序。" 
s5 ="写一段python函数，一个判断数字n是否大于100的。"
#s_test=sample["Q"]"
#input_ids = tokenizer(s_test, return_tensors="pt", truncation=True).input_ids.cuda()
list_s = [s1,s2,s3,s4,s5]
#print(list_s)
list_inputs_ids = [tokenizer_trained(s, return_tensors="pt", truncation=True).input_ids.cuda() for s in list_s]
#print(list_inputs_ids)
#top_p 越高 生成结果越多样 temperature越低 分布越冻结趋向于集中的确定性的结果 所以temperature=1.0（默认） or smaller结果就越确定
with torch.inference_mode():
  for input_ids_index in range(len(list_inputs_ids)):
    #print(tokenizer.batch_decode(list_inputs_ids[input_ids_index], skip_special_tokens=True)[0])
    outputs = model_trained.generate(input_ids=list_inputs_ids[input_ids_index], max_new_tokens=100, do_sample=True,top_p=0.9,temperature=1.7)
    print(f"\n{'---'*20}")
    print(f"Q:{list_s[input_ids_index]}")
    print(f"A:{tokenizer_trained.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

# 没有fintune的模型infer
# 对比原始没有finetune的model输出

#model_id是最开始保存原始模型的路径
remote_model = "/mnt/workspace/demos/lora_bnb_int8/PromptCLUE-base-v1-5"
model_id = remote_model
model_ori = T5ForConditionalGeneration.from_pretrained(model_id,  load_in_8bit=False, device_map='auto' )#device_map={"":0}
tokenizer_ori = T5Tokenizer.from_pretrained(model_id)

model_ori.eval()
print("original model loaded")

from datasets import load_dataset
from random import randrange

model_ori.config.use_cache = True

# Load dataset from the hub and get a sample
#test_dataset = load_dataset("json" ,data_files={"test":'test_data.json'} ,field='data' )
sample = my_datasets['test'][randrange(len(my_datasets["test"]))]['question']

s1='你是谁？' #"你觉得抖音公司会对接化发的梗进行限流吗？"
#s1 = "###Question###" + s1 +"###Answer###" #与训练时的prompt保持风格一致
s2 ="苹果是什么颜色?"
s3 ="请介绍一下深度学习有哪些模型?"
s4 ="请写一个判断数字n奇数还是偶数的python程序。"
s5 ="写一段python函数，一个判断数字n是否大于100的。"
#s_test=sample["Q"]"
#input_ids = tokenizer(s_test, return_tensors="pt", truncation=True).input_ids.cuda()
list_s = [s1,s2,s3,s4,s5]
#print(list_s)
list_inputs_ids = [tokenizer_ori(s, return_tensors="pt", truncation=True).input_ids.cuda() for s in list_s]
#print(list_inputs_ids)
#top_p 越高 生成结果越多样 temperature越低 分布越冻结趋向于集中的确定性的结果 所以temperature=1.0（默认） or smaller结果就越确定
with torch.inference_mode():
  for input_ids_index in range(len(list_inputs_ids)):
    #print(tokenizer.batch_decode(list_inputs_ids[input_ids_index], skip_special_tokens=True)[0])
    outputs = model_ori.generate(input_ids=list_inputs_ids[input_ids_index], max_new_tokens=100, do_sample=True,top_p=0.9,temperature=1.7)
    print(f"\n{'---'*20}")
    print(f"Q:{list_s[input_ids_index]}")
    print(f"A:{tokenizer_ori.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

资料来源：
https://github.com/valkryhx/lora_bnb_int8
