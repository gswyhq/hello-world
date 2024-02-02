

查看环境信息：
# 查看GPU的信息
!nvidia-smi
Sun Nov  5 17:20:42 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.8     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA A10          Off  | 00000000:00:08.0 Off |                    0 |
|  0%   27C    P8     8W / 150W |      0MiB / 22731MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+

# 使用以下命令清除训练中残存的GPU显存缓存
# torch.cuda.empty_cache()

# 安装需要的包 install libraries
!pip install sentencepiece
#!pip install transformers
!pip install git+https://github.com/huggingface/transformers.git  #安装使用的是github上的transformer版本
!pip install torch
!pip install rich[jupyter]
!pip install peft   #这个是用于peft加速的 会使用到其中的LoRA低秩适应技术

!pip install datasets
!pip install  "accelerate==0.17.1" "evaluate==0.4.0" "bitsandbytes==0.37.1" loralib #bitsandbytes用于8int量化，因为原始的fp32精度模型加载或训练会OOM
!pip install rouge-score tensorboard py7zr

import os,json
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import os,time
# 导入 T5 modules，chatYuan本质是T5模型
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM,TrainingArguments, Trainer
#下面这些依赖是LoRA相关的 peft是hf提供的第三方库
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
# rich: 让终端显示的更符合观感，可以不使用
from rich.table import Column, Table
from rich import box
from rich.console import Console
print("end....")

# 下载模型到本地
#[如果本地已经下载好模型文件则不用重复执行]这里先从hf上下载chatyuan到本地保存 方便后续使用
#模型bin文件 config文件 tokenizer文件 其他附属文件 一并保存 后续这些文件都要存在
#ClueAI/ChatYuan-large-v2 modelsize = 3.3G OOM
#ClueAI/PromptCLUE-base-v1-5 modelsize = 1G ok
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# model_id='ClueAI/ChatYuan-large-v2' #远程加载

local_base_model_path = "/mnt/workspace/demos/lora_bnb_int8/ChatYuan-large-v2"
model_id = local_base_model_path

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
# Save tokenizer and model locally
tokenizer.save_pretrained(local_base_model_path)
model.save_pretrained(local_base_model_path)
print("tokenizer and base model saved")


# 查看一下LoRA的trainable param的占比
#LoRA会对模型的某些注意力层做低rank矩阵分解，训练时主模型的参数冻结不变，训练的是那些低秩的矩阵。
peft_config = LoraConfig(  #TaskType.CAUSAL_LM TaskType.SEQ_2_SEQ_LM
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q','v'],
    bias="none",
)

print(model)
#LoRA会在target_modules=['q','v']也就是名为q，v的layer建立可训练的旁路矩阵
model_peft = get_peft_model(model, peft_config)
model_peft.print_trainable_parameters()


#这是一种正确的数据组织格式
#是下面cell中preprocess_function函数输出参数的sample格式，每个特征一个list 构建train/test数据集  注意数据集的格式要符合datasets的格式要求
import json
from datasets import load_dataset
idx=[]
Q=[]
A=[]
for i in range(23):#大小为23的数据集 被分成两部分：train和test
  idx.append(i)
  Q.append("你好？")
  A.append('我不好，今天是星期八！')


train_data = { 'data':{'idx':idx[:20],  'Q':Q[:20] ,'A':A[:20]} }
test_data =  {  'data':{'idx':idx[20:], 'Q':Q[20:] ,'A':A[20:]} }
with open('./train_data.json','w',encoding='utf8' ) as f:
  json.dump(train_data ,f)
with open('./test_data.json','w',encoding='utf8' ) as f:
  json.dump(test_data ,f)
print(train_data)
print(test_data)

#下载alapaca中文数据集
#这次只使用其中的传统文化类语料进行测试
#一共31个instruct_input_output格式的问答
# 如果没有下载这个仓库，可以使用下面命令进行clone

!git clone https://github.com/hikariming/alpaca_chinese_dataset.git

%mkdir -p my_data
%cp alpaca_chinese_dataset/其他中文问题补充/传统诗词及文化常识问题.json my_data

%mv my_data/传统诗词及文化常识问题.json my_data/culture.json

#将alpaca格式的原始语料处理成json line格式
#alpaca 原始结构是一个list，其中每个元素是一个json，
#即[{"instruction":"XXX"},{"input":"YYY"},{"output":"ZZZZ"}]
#json line格式就是每行一个json结构
#{'en': 'The Babel & konqueror; plugin', 'fr': 'Le module externe Babel pour & konqueror;'}
#{'en': 'Using the Babelfish plugin', 'fr': 'Utilisation du module externe Babelfish'}
import json
import os

def load_json_from_file(file_name):
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

#alpaca data 预处理成json line 格式的函数
def convert_to_target_format(data, start_id):
    result = []

    for entry in data:
        question = entry["instruction"]
        input_info=entry["input"]
        raw_answer = entry["output"].replace("\\n\\n", "\n \n")
        #这份代码原始处理方式把结果整的太复杂 我修改下
        #answer = raw_answer.split("\n")
        #result.append({"id": start_id, "paragraph": [{"q": question,'i':inputs, "a": answer}]})
        answer = raw_answer
        result.append({'q':question,'i':input_info,'a':answer})
        start_id += 1

    return result

def save_json_to_file(data, file_name):
    with open(file_name, 'w', encoding='utf-8') as file:
        for item in data:
            json_line = json.dumps(item, ensure_ascii=False)
            file.write(json_line + "\n")


#调用
corpus_folder = './my_data'
files = os.listdir(corpus_folder)

merged_data = []
#这里会把corpus_folder下所有的json文件都进行处理，整合成一个结果json文件
for file in files:
    if file.endswith('.json'):
        file_path = os.path.join(corpus_folder, file)
        data = load_json_from_file(file_path)
        merged_data.extend(data)

start_id = 1
converted_data = convert_to_target_format(merged_data, start_id)
#保存处理好的数据集文件
#每一行结构如下
#{"q": "“春江潮水连海平，海上明月共潮生。”出自哪首诗？", "i": "", "a": "这句诗出自张若虚的《春江花月夜》。"}
save_json_to_file(converted_data, './data_samples.json')

#针对本身就存在train和test的数据集，可以这样加载
#dataset = load_dataset("json" ,data_files={'train':"output1.json",'test':"output2.json"} )
#dataset

#如果json是嵌套结构，可以指定json文件中具体的key值对应的value作为加载对象,
#比如，field='data'
dataset = load_dataset("json" ,field='data',data_files={'train':"train_data.json" ,"test":'test_data.json'}  )
#如果本身数据集较小，只有一个文件，可以先加载再切分成train和test

from datasets import load_dataset , load_metric
#注意 split='train'不要忘记写
raw_datasets = load_dataset('json', data_files='./data_samples.json',split='train')
my_datasets = raw_datasets.train_test_split(test_size=0.2)
print(my_datasets)


#这里是找到合适的长度进行padding，小于该长度的句子被padding，超过的被截断
#注意max_source_length 是token级别的 一个token可能对应多个词
#所以别看max source length好像对应不上原句长度 实际是能对应的
#所有的句子在tokenize之后 都会被处理成固定长度，超过这个长度的会截断，不足这个长度的会padding
from datasets import concatenate_datasets
import numpy as np
# The maximum total input sequence length after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded.
# 注意 x["q"],x['i'] 不能直接x["q"]+x['i'] ，那样是在尾部续上，长度变成原先2倍，要用zip的方式按照index对应做合并，长度不变。
tokenized_inputs = concatenate_datasets([my_datasets["train"], my_datasets["test"]]).map(lambda x: tokenizer([q+i for (q,i) in zip(x["q"],x['i'])], truncation=True), batched=True, remove_columns=["q","i","a"])
input_lenghts = [len(x) for x in tokenized_inputs["input_ids"]]
# take 95 percentile of max length for better utilization
max_source_length =int(np.percentile(input_lenghts, 100))
print(f"Max source length: {max_source_length}")

# The maximum total sequence length for target text after tokenization.
# Sequences longer than this will be truncated, sequences shorter will be padded."
tokenized_targets = concatenate_datasets([my_datasets["train"], my_datasets["test"]]).map(lambda x: tokenizer(x["a"], truncation=True), batched=True, remove_columns=["q","i","a"])
target_lenghts = [len(x) for x in tokenized_targets["input_ids"]]
# take 90 percentile of max length for better utilization
max_target_length = int(np.percentile(target_lenghts, 100)) #max(target_lenghts)
print(f"Max target length: {max_target_length}")
#print(my_datasets['test'][0])
#print(len(my_datasets['test'][0])) #这个只会输出my_datasets['test'][0]的键值对的个数，不是文字长度，注意!


in_text_idx=tokenized_inputs[-1]['input_ids']
print(in_text_idx ,'\ntoken_len=',len(in_text_idx))
in_text= tokenizer.decode(in_text_idx)
print(in_text,"\nchar_len=",len(in_text))
out_text_idx=tokenized_targets[-1]['input_ids']
print(out_text_idx ,'\ntoken_len=',len(out_text_idx))
out_text= tokenizer.decode(out_text_idx)
print(out_text,"\nchar_len=",len(out_text))
#应该是一个token可以对应多个汉字 所以token_ken和char_len长度不同 这是符合设计的

#注意sample的格式【不是！】[{'q':xxx, 'i':yyy ,'a':zzz}]
#而是一个字典sample={"q":[q1,q2,....] ,"i":[i1,i2,....] ,"a":[a1,a2,.....]}
#print(sample)看看加深理解
def preprocess_function(sample,padding="max_length"):
    print(f"sample={sample}")
    # add prefix to the input for t5
    #这里对输入做一个修饰 相当于提示这种格式才用这个风格回答 也就是prompt
    #zip(sample['q'], sample['i'])结果是 [(q1,i1) ,(q2,i2)....]
    #这里将tuple_item中[0]位置key_q对应的instruction和[1]位置key_i对应的input_info进行拼接
    #拼接成 [q]instruction[i]input_info[a] 的形式，后续进行tokenize
    text_inputs = ["[q]" + tuple_item[0]  +"[i]" + tuple_item[1] + '[a]' for tuple_item in zip(sample['q'], sample['i'])  ]
    print(text_inputs)
    # tokenize inputs
    model_inputs = tokenizer(text_inputs, max_length=max_source_length, padding=padding, truncation=True)
    #print(2)
    text_targets = sample['a']
    print(text_targets)
    # Tokenize targets with the `text_target` keyword argument
    #labels = tokenizer(text_target=sample["A"], max_length=max_target_length, padding=padding, truncation=True)
    labels = tokenizer(text_target=text_targets, max_length=max_target_length, padding=padding, truncation=True)
    #print(3)
    # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
    # padding in the loss.
    #将padding token对应的数字改成-100，这样在计算loss时 padding token会被忽略
    #这个细节在下面定义data_collato时会用到
    '''
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
    #tokenizer,
    #model=model,
    #label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
      )
    '''
    if padding == "max_length":
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#调用preprocess_function
tokenized_dataset = my_datasets.map(preprocess_function, batched=True, remove_columns=["q", "i", "a"])
print(f"Keys of tokenized dataset: {list(tokenized_dataset['train'].features)}")

#保存datasets 后续备用
tokenized_dataset["train"].save_to_disk("culture_train")
tokenized_dataset["test"].save_to_disk("culture_eval")

# 模型训练:LoRA + bitsandbytes int8模型 微调训练
from transformers import AutoModelForSeq2SeqLM
#base_model_id = "ClueAI/ChatYuan-large-v2" #hf远程的模型
base_model_id =local_base_model_path  #已经放好在本地文件的模型


#使用int8量化的方式加载 注意模型本身还是f32的
from transformers import AutoModelForSeq2SeqLM

# huggingface hub model id
#model_id = "ClueAI/ChatYuan-large-v2" #远程
model_id = local_base_model_path
# load model from the hub or locally
#ValueError: ('You cannot load weights that are saved in int8 using `load_in_8bit=True`, make sure you are', ' using
#`load_in_8bit=True` on float32/float16/bfloat16 weights.')
#本地模型如果已经是int8 那就不能再次设置load_in_8bit=True


'''重要！这个model加载不能.cuda() 因为下面有model = prepare_model_for_int8_training(model)
  带.cuda直接会导致模型训练loss不降 学不到东西'''
model = AutoModelForSeq2SeqLM.from_pretrained(model_id, load_in_8bit=True, device_map="auto")#.cuda()

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

#Define LoRA Config
lora_config = LoraConfig(
    r=8, # 可调16
    lora_alpha=32, # 可调
    target_modules=["q", "v"], #注意回顾上面print 模型参数时的layers
    lora_dropout=0.1, #0.05 bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM
    )

#以int8精度准备模型
model = prepare_model_for_int8_training(model)

#加入LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

from transformers import DataCollatorForSeq2Seq

# we want to ignore tokenizer pad token in the loss
# 计算loss时忽略pad token，即label_pad_token_id=-100的token不参与loss计算
label_pad_token_id = -100
# Data collator
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

#使用了auto_find_batch_size=True  也可以自己定batch_size
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

log_output_dir="./log_LoRA_chatyuan_finetune_0327" #训练日志位置

# Define training args
training_args = Seq2SeqTrainingArguments(
    output_dir=log_output_dir,
	auto_find_batch_size=True,
    #per_device_train_batch_size =3 ,
    learning_rate=1e-3, # higher learning rate
    num_train_epochs=100,
    #lr_scheduler_type="linear",#"cosine loss下降很慢,
    logging_dir=f"{log_output_dir}/logs",
    logging_strategy="steps",
    logging_steps=20,
    save_strategy="no",
    report_to="tensorboard",
    gradient_accumulation_steps=1,#8
    #weight_decay=0.1,
    #warmup_steps=10,
    #lr_scheduler_type="cosine",

    #fp16=True,
)

# Create Trainer instance
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset["train"],
)
model.config.use_cache = False  # silence the warnings. Re-enable for inference

# train model
trainer.train() #这个输出的train_loss 貌似有bug 首次train的最终loss很大 需要训练两次才能正确？

#保存lora模型参数
peft_model_id="./lora_chatyuan_0327_model"
trainer.model.save_pretrained(peft_model_id)
#tokenizer.save_pretrained(peft_model_id)
# if you want to save the base model to call

#保存基座模型
base_model = "./out_base_model_0327" #这个是int8的模型 与原始的fp32模型不是一个
trainer.model.base_model.save_pretrained(base_model)
tokenizer.save_pretrained(base_model)

import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer,T5ForConditionalGeneration
from torch import cuda
# Load peft config for pre-trained checkpoint etc.
#这是lora模型参数的路径
peft_model_id = "./lora_chatyuan_0327_model"
#这是原始fp32精度模型的路径
# original_f32_base_model = "./base_model_chatyuan"
original_f32_base_model = "/mnt/workspace/demos/lora_bnb_int8/ChatYuan-large-v2"
#这是int8精度模型的路径
int8_base_model = "./out_base_model_0327"

config = PeftConfig.from_pretrained(peft_model_id)
print("config=",config)

# 加载基座模型
#model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path,  load_in_8bit=True,  device_map={"":0})
#tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
#上面那个 config.base_model_name_or_path 是远程的model 这里换成本地的
# device = 'cuda' if cuda.is_available() else 'cpu'
model = AutoModelForSeq2SeqLM.from_pretrained(original_f32_base_model,  load_in_8bit=True,  device_map='auto')
tokenizer = AutoTokenizer.from_pretrained(original_f32_base_model)

# 加载LoRA模型
model = PeftModel.from_pretrained(model, peft_model_id, device_map='auto').float()
model.eval()
print("Peft model loaded")


model.config.use_cache = True  # silence the warnings. Re-enable for inference

from datasets import load_dataset
from random import randrange

# Load dataset from the hub and get a sample
#dataset = load_dataset("json" ,data_files={"test":'test_data.json'} ,field='data' )
#sample = dataset['test'][randrange(len(dataset["test"]))]


#测一下新学的知识
s1="写出关于秋天的名句?"
# s1 = "###Question###" + s1 +"###Answer###" #与训练时的prompt保持风格一致
#下面这些是原来的知识
s2 ="作为一个智能语言模型，请介绍一下你自己?"
s3 ="请介绍一下深度学习有哪些模型?"
s4 ="请写一个判断数字n奇数还是偶数的python程序。"
s5 ="写出关于秋天的古诗，请多写一些不一样的给我参考"
#s_test=sample["Q"]"
#input_ids = tokenizer(s_test, return_tensors="pt", truncation=True).input_ids.cuda()
list_s = [s1,s2,s3,s4,s5]
#print(list_s)
list_inputs_ids = [tokenizer(s, return_tensors="pt", truncation=True).input_ids.cuda() for s in list_s]
#print(list_inputs_ids)
#top_p 越高 生成结果越多样 temperature越低 分布越冻结趋向于集中的确定性的结果 所以temperature=1.0（默认） or smaller结果就越确定
with torch.inference_mode():
  for input_ids_index in range(len(list_inputs_ids)):
    #print(tokenizer.batch_decode(list_inputs_ids[input_ids_index], skip_special_tokens=True)[0])
    outputs = model.generate(input_ids=list_inputs_ids[input_ids_index], max_new_tokens=400, do_sample=True,top_p=0.9,temperature=0.8)
    print(f"\n{'---'*20}")
    print(f"Q:{list_s[input_ids_index]}")
    print(f"A:{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}".replace('\\n','\n').replace('%20',' '))

#20230329测试 下面的代码可以在训练后动态量化 post dynamic quantize 的模式下
#加载fp32模型原始模型（不带LoRA参数） 量化为int8 实施推理
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
print(device)
saved_int8_base_model = "./out_base_model_0327"
base_model_chatyuan = "/mnt/workspace/demos/lora_bnb_int8/ChatYuan-large-v2"
tokenizer = T5Tokenizer.from_pretrained(base_model_chatyuan)
model = T5ForConditionalGeneration.from_pretrained(base_model_chatyuan,device_map="cpu")
#int8 model目前只能cpu加载
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
# model = model.to(device)
input_text = "苹果的颜色是？"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#output = model.generate(input_ids)
with torch.inference_mode():
  outputs = model.generate(input_ids=input_ids, max_new_tokens=400, do_sample=True,top_p=0.9,temperature=1.0)
  print(f"Q: {input_text}\n{'---'* 20}")
  print(f"A:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")



  #注意这里能看到原始没有微调的模型的表现
input_text = "写出关于秋天的诗句?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
#output = model.generate(input_ids)
with torch.inference_mode():
  outputs = model.generate(input_ids=input_ids, max_new_tokens=400, do_sample=True,top_p=0.9,temperature=2.0)
  print(f"Q: {input_text}\n{'---'* 20}")
  print(f"A:\n{tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")

资料来源：
https://github.com/valkryhx/lora_bnb_int8

!pip3 list
Package                       Version
----------------------------- --------------------
absl-py                       1.4.0
accelerate                    0.22.0
adaseq                        0.6.5
addict                        2.4.0
aiobotocore                   2.7.0
aiofiles                      23.2.1
aiohttp                       3.8.5
aioitertools                  0.11.0
aiosignal                     1.3.1
albumentations                1.3.1
aliyun-python-sdk-core        2.14.0
aliyun-python-sdk-kms         2.16.2
altair                        5.1.2
aniso8601                     9.0.1
annotated-types               0.5.0
antlr4-python3-runtime        4.9.3
anyio                         3.7.1
apex                          0.1
appdirs                       1.4.4
argon2-cffi                   23.1.0
argon2-cffi-bindings          21.2.0
arrow                         1.2.3
astropy                       5.2.2
asttokens                     2.2.1
astunparse                    1.6.3
async-lru                     2.0.4
async-timeout                 4.0.3
attrs                         23.1.0
audioread                     3.0.0
auto-gptq                     0.4.2
av                            10.0.0
Babel                         2.12.1
backcall                      0.2.0
backoff                       2.2.1
backports.zoneinfo            0.2.1
basicsr                       1.4.2
beautifulsoup4                4.12.2
bidict                        0.22.1
biopython                     1.81
bitarray                      2.8.1
bitsandbytes                  0.41.1
bitstring                     4.1.2
black                         23.9.1
bleach                        6.0.0
blinker                       1.6.3
blis                          0.7.11
blobfile                      2.0.2
bmt-clipit                    1.0
boltons                       23.0.0
boto3                         1.28.54
botocore                      1.31.54
Bottleneck                    1.3.7
cachetools                    5.3.1
catalogue                     2.0.10
certifi                       2023.7.22
cffi                          1.15.1
cfgv                          3.4.0
chardet                       5.2.0
charset-normalizer            2.0.4
chumpy                        0.70
cityscapesScripts             2.2.2
click                         8.1.7
clip                          1.0
cloudpickle                   2.2.1
cmake                         3.27.2
colorama                      0.4.6
coloredlogs                   14.0
comm                          0.1.4
conda                         23.7.2
conda-content-trust           0+unknown
conda-libmamba-solver         23.7.0
conda-package-handling        1.9.0
confection                    0.1.3
ConfigArgParse                1.7
contextlib2                   21.6.0
contourpy                     1.1.1
control-ldm                   0.0.1
cpm-kernels                   1.0.11
crcmod                        1.7
cryptography                  41.0.2
cycler                        0.11.0
cymem                         2.0.8
Cython                        0.29.36
dataclasses                   0.6
dataclasses-json              0.6.1
datasets                      2.13.0
ddpm-guided-diffusion         0.0.0
debugpy                       1.8.0
decorator                     4.4.2
decord                        0.6.0
deepspeed                     0.10.3
defusedxml                    0.7.1
descartes                     1.1.0
detectron2                    0.6
dgl                           1.1.1+cu118
diffusers                     0.21.2
dill                          0.3.6
diskcache                     5.6.3
Distance                      0.1.3
distlib                       0.3.7
dnspython                     2.3.0
easydict                      1.10
easyrobust                    0.2.4
edit-distance                 1.0.6
editdistance                  0.6.2
effdet                        0.4.1
einops                        0.6.1
embeddings                    0.0.8
emoji                         2.8.0
espnet-tts-frontend           0.0.3
et-xmlfile                    1.1.0
eventlet                      0.33.3
exceptiongroup                1.1.3
executing                     1.2.0
expecttest                    0.1.6
face-alignment                1.4.1
fairscale                     0.4.13
fairseq                       0.12.2
faiss                         1.7.2+cu113
faiss-cpu                     1.7.4
fastai                        2.7.12
fastapi                       0.104.0
fastcore                      1.5.29
fastdownload                  0.0.7
fastjsonschema                2.18.0
fastprogress                  1.0.3
fasttext                      0.9.2
ffmpeg                        1.4
ffmpeg-python                 0.2.0
ffmpy                         0.3.1
filelock                      3.12.2
filetype                      1.2.0
fire                          0.5.0
flake8                        6.1.0
Flask                         2.2.5
Flask-Cors                    4.0.0
Flask-RESTful                 0.3.10
Flask-SocketIO                5.3.6
flask-talisman                1.1.0
flatbuffers                   23.5.26
fonttools                     4.42.1
fqdn                          1.5.1
frozenlist                    1.4.0
fschat                        0.2.31
fsspec                        2023.9.2
ftfy                          6.1.1
funasr                        0.7.8
funtextprocessing             0.1.1
future                        0.18.3
fvcore                        0.1.5.post20221221
g2p                           1.1.20230822
g2p-en                        2.1.0
gast                          0.4.0
gitdb                         4.0.11
GitPython                     3.1.40
google-auth                   2.22.0
google-auth-oauthlib          1.0.0
google-pasta                  0.2.0
gradio                        3.50.2
gradio_client                 0.6.1
greenlet                      2.0.2
grpcio                        1.57.0
h11                           0.14.0
h5py                          3.9.0
hdbscan                       0.8.33
healpy                        1.16.5
hjson                         3.1.0
httpcore                      0.18.0
httptools                     0.6.1
httpx                         0.25.0
huggingface-hub               0.17.3
humanfriendly                 10.0
hydra-core                    1.3.2
HyperPyYAML                   1.2.2
identify                      2.5.29
idna                          3.4
imageio                       2.31.4
imageio-ffmpeg                0.4.9
imgaug                        0.4.0
importlib-metadata            6.8.0
importlib-resources           6.1.0
inflect                       7.0.0
iniconfig                     2.0.0
iopath                        0.1.9
ipdb                          0.13.13
ipykernel                     6.25.2
ipython                       8.12.2
isoduration                   20.11.0
isort                         5.12.0
itsdangerous                  2.1.2
jaconv                        0.3.4
jamo                          0.4.1
jedi                          0.19.0
jieba                         0.42.1
Jinja2                        3.1.2
jmespath                      0.10.0
joblib                        1.3.2
json-tricks                   3.17.3
json5                         0.9.14
jsonpatch                     1.33
jsonplus                      0.8.0
jsonpointer                   2.1
jsonschema                    4.19.1
jsonschema-specifications     2023.7.1
jupyter_client                8.3.1
jupyter_core                  5.3.1
jupyter-events                0.7.0
jupyter-lsp                   2.2.0
jupyter_server                2.7.3
jupyter_server_terminals      0.4.4
jupyterlab                    4.0.6
jupyterlab-pygments           0.2.2
jupyterlab_server             2.25.0
kaldiio                       2.18.0
kantts                        1.0.1
keras                         2.13.1
kiwisolver                    1.4.5
kornia                        0.7.0
kwsbp                         0.0.6
langchain                     0.0.326
langchain-experimental        0.0.36
langcodes                     3.3.0
langdetect                    1.0.9
langsmith                     0.0.54
lap                           0.4.0
layoutparser                  0.3.4
libclang                      16.0.6
libmambapy                    1.4.1
librosa                       0.9.2
lightning-utilities           0.9.0
lit                           16.0.6
llama_cpp_python              0.2.11
llvmlite                      0.41.0
lmdb                          1.4.1
lpips                         0.1.4
lxml                          4.9.3
lyft-dataset-sdk              0.0.8
Markdown                      3.4.4
markdown-it-py                3.0.0
markdown2                     2.4.10
MarkupSafe                    2.1.3
marshmallow                   3.20.1
matplotlib                    3.5.3
matplotlib-inline             0.1.6
mccabe                        0.7.0
mdurl                         0.1.2
megatron-util                 1.3.2
MinDAEC                       0.0.2
mir-eval                      0.7
mistune                       3.0.1
ml-collections                0.1.1
mmcls                         0.25.0
mmcv-full                     1.7.0
mmdet                         2.28.2
mmdet3d                       1.0.0a1
mmsegmentation                0.30.0
mock                          5.1.0
modelscope                    1.9.4
moviepy                       1.0.3
mpi4py                        3.1.4
mpmath                        1.3.0
ms-swift                      1.2.0
msg-parser                    1.2.0
msgpack                       1.0.6
multidict                     6.0.4
multiprocess                  0.70.14
MultiScaleDeformableAttention 1.0
munkres                       1.1.4
murmurhash                    1.0.10
mypy-extensions               1.0.0
nara-wpe                      0.0.9
nbclient                      0.8.0
nbconvert                     7.8.0
nbformat                      5.9.2
nerfacc                       0.2.2
nest-asyncio                  1.5.8
networkx                      2.8.4
nh3                           0.2.14
ninja                         1.11.1
nltk                          3.8.1
nodeenv                       1.8.0
notebook_shim                 0.2.3
numba                         0.58.0
numexpr                       2.8.6
numpy                         1.24.4
nuscenes-devkit               1.1.11
nvdiffrast                    0.3.1
nvidia-cublas-cu12            12.1.3.1
nvidia-cuda-cupti-cu12        12.1.105
nvidia-cuda-nvrtc-cu12        12.1.105
nvidia-cuda-runtime-cu12      12.1.105
nvidia-cudnn-cu12             8.9.2.26
nvidia-cufft-cu12             11.0.2.54
nvidia-curand-cu12            10.3.2.106
nvidia-cusolver-cu12          11.4.5.107
nvidia-cusparse-cu12          12.1.0.106
nvidia-nccl-cu12              2.18.1
nvidia-nvjitlink-cu12         12.3.52
nvidia-nvtx-cu12              12.1.105
oauthlib                      3.2.2
olefile                       0.46
omegaconf                     2.3.0
onnx                          1.14.1
onnxruntime                   1.15.1
onnxsim                       0.4.33
open-clip-torch               2.20.0
openai                        0.28.1
opencv-python                 4.8.0.76
opencv-python-headless        4.8.0.76
openpyxl                      3.1.2
opt-einsum                    3.3.0
optimum                       1.13.2
orjson                        3.9.9
oss2                          2.18.2
overrides                     7.4.0
packaging                     23.0
pai-easycv                    0.11.4
paint-ldm                     0.0.0
pandas                        2.0.3
pandocfilters                 1.5.0
panopticapi                   0.1
panphon                       0.20.0
parso                         0.8.3
pathlib                       1.0.1
pathspec                      0.11.2
pathy                         0.10.2
pdf2image                     1.16.3
pdfminer.six                  20221105
pdfplumber                    0.10.3
peft                          0.5.0
pexpect                       4.8.0
pickleshare                   0.7.5
Pillow                        9.5.0
pip                           23.2.1
pkgutil_resolve_name          1.3.10
platformdirs                  3.10.0
plotly                        5.17.0
pluggy                        1.0.0
plyfile                       1.0.1
pointnet2                     0.0.0
pooch                         1.7.0
portalocker                   2.7.0
pre-commit                    3.4.0
preshed                       3.0.9
prettytable                   3.9.0
proglog                       0.1.10
prometheus-client             0.17.1
prompt-toolkit                3.0.39
protobuf                      4.24.4
psutil                        5.9.5
ptflops                       0.7
ptyprocess                    0.7.0
pure-eval                     0.2.2
py-cpuinfo                    9.0.0
py-sound-connect              0.2.1
pyarrow                       13.0.0
pyasn1                        0.5.0
pyasn1-modules                0.3.0
pybind11                      2.11.1
pyclipper                     1.3.0.post5
pycocoevalcap                 1.2
pycocotools                   2.0.7
pycodestyle                   2.11.0
pycosat                       0.6.4
pycparser                     2.21
pycryptodome                  3.19.0
pycryptodomex                 3.19.0
pydantic                      1.10.12
pydantic_core                 2.10.0
pydeck                        0.8.1b0
pyDeprecate                   0.3.2
pydot                         1.4.2
pydub                         0.25.1
pyerfa                        2.0.0.3
pyflakes                      3.1.0
Pygments                      2.16.1
PyMCubes                      0.1.4
PyMuPDF                       1.22.5
PyMuPDFb                      1.23.5
pynini                        2.1.5
pynndescent                   0.5.10
pyOpenSSL                     23.2.0
pypandoc                      1.12
pyparsing                     3.1.1
pypdfium2                     4.22.0
pypinyin                      0.49.0
pyquaternion                  0.9.9
PySocks                       1.7.1
pysptk                        0.1.18
pytesseract                   0.3.10
pytest                        7.4.2
pythainlp                     4.0.2
python-crfsuite               0.9.9
python-dateutil               2.8.2
python-decouple               3.8
python-docx                   1.0.1
python-dotenv                 1.0.0
python-engineio               4.7.1
python-iso639                 2023.6.15
python-json-logger            2.0.7
python-magic                  0.4.27
python-multipart              0.0.6
python-pptx                   0.6.21
python-socketio               5.9.0
pytorch-lightning             1.7.7
pytorch-metric-learning       2.3.0
pytorch-wavelets              1.3.0
pytorch-wpe                   0.0.1
pytorch3d                     0.7.4
pytz                          2023.3.post1
pyvi                          0.1.1
PyWavelets                    1.4.1
PyYAML                        6.0.1
pyzmq                         25.1.1
qudida                        0.0.4
rapidfuzz                     3.3.1
rapidocr-onnxruntime          1.3.8
ray                           2.7.1
referencing                   0.30.2
regex                         2023.8.8
requests                      2.31.0
requests-oauthlib             1.3.1
resampy                       0.4.2
rfc3339-validator             0.1.4
rfc3986-validator             0.1.1
rich                          13.5.3
rotary-embedding-torch        0.3.0
rouge                         1.0.1
rouge-score                   0.0.4
rpds-py                       0.10.3
rsa                           4.9
ruamel.yaml                   0.17.32
ruamel.yaml.clib              0.2.7
s3fs                          2023.10.0
s3transfer                    0.6.2
sacrebleu                     2.3.1
sacremoses                    0.0.53
safetensors                   0.3.3
scikit-image                  0.19.3
scikit-learn                  1.3.1
scipy                         1.10.1
seaborn                       0.12.2
semantic-version              2.10.0
Send2Trash                    1.8.2
sentence-transformers         2.2.2
sentencepiece                 0.1.99
seqeval                       1.2.2
setuptools                    68.0.0
Shapely                       1.8.4
shortuuid                     1.0.11
shotdetect-scenedetect-lgss   0.0.4
simple-websocket              0.10.1
simplejson                    3.19.1
six                           1.16.0
sklearn-crfsuite              0.3.6
smart-open                    6.4.0
smmap                         5.0.1
smplx                         0.1.28
sniffio                       1.3.0
sortedcontainers              2.4.0
soundfile                     0.12.1
soupsieve                     2.5
sox                           1.4.1
spacy                         3.6.1
spacy-legacy                  3.0.12
spacy-loggers                 1.0.5
speechbrain                   0.5.15
SQLAlchemy                    2.0.19
srsly                         2.4.8
stack-data                    0.6.2
stanza                        1.5.1
starlette                     0.27.0
streamlit                     1.28.0
streamlit-aggrid              0.3.4.post3
streamlit-antd-components     0.2.3
streamlit-chatbox             1.1.10
streamlit-option-menu         0.3.6
subword-nmt                   0.3.8
svgwrite                      1.4.3
sympy                         1.12
tabulate                      0.9.0
taming-transformers-rom1504   0.0.6
tb-nightly                    2.14.0a20230808
tbb                           2021.10.0
tblib                         3.0.0
tenacity                      8.2.3
tensorboard                   2.13.0
tensorboard-data-server       0.7.1
tensorboardX                  2.6.2
tensorflow                    2.13.0
tensorflow-estimator          2.13.0
tensorflow-io-gcs-filesystem  0.33.0
tensorrt                      8.6.1.post1
tensorrt-bindings             8.6.1
tensorrt-libs                 8.6.1
termcolor                     2.3.0
terminado                     0.17.1
terminaltables                3.1.10
text-unidecode                1.3
text2sql-lgesql               1.3.0
TextGrid                      1.5
tf-slim                       1.1.0
thinc                         8.1.12
thop                          0.1.1.post2209072238
threadpoolctl                 3.2.0
tifffile                      2023.7.10
tiktoken                      0.5.1
timm                          0.9.8
tinycss2                      1.2.1
tinycudann                    1.6
tokenizers                    0.14.1
toml                          0.10.2
tomli                         2.0.1
toolz                         0.12.0
torch                         2.0.1+cu118
torch-complex                 0.4.3
torch-scatter                 2.1.1
torchaudio                    2.0.2+cu118
torchkeras                    3.9.4
torchmetrics                  0.11.4
torchsummary                  1.5.1
torchvision                   0.15.2+cu118
tornado                       6.3.3
tqdm                          4.65.0
traitlets                     5.9.0
transformers                  4.34.1
transformers-stream-generator 0.0.4
trimesh                       2.35.39
triton                        2.0.0
ttsfrd                        0.2.1
typeguard                     2.13.3
typer                         0.9.0
typing                        3.7.4.3
typing_extensions             4.8.0
typing-inspect                0.9.0
tzdata                        2023.3
tzlocal                       5.2
ujson                         5.8.0
umap-learn                    0.5.4
unicodecsv                    0.14.1
unicodedata2                  15.0.0
unicore                       0.0.1
Unidecode                     1.3.7
unstructured                  0.10.28
unstructured-inference        0.7.10
unstructured.pytesseract      0.3.12
uri-template                  1.3.0
urllib3                       1.26.16
utils                         1.0.1
uvicorn                       0.23.2
uvloop                        0.19.0
validators                    0.22.0
videofeatures-clipit          1.0
virtualenv                    20.24.5
vllm                          0.2.0
wasabi                        1.1.2
watchdog                      3.0.0
watchfiles                    0.21.0
Wave                          0.0.2
wavedrom                      2.0.3.post3
wcwidth                       0.2.6
webcolors                     1.13
webencodings                  0.5.1
websocket-client              1.6.3
websockets                    11.0.3
wenetruntime                  1.11.0
Werkzeug                      2.2.3
wget                          3.2
wheel                         0.38.4
wrapt                         1.15.0
wsproto                       1.2.0
xformers                      0.0.22.post7
xinference                    0.5.4
xlrd                          2.0.1
XlsxWriter                    3.1.9
xorbits                       0.7.0
xoscar                        0.1.3
xtcocotools                   1.13
xxhash                        3.3.0
yacs                          0.1.8
yapf                          0.30.0
yarl                          1.9.2
zhconv                        1.4.3
zipp                          3.16.2

