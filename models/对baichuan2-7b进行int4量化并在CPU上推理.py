#!/usr/lib/python3
# -*- coding: utf-8 -*-


# 第一步，下载模型：

import torch
from modelscope import snapshot_download, Model
# model_dir = snapshot_download("baichuan-inc/Baichuan2-7B-Chat-4bits", revision='v1.0.1')

model_dir = "/mnt/workspace/.cache/modelscope/baichuan-inc/Baichuan2-7B-Chat"

model = Model.from_pretrained(model_dir, device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
messages = []
messages.append({"role": "user", "content": "讲解一下“温故而知新”"})
response = model(messages)
print(response)
messages = response['history'].copy()
messages.append({"role": "user", "content": "背诵一下将进酒"})
response = model(messages)
print(response)

# git clone https://github.com/ggerganov/llama.cpp
# cd llama.cpp


# 第二步：将 pytorch模型转换为 f16-gguf文件

# python3 convert.py /mnt/workspace/.cache/modelscope/baichuan-inc/Baichuan2-7B-Chat

# 编译，构建量化命令；
# root@dsw-30793-5bb9f5b986-k7442:/mnt/workspace/demos/llama.cpp# make

# 第三步：int4量化
# root@dsw-30793-5bb9f5b986-k7442:/mnt/workspace/demos/llama.cpp# ./quantize /tmp/Baichuan2-7B-Chat/ggml-model-f16.gguf /tmp/Baichuan2-7B-Chat-q4_0.bin q4_0
# root@dsw-30793-5bb9f5b986-k7442:/mnt/workspace/demos/llama.cpp# du -sh /tmp/Baichuan2-7B-Chat/ggml-model-f16.gguf
# 14G     /tmp/Baichuan2-7B-Chat/ggml-model-f16.gguf
# root@dsw-30793-5bb9f5b986-k7442:/mnt/workspace/demos/llama.cpp# du -sh /tmp/Baichuan2-7B-Chat*
# 28G     /tmp/Baichuan2-7B-Chat
# 4.1G    /tmp/Baichuan2-7B-Chat-q4_0.bin

# cpu上面加载模型进行预测：
# gguf_file = "/tmp/Baichuan2-7B-Chat/ggml-model-f16.gguf"

from langchain.llms import CTransformers

# Local CTransformers wrapper for Llama-2-7B-Chat
llm = CTransformers(model='/tmp/Baichuan2-7B-Chat-q4_0.bin', # Location of downloaded GGML model
                    model_type='baichuan', # Model type Llama
                    config={'max_new_tokens': 1024,
                            'temperature': 0.01})

print(llm("你好"))

print(llm( '''<指令>假设你是一位出色的保险顾问，根据已知信息，简洁和专业的来回答问题。如果无法从中得到答案，请友好地拒绝回答。</指令>
 <已知信息>Q:本合同接受的首次投保年龄范围是多少？A:本合同接受的首次投保年龄范围为出生满30天至60周岁。</已知信息>
 <问题>我爸爸今年52岁能否投保呢？</问题><答案>：'''))


def main():
    pass


if __name__ == '__main__':
    main()
