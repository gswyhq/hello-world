#!/usr/bin/env python
# coding=utf-8

# bitsandbytes是对CUDA自定义函数的轻量级封装，特别是针对8位优化器、矩阵乘法（LLM.int8()）和量化函数。
# 故而bitsandbytes量化，只能是在GPU上面进行。
# bitsandbytes的安装，整体要求：Python >= 3.8。Linux发行版（如Ubuntu、MacOS等）+ CUDA > 10.0。（已弃用：CUDA 10.0已被弃用，只支持CUDA >= 11.0，将在0.39.0版本发布中支持）
# 库依赖要求：anaconda、cudatoolkit、pytorch

# pip install bitsandbytes
# 或者
# pip install -i https://pypi.tuna.tsinghua.edu.cn/simple bitsandbytes

# 对模型进行量化：
# 你只需要在AutoModelForCausalLM.from_pretrained中添加你的量化配置，即可使用量化模型。如下所示：

from transformers import BitsAndBytesConfig

# 将模型量化为NF4 (4 bits)的配置
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

# 将模型量化成Int8精度的配置 Int8 (8 bits)
quantization_config = BitsAndBytesConfig(load_in_8bit=True)

model = AutoModelForCausalLM.from_pretrained(
    args.checkpoint_path,
    device_map="cuda:0",
    quantization_config=quantization_config, # 量化的精度配置
    max_memory=max_memory,
    trust_remote_code=True,
).eval()

# 上述方法可以将模型量化成NF4和Int8精度的模型进行读取，

# 在HuggingFace Transformers中使用Int8推断
# 更详细的示例可以在examples / int8_inference_huggingface.py中找到

from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    'decapoda-research/llama-7b-hf',
    device_map = 'auto',
    load_in_8bit = True,
    max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB')

def main():
    pass


if __name__ == "__main__":
    main()
