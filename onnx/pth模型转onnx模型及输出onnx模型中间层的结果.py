#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn
import onnx

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

model = torch.load('***.pth', map_location=device)
model.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化

input_names = ['input']
output_names = ['output']

x = torch.randn(1, 3, 224, 224, device=device)

torch.onnx.export(model, x, '***.onnx', input_names=input_names, output_names=output_names, verbose='True')

#########################################################################################################################

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification, AutoConfig, convert_graph_to_onnx
# 模型来源：https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese
# model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
# tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
USERNAME = os.getenv('USERNAME')

model = AutoModelForMaskedLM.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese', output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')
model.config.output_hidden_states = True
model.eval()
onnx_model_file = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/teacher_model.onnx'
texta = '今天天气真不错，我们去散步吧！'
textb = '今天天气真糟糕，还是在宅家里写bug吧！'
inputs_a = tokenizer(texta, return_tensors="pt")
inputs_b = tokenizer(textb, return_tensors="pt")


outputs_a = model(**inputs_a, output_hidden_states=True)
texta_embedding = outputs_a.hidden_states[-1][:, 0, :].squeeze()

# 方法一：
torch.onnx.export(
            model,
            (inputs_a['input_ids'], inputs_a['attention_mask'], inputs_a['token_type_ids']),
            onnx_model_file,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask', 'token_type_ids'],
            output_names=['predictions'],
            dynamic_axes=None,
            verbose=False,
            keep_initializers_as_inputs=None)

# model：需要导出的 PyTorch 模型
# args：PyTorch模型输入数据的尺寸，指定通道数、长和宽。可以是单个 Tensor 或元组，也可以是元组列表。
# f：导出的 ONNX 文件路径和名称，mymodel.onnx。
# export_params：是否导出模型参数。如果设置为 False，则不导出模型参数。
# opset_version：导出的 ONNX 版本。默认值为 10。
# do_constant_folding：是否对模型进行常量折叠。如果设置为 True，不加载模型的权重。
# input_names：模型输入数据的名称。默认为 'input'。
# output_names：模型输出数据的名称。默认为 'output'。
# dynamic_axes：动态轴的列表，允许在导出的 ONNX 模型中创建变化的维度。
# verbose：是否输出详细的导出信息。
# example_outputs：用于确定导出 ONNX 模型输出形状的样本输出。
# keep_initializers_as_inputs：是否将模型的初始化器作为输入导出。如果设置为 True，则模型初始化器将被作为输入的一部分导出。

# 导出的时候，报错：
# ValueError: You have to specify either decoder_input_ids or decoder_inputs_embeds
# 解决方法：
# 在input_names参数中添加一个参数项：decoder_input_ids
# 例如：
# torch.onnx.export(
#             model,
#             (encoding['input_ids'], encoding['attention_mask'], encoding['input_ids']),
#             onnx_model_file,
#             export_params=True,
#             opset_version=11,
#             do_constant_folding=True,
#             input_names=['input_ids', 'attention_mask', 'decoder_input_ids'],
#             output_names=['sequences'],
#             dynamic_axes=None,
#             verbose=False,
#             keep_initializers_as_inputs=None)
#########################################################################################################################
# transformers model to onnx模型
from transformers.models.bert.configuration_bert import BertOnnxConfig
from pathlib import Path
onnx_config = BertOnnxConfig(model.config)
onnx_path = Path(onnx_model_file)

# 方法二：
onnx_inputs, onnx_outputs = transformers.onnx.export(tokenizer, model, onnx_config, onnx_config.default_onnx_opset, onnx_path)

#########################################################################################################################
# 加载转换的onnx模型进行预测：
from onnxruntime import InferenceSession, get_available_providers
session = InferenceSession(onnx_model_file, providers=get_available_providers())
input_feed = {k: v.numpy() for k, v in inputs_a.items()}
output_name = session.get_outputs()[0].name
ret = session.run([output_name], input_feed)[0]

# 上面预测结果是 output_hidden_states=False时的结果，即默认获取的是 last_hidden_state, 要想获取hidden_states结果，则需要输出onnx中间层的结果；
# bert的输出是由四部分组成：
# last_hidden_state：shape是(batch_size, sequence_length, hidden_size)，hidden_size=768,它是模型最后一层输出的隐藏状态。（通常用于命名实体识别）
# pooler_output：shape是(batch_size, hidden_size)，这是序列的第一个token(classification token)的最后一层的隐藏状态，它是由线性层和Tanh激活函数进一步处理的。（通常用于句子分类，至于是使用这个表示，还是使用整个输入序列的隐藏状态序列的平均化或池化，视情况而定）
# hidden_states：这是输出的一个可选项，如果输出，需要指定config.output_hidden_states=True,它也是一个元组，它的第一个元素是embedding，其余元素是各层的输出，每个元素的形状是(batch_size, sequence_length, hidden_size)
# attentions：这也是输出的一个可选项，如果输出，需要指定config.output_attentions=True,它也是一个元组，它的元素是每一层的注意力权重，用于计算self-attention heads的加权平均值。

############################################################################################################################################################
# onnx 模型输出所有中间层的结果
# 默认情况下获取到的是 last_hidden_state, 要想获取hidden_states结果，则需要输出onnx中间层的结果；
import copy
import onnx
from collections import OrderedDict
onnx_model = onnx.load(onnx_model_file)
ori_output = copy.deepcopy(onnx_model.graph.output)
for node in onnx_model.graph.node:
    for output in node.output:
        onnx_model.graph.output.extend([onnx.ValueInfoProto(name=output)])
ort_session = InferenceSession(onnx_model.SerializeToString())
outputs = [x.name for x in ort_session.get_outputs()]
ort_inputs = input_feed
ort_outs = ort_session.run(outputs, ort_inputs)
ort_outs = OrderedDict(zip(outputs, ort_outs))
ort_outs = OrderedDict(zip(outputs, ort_outs))

del onnx_model.graph.output[:]
onnx_model.graph.output.extend(ori_output)
# 保存可以输出中间层结果的模型，这样下次load 后，直接输入中间层名称，即可获取对应的结果
multi_output_onnx_model_file = onnx_model_file.replace('.onnx', '_nul_output.onnx')
onnx.save(onnx_model, multi_output_onnx_model_file)

# 遍历所有中间层，输出对应的结果, 根据结果，选择对应的层的key
for name in ort_outs.keys():
    ret = ort_session.run([name], ort_inputs)
    if len(ret)==1 and ret[0].shape == (1, 17, 768):
        print(name, ret[0][:,0,:].squeeze()[:10])

ret = ort_session.run(['onnx::MatMul_1606'], ort_inputs)

########################################################################################################################
# 改变模型精度
import onnxmltools
# 加载float16_converter转换器
from onnxmltools.utils.float16_converter import convert_float_to_float16

fp16_model_file = onnx_model_file.replace('.onnx', '_fp16.onnx')

# 使用convert_float_to_float16()函数将fp32模型转换成半精度fp16
onnx_model_fp16 = convert_float_to_float16(onnx_model)
# 使用onnx.utils.save_model()函数来保存，
onnxmltools.utils.save_model(onnx_model_fp16, fp16_model_file)

def main():
    pass


if __name__ == '__main__':
    main()
