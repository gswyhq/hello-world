#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# onnx模型转tensorflow的pb格式
# pip install onnx_tf

import onnx
from onnx_tf.backend import prepare
import os
import tensorflow as tf

USERNAME = os.getenv('USERNAME')
onnx_input_path = rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese/teacher_model.onnx'
pb_output_path = onnx_input_path.replace('.onnx', '.pb')

def onnx2pb(onnx_input_path, pb_output_path):
    onnx_model = onnx.load(onnx_input_path)  # load onnx model
    tf_exp = prepare(onnx_model)  # prepare tf representation
    tf_exp.export_graph(pb_output_path)  # export the model
    # 保存之后会在 pb_output_path 下生成如下三个文件: assets  saved_model.pb  variables

##########################################################################################################################
# 加载pb模型进行测试
from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification, AutoConfig
# 模型来源：https://huggingface.co/IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese
# model =AutoModelForMaskedLM.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
# tokenizer = AutoTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-SimCSE-110M-Chinese')
USERNAME = os.getenv('USERNAME')

# model = AutoModelForMaskedLM.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese', output_hidden_states=True)
tokenizer = AutoTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data/Erlangshen-SimCSE-110M-Chinese')

texta = '今天天气真不错，我们去散步吧！'
textb = '今天天气真糟糕，还是在宅家里写bug吧！'
inputs_a = tokenizer(texta, return_tensors="pt")
inputs_b = tokenizer(textb, return_tensors="pt")
pb_model = tf.keras.models.load_model(pb_output_path)

# 这里onnx模型是由pytorch模型转换而来，若是keras模型的话，可能预测的时候是使用接口：model.predict
ret_a = pb_model(**inputs_a)
ret_b = pb_model(**inputs_b)
texta_embedding = ret_a['onnx::MatMul_1606'][:, 0, :].numpy().squeeze()
textb_embedding = ret_b['onnx::MatMul_1606'][:, 0, :].numpy().squeeze()

def main():
    onnx2pb(onnx_input_path, pb_output_path)


if __name__ == '__main__':
    main()
