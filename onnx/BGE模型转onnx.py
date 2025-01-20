#!/usr/bin/env python
# coding=utf-8
import os
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig

USERNAME = os.getenv("USERNAME")
# 模型来源：https://www.modelscope.cn/models/AI-ModelScope/bge-small-zh-v1.5/files

BGE_MODEL_PATH = rf"D:\Users\{USERNAME}\data\bge-small-zh-v1.5"
tokenizer = AutoTokenizer.from_pretrained(BGE_MODEL_PATH)
config = AutoConfig.from_pretrained(BGE_MODEL_PATH)
bge_model = AutoModel.from_pretrained(BGE_MODEL_PATH)
bge_model.eval()

# 设置线程池的线程数
torch.set_num_threads(4)
with torch.no_grad():
    try:
        model_output = bge_model(**tokenizer(['我喜欢吃牛肉面，你喜欢吃什么'], padding=True, truncation=True, return_tensors='pt'))
        sentence_embeddings = model_output[0][:, 0]
    except RuntimeError as e:
        print(e)
sentence_embeddings
# Out[11]:
# tensor([[ 5.4689e-01, -4.4698e-01,  5.5987e-01,  5.0286e-01,  3.6766e-01,
#          -1.9772e-01, -5.1068e-02,  4.6330e-02, -3.4700e-01,  3.0879e-01,
#           3.6935e-01, -1.6736e+00, -2.0509e-01, -2.6857e-01, -2.6716e-02,

###########################################################################################################################
# 导出 onnx 模型
# !pip install onnx==1.16.1 onnxruntime==1.20.1 -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 导出 onnx 模型
import onnxruntime
from itertools import chain
from transformers.onnx.features import FeaturesManager
onnx_config = FeaturesManager._SUPPORTED_MODEL_TYPE['bert']['sequence-classification'](config)
dummy_inputs = onnx_config.generate_dummy_inputs(tokenizer, framework='pt')
output_onnx_path = "bge-small-zh-v1.5.onnx"
model = bge_model

torch.onnx.export(
    bge_model,
    (dummy_inputs,),
    f=output_onnx_path,
    input_names=list(onnx_config.inputs.keys()),
    output_names=list(onnx_config.outputs.keys()),
    dynamic_axes={
        name: axes for name, axes in chain(onnx_config.inputs.items(), onnx_config.outputs.items())
    },
    do_constant_folding=True,
    # use_external_data_format=onnx_config.use_external_data_format(model.num_parameters()),
    # enable_onnx_checker=True,
    # opset_version=onnx_config.default_onnx_opset,
)

# 加载运行 onnx 模型
from onnxruntime import SessionOptions, GraphOptimizationLevel, InferenceSession

output_onnx_path = "bge-small-zh-v1.5.onnx"
options = SessionOptions() # initialize session options
options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
# 设置线程数
options.intra_op_num_threads = 4


# 这里的路径传上一节保存的onnx模型地址
session = InferenceSession(
    output_onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
)

# disable session.run() fallback mechanism, it prevents for a reset of the execution provider
session.disable_fallback()

text = ['我喜欢吃牛肉面，你喜欢吃什么']
inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
inputs_detach = {k: v.detach().cpu().numpy() for k, v in inputs.items()}

# 运行 ONNX 模型
# 这里的logits要有export的时候output_names相对应

output = session.run(output_names=['logits'], input_feed=inputs_detach)
embeddings = output[0][:,0]
embeddings

# array([[ 5.46887517e-01, -4.46978569e-01,  5.59869409e-01,
#          5.02857625e-01,  3.67655814e-01, -1.97720945e-01,
#         -5.10679260e-02,  4.63301912e-02, -3.47001076e-01,
#          3.08788389e-01,  3.69348496e-01, -1.67355609e+00,
#         -2.05090344e-01, -2.68571585e-01, -2.67164223e-02,

def main():
    pass


if __name__ == "__main__":
    main()
