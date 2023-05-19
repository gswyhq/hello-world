#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 通过ONNX来进行Pytorch到TFlite的模型转换，也就是：Pytorch—>ONNX—>Tensorflow—>TFlite
# ONNX（Open Neural Network Exchange）是一种针对机器学习所设计的开放式的文件格式，用于存储训练好的模型。
# 它使得不同的人工智能框架（如Pytorch、MXNet）可以采用相同格式存储模型数据并交互。
# 目前官方支持加载ONNX模型并进行推理的深度学习框架有： Caffe2, PyTorch, MXNet，ML.NET，TensorRT 和 Microsoft CNTK，并且 TensorFlow 也非官方地支持ONNX。
#


# 第一步：由Pytorch得到ONNX这里给出一个Pytorch的mobilenet_v2的模型转ONNX的例子，并且验证模型的输出是否相同。
import os.path as osp
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchvision

# torch --> onnx

test_arr = np.random.randn(10, 3, 224, 224).astype(np.float32)
dummy_input = torch.tensor(test_arr)
model = torchvision.models.mobilenet_v2(pretrained=True).eval()
torch_output = model(torch.from_numpy(test_arr))

input_names = ["input"]
output_names = ["output"]
torch.onnx.export(model,
                  dummy_input,
                  "mobilenet_v2.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)

model = onnx.load("mobilenet_v2.onnx")
ort_session = ort.InferenceSession('mobilenet_v2.onnx')
onnx_outputs = ort_session.run(None, {'input': test_arr})
print('Export ONNX!')

# 由pytorch转出onnx时，会看到onnx中多出了一些gather，concat等算子
# 可以在导出onnx前都先执行一遍去冗，模型会清爽不少。
# pip install onnx-simplifier
# load初始的onnx模型，通过model_simp, check = simplify(model)这一行代码即可得到去除冗余后的模型。
import onnx
from onnxsim import simplify

# 加载预训练的 ONNX model
model = onnx.load(path + model_name + '.onnx')
# 对ONNX模型进行去冗余
model_simp, check = simplify(model)
assert check, "Simplified ONNX model could not be validated"


# 第二步：由ONNX转Tensorflow，得到.pb文件
from onnx_tf.backend import prepare
import onnx

TF_PATH = "tf_model"  # where the representation of tensorflow model will be stored
ONNX_PATH = "mobilenet_v2.onnx"  # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)

# 第三步：由.pb得到TFlite
import tensorflow as tf

TF_PATH = "tf_model"
TFLITE_PATH = "mobilenet_v2.tflite"
converter = tf.lite.TFLiteConverter.from_saved_model(TF_PATH)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tf_lite_model = converter.convert()
with open(TFLITE_PATH, 'wb') as f:
    f.write(tf_lite_model)

# 来源：https://zhuanlan.zhihu.com/p/363317178

def main():
    pass


if __name__ == '__main__':
    main()
