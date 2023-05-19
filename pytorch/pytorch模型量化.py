#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch

# 方法1. 静态量化 ：torch.quantize_per_tensor
# scale （标度）和 zero_point（零点位置）需要自定义。量化后的模型，不能训练（不能反向传播），也不能推理，需要解量化后，才能进行运算
inputs = torch.rand(2, 2)
q = torch.quantize_per_tensor(inputs,scale=0.025, zero_point=0, dtype=torch.quint8)
q.dtype
q.dequantize().dtype

# 以zero_point为中心，用8位数Q代表input离中心有多远，scale为距离单位
# 即input ≈ zero_point + Q * scale.
#
# dtype为量化类型，quint8代表8位无符号数，qint8代表8位带符号数，最高位是符号位
# 计算机运算时，默认32位浮点数，若将32位浮点数，变成8位定点数，会快很多。
# 目前pytorch中的反向传播不支持量化，所以该量化只用于评估训练好的模型，或者将32位浮点数模型存储为8位定点数模型，读取8位定点数模型后需要转换为32位浮点数才能进行神经网络参数的训练。
#
# 量化函数原型：Q = torch.quantize_per_tensor(input,scale = 0.025 , zero_point = 0, dtype = torch.quint8)
# >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8)
# tensor([-1.,  0.,  1.,  2.], size=(4,), dtype=torch.quint8,
#        quantization_scheme=torch.per_tensor_affine, scale=0.1, zero_point=10)
# >>> torch.quantize_per_tensor(torch.tensor([-1.0, 0.0, 1.0, 2.0]), 0.1, 10, torch.quint8).int_repr()
# tensor([ 0, 10, 20, 30], dtype=torch.uint8)
####################################################################################################################################

# 方法2. 动态量化 ： torch.quantization.quantize_dynamic
# 系统自动选择最合适的scale （标度）和 zero_point（零点位置），不需要自定义。量化后的模型，可以推理运算，但不能训练（不能反向传播）
# 在模型上调用torch.quantization.quantize_dynamic
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.LSTM, nn.Linear}, dtype=torch.qint8
)
print(quantized_model)


import torch

# 定义一个浮点模型，其中一些层可以被静态量化(statically quantized)
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub 将张量从浮点转为量化
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub 将张量从量化转为浮点
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # 手动指定从浮点转为量化
        x = self.quant(x)
        x = self.conv(x)
        x = self.relu(x)
        # 将量化转换模型需要的浮点数；
        x = self.dequant(x)
        return x

# 创建模型实例
model_fp32 = M()

# model必须设置为eval模式，静态量化逻辑才能工作；
model_fp32.eval()

# 附加一个全局变量，其中包含有关类型信息等
model_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

model_fp32_fused = torch.quantization.fuse_modules(model_fp32, [['conv', 'relu']])

# 准备静态量化模型
# 在校准期间观察激活张量模型
model_fp32_prepared = torch.quantization.prepare(model_fp32_fused)

# 校准准备好的模型以确定激活的量化参数
# 实际会用代表性的参数进行；
input_fp32 = torch.randn(4, 1, 4, 4)
model_fp32_prepared(input_fp32)

# 将观测模型转换为量化模型
# 量化权重，计算存储比例和偏置值，激活张量并量化
model_int8 = torch.quantization.convert(model_fp32_prepared)

# 运行量化的模型，相关计算将在in8中进行
res = model_int8(input_fp32)

# https://pytorch.org/docs/stable/quantization.html
# https://pytorch.apachecn.org/#/docs/1.7/55

#####################################################################################################################################
# 方法3. 量化意识训练
# 系统自动选择最合适的scale （标度）和 zero_point（零点位置），不需要自定义。但这是一种伪量化，量化后的模型权重仍然是32位浮点数，但大小和8位定点数权重的大小相同。伪量化后的模型可以进行训练。
# 虽然是以32位浮点数进行的训练，但结果与8位定点数的结果一致。
# 静态量化(Static Quantization)的量化感知训练
# Quantization Aware Training (QAT)
# 量化感知训练（QAT）对训练期间的量化效果进行建模，与其他量化方法相比。我们可以对静态、动态或仅加权量化进行QAT。
# 在训练过程中，所有计算都在浮点中完成，fake_quant模块通过钳位和舍入来模拟量化的效果，以模拟INT8的效果。
# 在模型转换之后，量化权重和激活，并在可能的情况下将激活融合到前一层。它通常与神经网络一起使用，并且与静态量化相比具有更高的精度。

import torch

# 定义一个浮点模型，某些层可以QAT
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # QuantStub 用于张量由浮点->量化
        self.quant = torch.quantization.QuantStub()
        self.conv = torch.nn.Conv2d(1, 1, 1)
        self.bn = torch.nn.BatchNorm2d(1)
        self.relu = torch.nn.ReLU()
        # DeQuantStub 用于张量由量化-> 浮点
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dequant(x)
        return x

model_fp32 = M()

model_fp32.eval()

model_fp32.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

model_fp32_fused = torch.quantization.fuse_modules(model_fp32,
    [['conv', 'bn', 'relu']])

# 准备QAT模型
model_fp32_prepared = torch.quantization.prepare_qat(model_fp32_fused.train())

# 训练观测模型
training_loop(model_fp32_prepared)

# 将观测模型转换为训练模型
model_fp32_prepared.eval()
model_int8 = torch.quantization.convert(model_fp32_prepared)

# 运行int8量化模型
res = model_int8(input_fp32)

####################################################################################################################################
pytorch模型量化，有时候，也可以通过half方法进行，这个时候，可以将参数类型由 float32-> float16, 对应的内存消耗，可减小一半：
half()
方法: half()
    Casts all floating point parameters and buffers to half datatype.
    将所有的浮点参数和缓冲转换为半浮点(half)数据类型.
    Returns  函数返回
        self  自身self
    Return type  返回类型
        Module  模块Module类型

FP16：.half().cuda()
INT8：.half().quantize(8).cuda()
INT4：.half().quantize(4).cuda()

举例：
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2")
# 该加载方式，在最大长度为512时 大约需要6G多显存
# 如显存不够，可采用以下方式加载，进一步减少显存需求，约为3G
model = T5ForConditionalGeneration.from_pretrained("ClueAI/ChatYuan-large-v2").half()
来源：https://huggingface.co/ClueAI/ChatYuan-large-v2

half异常：
有时候加载模型正常，但预测时候报错：
RuntimeError: "LayerNormKernelImpl" not implemented for 'Half'
import torch
torch.layer_norm(torch.tensor([1., 2., 3.], dtype=torch.float16, device='cpu'), normalized_shape=(3,))
PyTorch 的 LayerNorm 暂时不支持 half 类型。

####################################################################################################################################
# post-quantization是直接训练出一个浮点模型直接对模型的参数进行直接量化。这种方法比较常见于对一个大模型进量化，而对小模型会导致大幅度的性能降低。主要原因有两个：1）post-training对参数取值的要求需要比较大的范围。如果参数的取值范围比较小，在量化过程中非常容易导致很高的相对误差。2）量化后的权重中的一些异常的权重会导致模型参数量的降低。
#
# training-aware-quantization是在训练中模拟量化行为，在训练中用浮点来保存定点参数，最后inference的时候，直接采用定点参数。

def main():
    pass


if __name__ == '__main__':
    main()
