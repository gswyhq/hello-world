#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 考虑到在不同的计算设备上，半精度和双精度锁带来的性能提升是显而易见的。
# 将现有的ONNX模型从fp32模型转换成fp16模型
# 首先我们需要准备一个叫onnxmltools的库。可以通过pip来进行安装。
# pip install onnxmltools
# 确认安装好onnxmltools后，我们通过如下的一段脚本进行精度的转换：

import onnxmltools
# 加载float16_converter转换器
from onnxmltools.utils.float16_converter import convert_float_to_float16
# 使用onnxmltools.load_model()函数来加载现有的onnx模型
# 但是请确保这个模型是一个fp32的原始模型
fp32_model_file = './result/20230303091758/classify-05-0.2407-0.1039-0.1230.onnx'
fp16_model_file = './result/20230303091758/classify-05-0.2407-0.1039-0.1230_fp16.onnx'
onnx_model = onnxmltools.load_model(fp32_model_file)
# 使用convert_float_to_float16()函数将fp32模型转换成半精度fp16
onnx_model_fp16 = convert_float_to_float16(onnx_model)
# 使用onnx.utils.save_model()函数来保存，
onnxmltools.utils.save_model(onnx_model_fp16, fp16_model_file)


# In [46]: !du -sh ./result/20230303091758/classify-05-0.2407-0.1039-0.1230*
# 9.6M    ./result/20230303091758/classify-05-0.2407-0.1039-0.1230.hdf5
# 3.2M    ./result/20230303091758/classify-05-0.2407-0.1039-0.1230.onnx
# 1.6M    ./result/20230303091758/classify-05-0.2407-0.1039-0.1230_fp16.onnx

################################################################################################################################


# Dynamic quantization  动态量化
# Post Training Dynamic Quantization，模型训练完毕后的动态量化；
# Post Training Dynamic Quantization，简称为 Dynamic Quantization，也就是动态量化，或者叫作Weight-only的量化，
# 是提前把模型中某些 op 的参数量化为 INT8，然后在运行的时候动态的把输入量化为 INT8，然后在当前 op 输出的时候再把结果 requantization 回到 float32 类型。
# 动态量化默认只适用于 Linear 以及 RNN 的变种。
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

fp32_model_file = './result/20230303091758/classify-05-0.2407-0.1039-0.1230.onnx'
model_quant_dynamic = './result/20230303091758/classify-05-0.2407-0.1039-0.1230.quant.onnx'
# 动态量化
quantize_dynamic(
    model_input=fp32_model_file, # 输入模型
    model_output=model_quant_dynamic, # 输出模型
    activation_type = QuantType.QUInt8,
    weight_type=QuantType.QUInt8, # 参数类型 Int8 / UInt8
    optimize_model=True # 是否优化模型
)


# --------------------------
# QAT quantization  QAT量化
# QAT（Quantization Aware Training），模型训练中开启量化。
import onnx
from onnxruntime.quantization import quantize_qat, QuantType

model_qat_quant_file = './result/20230303091758/classify-05-0.2407-0.1039-0.1230.qat.onnx'
quantize_qat(fp32_model_file, model_qat_quant_file)

# --------------------------
# Post Training Static Quantization，模型训练完毕后的静态量化；
# 因为静态需要额外的数据用于校准模型，所以相比动态量化，静态量化更加复杂一些
# 需要先编写一个校准数据的读取器，然后再调用 ONNXRuntime 的 quantize_static 接口进行静态量化

##########################################################################################################################
# 使用量化的模型进行预测，但推理结果跟量化前后不一致，原因暂不明

from onnxruntime import InferenceSession, get_available_providers
session_quant = InferenceSession(model_quant_dynamic, providers=get_available_providers())
input_feed = {t.name: image.astype(np.float32) for t in session_quant.get_inputs()}
output_name = session_quant.get_outputs()[0].name
session_quant.run([output_name], input_feed)[0]

def main():
    pass


if __name__ == '__main__':
    main()
