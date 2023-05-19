#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 在部署ONNX模型阶段，我们将使用onnxruntime这个模块。
#
# 针对你所将使用的计算设备，如果你是CPU用户，那么你需要使用如下的指令来安装onnxruntime
#
# pip install onnxruntime
#
# 反之，如果你的计算设备是是GPU，那么你需要使用如下的指令来安装onnxruntime
#
# pip install onnxruntime-gpu
#
# 确认好onnxruntime安装完成后，你只需要使用如下的指令来加载你的ONNX模型即可


import onnxruntime as ort
# 指定onnx模型所在的位置
fp32_model_file = './result/20230303091758/classify-05-0.2407-0.1039-0.1230.onnx'
# 创建providers参数列表
providers = [
        # 指定模型可用的CUDA计算设备参数
        ('CUDAExecutionProvider', {
            # 因为这里笔者只有一张GPU，因此GPU ID序列就为0
            'device_id': 0,
            # 这里网络额外策略使用官方默认值
            'arena_extend_strategy': 'kNextPowerOfTwo',
            # 官方这里默认建议的GPU内存迭代上限是2GB，如果你的GPU显存足够大
            # 可以将这里的2修改为其它数值
            'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
            # cudnn转换算法的调用参数设置为完整搜索
            'cudnn_conv_algo_search': 'EXHAUSTIVE',
            # 确认从默认流进行CUDA流赋值
            'do_copy_in_default_stream': True,
        }),
        'CPUExecutionProvider',
    ]
# 使用onnxruntime.InferenceSession()函数创建Session
# 第一参数为模型所在的路径，第二参数为模型的providers参数列表
session = ort.InferenceSession(fp32_model_file, providers=providers)
# 通过get_input()函数和get_output()函数获取网络的输入和输出名称
# input_name = session.get_inputs()[0].name
input_feed = {t.name: image.astype(np.float32) for t in session.get_inputs()}
output_name = session.get_outputs()[0].name
# 使用session.run()函数执行ONNX任务
# 值得注意的是，这里演示使用的ONNX模型是FP32精度的模型
# 如果你使用的fp16模型但传入的数据是fp32类型的会抛出数据异常的错误
# 另外ONNX的异常抛出是十分人性化的，它会指明你在推理是发生异常的具体位置以及应对策略

result = session.run([output_name], input_feed)[0]
result = result.argmax()



def main():
    pass


if __name__ == '__main__':
    main()
