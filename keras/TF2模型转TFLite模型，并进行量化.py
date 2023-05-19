#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# TF2有量化方式有两种，训练后量化和感知量化。训练后量化是指模型训练好之后再进行对权重量化；感知量化是指在训练过程中自动根据训练数据的反馈进行量化。感知量化对模型的精度影响最小。
####################################################################################################################################
# 一、训练后量化
# 链接：https://zhuanlan.zhihu.com/p/444746800

import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

graph_def_file = "frozen_inference_graph.pb"

input_names = ["FeatureExtractor/MobilenetV2/MobilenetV2/input"]
output_names = ["concat", "concat_1"]
input_tensor = {input_names[0]: [1, 300, 300, 3]}

# uint8 quant
converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_names, output_names, input_tensor)
converter.target_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
converter.allow_custom_ops = True

converter.inference_type = tf.uint8  # tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0]: (127.5, 127.5)}  # mean, std_dev
converter.default_ranges_stats = (0, 255)

tflite_uint8_model = converter.convert()
open("uint8.tflite", "wb").write(tflite_uint8_model)

#-------------------------------------------------------------------------------------------------------------------------

import tensorflow_model_optimization as tfmot

quantize_model = tfmot.quantization.keras.quantize_model

# q_aware stands for for quantization aware.
q_aware_model = quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

q_aware_model.summary()

# https://tensorflow.google.cn/model_optimization/guide/quantization/training_example
# https://tensorflow.google.cn/lite/performance/post_training_quantization?hl=zh-cn

####################################################################################################################################
# 二、感知量化
# 链接：https://zhuanlan.zhihu.com/p/503075176

# tf中有一些layer不支持量化，因此需要进行规避（不对该层量化）可以根据类型进行规避，也可以根据layer的name进行规避from keras import layers
from keras.models import load_model
import tensorflow as tf
model=load_model('model_name.h5')
import tensorflow_model_optimization as tfmot
from tensorflow.keras import layers

# Helper function uses `quantize_annotate_layer` to annotate that only the
# Dense layers should be quantized.
def apply_quantization_to_dense(layer):
  if isinstance(layer, tf.keras.layers.Conv2DTranspose):#根据类型进行规避
      return layer
  elif isinstance(layer, tf.keras.layers.BatchNormalization):
      return layer
  elif isinstance(layer, tf.keras.layers.Concatenate):
      return layer
  elif isinstance(layer, tf.keras.layers.Activation):
      return layer
  elif layer.name=='img_out':#根据name进行规避
      return layer
  return tfmot.quantization.keras.quantize_annotate_layer(layer)
annotated_model = tf.keras.models.clone_model(
    model,
    clone_function=apply_quantization_to_dense,
)
#已经被量化的模型
model = tfmot.quantization.keras.quantize_apply(annotated_model)
model.summary()

# https://tensorflow.google.cn/model_optimization/guide/quantization/training?hl=zh-cn

def main():
    pass


if __name__ == '__main__':
    main()
