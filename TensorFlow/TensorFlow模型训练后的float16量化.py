#!/usr/bin/env python
# coding=utf-8

# TensorFlow Lite 支持在模型从 TensorFlow 转换到 TensorFlow Lite FlatBuffer 格式期间将权重转换为 16 位浮点值。这样可以将模型的大小缩减至原来的二分之一。
# 某些硬件（如 GPU）可以在这种精度降低的算术中以原生方式计算，从而实现比传统浮点执行更快的速度。可以将 Tensorflow Lite GPU 委托配置为以这种方式运行。
# 但是，转换为 float16 权重的模型仍可在 CPU 上运行而无需其他修改：float16 权重会在首次推断前上采样为 float32。
# 这样可以在对延迟和准确率造成最小影响的情况下显著缩减模型大小。

# 构建 MNIST 模型

import logging
logging.getLogger("tensorflow").setLevel(logging.DEBUG)
import os
import time
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pathlib

USERNAME = os.getenv('USERNAME')


# 处理器：11th Gen Intel(R) Core(TM) i7-1165G7 @ 2.80GHz   2.80 GHz
# tf.__version__
# Out[19]: '2.9.1'

# float16 量化的优点如下：
# 将模型的大小缩减一半（因为所有权重都变成其原始大小的一半）。
# 实现最小的准确率损失。
# 支持可直接对 float16 数据进行运算的部分委托（例如 GPU 委托），从而使执行速度比 float32 计算更快。

# float16 量化的缺点如下：
# 它不像对定点数学进行量化那样减少那么多延迟。
# 默认情况下，float16 量化模型在 CPU 上运行时会将权重值“反量化”为 float32。（请注意，GPU 委托不会执行此反量化，因为它可以对 float16 数据进行运算。）

# 训练并导出模型

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model architecture
model = keras.Sequential([
  keras.layers.InputLayer(input_shape=(28, 28)),
  keras.layers.Reshape(target_shape=(28, 28, 1)),
  keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation=tf.nn.relu),
  keras.layers.MaxPooling2D(pool_size=(2, 2)),
  keras.layers.Flatten(),
  keras.layers.Dense(10)
])

# Train the digit classification model
model.compile(optimizer='adam',
              loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(
  train_images,
  train_labels,
  epochs=1,
  validation_data=(test_images, test_labels)
)


# 在此示例中，您只对模型进行了一个周期的训练，因此只训练到约 96% 的准确率。

# 转换为 TensorFlow Lite 模型
# 现在，您可以使用 TensorFlow Lite Converter 将训练后的模型转换为 TensorFlow Lite 模型。

# 现在使用 TFLiteConverter 加载模型：


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 将其写入 .tflite 文件：


tflite_models_dir = pathlib.Path(rf"D:\Users\{USERNAME}\Downloads\mnist_tflite_models")
tflite_models_dir.mkdir(exist_ok=True, parents=True)

tflite_model_file = tflite_models_dir/"mnist_model.tflite"
tflite_model_file.write_bytes(tflite_model)


# 要改为在导出时将模型量化为 float16，首先将 optimizations 标记设置为使用默认优化。然后将 float16 指定为目标平台支持的类型：


converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
# 最后，像往常一样转换模型。请注意，为了方便调用，转换后的模型默认仍将使用浮点输入和输出。


tflite_fp16_model = converter.convert()
tflite_model_fp16_file = tflite_models_dir/"mnist_model_quant_f16.tflite"
tflite_model_fp16_file.write_bytes(tflite_fp16_model)

#
# 请注意，生成文件的大小约为 1/2。


# ls -lh {tflite_models_dir}

# total 128K
# -rw-rw-r-- 1 kbuilder kbuilder 83K Aug 11 18:54 mnist_model.tflite
# -rw-rw-r-- 1 kbuilder kbuilder 44K Aug 11 18:54 mnist_model_quant_f16.tflite
# 运行 TensorFlow Lite 模型
# 使用 Python TensorFlow Lite 解释器运行 TensorFlow Lite 模型。
#
# 将模型加载到解释器中

interpreter = tf.lite.Interpreter(model_path=str(tflite_model_file))
interpreter.allocate_tensors()


interpreter_fp16 = tf.lite.Interpreter(model_path=str(tflite_model_fp16_file))
interpreter_fp16.allocate_tensors()
# 在单个图像上测试模型

test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

interpreter.set_tensor(input_index, test_image)
interpreter.invoke()
predictions = interpreter.get_tensor(output_index)

import matplotlib.pylab as plt

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_labels[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)



test_image = np.expand_dims(test_images[0], axis=0).astype(np.float32)

input_index = interpreter_fp16.get_input_details()[0]["index"]
output_index = interpreter_fp16.get_output_details()[0]["index"]

interpreter_fp16.set_tensor(input_index, test_image)
interpreter_fp16.invoke()
predictions = interpreter_fp16.get_tensor(output_index)

plt.imshow(test_images[0])
template = "True:{true}, predicted:{predict}"
_ = plt.title(template.format(true= str(test_labels[0]),
                              predict=str(np.argmax(predictions[0]))))
plt.grid(False)


# 评估模型

# A helper function to evaluate the TF Lite model using "test" dataset.
def evaluate_model(interpreter):
  input_index = interpreter.get_input_details()[0]["index"]
  output_index = interpreter.get_output_details()[0]["index"]

  # Run predictions on every image in the "test" dataset.
  prediction_digits = []
  for test_image in test_images:
    # Pre-processing: add batch dimension and convert to float32 to match with
    # the model's input data format.
    test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
    interpreter.set_tensor(input_index, test_image)

    # Run inference.
    interpreter.invoke()

    # Post-processing: remove batch dimension and find the digit with highest
    # probability.
    output = interpreter.tensor(output_index)
    digit = np.argmax(output()[0])
    prediction_digits.append(digit)

  # Compare prediction results with ground truth labels to calculate accuracy.
  accurate_count = 0
  for index in range(len(prediction_digits)):
    if prediction_digits[index] == test_labels[index]:
      accurate_count += 1
  accuracy = accurate_count * 1.0 / len(prediction_digits)

  return accuracy

start_time = time.time()
print(evaluate_model(interpreter))
print("总耗时（s）：", time.time()-start_time)
# 0.9662
# 总耗时（s）： 1.8334996700286865

# 在 float16 量化模型上重复评估，以获得如下结果：

start_time = time.time()
print(evaluate_model(interpreter_fp16))
print("总耗时（s）：", time.time()-start_time)
# 0.9662
# 总耗时（s）： 1.8544859886169434
# 在此示例中，您已将模型量化为 float16，但准确率没有任何差别。


# 资料来源：https://tensorflow.google.cn/lite/performance/post_training_float16_quant?hl=zh-cn

# 动态量化：
# 动态范围量化
# 训练后量化最简单的形式是仅将权重从浮点静态量化为整数（具有 8 位精度）：
import tensorflow as tf
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
tflite_model_quant_file = tflite_models_dir/"mnist_model_quant.tflite"
tflite_model_quant_file.write_bytes(tflite_quant_model)
interpreter_quant = tf.lite.Interpreter(model_path=str(tflite_model_quant_file))
interpreter_quant.allocate_tensors()

start_time = time.time()
print(evaluate_model(interpreter_quant))
print("总耗时（s）：", time.time()-start_time)
# 0.9576
# 总耗时（s）： 2.0016911029815674
# 模型大小降低到1/4;

def main():
    pass


if __name__ == "__main__":
    main()
