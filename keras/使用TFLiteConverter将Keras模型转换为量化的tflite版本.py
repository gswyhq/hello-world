#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import tensorflow.python.keras.backend as K
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np

x_train = np.array([[0.6171875,  0.59791666],[0.6171875 , 0.59791666],[0.6171875,  0.59791666]])
y_train = np.array([[0.6171875,  0.59791666],[0.6171875 , 0.59791666],[0.6171875,  0.59791666]])


def representative_dataset_gen():
    for i in range(1):
        # Get sample input data as a numpy array in a method of your choosing.
        sample = np.array([0.5,0.6])
        sample = np.expand_dims(sample, axis=0)
        sample = sample.astype(np.float32)
        yield [sample]



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2,)),
  tf.keras.layers.Dense(12, activation='relu'),
  tf.keras.layers.Dense(2, activation='softmax')
])


model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)

# converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.float32
converter.inference_output_type = tf.float32
converter.representative_dataset = representative_dataset_gen

tflite_quant_model = converter.convert()

def main():
    pass


if __name__ == '__main__':
    main()
