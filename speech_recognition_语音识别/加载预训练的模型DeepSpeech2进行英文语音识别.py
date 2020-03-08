#!/usr/bin/python3.7
# coding: utf-8

# pip3 install automatic-speech-recognition==1.0.2 -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

import os
import automatic_speech_recognition as asr

path = "/data/LibriSpeech/dev-clean/1272/128104"
file = 'tests/sample-en.wav'  # sample rate 16 kHz, and 16 bit depth
# https://github.com/rolczynski/Automatic-Speech-Recognition/blob/master/tests/sample-en.wav

sample = asr.utils.read_audio(file)
pipeline = asr.load('deepspeech2', lang='en')
pipeline.model.summary()     # TensorFlow model
sentences = pipeline.predict([sample])
print(sentences)

for file in os.listdir(path):
    if not file.endswith(".wav"):
        continue
    abs_file = os.path.join(path, file)
    sample = asr.utils.read_audio(abs_file)
    sentences = pipeline.predict([sample])
    print(file, sentences)
