#!/usr/lib/python3
# -*- coding: utf-8 -*-

# 第一步：通过镜像站下载模型
# ~$ export HF_ENDPOINT=https://hf-mirror.com
# ~$ huggingface-cli download --resume-download facebook/hf-seamless-m4t-medium

import transformers
transformers.__version__
'4.35.0.dev0'

from transformers import AutoProcessor, SeamlessM4TModel, SeamlessM4TForTextToSpeech

model_dir = "/root/.cache/huggingface/hub/models--facebook--hf-seamless-m4t-medium/snapshots/1cba099410d9e41736f12cad8e54e13fe37bec80/"
processor = AutoProcessor.from_pretrained(model_dir)
model = SeamlessM4TModel.from_pretrained(model_dir)

text_inputs = processor(text = "窗前明月光，疑是地上霜；举头望明月，低头思故乡", src_lang="cmn", return_tensors="pt")
# from text
output_tokens = model.generate(**text_inputs, tgt_lang="cmn", generate_speech=False)
audio_array_from_text = model.generate(**text_inputs, tgt_lang="cmn")[0].cpu().numpy().squeeze()

import numpy as np

def vec2wav(pcm_vec, wav_file, framerate = 16000):
    """ 将numpy数组转为单通道wav文件
    :param pcm_vec: 输入的numpy向量
    :param wav_file: wav文件名
    :param framerate: 采样率 :return: """
    import wave
    # pcm_vec = np.clip (pcm_vec, -32768, 32768)
    if np.max(np.abs(pcm_vec)) > 1.0:
        pcm_vec *= 32767 / max(0.01, np.max(np.abs(pcm_vec)))
    else:
        pcm_vec = pcm_vec * 32768
    pcm_vec = pcm_vec.astype(np.int16)
    wave_out = wave.open(wav_file, 'wb')
    wave_out.setnchannels(1)
    wave_out.setsampwidth(2)
    wave_out.setframerate(framerate)
    wave_out.writeframes(pcm_vec)

vec2wav(audio_array_from_text, "123.wav")

# 实测效果不是很好，几乎听不清楚；


def main():
    pass


if __name__ == '__main__':
    main()
