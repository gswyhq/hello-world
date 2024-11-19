#!/usr/bin/env python
# coding=utf-8

import os

USERNAME = os.getenv("USERNAME")

# https://modelscope.cn/models/iic/speech_eres2netv2_sv_zh-cn_16k-common/files
model_dir = rf"D:\Users\{USERNAME}\data\speech_eres2netv2_sv_zh-cn_16k-common"

from modelscope.pipelines import pipeline
sv_pipline = pipeline(
    task='speaker-verification',
    model=model_dir,
    model_revision='v1.0.2'
)
speaker1_a_wav = rf"D:\Users\{USERNAME}\Downloads\test/speaker1_a_cn_16k.wav"
speaker1_b_wav = rf"D:\Users\{USERNAME}\Downloads\test/speaker1_b_cn_16k.wav"
speaker2_a_wav = rf"D:\Users\{USERNAME}\Downloads\test/speaker2_a_cn_16k.wav"
# 相同说话人语音
result = sv_pipline([speaker1_a_wav, speaker1_b_wav])
print(result)
# 不同说话人语音
result = sv_pipline([speaker1_a_wav, speaker2_a_wav])
print(result)
# 可以自定义得分阈值来进行识别
result = sv_pipline([speaker1_a_wav, speaker2_a_wav], thr=0.365)
print(result)



def main():
    pass


if __name__ == "__main__":
    main()
