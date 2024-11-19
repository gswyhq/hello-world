#!/usr/bin/env python
# coding=utf-8

import os
from pydub import AudioSegment
USERNAME = os.getenv("USERNAME")

# 读取音频文件
audio = AudioSegment.from_wav('audio.wav')

# 获取采样率
sample_rate = audio.frame_rate

# 改变采样率
if sample_rate!=16000:
    audio_16k = audio.set_frame_rate(16000)
    # 保存音频文件
    audio_16k.export('audio_16k.wav', format='wav')


def main():
    pass


if __name__ == "__main__":
    main()
