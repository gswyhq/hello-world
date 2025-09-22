'''
在处理 双声道（立体声） 的语音文件时，如果左右声道内容不一致（比如一个声道是说话人A，另一个是说话人B），使用耳机时会出现“只有一边有声音”的情况。
需要对音频进行 左右声道混合（mix），使得：

左右声道内容 相同，即两个耳朵都能听到 相同的声音
或者，将两个声道的内容 混合成一个声道，然后复制到左右声道
'''

import soundfile as sf
import numpy as np

def stereo_to_dual_ear(input_path, output_path):
    # 读取音频文件
    data, samplerate = sf.read(input_path, dtype='float32')

    # 检查是否为双声道
    if data.ndim == 1:
        print("⚠️ 输入音频是单声道，无需处理。")
        sf.write(output_path, data, samplerate)
        return

    if data.shape[1] != 2:
        print("⚠️ 输入音频不是双声道，无法处理。")
        return

    left_channel = data[:, 0]  # 左声道
    right_channel = data[:, 1]  # 右声道

    # 混合左右声道为一个声道（平均）
    mixed = (left_channel + right_channel) / 2.0

    # 将混合后的声道复制到左右声道
    stereo_mixed = np.column_stack((mixed, mixed))

    # 保存为双声道音频
    sf.write(output_path, stereo_mixed, samplerate, format='WAV')
    print(f"✅ 已处理并保存为双声道一致的音频文件: {output_path}")

# 示例调用
if __name__ == "__main__":
    input_wav = "input.wav"       # 输入文件
    output_wav = "output_dual_ear.wav"  # 输出文件
    stereo_to_dual_ear(input_wav, output_wav)


