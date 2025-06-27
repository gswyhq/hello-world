from pydub import AudioSegment
import os

def split_audio_file(input_file, output_dir, num_splits=10):
    # 读取音频文件
    audio = AudioSegment.from_wav(input_file)
    
    # 获取音频总时长（毫秒）
    total_duration = len(audio)
    
    # 计算每个小文件的时长
    split_duration = total_duration // num_splits
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 分割音频并保存
    for i in range(num_splits):
        start_time = i * split_duration
        end_time = (i + 1) * split_duration if i < num_splits - 1 else total_duration
        split_audio = audio[start_time:end_time]
        output_file = os.path.join(output_dir, f"split_{i+1}.wav")
        split_audio.export(output_file, format="wav")
        print(f"Saved {output_file}")

# 示例用法
input_file = "input.wav"
output_dir = "output_splits"
split_audio_file(input_file, output_dir)

