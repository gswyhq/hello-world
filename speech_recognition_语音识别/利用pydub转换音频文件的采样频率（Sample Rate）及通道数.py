

from pydub import AudioSegment as am

# 采样频率，由44.1kHz -> 16kHz
sound = am.from_file('作品1号.wav', format='wav')
print('原始文件的采样频率为：{}'.format(sound.frame_rate))
# Out[8]: 原始文件的采样频率为：44100
sound = sound.set_frame_rate(16000)  # 设置文件的采样频率为16kHz
sound.export('作品1号_16kHz.wav', format='wav')


# 由多通道转换为单通道

print('原始文件通道数为：{}'.format(sound.channels))
# Out[8]: 2
sound = sound.set_channels(1)  # 设置文件为单通道
sound.export('作品1号_16kHz_1_channels.wav', format='wav')
