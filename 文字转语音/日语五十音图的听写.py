#!/usr/bin/python3
# coding: utf-8

from pydub import AudioSegment
from pydub.silence import split_on_silence
import random
import os

file = '/home/gswyhq/data/五十音.mp3'
EXPORT_PATH = '/home/gswyhq/data/五十音图'
time_start = "00:16"
time_end = "01:35"

song = AudioSegment.from_mp3(file)
start = (int(time_start.split(':')[0])*60 + int(time_start.split(':')[1]))*1000
end = (int(time_end.split(':')[0])*60 + int(time_end.split(':')[1]))*1000
# print(start, end)
# 剪切时间：是按ms 毫秒来的，所以时间格式的转换就要到毫秒级的。
word = song[start:end]

# 这里silence_thresh是认定小于-42dBFS以下的为silence，然后需要保持小于-42dBFS超过 700毫秒。这样子分割成一段一段的。
# 最关键的就是这两个值的确定，这里需要我们用到foobar的一个功能：视图——可视化———音量计
# 可以观察一段音频的dBFS大小，正常的音量差不多都是在-25dBFS到-10dBFS。这个单位是从-96dBFS到0dBFS的，越靠近0，音量越大。
# 我们这里取-42dBFS以下的，认为是静音。然后可以用foobar估算每个单词中间的间隙时间，大概是在900ms也就是0.9s。我们还是取小一些 0.7s分割。
words = split_on_silence(word, min_silence_len=700, silence_thresh=-42)

# 再来就是生成一个乱序的序列，然后把单词对应进去，然后中间插入空白静音1s。
silent = AudioSegment.silent(duration=1000)

print("共分割出{}个音".format(len(words)))
wushiyintu = ['あ', 'い', 'う', 'え', 'お',
              'か', 'き', 'く', 'け', 'こ',
              'さ', 'し', 'す', 'せ', 'そ',
              'た', 'ち', 'つ', 'て', 'と',
              'な', 'に', 'ぬ', 'ね', 'の',
              'は', 'ひ', 'ふ', 'へ', 'ほ',
              'ま', 'み', 'む', 'め', 'も',
              'や', 'ゆ', 'よ',
              'ら', 'り', 'る', 'れ', 'ろ',
              'わ', 'を', 'ん']

# 每个音写一个文件
for index, name in enumerate(wushiyintu):
    yin = words[index]
    yin.export(os.path.join(EXPORT_PATH, '{}.mp3'.format(name)), format='mp3')


fenhangduan_dict = {
    'あ段': ['あ', 'か', 'さ', 'た', 'な', 'は', 'ま', 'や', 'ら', 'わ'],
    'い段': ['い', 'き', 'し', 'ち', 'に', 'ひ', 'み', 'り'],
    'う段': ['う', 'く', 'す', 'つ', 'ぬ', 'ふ', 'む', 'ゆ', 'る'],
    'え段': ['え', 'け', 'せ', 'て', 'ね', 'へ', 'め', 'れ'],
    'お段': ['お', 'こ', 'そ', 'と', 'の', 'ほ', 'も', 'よ', 'ろ', 'を', 'ん'],
    'あ行': ['あ', 'い', 'う', 'え', 'お'],
    'か行': ['か', 'き', 'く', 'け', 'こ'],
    'さ行': ['さ', 'し', 'す', 'せ', 'そ'],
    'た行': ['た', 'ち', 'つ', 'て', 'と'],
    'な行': ['な', 'に', 'ぬ', 'ね', 'の'],
    'は行': ['は', 'ひ', 'ふ', 'へ', 'ほ'],
    'ま行': ['ま', 'み', 'む', 'め', 'も'],
    'や行': ['や', 'ゆ', 'よ'],
    'ら行': ['ら', 'り', 'る', 'れ', 'ろ'],
    'わ行': ['わ', 'を'],
}

def run(key, yin_list):
    new = AudioSegment.empty()
    for name in yin_list:
        index = wushiyintu.index(name)
        new += words[index]+silent
    # 每列保存为一个音频文件
    new.export(os.path.join(EXPORT_PATH, "{}.mp3".format(key)), format='mp3')

for key, yin_list in fenhangduan_dict.items():
    run(key=key, yin_list=yin_list)

random.shuffle(words) # 对五十音进行随机排列
new = AudioSegment.empty()
for i in words:
    new += i + silent
# 保存为一个用来听写的音频文件
new.export(os.path.join(EXPORT_PATH, 'listening.mp3'), format='mp3')

# 作者：ClarenceYK
# 链接：http://www.jianshu.com/p/a9291fa603f6
# 來源：简书

def main():
    pass


if __name__ == '__main__':
    main()

