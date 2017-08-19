#----coding:utf-8----------------------------------------------------------------
# 名称:        模块1
# 目的:
# 参考:
# 作者:      gswyhq
#
# 日期:      2017-08-13
# 版本:      Python 3.3.5
# 系统:      deepin15.3
# Email:     gswyhq@126.com
#-------------------------------------------------------------------------------
"""
文字转换成语音
"""
import os
from pypinyin import lazy_pinyin, TONE3
from pydub import AudioSegment
from pydub.silence import split_on_silence

# 中间插入空白静音1s。
silent = AudioSegment.silent(duration=1000)

PINYIN_VOICE_PATH = '/home/gswyhq/ekho-7.5.1/ekho-data/pinyin'

EXPORT_PATH = '/home/gswyhq/data/新文件夹'

def load_voice_dict():
    voice_file_list = [f for f in os.listdir(PINYIN_VOICE_PATH) if f.endswith('.wav')]
    voice_dict = {}

    for voice_file in voice_file_list:
        name = voice_file[:-4]
        song = AudioSegment.from_wav(os.path.join(PINYIN_VOICE_PATH, voice_file))
        voice_dict.setdefault(name, song)
    return voice_dict

VOICE_DICT = load_voice_dict()

def txt_to_voice(text, name='test'):
    pinyin_list = lazy_pinyin(text, style=TONE3)
    new = AudioSegment.empty()
    for piny in pinyin_list:
        piny_song = VOICE_DICT.get(piny)
        if piny_song is None and piny and piny[-1] not in '0123456789':
            # 没有音调
            piny = piny + '5'
            piny_song = VOICE_DICT.get(piny, silent)

        # new += piny_song
        new = new.append(piny_song, crossfade=1)
    new.export(os.path.join(EXPORT_PATH, "{}.mp3".format(name)), format='mp3')

def main():
    text = '''《御天神帝》全集
        作者：乱世狂刀
        声明:本书由奇书网(www.Qisuu.com)自网络收集整理制作,仅供交流学习使用,版权归原作者和出版社所有,如果喜欢,请支持正版.
        序章
        序言·守墓的少年
        “小羽，不要哭，人终有一死，我和你娘的日子到了，老伙计们在星辰的怀抱之中等着我们呢！”
        “呵呵，比起那些已经提前走了的伙计们，能亲眼看着你，一天一天从一个小婴儿成长到十岁，我们已经很幸运了！”
        夕阳如血。'''
    txt_to_voice(text)

if __name__ == '__main__':
    main()
