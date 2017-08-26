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
文字转换成语音, 原理，将每个汉字转换成对应的拼音，再通过拼音调用对应的音频文件，拼接成最后的音频文件；
"""

import os
from pypinyin import lazy_pinyin, TONE3
from pydub import AudioSegment
import traceback

# 中间插入空白静音500ms。
silent = AudioSegment.silent(duration=500)

# 单字音频文件的来源：
# 从Ekho Voice Data的下载页面(https://sourceforge.net/projects/e-guidedog/files/Ekho%20Voice%20Data/0.2/)获取它们。
# 下载音频文件解压；
# gswyhq@gswyhq-pc:~$ tar xJvf pinyin-huang-44100-wav-v2.tar.xz
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

def txt_to_voice(text, name='test', export_path=EXPORT_PATH):
    """
    将文字转换为音频
    :param text: 需要转换的文字
    :param name: 生成的音频文件名
    :return: 
    """
    pinyin_list = lazy_pinyin(text, style=TONE3)
    new = AudioSegment.empty()
    for piny in pinyin_list:
        piny_song = VOICE_DICT.get(piny)
        if piny_song is None and piny and piny[-1] not in '0123456789':
            # 没有音调
            piny = piny + '5'
            piny_song = VOICE_DICT.get(piny, silent)

        # 交叉渐入渐出方法
        # with_style = beginning.append(end, crossfade=1500)
        # crossfade 就是让一段音乐平缓地过渡到另一段音乐，crossfade = 1500 表示过渡的时间是1.5秒。
        # if new and piny_song:
        #     crossfade = min(len(new), len(piny_song), 1500)/60
        #     new = new.append(piny_song, crossfade=crossfade)
        if not piny_song:
            continue
        new += piny_song

    new.export(os.path.join(export_path, "{}.mp3".format(name)), format='mp3')

def main():
    text = '''    红海早过了。船在印度洋面上开驶着。但是太阳依然不饶人地迟落早起侵占去大部分的夜。
    夜仿佛纸浸了油，变成半透明体；它给太阳拥抱住了，分不出身来，也许是给太阳陶醉了，
    所以夕照霞隐褪后的夜色也带着酡红。到红消醉醒，船舱里的睡人也一身腻汗地醒来，洗了澡赶到甲板上吹海风，
    又是一天开始。这是七月下旬，合中国旧历的三伏，一年最热的时候。在中国热得更比常年利害，
    事后大家都说是兵戈之象，因为这就是民国二十六年【一九三七年】。'''
    txt_to_voice(text)

def test():
    path = '/home/gswyhq/下载/御天神帝_qisuu.com/御天神帝'
    fts = os.listdir(path)
    fts.sort()
    export_path = '/home/gswyhq/下载/御天神帝_qisuu.com/mp3'

    for ft  in fts:
        with open(os.path.join(path, ft))as f:
            text = f.read()

        try:
            if os.path.isfile(os.path.join(export_path, ft.replace('.txt', '.mp3'))):
                continue
            txt_to_voice(text, name=ft[:-4], export_path=export_path)
            print(ft)
        except Exception as e:
            print('转换文件出错：{}'.format(ft))
            print("错误详情：{}".format(traceback.format_exc()))
            print(e)

    print('ok')

if __name__ == '__main__':
    # main()
    test()