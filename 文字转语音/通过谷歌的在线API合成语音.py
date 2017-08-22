#!/usr/bin/python3
# coding: utf-8

# Create an mp3 file from spoken text via the Google TTS (Text-to-Speech) API

# pip3 install gTTS

from gtts import gTTS

def main():
    tts = gTTS(text='Hello', lang='en', slow=True)
    tts.save("hello.mp3")

    # 快音
    tts = gTTS(text='Hello', lang='en', slow=False)
    tts.save("hello2.mp3")

    tts = gTTS(text='我是中国人，我深深爱着我的祖国和人民；', lang='zh-cn', slow=False)
    tts.save("hello3.mp3")


if __name__ == '__main__':
    main()