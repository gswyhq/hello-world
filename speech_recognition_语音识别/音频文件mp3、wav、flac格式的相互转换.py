#!/usr/bin/python3
# coding: utf-8

# apt-get install -y ffmpeg libavcodec-extra
# pip3 install pydub -i http://pypi.douban.com/simple --trusted-host=pypi.douban.com

# 方案一：通过Python程序转换；

from pydub import AudioSegment
from pydub.playback import play

sound = AudioSegment.from_file("mysound.wav", format="wav")
play(sound)

song = AudioSegment.from_wav("never_gonna_give_you_up.wav")

flac_sound = AudioSegment.from_file("/root/sample-en.flac", format="flac")

song = AudioSegment.from_mp3("never_gonna_give_you_up.mp3")

sound.export("/root/sample-en.flac", format="flac")

ogg_version = AudioSegment.from_ogg("never_gonna_give_you_up.ogg")
flv_version = AudioSegment.from_flv("never_gonna_give_you_up.flv")

mp4_version = AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
wma_version = AudioSegment.from_file("never_gonna_give_you_up.wma", "wma")
aac_version = AudioSegment.from_file("never_gonna_give_you_up.aiff", "aac")

awesome.export("mashup.mp3", format="mp3")

awesome.export("mashup.mp3", format="mp3", tags={'artist': 'Various artists', 'album': 'Best of 2011', 'comments': 'This album is awesome!'})


awesome.export("mashup.mp3", format="mp3", bitrate="192k")

# Use preset mp3 quality 0 (equivalent to lame V0)
awesome.export("mashup.mp3", format="mp3", parameters=["-q:a", "0"])

# Mix down to two channels and set hard output volume
awesome.export("mashup.mp3", format="mp3", parameters=["-ac", "2", "-vol", "150"])

# 方案二，通过命令行转换
# root@b56138676fb3:~# ffmpeg -i sample-en.wav sample-en.mp3
# root@b56138676fb3:~# ffmpeg -i sample-en.wav sample-en.flac
# root@b56138676fb3:~# ffmpeg -i sample-en.mp3 from_mp3.flac
# root@b56138676fb3:~# ffmpeg -i sample-en.mp3 from_mp3.wav
# root@b56138676fb3:~# ffmpeg -i sample-en.flac from_flac.wav
# root@b56138676fb3:~# ffmpeg -i sample-en.flac from_flac.mp3
# 其他格式转换方法类似：
# ffmpeg -i input.mp4 out.mp3
# ffmpeg ffmpeg -i input.flac -id3v2_version 3 out.mp3

# 方案三：通过docker镜像：
# docker run --rm -it -v $PWD:/data gswyhq/ffmpeg:v4.1.4-1 ffmpeg -i /data/sample-en.wav /data/sample-en.mp3

def main():
    pass


if __name__ == '__main__':
    main()
