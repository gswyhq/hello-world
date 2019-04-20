#!/usr/bin/python3
# coding: utf-8

# sudo apt-get install ffmpeg
# sudo apt-get install pydub

from pydub import AudioSegment

# 加载一个wav文件
song = AudioSegment.from_wav("never_gonna_give_you_up.wav")

# 加载一个MP3文件
song = AudioSegment.from_mp3("never_gonna_give_you_up.mp3")

# 加载其它格式的文件
ogg_version = AudioSegment.from_ogg("never_gonna_give_you_up.ogg")
flv_version = AudioSegment.from_flv("never_gonna_give_you_up.flv")

mp4_version = AudioSegment.from_file("never_gonna_give_you_up.mp4", "mp4")
wma_version = AudioSegment.from_file("never_gonna_give_you_up.wma", "wma")
aac_version = AudioSegment.from_file("never_gonna_give_you_up.aiff", "aac")

# 音频切片
# pydub 是以毫秒为单位进行处理
ten_seconds = 10 * 1000  # 十秒

first_10_seconds = song[:ten_seconds]

last_5_seconds = song[-5000:]

# 使开始的时候音量更大，结束时更小
# 开始的时候增加 6dB
beginning = first_10_seconds + 6

# 结束的时候减小 3dB
end = last_5_seconds - 3

# 音频文件的拼接
without_the_middle = beginning + end


without_the_middle.duration_seconds == 15.0

# 1、对于多个音频的计算，需要多个音频之间的通道数、帧数、采样率以及比特数都一样，
# 否则低质量的音频会向高质量的转换，单声道会向立体声转换，低帧数向高帧数转换。
# 2、AudioSegment原生就支持wav和raw，如果其他文件需要安装ffmpeg。
# raw还需要，sample_width，frame_rate，channels三个参数。

# 生成文件：
# export()方法可以使一个AudioSegment对象转化成一个文件。
sound = AudioSegment.from_file("/path/to/sound.wav", format="wav")
file_handle = sound.export("/path/to/output.mp3", format="mp3")     # 简单输出
file_handle2 = sound.export("/path/to/output.mp3",
                           format="mp3",
                           bitrate="192k",
                           tags={"album": "The Bends", "artist": "Radiohead"})         # 复杂输出

# AudioSegment.empty()用于生成一个长度为0的AudioSegment对象，一般用于多个音频的合并。

sounds = [
  AudioSegment.from_wav("sound1.wav"),
  AudioSegment.from_wav("sound2.wav"),
  AudioSegment.from_wav("sound3.wav"),
]
playlist = AudioSegment.empty()
for sound in sounds:
  playlist += sound



# AudioSegment.silent()：
ten_second_silence = AudioSegment.silent(duration=10000)  # 产生一个持续时间为10s的无声AudioSegment对象

# 此外，还能通过AudioSegment获取音频的参数，同时还能修改原始参数。


# AudioSegments are immutable
#
# # song is not modified
# backwards = song.reverse()
# Crossfade (again, beginning and end are not modified)
#
# # 1.5 second crossfade
# with_style = beginning.append(end, crossfade=1500)
# Repeat
#
# # repeat the clip twice
# do_it_over = with_style * 2
# Fade (note that you can chain operations because everything returns an AudioSegment)
#
# # 2 sec fade in, 3 sec fade out
# awesome = do_it_over.fade_in(2000).fade_out(3000)
# Save the results (again whatever ffmpeg supports)
#
# awesome.export("mashup.mp3", format="mp3")
# Save the results with tags (metadata)
#
# awesome.export("mashup.mp3", format="mp3", tags={'artist': 'Various artists', 'album': 'Best of 2011', 'comments': 'This album is awesome!'})
# You can pass an optional bitrate argument to export using any syntax ffmpeg supports.
#
# awesome.export("mashup.mp3", format="mp3", bitrate="192k")
# Any further arguments supported by ffmpeg can be passed as a list in a 'parameters' argument, with switch first, argument second. Note that no validation takes place on these parameters, and you may be limited by what your particular build of ffmpeg/avlib supports.
#
# # Use preset mp3 quality 0 (equivalent to lame V0)
# awesome.export("mashup.mp3", format="mp3", parameters=["-q:a", "0"])
#
# # Mix down to two channels and set hard output volume
# awesome.export("mashup.mp3", format="mp3", parameters=["-ac", "2", "-vol", "150"])



def main():
    pass


if __name__ == '__main__':
    main()