#!/usr/bin/python3
# coding: utf-8

from pydub import AudioSegment
file_name = "/home/gswyhq/音乐/与你同行.MP3"
save_name = "/home/gswyhq/音乐/与你同行.mp3"

sound = AudioSegment.from_mp3(file_name)
start_time = "0:05"
stop_time = "5:00"
print("time:", start_time, "~", stop_time)
start_time = (int(start_time.split(':')[0])*60+int(start_time.split(':')[1]))*1000
stop_time = (int(stop_time.split(':')[0])*60+int(stop_time.split(':')[1]))*1000
print("ms:", start_time, "~", stop_time)
word = sound[start_time:stop_time]
# save_name = "word"+file_name[6:]
print(save_name)
# word.export(save_name, format="mp3",tags={'artist': 'AppLeU0', 'album': save_name[:-4]})
word.export(save_name, format="mp3")

def main():
    pass


if __name__ == '__main__':
    main()