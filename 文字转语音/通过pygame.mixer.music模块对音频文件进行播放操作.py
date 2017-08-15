#!/usr/bin/python3
# coding: utf-8

import pygame

# pygame.mixer.music.load
# 说明：加载音乐文件
# 原型：pygame.mixer.music.load(filename):
# return None
#
# pygame.mixer.music.play
# 说明：播放音乐
# 原型：pygame.mixer.music.play(loops=0, start=0.0):
# return None，
# 其中loops表示循环次数，如设置为 - 1，表示不停地循环播放；如loops = 5,
# 则播放5 + 1 = 6
# 次，start参数表示从音乐文件的哪一秒开始播放，设置为0表示从开始完整播放
#
# pygame.mixer.music.rewind
# 说明：重新播放
# 原型：pygame.mixer.music.rewind():
# return None
#
# pygame.mixer.music.stop
# 说明：停止播放
# 原型：pygame.mixer.music.stop():
# return None
#
# pygame.mixer.music.pause
# 说明：暂停
# 原型pygame.mixer.music.pause():
# return None
# 可通过pygame.mixer.music.unpause恢复播放
#
# pygame.mixer.music.unpause
# 说明：恢复播放
# 原型：pygame.mixer.music.unpause():
# return None
#
# pygame.mixer.music.fadeout
# 说明：暂停指定的时间，然后接着播放
# 原型：pygame.mixer.music.fadeout(time):
# return None，
# 单位为毫秒
#
# pygame.mixer.music.set_volume
# 说明：设置音量
# 原型：pygame.mixer.music.set_volume(value):
# return None
# 取值0
# .0
# ~1.0
#
# pygame.mixer.music.get_volume
# 说明：获取音量
# 原型：pygame.mixer.music.get_volume():
# return value
#
# pygame.mixer.music.get_busy
# 说明：判断当前是否有音乐在播放
# 原型：pygame.mixer.music.get_busy():
# return bool
#
# pygame.mixer.music.get_pos
# 说明：获取当前播放了多长时间
# 原型：pygame.mixer.music.get_pos():
# return time
#
# pygame.mixer.music.queue
# 说明：将其他音乐文件放入播放队列，当前音乐播放完成后，自动播放队列中其他的音乐文件
#
# pygame.mixer.music.set_endevent
# 说明：播放完成后的事件通知
# 原型：pygame.mixer.music.set_endevent():
# return None
# pygame.mixer.music.set_endevent(type):
# return None
#
# pygame.mixer.music.get_endevent
# 说明：获取播放完成后的事件，如果没有，返回pygame.NOEVENT.
# 原型：pygame.mixer.music.get_endevent():
# return type

def main():
    pass


if __name__ == '__main__':
    main()