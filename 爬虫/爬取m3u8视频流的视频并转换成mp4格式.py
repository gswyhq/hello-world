#!/usr/bin/python3
# coding: utf-8

import os
import requests
import time
from urllib import parse
from datetime import datetime

# chrome浏览器 -> 右键 -> Network -> ALL -> 在视频所在页面刷新 -> 在‘Name’中找到 index.m3u8 -> 右键 -> Copy -> Copy as cRUL

# gswyhq@gswyhq-PC:~$ curl 'https://abc.com/20190606/xp8D5cWd/index.m3u8'

# 直接下载这个链接，会得到一个.m3u8的文件，内容如下：
# #EXTM3U
# #EXT-X-STREAM-INF:PROGRAM-ID=1,BANDWIDTH=1000000,RESOLUTION=1280x536
# /20190606/xp8D5cWd/hls/index.m3u8
#
# m3u8链接分层的，所以还需要解析，最后一行的 /20190606/xp8D5cWd/hls/index.m3u8 其实是获取第二层KEY。
# 即要用它替换掉之前解析出来的链接的最后的"index.m3u8"，构成新的链接。
#
# gswyhq@gswyhq-PC:~$ curl 'https://abc.com/20190606/xp8D5cWd/hls/index.m3u8'

# 这个链接会下载一个.m3u8的文件，内容如下（实际上是很多ts链接）：

# #EXTM3U
# #EXT-X-VERSION:3
# #EXT-X-TARGETDURATION:10
# #EXT-X-MEDIA-SEQUENCE:0
# #EXTINF:5,
# /20190606/xp8D5cWd/hls/ymGV3236000.ts
# #EXTINF:5.88,
# /20190606/xp8D5cWd/hls/ymGV3236001.ts
# #EXTINF:5.92,
# /20190606/xp8D5cWd/hls/ymGV3236002.ts
# ....
# /20190606/xp8D5cWd/hls/ymGV3236229.ts
# #EXTINF:2.16,
# /20190606/xp8D5cWd/hls/ymGV3236230.ts
# #EXT-X-ENDLIST

# ts文件会按照顺序编号的
# 批量下载ts文件
#
# 注意：下载生成ts文件时，命名一定要规范，
# 如：0000.ts 0001.ts 0002.ts … … 1806.ts
# 不能 1.ts 2.ts 3.ts … … 1806.ts
# 因为下面合并的时候是按照字符匹配的

# 提取ts列表文件的内容，逐个拼接ts的url，形成list
def getPlayList(url_root, m3u8_url_2):
    r2 = requests.get(m3u8_url_2)
    content = r2.text
    ts_url_list = []
    for line in content.split('\n'):
        if line.endswith('.ts'):
            ts_url = url_root + line
            ts_url_list.append(ts_url)
    if not ts_url_list and any(line for line in content.split('\n') if line.endswith('index.m3u8')):
        m3u8_url_2 = url_root + [line for line in content.split('\n') if line.endswith('index.m3u8')][0]
        return getPlayList(url_root, m3u8_url_2)
    return ts_url_list

# 批量下载ts文件
def download_ts(ts_url_list, download_path):
    if not os.path.exists(download_path):
        os.mkdir(download_path)
    test_num = 3  # 下载失败重试次数
    print('共`{}`个ts文件'.format(len(ts_url_list)))
    for ts_url in ts_url_list:
        ts_name = ts_url.rsplit('/', maxsplit=1)[-1]
        path = os.path.join(download_path, ts_name)
        for i in range(test_num):
            try:
                if os.path.isfile(path):
                    break
                r = requests.get(ts_url, timeout=(10, 60 * 60))
                assert r.status_code == 200, '`{}`下载失败，重试{}'.format(ts_url, i+1)
                with open(path, 'wb') as f:
                    f.write(r.content)
                print('`{}`下载成功！'.format(ts_url))
                break
            except:
                print('`{}`下载失败，重试{}'.format(ts_url, i+1))
        time.sleep(1)

# ts合并成mp4
# 整合所有ts文件，保存为mp4格式
def tsToMp4(mp4_name, download_path):
    print("开始合并...")

    # os.system("copy /b {}/*.ts new.mp4".format(download_path))  # widows 系统
    os.system('cat {0}/*.ts > {0}/{1}'.format(download_path, mp4_name)) # linux 系统
    os.system('rm {0}/*.ts'.format(download_path))  # linux 系统
    print("结束合并...")

# url解码
# parse.unquote('https%3A%2F%2F2.123.com%2F20190825%2FPpeJJhQI%2Findex.m3u8')
# Out[28]: 'https://2.123.com/20190825/PpeJJhQI/index.m3u8'

def download_mp4(url_root, m3u8_url_2, download_path):
    print('开始下载`{}`'.format(m3u8_url_2))
    ts_url_list = getPlayList(url_root, m3u8_url_2)
    if not ts_url_list:
        print('`{}`未能获取ts文件'.format(m3u8_url_2))
        return False
    ts_name = ts_url_list[0].rsplit('/', maxsplit=1)[-1]
    mp4_name = ts_name.replace('.ts', '.mp4')
    if os.path.isfile(os.path.join(download_path, mp4_name)):
        print('`{},{}`文件已经存在！'.format(m3u8_url_2, mp4_name))
        return False
    download_ts(ts_url_list, download_path)
    tsToMp4(mp4_name, download_path)
    return True

def main():
    download_path = '/home/gswyhq/Downloads/output'

    download_urls = [
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190412/CeLwp2bn/hls/index.m3u8'],
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190411/7pBRXE7g/hls/index.m3u8'],
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190411/Z4NKyMee/hls/index.m3u8'],
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190808/Dt6AK4HN/hls/index.m3u8'],
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190411/KwWwEPYX/hls/index.m3u8'],
    ['http://xyz.com:2100', 'http://xyz.com:2100/20190411/Lg6SYyyI/hls/index.m3u8']
    ]

    for url_root, m3u8_url_2 in download_urls:
        download_mp4(url_root, m3u8_url_2, download_path)


if __name__ == '__main__':
    main()