#!/usr/bin/python
#-*- coding:UTF-8 -*-

import sys
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import quote
import urllib.request

# 中文-> 英文
zh_en_url = 'https://translate.google.cn/#zh-CN/en/'

# 英文-> 中文
en_zh_url = 'https://translate.google.cn/#en/zh-CN/'

# 命令行打开百度搜索使用示例：
# gswyhq@gswyhq-pc:~$ ./baidu.py
# gswyhq@gswyhq-pc:~$ ./baidu.py 怎么上网 python


if len(sys.argv) > 1:
    argv_list = sys.argv[1:]
else:
    argv_list = []

if all(re.search('^([a-zA-Z])+$', t) for t in argv_list):
    url = en_zh_url
else:
    url = zh_en_url

url = url + quote(' '.join(argv_list))

req = urllib.request.Request(url, headers={
    'age': 0,
    'cache-control': 'max-age=14400',
    'Connection': 'keep-alive',
    # 'Content-Encoding': 'gzip',
    'Content-Type': 'text/html',
    'Server': 'nginx',
    # 'Transfer-Encoding':'chunked',
    'VAR-Cache': 'HIT',
    'Vary': 'Accept-Encoding',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept-Language': 'zh-CN,zh;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
    'Upgrade-Insecure-Requests': 1,
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/49.0.2623.75 Safari/537.36'})

oper = urllib.request.urlopen(req)
data = oper.read()
u_data = data.decode('utf8')
soup = BeautifulSoup(data, 'lxml')
soup.find('//*[@id="result_box"]/span')

