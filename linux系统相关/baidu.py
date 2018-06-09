#!/usr/bin/python
#-*- coding:UTF-8 -*-


# 命令行打开百度搜索使用示例：
# gswewf@gswewf-pc:~$ ./baidu.py
# gswewf@gswewf-pc:~$ ./baidu.py 怎么上网 python

import time
import sys
import webbrowser

url = 'https://www.baidu.com/'

if len(sys.argv) > 1:
    url = url + 's?ie=utf-8&f=8&rsv_bp=1&tn=baidu&wd=' + '%20'.join(sys.argv[1:])

webbrowser.open(url, autoraise=False)

time.sleep(3)


