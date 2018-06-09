#!/usr/bin/python
#-*- coding:UTF-8 -*-


# 命令行获取zip使用示例：
# gswewf@gswewf-pc:~$ ./cmd_help.py zip

import time
import sys
import webbrowser
# sys.path.append("libs")
url = 'http://linux.51yip.com/search/'

if len(sys.argv) > 1:
    url = url + sys.argv[1]

webbrowser.open(url, autoraise=False)

time.sleep(3)


