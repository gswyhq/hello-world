#!/usr/bin/python
#-*- coding:UTF-8 -*-

import sys
import os

savedStdout = sys.stdout  #保存标准输出流
with open(os.devnull, 'w+') as file:
    sys.stdout = file  #标准输出重定向至文件
    sys.stderr = file
    print ('这里的信息被重定向了，输出到devnull')
    print('2334323432')


sys.stdout = savedStdout  #恢复标准输出流
print ('这里的信息恢复到输出到终端')

