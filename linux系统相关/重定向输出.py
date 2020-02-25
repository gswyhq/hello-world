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


# 解决 sudo echo x > 时的 Permission denied错误
运行下面语句清缓存时，报Permission denied错误：-bash: /proc/sys/vm/drop_caches: Permission denied
sudo echo 1 > /proc/sys/vm/drop_caches
问题原因：
bash 拒绝这么做，提示权限不够，是因为重定向符号 “>” 也是 bash 的命令。sudo 只是让 echo 命令具有了 root 权限，
但是没有让 “>” 命令也具有root 权限，所以 bash 会认为这个命令没有写入信息的权限。
解决方法：
"sh -c" 命令,它可以让 bash 将一个字串作为完整的命令来执行
sudo sh -c "echo 1 > /proc/sys/vm/drop_caches"

或者
echo 1 |sudo tee /proc/sys/vm/drop_caches 


