#!/bin/bash

#ssh执行远程操作
#命令格式：ssh -p $port $user@$ip 'cmd'

#$port : ssh连接端口号
#$user: ssh连接用户名
#$ip:ssh连接的ip地址
#cmd:远程服务器需要执行的操作，cmd如果是脚本，注意绝对路径问题

#ssh的-t参数
#-t：就是可以提供一个远程服务器的虚拟tty终端，加上这个参数我们就可以在远程服务器的虚拟终端上输入自己的提权密码了，非常安全

# 远程服务器上的程序的绝对路径
remote_cmd='/home/zy/yhb/remotely_git.sh'

#变量定义
ip_array=("192.168.3.103")
user="zy"

#本地通过ssh执行远程服务器的脚本
for ip in ${ip_array[*]}
do
    if [ $ip = "192.168.1.1" ]; then
        port="7777"
    else
        port="22"
    fi
    ssh -t -p $port $user@$ip $remote_cmd
done

# 本地shell文件，远程执行（将 time.sh 脚本放到远程服务器192.168.3.103  上运行）：
ssh zy@192.168.3.103 < time.sh


Windows系统利用putty远程执行本地脚本：
putty.exe -ssh -pw 123456 zy@192.168.3.103 -m "time.sh"

Windows系统利用pscp复制远程文件到本地：
pscp.exe -pw 123456 zy@192.168.3.103:/home/zy/remote_command/dump_bak.xls .

