#!/bin/bash

a=3;
b=4;
c=4;

if [ $a -gt $b ]
	then echo "参数$a大于参数$b"
else 
	echo "参数$a小于参数$b"
fi

x=`ls |grep shell|wc -l`
y=10;

if [ $x -eq $y ];then
    echo "$x等于$y"
elif [ $x -gt $y ];then
    echo "$x 大于 $y"
else
    echo "$ 小于 $y"
fi

#数字判断一些命令：
#-gt是大于
#-lt是小于
#-eq是等于
#-ne是不等于
#-ge是大于等于
#-le是小于等于 

