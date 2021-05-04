#!/bin/bash

path=`pwd`
echo ${path}

cd ${path}

# :>file可以创建空文件，如果file存在，则把file截断为0字节
:>log/myapp.log
:>log/myapp_err.log

# 需要设置LANG变量，默认情况下是空
export LANG="zh_CN.UTF-8"

cp supervisor_tornado.conf /etc/supervisor/conf.d/supervisor_tornado.conf
cp nginx_nlp.conf /etc/nginx/conf.d/nginx_nlp.conf

sed -i "s#MYDIR#$MYDIR#g" /etc/supervisor/conf.d/supervisor_tornado.conf

service nginx start
service supervisor start

# 需要添加此项， 不然启动容器的时候，一直在重启
/bin/bash

#gswewf@gswewf-PC:~/xinch_zhongjixian$ docker run -it --restart=always --detach --name=xinch_zhongjixian_18000 -e MYDIR=/xinch_zhongjixian -p 18000:8000 --volume=$PWD:/xinch_zhongjixian --workdir=/xinch_zhongjixian -v /etc/localtime:/etc/localtime ubuntu:1221 /xinch_zhongjixian/start.sh
#7ba49adfd9ea465fdf6cf3a397a690064f4f57bf9df3927fd4020421053bea56