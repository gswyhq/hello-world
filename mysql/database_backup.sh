#!/bin/bash

cd /home/gswyhq/yhb/mysql_connect

#备份地址
backupdir=`pwd`/data

if [ ! -e $backupdir ]
then
    mkdir $backupdir
    echo "创建目录$backupdir"
else
    echo "数据保存在$backupdir"
fi

#备份文件后缀时间
time=_` date +%Y_%m_%d_%H_%M_%S `
#需要备份的数据库名称
db_name=yhb
table_name=question

#mysql 用户名
#db_user=
#mysql 密码
#db_pass=
#mysqldump命令使用绝对路径
/usr/bin/mysqldump -hlocalhost $db_name $table_name | gzip > $backupdir/$table_name$time.sql.gz
# /home/server/mysql-5.6.21/bin/mysqldump $db_name | gzip > $backupdir/$table_name$time.sql.gz
#删除1天之前的备份文件
find $backupdir -name $table_name"*.sql.gz" -type f -mtime +1 -exec rm -rf {} \; > /dev/null 2>&1

# mysql@gswyhq-pc:~$ /usr/bin/mysqldump  -uroot -p123456 --host=localhost --databases yhb --tables python_tstamps > python_tstamps.sql
# mysqldump: [Warning] Using a password on the command line interface can be insecure.
# mysql@gswyhq-pc:~$ /usr/bin/mysqldump --host=localhost --databases yhb --tables python_tstamps > python_tstamps.sql
