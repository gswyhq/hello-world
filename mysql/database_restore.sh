#!/bin/bash

##################################################################
# 还原某个数据库  
##################################################################  
# 修改密码  
# mysqladmin -u root password "mypasssecret"  
# mysqladmin -u root password oldpass "mypasssecret"  
table_name=youhuibao_2017_05_05_15_14_04.sql.gz
path=`pwd`

#echo $path/$table_name
gunzip $table_name

/usr/bin/mysql --host=localhost -uroot -p123456 youhuibao < youhuibao_2017_05_05_15_14_04.sql
