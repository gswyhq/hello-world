#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

#1.使用pipeline进行批量导入数据。包括先使用rpush插入数据，然后使用expire修改过期时间
class Redis_Handler(Handler):
	def connect(self):
		#print self.host,self.port,self.table
		self.conn = Connection(self.host,self.port,self.table)	
		
	def execute(self, action_name):
		filename = "/tmp/temp.txt"
		batch_size = 10000
		with open(filename) as file:
			try:
				count = 0
				pipeline_redis = self.conn.client.pipeline()
				for lines in file:
					(key,value) = lines.split(',')
						count = count + 1
						if len(key)>0:
							pipeline_redis.rpush(key,value.strip())
							if not count % batch_size:
								pipeline_redis.execute()
								count = 0
			
	
				#send the last batch
				pipeline_redis.execute()
			except Exception:
				print 'redis add error'

def main():
    pass


if __name__ == "__main__":
    main()
    
日常中有时为了做实验需要向redis导入大量数据

下面提供一些思路：

1、循环导入

key的类型为string，下面是例子脚本

for((i=1;i<=1000000;i++))
do
redis-cli -a "password" -n 0 -p 6379 set "k"$i "v"$i
done

这样的坏处显而易见，每次都要建立连接，效率低下，不可取。

当然你导入几百几千个key倒是没什么问题。


2、pipeline方式导入


先把预先的命令导入到一个文本里面
for((i=1;i<=1000000;i++))
do
echo "set k$i v$i" >> /tmp/_t.txt
done


执行格式转换
unix2dos /tmp/_t.txt 


pipeline导入
cat /tmp/_t.txt | redis-cli -a "password" -n 0 -p 6379 --pipe


以上测试下来，100万的数据，在几十秒中搞定，速度嗷嗷的

参考：http://redis.io/topics/mass-insert

