#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
    import configparser
else:
    import cPickle as pickle
    from codecs import open
    import ConfigParser as configparser


class Db_Connector:
    def __init__(self, config_file_path):
        cf = configparser.ConfigParser()
        cf.read(config_file_path)

        #获取节点，返回的是一个list
        s = cf.sections()
        print ('section:', s)

        #获取某个节点下的键，返回的是一个list
        o = cf.options("db")
        print ('options:', o)

        #获取某个节点下的键值对，返回键值对元组列表
        v = cf.items("db")
        print ('db:', v)

        db_host = cf.get("db", "db_host")
        db_port = cf.getint("db", "db_port") #获取的是整形数值类型,同样的还有getfloat、getboolean。
        db_user = cf.get("db", "db_user")
        db_pwd = cf.get("db", "db_pass")

        print (db_host, db_port, db_user, db_pwd)

        #设置某个option 的值。（记得最后要写回）
        cf.set("db", "db_pass", "126")


        #添加一个section。（同样要写回）
        cf.add_section('liuqing')
        cf.set('liuqing', 'int', '15')
        print(cf.options('liuqing'))

        #移除section 或者option 。（只要进行了修改就要写回的哦）
        cf.remove_option('liuqing','int')
        print(cf.options('liuqing'))
        cf.remove_section('liuqing')
        cf.write(open(config_file_path, "w"))


if __name__ == "__main__":
    f = Db_Connector("/home/gswyhq/music/test.conf")

'''
class Config:
    def __init__(self, path):
        self.path = path
        self.cf = ConfigParser.ConfigParser()
        self.cf.read(self.path)
    def get(self, field, key):
        result = ""
        try:
            result = self.cf.get(field, key)
        except:
            result = ""
        return result
    def set(self, filed, key, value):
        try:
            self.cf.set(field, key, value)
            cf.write(open(self.path,'w'))
        except:
            return False
        return True



def read_config(config_file_path, field, key):
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(config_file_path)
        result = cf.get(field, key)
    except:
        sys.exit(1)
    return result

def write_config(config_file_path, field, key, value):
    cf = ConfigParser.ConfigParser()
    try:
        cf.read(config_file_path)
        cf.set(field, key, value)
        cf.write(open(config_file_path,'w'))
    except:
        sys.exit(1)
    return True

if __name__ == "__main__":
   if len(sys.argv) < 4:
      sys.exit(1)

   config_file_path = sys.argv[1]
   field = sys.argv[2]
   key = sys.argv[3]
   if len(sys.argv) == 4:
      print read_config(config_file_path, field, key)
   else:
      value = sys.argv[4]
      write_config(config_file_path, field, key, value)

'''