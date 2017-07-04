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

Python中一般使用xlrd（excel read）来读取Excel文件，使用xlwt（excel write）来生成Excel文件（可以控制Excel中单元格的格式），需要注意的是，用xlrd读 取excel是不能对其进行操作的：xlrd.open_workbook()方法返回xlrd.Book类型，是只读的，不能对其进行操作。而 xlwt.Workbook()返回的xlwt.Workbook类型的save(filepath)方法可以保存excel文件。因此对于读取和生成Excel文件都非常容易处理，但是对于已经存在的Excel文件进行修改就比较麻烦了。不过，还有一个xlutils（依赖于xlrd和xlwt）提供复制excel文件内容和修改文件的功能。其实际也只是在xlrd.Book和xlwt.Workbook之间建立了一个管道而已，如下图：

 

 

 

xlutils.copy模块的copy()方法实现了这个功能，示例代码如下：

from xlrd import open_workbook
from xlutils.copy import copy
 
rb = open_workbook('m:\\1.xls')
 
#通过sheet_by_index()获取的sheet没有write()方法
rs = rb.sheet_by_index(0)
 
wb = copy(rb)
 
#通过get_sheet()获取的sheet有write()方法
ws = wb.get_sheet(0)
ws.write(0, 0, 'changed!')
 
wb.save('m:\\1.xls')


def main():
    pass


if __name__ == "__main__":
    main()
