#!/usr/bin/python3
# coding: utf-8

import sys
import os

'''
通过iconv -l 命令查看，其支持的编码格式还不少，之间可以互相转换

1. enca (如果你的系统中没有安装这个命令，可以用sudo yum install -y enca 安装 )查看文件编码
$ enca filename

2. enconv 转换文件编码，比如要将一个GBK编码的文件转换成UTF-8编码，操作如下
enconv -L zh_CN -x UTF-8 filename

3. iconv 转换，iconv的命令格式如下：
iconv -f encoding -t encoding inputfile
比如将一个UTF-8 编码的文件转换成GBK编码
iconv -f UTF-8 -t GBK file1 -o file2

转换gbk编码文件为utf-8编码文件
简洁命令：iconv  -f gbk -t utf-8 index.html > aautf8.html
其中-f指的是原始文件编码，-t是输出编码  index.html 是原始文件   aautf8.html是输出结果文件

转换gbk编码文件为utf-8编码文件
详细命令：iconv -c --verbose  -f gbk -t utf-8 index.html -o index_utf8.html
-c 指的是从输出中忽略无效的字符， --verbose指的是打印进度信息   -o是输出文件

转换utf-8编码文件为gb2312编码文件
详细命令：iconv -c --verbose  -f utf-8 -t gb2312 index_utf8.html -o index_gb2312.html


'''
def gbk_to_utf8(file_path):
    if os.path.isfile(file_path):
        input_files = [file_path]
    elif os.path.isdir(file_path):
        input_files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    else:
        raise ValueError('输出参数有误，参数应该为一个文件路径或文件目录')
    for input_file in input_files:
        try:
            with open(input_file, encoding='gb18030')as f:
                data = f.read()
            with open(input_file, 'w', encoding='utf-8')as f:
                f.write(data)
        except Exception as e:
            print('`{}`转换失败：{}'.format(input_file, e))

def main():
    # input_file = '/home/gswyhq/下载/测试文件.txt'
    file_path = sys.argv[1]
    gbk_to_utf8(file_path)

if __name__ == '__main__':
    main()
