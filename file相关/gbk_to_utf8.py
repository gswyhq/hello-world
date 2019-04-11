#!/usr/bin/python3
# coding: utf-8

import sys

'''
通过iconv -l 命令查看，其支持的编码格式还不少，之间可以互相转换

转换gbk编码文件为utf-8编码文件
简洁命令：iconv  -f gbk -t utf-8 index.html > aautf8.html
其中-f指的是原始文件编码，-t是输出编码  index.html 是原始文件   aautf8.html是输出结果文件

转换gbk编码文件为utf-8编码文件
详细命令：iconv -c --verbose  -f gbk -t utf-8 index.html -o index_utf8.html
-c 指的是从输出中忽略无效的字符， --verbose指的是打印进度信息   -o是输出文件

转换utf-8编码文件为gb2312编码文件
详细命令：iconv -c --verbose  -f utf-8 -t gb2312 index_utf8.html -o index_gb2312.html


'''
def gbk_to_utf8(input_file):
    with open(input_file, encoding='gb18030')as f:
        data = f.read()
    with open(input_file, 'w', encoding='utf-8')as f:
        f.write(data)

def main():
    # input_file = '/home/gswyhq/下载/测试文件.txt'
    input_file = sys.argv[1]
    gbk_to_utf8(input_file)

if __name__ == '__main__':
    main()