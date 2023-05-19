#!/usr/bin/python3
# coding: utf-8

import sys
import os
import zhconv
from tqdm import tqdm

# pip3 install zhconv

def convert_hans(file_path):
    if os.path.isfile(file_path):
        input_files = [file_path]
    elif os.path.isdir(file_path):
        input_files = [os.path.join(file_path, f) for f in os.listdir(file_path)]
    else:
        raise ValueError('输出参数有误，参数应该为一个文件路径或文件目录')
    for input_file in input_files:
        try:
            with open(input_file, encoding='utf-8')as f:
                datalines = f.readlines()
            with open(input_file, 'w', encoding='utf-8')as f:
                for line in tqdm(datalines):
                    line = line.strip()
                    if not line:
                        continue
                    f.write(zhconv.convert(line, 'zh-cn')+'\n')
        except Exception as e:
            print('`{}`转换失败：{}'.format(input_file, e))

def main():
    # input_file = '/home/gswyhq/下载/测试文件.txt'
    file_path = sys.argv[1]
    convert_hans(file_path)

if __name__ == '__main__':
    main()
