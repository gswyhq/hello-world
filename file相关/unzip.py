#!/usr/bin/python3
# coding: utf-8

import os
import zipfile

# 解压缩文件

def files_to_zip(zip_file, datas):
    """
    将多个文件压缩成zip格式文件
    :param zip_file: 压缩文件存放路径
    :param datas: 
    :return: 
    """
    with zipfile.ZipFile(zip_file,'w',zipfile.ZIP_DEFLATED) as f:
        for filename,file_url in datas:
            f.write(file_url, filename)


def add_file_to_zip(zip_file, datas):
    """
    压缩新文件到已有ZIP文件中
    :param zip_file: 压缩文件存放路径
    :param datas: 
    :return: 
    """
    with zipfile.ZipFile(zip_file, 'a', zipfile.ZIP_DEFLATED) as f:
        for filename,file_url in datas:
            f.write(filename,file_url)

def un_zip_files(zip_file):
    """
    解压当前目录下的zip文件到当前目录同zip文件名目录下，并删除原有的zip文件
    :param zip_file: zip格式的压缩文件
    :return: 
    """
    assert os.path.isfile(zip_file), "文件不存在：{}".format(zip_file)
    assert os.path.splitext(zip_file)[1] == '.zip', "输入的应该为.zip格式文件： {}".format(zip_file)
    zip_file_path, _ = os.path.splitext(zip_file)
    assert not os.path.isdir(zip_file_path), "解压的文件路径已经存在：　{}".format(zip_file_path)
    os.mkdir(zip_file_path)

    with zipfile.ZipFile(zip_file, 'r') as file_zip:
        for file in file_zip.namelist():
            file_zip.extract(file, zip_file_path)
    os.remove(zip_file)

def main():
    zip_file = '/home/gswyhq/Downloads/最新上传数据.zip'
    datas = [
        ('实体及其同义词.csv', '/home/gswyhq/graph_qa/input/实体及其同义词.csv'),
        ('neo4j三元组.csv', '/home/gswyhq/graph_qa/input/neo4j三元组.csv'),
        ('中信银行标注意图.csv', '/home/gswyhq/graph_qa/input/中信银行标注意图.csv')
    ]
    # files_to_zip(zip_file, datas)  # 文件压缩成zip格式

    un_zip_files(zip_file)  # 解压zip格式文件

if __name__ == '__main__':
    main()
