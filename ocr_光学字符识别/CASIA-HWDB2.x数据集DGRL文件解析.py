#!/usr/bin/env python
# coding=utf-8

# 公开的手写文本识别数据集，包含Chinese OCR, 中科院自动化研究所-手写中文数据集CASIA-HWDB2.x(http://www.nlpr.ia.ac.cn/databases/handwriting/Download.html)，
# 以及由中科院手写数据和网上开源数据合并组合的数据集(https://aistudio.baidu.com/aistudio/datasetdetail/102884/0)等

# CASIA-HWDB 数据集是最常见的手写汉字识别数据集，它包含脱机、联机两部分，分单字、文本行两种类型：
# HWDB1.x：脱机单字，1.0~1.2 三个版本，数据格式为 .gnt
# OLHWDB1.x：联机单字，1.0~1.2 三个版本，
# HWDB2.x：脱机文本行，1.0~1.2 三个版本，数据格式为 .dgrl
# OLHWDB1.x：联机文本行，1.0~1.2 三个版本，
# 一般常用的汉字识别多为脱机单字识别，该部分数据格式为 .gnt;
# 文本行识别数据集，它的格式是 .dgrl，它的解析类似于 .gnt

# .dgrl 格式，一张图对应一个 DGRL 文件，大部分内容都有固定的长度，部分内容长度不固定但是也能通过其他数据推导出来，我们可以通过访问文件特定位置的数据得到我们需要的内容：行文本标注，行图像。
#

# HWDB2.0Train (709MB)
# HWDB2.0Test (175MB)
# HWDB2.1Train (540MB)
# HWDB2.1Test (136MB)
# HWDB2.2Train (515MB)
# HWDB2.2Test (128MB)

# mkdir HWDB2.xTrain HWDB2.xTest

# unzip HWDB2.0Test.zip -d HWDB2.xTest
# unzip HWDB2.1Test.zip -d HWDB2.xTest
# unzip HWDB2.2Test.zip -d HWDB2.xTest

# unzip HWDB2.0Train.zip -d HWDB2.xTrain
# unzip HWDB2.1Train.zip -d HWDB2.xTrain
# unzip HWDB2.3Train.zip -d HWDB2.xTrain

import struct
import os
import cv2 as cv
import numpy as np
from glob import glob
from tqdm import tqdm
USERNAME = os.getenv("USERNAME")

def read_from_agrl(dgrl):
    if not os.path.exists(dgrl):
        print('DGRL not exis!')
        return

    dir_name, base_name = os.path.split(dgrl)
    label_dir = dir_name + '_label'
    image_dir = dir_name + '_images'
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    # 读取 .dgrl 时先以二进制方式打开文件
    with open(dgrl, 'rb') as f:
        # 读取表头尺寸
        # 用 numpy 挨个去读取，它的读取方式跟 f.readline() 类似，一个一个读，所以之前读了多少个数据就很重要！所以要指定读的格式、数量：
        # 一般 dtype 都选择 uint8，count 需要根据上图结构中的长度 Length 做相应变化。
        header_size = np.fromfile(f, dtype='uint8', count=4)
        header_size = sum([j << (i * 8) for i, j in enumerate(header_size)])
        # print(header_size)

        # 读取表头剩下内容，提取 code_length
        header = np.fromfile(f, dtype='uint8', count=header_size - 4)
        code_length = sum([j << (i * 8) for i, j in enumerate(header[-4:-2])])
        # print(code_length)

        # 读取图像尺寸信息，提取图像中行数量
        image_record = np.fromfile(f, dtype='uint8', count=12)
        height = sum([j << (i * 8) for i, j in enumerate(image_record[:4])])
        width = sum([j << (i * 8) for i, j in enumerate(image_record[4:8])])
        line_num = sum([j << (i * 8) for i, j in enumerate(image_record[8:])])
        print('图像尺寸:')
        print(height, width, line_num)

        # 读取每一行的信息
        for k in range(line_num):
            print(k + 1)

            # 读取该行的字符数量
            char_num = np.fromfile(f, dtype='uint8', count=4)
            char_num = sum([j << (i * 8) for i, j in enumerate(char_num)])
            print('字符数量:', char_num)

            # 读取该行的标注信息
            label = np.fromfile(f, dtype='uint8', count=code_length * char_num)
            label = [label[i] << (8 * (i % code_length)) for i in range(code_length * char_num)]
            label = [sum(label[i * code_length:(i + 1) * code_length]) for i in range(char_num)]

            # 要注意的一个地方是：行文本标注读取出来以后，是一个 int 列表，要把它还原成汉字，一个汉字占用两个字节（具体由 code_length 决定），使用 struct 将其还原：
            label = [struct.pack('I', i).decode('gbk', 'ignore')[0] for i in label]
            # 上面的 i 就是提取出来的汉字编码，解码格式为 gbk，有些行文本会有空格，解码可能会出错，使用 ignore 忽略。
            print('合并前：', label)
            label = ''.join(label)
            print('合并后：', label)

            # 读取该行的位置和尺寸
            pos_size = np.fromfile(f, dtype='uint8', count=16)
            y = sum([j << (i * 8) for i, j in enumerate(pos_size[:4])])
            x = sum([j << (i * 8) for i, j in enumerate(pos_size[4:8])])
            h = sum([j << (i * 8) for i, j in enumerate(pos_size[8:12])])
            w = sum([j << (i * 8) for i, j in enumerate(pos_size[12:])])
            # print(x, y, w, h)

            # 读取该行的图片
            bitmap = np.fromfile(f, dtype='uint8', count=h * w)
            bitmap = np.array(bitmap).reshape(h, w)

            # 保存信息
            label_file = os.path.join(label_dir, base_name.replace('.dgrl', '_' + str(k) + '.txt'))
            with open(label_file, 'w', encoding='utf-8') as f1:
                f1.write(label)
            bitmap_file = os.path.join(image_dir, base_name.replace('.dgrl', '_' + str(k) + '.jpg'))
            cv.imwrite(bitmap_file, bitmap)
            print(label_file)

def main():
    for dgrl_file in tqdm(glob(rf"D:\Users\{USERNAME}\data\CASIA-HWDB2.x\HWDB2.xTest\*.dgrl")):
        read_from_agrl(dgrl_file)
        # break

    for dgrl_file in tqdm(glob(rf"D:\Users\{USERNAME}\data\CASIA-HWDB2.x\HWDB2.xTrain\*.dgrl")):
        read_from_agrl(dgrl_file)
        # break

# 解析后的图片存于目录：
# HWDB2.xTest_images
# HWDB2.xTrain_images

# 解析后图片对应的文本存于目录：
# HWDB2.xTest_label
# HWDB2.xTrain_label

if __name__ == "__main__":
    main()
