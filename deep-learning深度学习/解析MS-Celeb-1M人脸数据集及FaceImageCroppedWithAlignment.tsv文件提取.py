#!/usr/bin/python3
# coding: utf-8

# 1，MS-Celeb-1M数据集：
# MSR IRC是目前世界上规模最大、水平最高的图像识别赛事之一，由MSRA（微软亚洲研究院）图像分析、大数据挖掘研究组组长张磊发起，每年定期举办。
# 从1M个名人中，根据他们的受欢迎程度，选择100K个。然后，利用搜索引擎，给100K个人，每人搜大概100张图片。共100K * 100 = 10M个图片。
# 测试集包括1000个名人，这1000个名人来自于1M个明星中随机挑选。而且经过微软标注。每个名人大概有20张图片，这些图片都是网上找不到的。
# 其他常用人脸数据集：CAISA - WebFace, VGG - Face, MegaFace.
#
# 数据有对齐版可以直接用于训练（共80G数据）：
#
#
#
# 数据下载地址：https://hyper.ai/datasets/5543
#
# 2，FaceImageCroppedWithAlignment.tsv文件提取参考： https://www.twblogs.net/a/5ba2faf12b71771a4daa0c47/
#
# 下载并解压微软的大型人脸数据集MS-Celeb-1M后，将FaceImageCroppedWithAlignment.tsv文件还原成JPG图片格式。代码如下：

import base64
import struct
import os


def read_line(line):
    m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = line.split("	")
    rect = struct.unpack("ffff", base64.b64decode(face_rectangle))
    return m_id, image_search_rank, image_url, page_url, face_id, rect, base64.b64decode(face_data)


def write_image(filename, data):
    with open(filename, "wb") as f:
        f.write(data)


def unpack(file_name, output_dir):
    i = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = read_line(line)
            img_dir = os.path.join(output_dir, m_id)
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            img_name = "%s-%s" % (image_search_rank, face_id) + ".jpg"
            write_image(os.path.join(img_dir, img_name), face_data)
            i += 1
            if i % 1000 == 0:
                print(i, "images finished")

            # 仅仅演示，提取100张图像
            if i > 20:
                break
        print("all finished")


def main():
    file_name = "/media/gswyhq/000F3553000267F4/g_pan/MS-Celeb-1M/data/aligned_face_images/FaceImageCroppedWithAlignment.tsv"
    output_dir = "/media/gswyhq/000F3553000267F4/g_pan/MS-Celeb-1M/test_data2"
    unpack(file_name, output_dir)



# 提取后数据总共800多万张人脸图像：
#
#
#
# 3，其中同一目录图像有很多数据并非是同一人
#
# 网上有一份清理的文档 MS-Celeb-1M_clean_list.txt（包含79076个人，5049824张人脸图像）



if __name__ == '__main__':
    main()