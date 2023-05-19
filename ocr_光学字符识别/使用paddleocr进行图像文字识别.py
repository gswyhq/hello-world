#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip3 install paddleocr

# 命令行识别：
# ~$ paddleocr --image_dir 68747.png --use_angle_cls true --use_gpu false

# 若离线识别，需要先下载好预训练模型，解压放置于：~/.paddleocr
# ~/.paddleocr 目录结构如下：
# .
# +--- whl
# |   +--- cls
# |   |   +--- ch_ppocr_mobile_v2.0_cls_infer
# |   |   |   +--- ._inference.pdmodel
# |   |   |   +--- ch_ppocr_mobile_v2.0_cls_infer.tar
# |   |   |   +--- inference.pdiparams
# |   |   |   +--- inference.pdiparams.info
# |   |   |   +--- inference.pdmodel
# |   +--- det
# |   |   +--- ch
# |   |   |   +--- ch_PP-OCRv3_det_infer
# |   |   |   |   +--- ch_PP-OCRv3_det_infer.tar
# |   |   |   |   +--- inference.pdiparams
# |   |   |   |   +--- inference.pdiparams.info
# |   |   |   |   +--- inference.pdmodel
# |   +--- rec
# |   |   +--- ch
# |   |   |   +--- ch_PP-OCRv3_rec_infer
# |   |   |   |   +--- ch_PP-OCRv3_rec_infer.tar
# |   |   |   |   +--- inference.pdiparams
# |   |   |   |   +--- inference.pdiparams.info
# |   |   |   |   +--- inference.pdmodel
#
#
# 模型下载地址：
# 检测模型：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_infer.tar
# 识别模型：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar
# 方向分类器：https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar

##########################################################################################################################

from paddleocr import PaddleOCR, draw_ocr
import unicodedata
import os

USERNAME = os.getenv("USERNAME")
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
ocr = PaddleOCR(use_angle_cls=True, #设置 true 使用方向分类器识别180度旋转文字
                det_db_box_thresh=0.6,  # 当使用DB模型会出现整行漏检，可适当调小该值
                det_db_unclip_ratio=1.5, # 如果觉得文本框不够紧凑，也可以把该参数调小。默认值1.5
                lang="ch")  # # 只需运行一次即可下载模型并将其加载到内存中
img_path = rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747.png"
result = ocr.ocr(img_path,
                 cls=True,  # 是否使用角度分类器,默认值为True。如果为 true，则可以识别旋转 180 度的文本。如果没有文本旋转 180 度，请使用 cls=False 以获得更好的性能。即使 cls=False，也可以识别旋转 90 度或 270 度的文本。
                 rec=True,  # 是否使用文本识别,默认值为True
                 det=True,  # 是否使用文本检测,默认值为True
                 )
# 结果是一个list，每个item包含了文本框(左上为零点，依次为：左上, 左下, 右下, 右上;)，文字和识别置信度
for idx in range(len(result)):
    res = result[idx]
    print("结果数：", len(res))
    for line in res:
        dt_boxes, (rec, score) = line
        rec = unicodedata.normalize('NFKD', rec)  # 将文本标准化
        print(dt_boxes, rec, score)

# 显示结果
# 如果本地没有simfang.ttf，可以在doc/fonts目录下下载
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path=rf'D:\Users\{USERNAME}\github_project/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save(rf"D:\Users\{USERNAME}\github_project\PJ_PREDICT_IMG\68747_4.png")

unicodedata.normalize('NFKD', '纳税人识别号：９１３３００００７１７６１２５２２B')  # 将文本标准化
# Out[41]: '纳税人识别号:91330000717612522B'