#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pip3 install paddleocr>=3.0.0 paddlepaddle>=3.0.0

# 运行 PP-OCRv5 推理
# paddleocr ocr -i https://paddle-model-ecology.bj.bcebos.com/paddlex/imgs/demo_image/general_ocr_002.png --use_doc_orientation_classify False --use_doc_unwarping False --use_textline_orientation False

# 若离线识别，需要先下载好预训练模型，解压放置于：.paddlex/official_models
# .paddlex/official_models 目录结构如下：
# .
# +--- PP-LCNet_x1_0_doc_ori
# |   +--- inference.json
# |   +--- inference.pdiparams
# |   +--- inference.yml
# +--- PP-LCNet_x1_0_doc_ori_infer.tar
# +--- PP-LCNet_x1_0_textline_ori
# |   +--- img_textline180_demo_res.jpg
# |   +--- img_textline180_demo_res.json
# |   +--- inference.json
# |   +--- inference.pdiparams
# |   +--- inference.yml
# +--- PP-OCRv5_server_det
# |   +--- check_dataset_result.json
# |   +--- inference.json
# |   +--- inference.pdiparams
# |   +--- inference.yml
# +--- PP-OCRv5_server_rec
# |   +--- inference.json
# |   +--- inference.pdiparams
# |   +--- inference.yml

# 模型下载地址：
# 检测模型：https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_det_infer.tar
# 识别模型：https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv5_server_rec_infer.tar
# 方向分类器：https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_textline_ori_infer.tar
# https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar
# 字体：https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf
# https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/simfang.ttf

##########################################################################################################################
# 仅检测模型使用示例：
import os

USERNAME = os.getenv("USERNAME")
os.environ['PADDLE_PDX_LOCAL_FONT_FILE_PATH']= rf'D:\Users\{USERNAME}/github_project/PaddleOCR/doc/fonts/PingFang-SC-Regular.ttf'

from paddleocr import PaddleOCR
import unicodedata

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    doc_orientation_classify_model_name='PP-LCNet_x1_0_doc_ori',
    text_detection_model_name='PP-OCRv5_server_det',
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_line_orientation_model_name="PP-LCNet_x1_0_textline_ori",
    use_doc_unwarping=False,
    use_textline_orientation=False)
# 对示例图像执行 OCR 推理
result = ocr.predict(
    use_doc_orientation_classify=True,
    input=rf"D:\Users\{USERNAME}/github_project/PaddleOCR/tests/test_files/general_ocr_002.png")
# 可视化结果并保存 json 结果
for res in result:
    res.print()
    res.save_to_img(rf"D:\Users\{USERNAME}/Downloads/output")
    res.save_to_json(rf"D:\Users\{USERNAME}/Downloads/output")


