
# 基于PP-OCRv3的手写文字识别
# pip3 install paddleocr>=3.2.0 paddlepaddle>=3.2.0


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
# 文本识别模块：
# https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0//PP-OCRv5_server_rec_infer.tar

##########################################################################################################################
# 仅检测模型使用示例：
import os, re
import cv2
import svgwrite
USER_DIR = os.path.expanduser('~')
USERNAME = os.getenv("USERNAME")
os.environ['PADDLE_PDX_LOCAL_FONT_FILE_PATH']= rf'{USER_DIR}/github_project/PaddleOCR/doc/fonts/PingFang-SC-Regular.ttf'

from paddleocr import PaddleOCR
import unicodedata
from PIL import Image
import numpy as np


def jpeg_to_png(jpeg_file):
    '''
    将 JPG 或 JPEG 格式的图片转换为 PNG 格式。
    '''
    # 检查路径中是否包含中文字符
    assert not re.search(r'[\u4e00-\u9fff]', jpeg_file), f"⚠️ 警告：路径中包含中文字符，可能影响某些图像处理库的兼容性。路径为：{jpeg_file}"

    if jpeg_file.lower().endswith('.png'):
        return jpeg_file
    assert any(jpeg_file.lower().endswith(ext) for ext in ('.jpg', '.jpeg')), '输入应该为jpg、jepg格式文件'
    save_file = jpeg_file.rsplit('.', maxsplit=1)[0]+'.png'
    im = Image.open(jpeg_file)
    im.save(save_file, "png")  # 保存图像为png格式
    return save_file

# 初始化 PaddleOCR 实例
ocr = PaddleOCR(
    doc_orientation_classify_model_name='PP-LCNet_x1_0_doc_ori',
    doc_orientation_classify_model_dir=f"{USER_DIR}/.paddlex/official_models/PP-LCNet_x1_0_doc_ori",
    text_detection_model_name='PP-OCRv5_server_det',
    text_detection_model_dir=f"{USER_DIR}/.paddlex/official_models/PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    text_recognition_model_dir=f"{USER_DIR}/.paddlex/official_models/PP-OCRv5_server_rec",
    textline_orientation_model_name="PP-LCNet_x1_0_textline_ori",
    textline_orientation_model_dir=f"{USER_DIR}/.paddlex/official_models/PP-LCNet_x1_0_textline_ori",
    use_doc_unwarping=False,
    use_textline_orientation=False)
# 对示例图像执行 OCR 推理
result = ocr.predict(
    use_doc_orientation_classify=True,
    input=jpeg_to_png(rf"{USER_DIR}\Pictures\2516320889_09_24_52.png"),
    return_word_box=True, # 按字粒度进行识别
)
# 可视化结果并保存 json 结果
for res in result:
    res.print()
    rec_texts = res['rec_texts']
    rec_scores = res['rec_scores']
    rec_boxes = res['rec_boxes']
    res.save_to_img(rf"{USER_DIR}/Downloads/output")
    res.save_to_json(rf"{USER_DIR}/Downloads/output")


