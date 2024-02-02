
#读光-文字识别-行识别模型-中英-手写文本领域

#来源：https://modelscope.cn/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/summary
# 手写汉字识别，百度也有对应项目，但需要自己下载预训练模型及数据集微调训练，对应地址：
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/applications/手写文字识别.md

import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

USERNAME = os.getenv('USERNAME')
# ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-handwritten_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model=rf"D:\Users\{USERNAME}\data\cv_convnextTiny_ocr-recognition-handwritten_damo")

# http://www.modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/revisions?EndTime=1684425540
# www.modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo/files?Recursive=

# 项目文件结构：
# .
# +--- .gitattributes
# +--- configuration.json
# +--- pytorch_model.pt
# +--- README.md
# +--- resources
# |   +--- ConvTransformer-Pipeline.jpg
# |   +--- rec_result_visu.png
# +--- vocab.txt

# modelscope模型文件离线下载地址：
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=resources%2Frec_result_visu.png
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=resources%2FConvTransformer-Pipeline.jpg
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=README.md
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=.gitattributes
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=configuration.json
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=pytorch_model.pt
# https://modelscope.cn/api/v1/models/damo/cv_convnextTiny_ocr-recognition-handwritten_damo/repo?Revision=master&FilePath=vocab.txt

### 使用远程图片预测
# img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition_handwritten.jpg'
# result = ocr_recognition(img_url)
# print(result)

### 使用本地图像文件预测
### 输入图片应为包含文字的单行文本图片。其它如多行文本图片、非文本图片等可能没有返回结果，此时表示模型的识别结果为空。
img_path = rf'D:\Users\{USERNAME}\data\cv_convnextTiny_ocr-recognition-handwritten_damo\resources\rec_result_visu.png'
img = cv2.imread(img_path)
result = ocr_recognition(img)
print(result)


##################################################### 多行手写汉字图片的识别 #####################################################################

import copy
import numpy as np
from paddleocr import PaddleOCR, draw_ocr
from paddleocr.tools.infer.utility import get_rotate_crop_image, get_minarea_rect_crop
from paddleocr.paddleocr import check_img
from paddleocr.tools.infer.predict_system import sorted_boxes
import unicodedata
import os

USERNAME = os.getenv("USERNAME")
# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`

# 这里使用baidu的文本框检测，也可以使用其他的文本框检测模型，如: “https://www.modelscope.cn/studios/damo/cv_ocr-text-spotting/summary”
ocr = PaddleOCR(use_angle_cls=True, #设置 true 使用方向分类器识别180度旋转文字
                det_db_box_thresh=0.6,  # 当使用DB模型会出现整行漏检，可适当调小该值
                det_db_unclip_ratio=1.5, # 如果觉得文本框不够紧凑，也可以把该参数调小。默认值1.5
                lang="ch")  # # 只需运行一次即可下载模型并将其加载到内存中
img_path = rf"D:\Users\{USERNAME}\data\协查通知元素识别\image\original\6.png"
result = ocr.ocr(img_path,
                 cls=False,  # 是否使用角度分类器,默认值为True。如果为 true，则可以识别旋转 180 度的文本。如果没有文本旋转 180 度，请使用 cls=False 以获得更好的性能。即使 cls=False，也可以识别旋转 90 度或 270 度的文本。
                 rec=False,  # 是否使用文本识别,默认值为True
                 det=True,  # 是否使用文本检测,默认值为True
                 )
img = check_img(img_path)
ori_im = img.copy()
result2 = []
for idx in range(len(result)):
    dt_boxes = result[idx]
    dt_boxes = sorted_boxes(np.array(dt_boxes))
    result2.append([])
    print("结果数：", len(dt_boxes))
    for bno in range(len(dt_boxes)):
        tmp_box = copy.deepcopy(dt_boxes[bno])
        img_crop = get_minarea_rect_crop(ori_im, tmp_box)
        text_list = ocr_recognition(img_crop).get('text', [])
        text = '\n'.join(text_list)
        text = unicodedata.normalize('NFKD', text)  # 将文本标准化
        result2[idx].append([tmp_box, (text, 0.8)])

# 最后结果是一个list，每个item包含了文本框(左上为零点，依次为：左上, 左下, 右下, 右上;)，文字和识别置信度

# 显示结果
# 如果本地没有simfang.ttf，可以在doc/fonts目录下下载
from PIL import Image
result = result2[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
print(result)
im_show = draw_ocr(image, boxes, txts, scores, font_path=rf'D:\Users\{USERNAME}\github_project/PaddleOCR/doc/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save(rf"D:\Users\{USERNAME}\data\协查通知元素识别\image\6_2.png")

unicodedata.normalize('NFKD', '纳税人识别号：９１３３００００７１７６１２５２２B')  # 将文本标准化
# Out[41]: '纳税人识别号:91330000717612522B'
