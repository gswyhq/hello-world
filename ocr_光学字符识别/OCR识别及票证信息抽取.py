
# 第一步，票证检测矫正
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
card_detection_correction = pipeline(Tasks.card_detection_correction, model='damo/cv_resnet18_card_correction')
result = card_detection_correction('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/card_detection_correction.jpg')
print(result)

cv2.imwrite('result.jpg', result['output_imgs'][0])

# 第二步，读光文字检测
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-db-line-level_damo')
result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
print(result)

# 第三步，读光文字识别
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import cv2

ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')

### 使用url
img_url = 'http://duguang-labelling.oss-cn-shanghai.aliyuncs.com/mass_img_tmp_20220922/ocr_recognition.jpg'
result = ocr_recognition(img_url)
print(result)

### 使用图像文件
### 请准备好名为'ocr_recognition.jpg'的图像文件
# img_path = 'ocr_recognition.jpg'
# img = cv2.imread(img_path)
# result = ocr_recognition(img)
# print(result)


# 文本检测及识别
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
import cv2
import math

# scripts for crop images
def crop_image(img, position):
    def distance(x1,y1,x2,y2):
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))
    position = position.tolist()
    for i in range(4):
        for j in range(i+1, 4):
            if(position[i][0] > position[j][0]):
                tmp = position[j]
                position[j] = position[i]
                position[i] = tmp
    if position[0][1] > position[1][1]:
        tmp = position[0]
        position[0] = position[1]
        position[1] = tmp

    if position[2][1] > position[3][1]:
        tmp = position[2]
        position[2] = position[3]
        position[3] = tmp

    x1, y1 = position[0][0], position[0][1]
    x2, y2 = position[2][0], position[2][1]
    x3, y3 = position[3][0], position[3][1]
    x4, y4 = position[1][0], position[1][1]

    corners = np.zeros((4,2), np.float32)
    corners[0] = [x1, y1]
    corners[1] = [x2, y2]
    corners[2] = [x4, y4]
    corners[3] = [x3, y3]

    img_width = distance((x1+x4)/2, (y1+y4)/2, (x2+x3)/2, (y2+y3)/2)
    img_height = distance((x1+x2)/2, (y1+y2)/2, (x4+x3)/2, (y4+y3)/2)

    corners_trans = np.zeros((4,2), np.float32)
    corners_trans[0] = [0, 0]
    corners_trans[1] = [img_width - 1, 0]
    corners_trans[2] = [0, img_height - 1]
    corners_trans[3] = [img_width - 1, img_height - 1]

    transform = cv2.getPerspectiveTransform(corners, corners_trans)
    dst = cv2.warpPerspective(img, transform, (int(img_width), int(img_height)))
    return dst

def order_point(coor):
    arr = np.array(coor).reshape([4, 2])
    sum_ = np.sum(arr, 0)
    centroid = sum_ / arr.shape[0]
    theta = np.arctan2(arr[:, 1] - centroid[1], arr[:, 0] - centroid[0])
    sort_points = arr[np.argsort(theta)]
    sort_points = sort_points.reshape([4, -1])
    if sort_points[0][0] > centroid[0]:
        sort_points = np.concatenate([sort_points[3:], sort_points[:3]])
    sort_points = sort_points.reshape([4, 2]).astype('float32')
    return sort_points

ocr_detection = pipeline(Tasks.ocr_detection, model='damo/cv_resnet18_ocr-detection-line-level_damo')
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
img_path = 'ocr_detection.jpg'
image_full = cv2.imread(img_path)
det_result = ocr_detection(image_full)
# det_result = ocr_detection('https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/ocr_detection.jpg')
det_result = det_result['polygons']
for i in range(det_result.shape[0]):
    pts = order_point(det_result[i])
    image_crop = crop_image(image_full, pts)
    result = ocr_recognition(image_crop)
    print("box: %s" % ','.join([str(e) for e in list(pts.reshape(-1))]))
    print("text: %s" % result['text'])

# 第四步：实体抽取
url = "https://modelscope.cn/api/v1/models/damo/cv_resnet18_card_correction/repo?Revision=master&FilePath=data/demo1.jpg"
det_result = ocr_detection(url)

# wget "https://modelscope.cn/api/v1/models/damo/cv_resnet18_card_correction/repo?Revision=master&FilePath=data/demo1.jpg" > t12324.jpg
img_path = "t12324.jpg"
o_img = cv2.imread(img_path)
text_all = ''
ocr_recognition = pipeline(Tasks.ocr_recognition, model='damo/cv_convnextTiny_ocr-recognition-general_damo')
for ori_pts in det_result['polygons']:
    pts = order_point(ori_pts)
    image_crop = crop_image(o_img, pts)
    line_result = ocr_recognition(image_crop)['text'][0]
    text_all = text_all+';'+line_result
print(text_all)

#;6217566000000000000;山东省人力资源和社会保障厅;服务电话(人力资源社会保障:12333;1234567890987654;社会保障号码;社会保障卡号;UnionPay;2019年11月;中国银行:95566);EP 1992 0614-HiCo;发卡日期;有效期限;K1234567;中国银行;BANK OF CHINA;读小光;银联;云保;姓名;10年;ATM;;力资,;社;源;龙;中;3

from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
from modelscope import snapshot_download

model_dir = snapshot_download("qwen/Qwen-7B-Chat", revision = 'v1.1.4')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
response, history = model.chat(tokenizer, "你好", history=None)
print(response)
response, history = model.chat(tokenizer, "请告诉我下面这段文字的社会保障卡号，发卡日期，有效期限："+text_all, history=None)
print(response)

model_dir = snapshot_download("qwen/Qwen-7B-Chat", revision = 'v1.1.4')
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
response, history = model.chat(tokenizer, "你好", history=None)
response, history = model.chat(tokenizer, "请告诉我下面这段文字的发票代码，发票号码，发票金额，发票印制地名称："+text_all, history=None)


# 资料来源：https://mp.weixin.qq.com/s/em9ZbYhPEINe-KgU4fYDyw
# https://zhuanlan.zhihu.com/p/670853734


