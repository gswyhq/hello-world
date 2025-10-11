#!/usr/bin/env python
# coding=utf-8

import json
import cv2
import os
import numpy as np
import svgwrite

USER_DIR = os.path.expanduser('~')


def remove_gray_background_for_chinese_char(image_path, output_path, threshold=150):
    '''去除灰色背景，让汉字笔迹更清晰、对比度更高'''
    # 打开图片并转为 RGBA（确保有透明通道）
    img = Image.open(image_path).convert("RGBA")
    data = img.getdata()
    new_data = []
    for item in data:
        r, g, b, a = item
        gray = (r + g + b) // 3
        if gray > threshold:
            new_data.append((0, 0, 0, 0))  # 完全透明
        else:
            new_data.append((0, 0, 0, 255))  # 黑色，保留汉字笔迹
    img.putdata(new_data)
    img.save(output_path, "PNG")

# OCR 结果转 SVG 的函数
def convert_image_to_svg(input_image_path, output_svg_path, max_size=64):
    # 读取图像并转为灰度图
    img = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"无法读取图像: {input_image_path}")
        return

    # 调整图像大小
    img = cv2.resize(img, (max_size, max_size), interpolation=cv2.INTER_AREA)

    # 二值化处理（字符为白色，背景为黑色）
    _, binary_img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # 边缘检测
    edges = cv2.Canny(binary_img, 50, 150)  # 调整 Canny 参数

    # 寻找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 调试输出：查看是否找到轮廓
    if not contours:
        print(f"未找到轮廓: {input_image_path}")
        return

    # 创建 SVG 文件
    dwg = svgwrite.Drawing(output_svg_path, size=(f"{max_size}px", f"{max_size}px"), viewBox=f"-{max_size//2} 0 {max_size} {max_size}")

    # 将每个轮廓转换为 SVG path
    for contour in contours:
        path_data = []
        for i, point in enumerate(contour):
            x, y = point[0][0], point[0][1]
            if i == 0:
                path_data.append(f"M {x},{y}")
            else:
                path_data.append(f"L {x},{y}")
        if path_data:
            dwg.add(dwg.path(d=" ".join(path_data) + " Z", fill="black"))

    dwg.save()

# 读取 OCR 结果
json_path = f"{USER_DIR}\Downloads\output/2516320889_09_24_52_res.json"
with open(json_path, "r", encoding="utf-8") as f:
    ocr_result = json.load(f)

# 读取原始图像
image_path = ocr_result['input_path']
image = cv2.imread(image_path)

return_word_box = ocr_result['return_word_box']

assert return_word_box, "原始识别结果不是字符级别，不进行转换"

# 创建输出目录
output_dir = f"{USER_DIR}/Downloads/output/output_svg"
os.makedirs(output_dir, exist_ok=True)

# 遍历每个字符
for i in range(len(ocr_result["text_word_boxes"])):
    word_boxes = ocr_result["text_word_boxes"][i]
    words = ocr_result["text_word"][i]

    for j in range(len(word_boxes)):
        if len(words[j]) > 1:
            continue

        # 获取字符的坐标 [x1, y1, x2, y2]
        x1, y1, x2, y2 = word_boxes[j]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 从图像中裁剪字符区域
        char_img = image[y1:y2, x1:x2]

        # 保存为临时 PNG（用于调试）
        temp_png_path = f"{output_dir}/temp_{ord(words[j])}.png"
        cv2.imwrite(temp_png_path, char_img)

        # 生成 SVG 文件
        svg_path = os.path.join(output_dir, f"{ord(words[j])}.svg")
        remove_gray_background_for_chinese_char(temp_png_path, temp_png_path)
        convert_image_to_svg(temp_png_path, svg_path)

        # 删除临时 PNG 文件
        os.remove(temp_png_path)

        print(f"Saved: {words[j]} -> {ord(words[j])}.svg")

def main():
    pass


if __name__ == "__main__":
    main()
