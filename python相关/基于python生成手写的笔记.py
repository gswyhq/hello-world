#!/usr/bin/env python
# coding=utf-8

import os
from PIL import Image, ImageFont
import docx
from handright import Template, handwrite
from PIL import Image

USERNAME = os.getenv("USERNAME")

def png2jpg(png_img):
    '''将png图片转换为jpg图片'''
    assert png_img.lower().endswith('.png'), "图片后缀错误，应该是：.png"
    jpg_img = png_img[:-4] + '.jpg'
    im = Image.open(png_img)
    im = im.convert("RGB")
    # 保存图片
    im.save(jpg_img)
    return jpg_img

text = ""
file = docx.Document(rf"D:\Users\{USERNAME}\Downloads\测试文档.docx")  ##word路径

for pare in file.paragraphs:
    text = text + '\n' + str(pare.text)

print(text)
font = ImageFont.truetype(rf"D:\Users\{USERNAME}\github_project/Handright\tests/fonts/Bo Le Locust Tree Handwriting Pen Chinese Font-Simplified Chinese Fonts.ttf", size=40)
# font_size = 40
# font.size = font_size
png_img = rf"D:\Users\{USERNAME}\Downloads\fac029de5367482c9855576fa9d5aa53.png"
template = Template(
    background=Image.open(rf"D:\Users\{USERNAME}\Downloads\fac029de5367482c9855576fa9d5aa53.png"),  ## 需要手动拍摄一张图片
    # font=ImageFont.truetype(r".\MILanProVF-Thin.ttf"),  ##字体
    font=font,  ##字体
    line_spacing=50,
    fill=0,
    left_margin=100,
    top_margin=100,
    right_margin=100,
    bottom_margin=100,
    word_spacing=2,
    line_spacing_sigma=0,
    font_size_sigma=2,
    word_spacing_sigma=2,
    end_chars=", :",
    perturb_x_sigma=4,
    perturb_y_sigma=4,
    perturb_theta_sigma=0.05,
)
image = handwrite(text, template)
for i, im in enumerate(image):
    assert isinstance(im, Image.Image)
    im = im.convert("RGB")
    im.save(rf"D:\Users\{USERNAME}\Downloads\1_{i}.jpg")  ## 生成的图片




from PIL import Image, ImageFont

from handright import Template, handwrite

text = "我能吞下玻璃而不伤身体。"
template = Template(
    background=Image.new(mode="1", size=(1024, 2048), color=1),
    font=ImageFont.truetype(rf"D:\Users\{USERNAME}\github_project/Handright\tests/fonts/Bo Le Locust Tree Handwriting Pen Chinese Font-Simplified Chinese Fonts.ttf", size=100),
)
images = handwrite(text, template)
for im in images:
    assert isinstance(im, Image.Image)
    im.show()


def main():
    pass


if __name__ == "__main__":
    main()
