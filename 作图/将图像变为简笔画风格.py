#!/usr/bin/python3
# coding: utf-8

from PIL import Image, ImageFilter, ImageOps

def dodge(a, b, alpha):
    return min(int(a * 255 / (256 - b * alpha)), 255)

def draw(img, blur=25, alpha=1.0):
    # 图片转换成灰色
    img1 = img.convert('L')
    img2 = img1.copy()
    img2 = ImageOps.invert(img2)
    # 模糊度
    for i in range(blur):
        img2 = img2.filter(ImageFilter.BLUR)
    width, height = img1.size
    for x in range(width):
        for y in range(height):
            a = img1.getpixel((x, y))
            b = img2.getpixel((x, y))
            img1.putpixel((x, y), dodge(a, b, alpha))
    img1.show()
    return img1



def main():
    jpg_path = r'/home/gswyhq/Downloads/000.jpg'
    img = Image.open(jpg_path)
    img1 = draw(img)
    img1.save(r'/home/gswyhq/Downloads/000_1.jpg')

if __name__ == '__main__':
    main()