#------encoding:utf-8----------------------------------
# Name: PIL简单使用实例
# 官网:http://effbot.org/imagingbook/
#
# Author:      lenove
#
# Created:     23-11-2015
# Copyright:   (c) lenove 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
from PIL import Image,ImageDraw,ImageFont
font = ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc', size=45,encoding='utf-8')

def main():
    img=Image.new('RGB',(200,200),(255,255,255))#200*200的白色背景
    draw=ImageDraw.Draw(img)
    draw.line((20,50,150,80),fill=(255,0,0))#红色线条
    draw.line((150,150,20,200),fill=(0,255,0))#绿色线条
    draw.text((40,80),'您好！',(0,0,0),font=font)#黑色文本
    #img.save(r'f:\python\data\test.jpg','JPEG')
    img.show()


if __name__ == '__main__':
    main()
