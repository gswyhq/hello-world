#!/usr/bin/env python
# -*- coding: utf-8 -*-”

import os
import time
from datetime import datetime
from xlwt import Workbook, XFStyle, Pattern, Borders

filename = 'TestData2.xls'                                      #检测当前目录下是否有TestData2.xls文件，如果有则清除以前保存文件
if os.path.exists(filename):
    os.remove(filename)

print (time.strftime("%Y-%m-%d",time.localtime(time.time()))) #打印读取到当前系统时间

wbk = Workbook(encoding='utf-8')
sheet = wbk.add_sheet('new sheet 1', cell_overwrite_ok=True)                 #第二参数用于确认同一个cell单元是否可以重设值。
style = XFStyle()                       #赋值style为XFStyle()，初始化样式

Line_data = ('测试表')              #创建一个Line_data列表，并将其值赋为测试表

for i in range(0x00,0xff):              # 设置单元格背景颜色
    pattern = Pattern()                 # 创建一个模式
    pattern.pattern = Pattern.SOLID_PATTERN     # 设置其模式为实型
    pattern.pattern_fore_colour = i
    # 设置单元格背景颜色 0 = Black, 1 = White, 2 = Red, 3 = Green, 4 = Blue, 5 = Yellow, 6 = Magenta,  the list goes on...
    style.pattern = pattern             # 将赋值好的模式参数导入Style
    sheet.write_merge(i, i, 0, 2, Line_data, style) #以合并单元格形式写入数据，即将数据写入以第1/2/3列合并德单元格内

for i in range(0x00,0xff):              # 设置单元格内字体样式
    fnt = Font()                        # 创建一个文本格式，包括字体、字号和颜色样式特性
    fnt.name = '微软雅黑'                # 设置其字体为微软雅黑, 'SimSun'    # 指定“宋体”
    fnt.colour_index = i                # 设置其字体颜色
    fnt.bold = True
    style.font = fnt                    #将赋值好的模式参数导入Style
    sheet.write_merge(i,i,3,5,Line_data,style)  #以合并单元格形式写入数据，即将数据写入以第4/5/6列合并德单元格内

for i in range(0, 0x53):                # 设置单元格下框线样式
    borders = Borders()
    borders.left = i
    borders.right = i
    borders.top = i
    borders.bottom = i
    style.borders = borders         #将赋值好的模式参数导入Style
    sheet.write_merge(i,i,6,8,Line_data,style)  #以合并单元格形式写入数据，即将数据写入以第4/5/6列合并德单元格内

for i in range(6, 80):                  # 设置单元格下列宽样式
    sheet.write(0,i,Line_data,style)
    sheet.col(i).width = 0x0d00 + i*50

path_py = "/home/gswewf/jian1.bmp"         #读取插入图片以.py运行时路径，images和.py在同一目录下

sheet.insert_bitmap(path_py, 2, 9)         #插入一个图片

wbk.save('TestData2.xls')               #保存TestData2.xls文件，保存到脚本或exe文件运行的目录下

import xlwt
def set_font(colour_index=2):
    """
    设置单元格字体颜色，0：黑色，1：无色，2：红色，3：绿色， 4：蓝色， 5： 黄色， 6，粉红色...
    :param colour_index: 颜色代号, 可以是整数，也可以是： 0x00～0xff
    :return:
    eg: ws.write(3, 0, '设置字体颜色', style0)
    """
    font0 = xlwt.Font()
    font0.name = 'Times New Roman'
    font0.colour_index = colour_index  # 设置字体颜色为红色
    font0.bold = True

    style0 = xlwt.XFStyle()
    style0.font = font0
    return style0

style1 = xlwt.XFStyle()
style1.num_format_str = 'D-MMM-YY'

wb = xlwt.Workbook()
ws = wb.add_sheet('测试工作表')


[ws.write(0, i, '字体颜色{}'.format(i), set_font(i)) for i in range(10)]

ws.write(1, 0, datetime.now(), style1)
ws.write(2, 0, 1)
ws.write(2, 1, 1)
ws.write(2, 2, xlwt.Formula("A3+B3"))

font0 = xlwt.Font()
font0.name = 'SimSun'
font0.colour_index = 0xff  # 设置字体颜色为红色
font0.bold = True

style0 = xlwt.XFStyle()
style0.font = font0
ws.write(3, 0, '设置字体颜色', style0)

wb.save('example.xls')