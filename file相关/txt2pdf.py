#!/usr/bin/env python
# coding=utf-8

import sys, os
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics, ttfonts
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, PageTemplate, Frame, Paragraph, BaseDocTemplate
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4, landscape, portrait
from reportlab.lib.units import cm, mm

'''
将txt文件转换为PDF文件
pip3 install reportlab==4.2.0
'''

def main():
    assert len(sys.argv) == 2, "输入参数有误，示例：python3 txt2pdf.py test.txt"
    input_file = sys.argv[1]
    assert input_file.lower().endswith('.txt'), "参数应该为txt文件"

    header_content = os.path.splitext(os.path.basename(input_file))[0]

    save_file = input_file[:-4]+'.pdf'
    with open(input_file, "r", encoding='utf-8') as file:
        contents = file.read().split('\n')

    # # 注册中文字体
    pdfmetrics.registerFont(ttfonts.TTFont('KaiTi', r'C:\Windows\Fonts\simkai.ttf'))
    # 或者下载中文字体SimSun.ttf/SimSun-Bold.ttf 2.把下载下来的字体放到解释器目录site-packages/reportlab/fonts文件夹
    # fc-list :lang=zh 查询有哪些中文字体

    Style = getSampleStyleSheet()
    # 字体的样式，大小
    bt = Style['Normal']
    bt.fontSize = 12
    bt.fontName = 'KaiTi'
    # 设置分词,自动换行
    bt.wordWrap = 'CJK'

    # 居左对齐
    bt.alignment = 0
    # 居中
    # bt.alignment = 1

    # 设置行距和缩进
    bt.firstLineIndent = 32  # 第一行开头空格
    bt.leading = 20 # 行距

    # 这是位置和颜色
    bt.textColor = colors.black

    doc = BaseDocTemplate(save_file, pagesize=A4, leftMargin=48, rightMargin=48, topMargin=72, bottomMargin=36)

    styleC = getSampleStyleSheet()["Normal"]
    styleC.fontName = 'KaiTi'
    styleC.fontSize=8

    # 页脚
    def footer(canvas, doc):
        canvas.saveState()
        page_num = canvas.getPageNumber()
        content = Paragraph(f"第{page_num}页", styleC)
        w, h = content.wrap(doc.width, doc.bottomMargin)
        content.drawOn(canvas, doc.leftMargin, h)
        canvas.restoreState()

    # 页眉
    def header(canvas, doc):
        canvas.saveState()
        p = Paragraph(header_content, styleC)
        w, h = p.wrap(doc.width, doc.topMargin)
        p.drawOn(canvas, (w+doc.leftMargin)/2, doc.height + doc.topMargin)
        canvas.line(doc.leftMargin, doc.bottomMargin + doc.height + 1 * cm, doc.leftMargin + doc.width,
                                    doc.bottomMargin + doc.height + 1 * cm)  # 画一条横线
        canvas.restoreState()

    frame_footer = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id='normal')  # 声明一块Frame，存放页码
    template = PageTemplate(id='test', frames=frame_footer, onPage=header,
                            onPageEnd=footer)  # 设置页面模板，在加载页面时先运行herder函数，在加载完页面后运行footer函数
    doc.addPageTemplates([template])
    doc.build([Paragraph(content, bt) for content in contents])

if __name__ == "__main__":
    main()
