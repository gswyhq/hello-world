#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# http://www.blog.pythonlibrary.org/2019/04/08/reportlab-adding-a-chart-to-a-pdf-with-python/

from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.validators import Auto
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib import colors

# reportlab, 中文乱码解决方法
# 1.下载中文字体[SimSun.ttf](https://github.com/StellarCN/scp_zh/blob/master/fonts/SimSun.ttf)
# 2.把下载下来的字体放到/Library/Python/2.7/site-packages/reportlab/fonts文件夹下。（文件夹根据自己安装的reportlab的路径来; python3,reportlab(3.4.0)无该路径，使用的绝对路径解决）
# 3.注册字体并使用

from reportlab.platypus import SimpleDocTemplate, Image, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# pdfmetrics.registerFont(TTFont('SimSun', 'SimSun.ttf'))  #注册字体
pdfmetrics.registerFont(TTFont('SimSun', '/home/gswyhq/github_projects/scp_zh/fonts/SimSun.ttf'))  #注册字体

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(fontName='SimSun', name='Song', leading=20, fontSize=12))  #自己增加新注册的字体

# Paragraph(describe, styles['Song']),  #使用新字体


def add_legend(draw_obj, chart, data):
    legend = Legend()
    legend.fontName = 'SimSun'
    legend.alignment = 'right'
    legend.x = 10
    legend.y = 70
    legend.colorNamePairs = Auto(obj=chart)
    draw_obj.add(legend)


def pie_chart_with_legend():
    data = list(range(15, 105, 15))
    drawing = Drawing(width=400, height=200)
    my_title = String(170, 40, '饼状图', fontSize=14, fontName='SimSun')
    pie = Pie()
    pie.sideLabels = True
    pie.x = 150
    pie.y = 65
    pie.data = data
    pie.labels = [letter for letter in '赤橙黄绿青蓝紫']
    pie.slices.strokeWidth = 0.5
    pie.slices.fontName = 'SimSun'
    pie.slices[0].fillColor = colors.pink
    pie.slices[1].fillColor = colors.magenta
    pie.slices[2].fillColor = colors.yellow
    pie.slices[3].fillColor = colors.cyan
    pie.slices[4].fillColor = colors.blue
    pie.slices[5].fillColor = colors.blueviolet
    drawing.add(my_title)
    drawing.add(pie)
    add_legend(drawing, pie, data)
    return drawing



def main():
    doc = SimpleDocTemplate('flowable_with_chart.pdf')

    elements = []
    # styles = getSampleStyleSheet()

    ptext = Paragraph('图表之前的文本', styles["Song"])
    elements.append(ptext)

    chart = pie_chart_with_legend()
    elements.append(chart)

    ptext = Paragraph('图表之后的文本', styles["Song"])
    elements.append(ptext)

    doc.build(elements)


if __name__ == '__main__':
    main()



