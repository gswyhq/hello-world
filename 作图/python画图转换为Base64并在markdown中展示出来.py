#!/usr/bin/env python
# coding=utf-8

import matplotlib.pyplot as plt
import base64
import io
from PIL import Image
from IPython.display import display, Markdown

from pylab import mpl
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
mpl.rcParams['axes.unicode_minus'] = False #解决保存图像是负号'-'显示为方块的问题

import os
USERNAME = os.getenv("USERNAME")

# 生成杜邦图
def generate_duPont_chart():
    # 示例数据
    roe = 0.15  # 净资产收益率
    profit_margin = 0.10  # 净利润率
    asset_turnover = 1.5  # 总资产周转率
    financial_leverage = 1.2  # 财务杠杆

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制杜邦分析图
    ax.barh(['ROE'], [roe], color='blue', label='ROE')
    ax.barh(['Profit Margin'], [profit_margin], color='green', label='Profit Margin')
    ax.barh(['Asset Turnover'], [asset_turnover], color='orange', label='Asset Turnover')
    ax.barh(['Financial Leverage'], [financial_leverage], color='red', label='Financial Leverage')

    # 添加数值标签
    for i, v in enumerate([roe, profit_margin, asset_turnover, financial_leverage]):
        ax.text(v + 0.01, i, f'{v:.2f}', color='black', va='center')

    # 设置图表标题和标签
    ax.set_title('杜邦分析图')
    ax.set_xlabel('比率')
    ax.set_ylabel('指标')
    ax.legend()

    # 保存图表到文件
    # plt.savefig(rf"D:\Users\{USERNAME}\MaxKB\docs\test3.png")
    # plt.show()
    return fig

# 将图像转换为 Base64 编码的 URL
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    image_bytes = buf.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    plt.close(fig)  # 确保 fig 对象关闭
    return base64_image

# 生成 Markdown 图像标签
def generate_markdown_image(base64_image):
    image_format = 'png'
    image_url = f'data:image/{image_format};base64,{base64_image}'
    markdown_image = f'![杜邦图]({image_url})'
    return markdown_image

# 主函数
def main():
    fig = generate_duPont_chart()
    base64_image = fig_to_base64(fig)
    markdown_image = generate_markdown_image(base64_image)
    print(markdown_image)  # 调试输出
    display(Markdown(markdown_image))

if __name__ == "__main__":
    main()

