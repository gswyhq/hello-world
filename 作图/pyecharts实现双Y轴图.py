#!/usr/bin/python3
# coding: utf-8

# pyecharts 实现双Y轴图

from pyecharts import options as opts
from pyecharts.charts import Bar, Line
from pyecharts.globals import ThemeType
from pyecharts.faker import Faker

def overlap_bar_line(v1, v2, v3, v4, v5):
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))  # 这里可以选择主题
            .add_xaxis(v1)
            .add_yaxis("总用户量", v2)
            .add_yaxis("完成输入用户量", v3)
            .extend_axis(
            yaxis=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}%"), interval=5
            )
        )
            .set_series_opts(label_opts=opts.LabelOpts(is_show=False))
            .set_global_opts(
            title_opts=opts.TitleOpts(title="周数据模拟"),
            yaxis_opts=opts.AxisOpts(
                axislabel_opts=opts.LabelOpts(formatter="{value}/人")
            ),
        )
    )
    line = Line()
    line.add_xaxis(v4).add_yaxis("覆盖率", v5, yaxis_index=1, )
    bar.overlap(line)
    bar.render('/home/gswyhq/financial_qa/123.html')

def bar_line():
    """矩形图+折线图"""
    v1 = ["10月30日", "10月31日", "11月01日", "11月02日", "11月03日", "11月04日", "11月05日"]  # x轴坐标
    v2 = [46, 39, 40, 34, 48, 54, 57]  # 总用户量
    v3 = [9, 6, 6, 9, 9, 5, 10]  # 完成输入用户量
    v4 = [i for i in range(0, 101)]  # y轴
    v5 = [19.57, 15.38, 15, 26, 18, 9.2, 17.5, 17.42]
    # y对应的数据
    overlap_bar_line(v1, v2, v3, v4, v5)

def double_line():
    # 双折线图；双y轴
    v1 = ["10月30日", "10月31日", "11月01日", "11月02日", "11月03日", "11月04日", "11月05日"]  # x轴坐标
    v2 = [946, 939, 940, 934, 948, 854, 1057]  # 总用户量
    v3 = [9, 6, 6, 9, 9, 5, 10]  # 完成输入用户量
    v4 = [i for i in range(0, 101)]  # y轴
    v5 = [19.57, 15.38, 15, 26, 18, 9.2, 17.5, 17.42]
    line1 = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.WESTEROS))
            .add_xaxis(v1)
            .add_yaxis('总用户量', v2, color="#749f83")
            .set_global_opts(title_opts=opts.TitleOpts(title='周数据模拟', subtitle=''))
            .set_series_opts(itemstyle_opts={ 'normal' : { 'color':'green', 'lineStyle':{ 'color':'red' } } })
    )
    line = Line()
    line.add_xaxis(v1).add_yaxis("覆盖率", v5, yaxis_index=1, color=Faker.rand_color())
    line.set_series_opts(itemstyle_opts={ 'normal' : { 'color':'#54ffff', 'lineStyle':{ 'color':'#65874f' } } })
    line1.extend_axis(yaxis=opts.AxisOpts())
    line1.overlap(line)
    line1.render('/home/gswyhq/financial_qa/123.html')

def main():
    # bar_line()
    double_line()


if __name__ == '__main__':
    main()

# 参考资料： http://pyecharts.org/#/zh-cn/rectangular_charts?id=overlap%EF%BC%9A%E5%B1%82%E5%8F%A0%E5%A4%9A%E5%9B%BE

