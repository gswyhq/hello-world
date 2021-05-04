#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from snapshot_selenium import snapshot
from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.render import make_snapshot
from selenium import webdriver
options = webdriver.ChromeOptions()
options.add_argument("headless")
driver = webdriver.Chrome(options=options,
                          executable_path=r'D:\Users\abcd\chromedriver_win32\chromedriver.exe')
def bar_chart() -> Bar:
    c = (
        Bar()
        .add_xaxis(["衬衫", "毛衣", "领带", "裤子", "风衣", "高跟鞋", "袜子"])
        .add_yaxis("商家A", [114, 55, 27, 101, 125, 27, 105])
        .add_yaxis("商家B", [57, 134, 137, 129, 145, 60, 49])
        .reversal_axis()
        .set_series_opts(label_opts=opts.LabelOpts(position="right"))
        .set_global_opts(title_opts=opts.TitleOpts(title="Bar-测试渲染图片"))
    )
    return c
# 需要安装 snapshot_selenium; 当然也可以使用： snapshot_phantomjs
make_snapshot(snapshot, bar_chart().render(path="render2.html"), "bar.png", driver=driver)

# Message: 'chromedriver' executable needs to be in PATH.解决办法（综合了网上的一些办法）
# 1.打开chrome 输入 “chrome://version/”来查看chrome版本 
# 
# 2.访问此网站 http://chromedriver.storage.googleapis.com/index.html 然后选择合适版本的driver。点击notes.txt就可查看其对应的版本号
# 
# 3.把chromedriver.exe文件放入chrome安装路径，也就是C:\Program Files (x86)\Google\Chrome\Application（一班都是这个，根据自己情况弄）
# 4.把chromedriver.exe文件放到环境的scripts文件里
# 5.把C:\Program Files (x86)\Google\Chrome\Application路径放到path的环境变量里
