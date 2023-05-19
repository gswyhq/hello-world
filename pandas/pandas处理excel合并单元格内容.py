#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import pandas as pd
from glob import glob
from tqdm import tqdm
import re
import json
import functools
USERNAME = os.getenv("USERNAME")

# 数据来源国家民政局（https://www.mca.gov.cn/article/sj/xzqh/1980/）
mz_dfs = pd.read_excel(rf"D:\Users\{USERNAME}\data\数据梳理\经纬度地理位置\民政部行政区划代码\民政部区划代码_20230428.xlsx", sheet_name=None, dtype=str)

df = mz_dfs['2022县下']
#将数据读取到DataFrame后，在有合并单元格的列中，将会有值为NaN的单元格出现

# 对指定列合并单元格进行填充(将合并单元格值重复填充到每个单元格)
df['省份'] = df['省份'].fillna(method='ffill')

# 也可以对多个列合并单元格一起填充；
df = df[['class_id', 'class_name', 'remark']].fillna(method='ffill')

# 或者整个表的所有合并单元格进行填充
df = df.fillna(method='ffill')

# 对数据进行分组, 将合并单元格列的其他列数据合并起来；
# 将各个省份(合并单元格列)的‘现名称’，按逗号分隔合并成一个单元格；
df = df.fillna(method='ffill').fillna('')
df_2 = df.groupby('省份').agg({"现名称": ', '.join})

# pandas  删除空行，或空列
# axis=0 代表去除带有一个或多个空值的行，对应的 axis=1 代表去除带有一个或多个空值的列，本例中我们需要去除带有空值的行
# 若删除空行，并删除列student_list为空值的数据
df = df.dropna(axis=0).drop('student_list', axis=1)

def main():
    pass


if __name__ == '__main__':
    main()
