#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

df1 = pd.DataFrame({'uid':[1,2,5], 'd1':[1,2,5]})
df2 = pd.DataFrame({'uid':[1,4,6], 'd2':[1,4,6]})
df3 = pd.DataFrame({'uid2':[2,5,7], 'd3':[2,5,7]})

# df1
# Out[21]:
#    uid  d1
# 0    1   1
# 1    2   2
# 2    5   5
# df2
# Out[22]:
#    uid  d2
# 0    1   1
# 1    4   4
# 2    6   6
# df3
# Out[23]:
#    uid2  d3
# 0    2   2
# 1    5   5
# 2    7   7

# 1.默认连接方式（会使用表之间相同的列名作为外键进行连接，如果有两边对不上的字段则会自动丢弃）
pd.merge(df1, df2)
# Out[24]:
#    uid  d1  d2
# 0    1   1   1

# 2.通过on指定外键
# on指定外键，如果有重名的列名就会自动更改名称，有不对应的数据也会丢弃
pd.merge(df1, df2, on='uid')
# Out[25]:
#    uid  d1  d2
# 0    1   1   1

# 4.指定左边表的外键left_on和右边表的外键right_on(处理两张表的外键不统一的情况)
pd.merge(df1, df3, left_on='uid', right_on='uid2')
# Out[31]:
#    uid  d1  uid2  d3
# 0    2   2     2   2
# 1    5   5     5   5

# pd.merge默认使用的是内连接
# 内连接：相当于取两个数据集的交集，即二者都有的部分
# 外连接：两个数据集的并集，即全部数据

# 3.通过how指定连接方式
# inner表示内链接，outer表示外连接（缺失数据会补充为NaN）
pd.merge(df1, df2, on='uid', how='outer')
# Out[26]:
#    uid   d1   d2
# 0    1  1.0  1.0
# 1    2  2.0  NaN
# 2    5  5.0  NaN
# 3    4  NaN  4.0
# 4    6  NaN  6.0

# left表示左连接，right表示右连接（尽量保证左表的数据完整或者右表的数据完整）
pd.merge(df1, df2, on='uid', how='left')
# Out[27]:
#    uid  d1   d2
# 0    1   1  1.0
# 1    2   2  NaN
# 2    5   5  NaN
pd.merge(df1, df2, on='uid', how='right')
# Out[28]:
#    uid   d1  d2
# 0    1  1.0   1
# 1    4  NaN   4
# 2    6  NaN   6

# 4.指定左边表的外键left_on和右边表的外键right_on(处理两张表的外键不统一的情况)
pd.merge(df1, df3, left_on='uid', right_on='uid2', how='outer')
# Out[32]:
#    uid   d1  uid2   d3
# 0  1.0  1.0   NaN  NaN
# 1  2.0  2.0   2.0  2.0
# 2  5.0  5.0   5.0  5.0
# 3  NaN  NaN   7.0  7.0

# 5.left_index和right_index(使用索引作为外键连接)
# 如下，左边表使用列名作为外键，右边表使用索引作为外键

df4 = pd.DataFrame({'d4': [6,8,9,7], 'key':list('abcd')})
df5 = pd.DataFrame({'d5':[4,2,1,6]}, index=list('acab'))
# df4
# Out[35]:
#    d4 key
# 0   6   a
# 1   8   b
# 2   9   c
# 3   7   d
# df5
# Out[36]:
#    d5
# a   4
# c   2
# a   1
# b   6

pd.merge(df4, df5, left_on='key', right_index=True)
# Out[37]:
#    d4 key  d5
# 0   6   a   4
# 0   6   a   1
# 1   8   b   6
# 2   9   c   2
pd.merge(df4, df5, left_on='key', right_index=True, how='outer')
# Out[38]:
#    d4 key   d5
# 0   6   a  4.0
# 0   6   a  1.0
# 1   8   b  6.0
# 2   9   c  2.0
# 3   7   d  NaN

# 如果合并之后两边表有重名的列，则自动会在列名后加上_x或者_y
# 要想自定义后缀，则需要加上suffixes参数，例如suffixes=["_left","_right"]

# 来源：https://blog.csdn.net/print_and_return/article/details/80577561


pandas dataframe的合并（append, merge, concat）

df1 = pd.DataFrame(np.ones((4, 4))*1, columns=list('DCBA'), index=list('4321'))
df2 = pd.DataFrame(np.ones((4, 4))*2, columns=list('FEDC'), index=list('6543'))
df3 = pd.DataFrame(np.ones((4, 4))*3, columns=list('FEBA'), index=list('6521'))

pd.concat(objs, axis=0, join='outer', join_axes=None, ignore_index=False,
          keys=None, levels=None, names=None, verify_integrity=False,
	  copy=True)

参数axis
默认值：axis=0
axis=0：竖方向（index）合并，如4行4列+4行4列=8行，合并方向index作列表相加，非合并方向columns取并集
axis=1：横方向（columns）合并，如4行4列+4行4列=8列，合并方向columns作列表相加，非合并方向index取并集

join
默认值：join=‘outer’
非合并方向的行/列名称：取交集（inner），取并集（outer）。
axis=0时join='inner'，columns取交集
axis=1时join='inner'，index取交集

join_axes
默认值：join_axes=None，取并集
合并后，可以设置非合并方向的行/列名称，使用某个df的行/列名称
axis=0时join_axes=[df1.columns]，合并后columns使用df1的
axis=1时axes=[df1.index]，合并后index使用df2的：
同时设置join和join_axes的，以join_axes为准

ignore_index
默认值：ignore_index=False
合并方向是否忽略原行/列名称，而采用系统默认的索引，即从0开始的int。
axis=0时ignore_index=True，index采用系统默认索引
axis=1时ignore_index=True，columns采用系统默认索引

keys
默认值：keys=None
可以加一层标签，标识行/列名称属于原来哪个df。

concat多个DataFrame
>>> pd.concat([df1, df2, df3], sort=False, join_axes=[df1.columns])

# append
append(self, other, ignore_index=False, verify_integrity=False)
竖方向合并df，没有axis属性
不会就地修改，而是会创建副本
示例：
>>> df1.append(df2)    # 相当于pd.concat([df1, df2])

# append多个DataFrame
和concat相同，append也支持append多个DataFrame
>>> df1.append([df2, df3], ignore_index=True)

def main():
    pass


if __name__ == '__main__':
    main()
