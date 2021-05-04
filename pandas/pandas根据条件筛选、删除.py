#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

df = pd.DataFrame({'A':[1,2,3,None], 'B': [None, 2,3,4], 'C': [19, None, 2, 5]})
df
Out[14]: 
    A    B     C
0  1.0  NaN  19.0
1  2.0  2.0   NaN
2  3.0  3.0   2.0
3  NaN  4.0   5.0

# 删除指定列的空值；
df.dropna(axis=0, subset=['B'])
Out[15]:
   A    B    C
1  2.0  2.0  NaN
2  3.0  3.0  2.0
3  NaN  4.0  5.0

# 删除指定行的空值
df.dropna(axis=1, subset=[1])
Out[16]:
    A    B
0  1.0  NaN
1  2.0  2.0
2  3.0  3.0
3  NaN  4.0

df = pd.read_excel(r'new_labels_20200826.xlsx')
df3 = df.head(15)
df3['labels'].values
# Out[197]:
# array(['餐饮美食', '其他', '餐饮美食', '其他额度 其他手续费 其他还款 提高额度 费率 客服评价 理财 其他评价 分期还款',
#        '其他', '餐饮美食', '餐饮美食', '购物活动', '购物活动', '餐饮美食', '额度高', '其他积分 积分兑换',
#        '其他', '健康权益', '其他'], dtype=object)
df3.shape
# Out[198]: (15, 17)

# 按行的名称，或列的名称取值：
df = pd.DataFrame([[1,2], [3,4], [5, 6]],columns=list('AB'), index=list('abc'))
df
Out[25]:
   A  B
a  1  2
b  3  4
c  5  6
df['A'].loc[['a', 'c']]
Out[26]:
a    1
c    5
Name: A, dtype: int64
df.loc[['a', 'c']]['A']
Out[27]:
a    1
c    5
Name: A, dtype: int64

# 目的：总共有15行，在15行标签列中，分离出多标签行和单标签行

df4 = df3.drop(df3[df3['labels'].map(lambda x: len(x.split())) > 1].index)
df4.shape
# Out[200]: (13, 17)
df5 = df3[df3['labels'].map(lambda x: len(x.split())) > 1]
df5.shape
# Out[202]: (2, 17)

df5['labels'].values
# Out[204]:
# array(['其他额度 其他手续费 其他还款 提高额度 费率 客服评价 理财 其他评价 分期还款', '其他积分 积分兑换'],
#       dtype=object)
df4['labels'].values
# Out[205]:
# array(['餐饮美食', '其他', '餐饮美食', '其他', '餐饮美食', '餐饮美食', '购物活动', '购物活动', '餐饮美食',
#        '额度高', '其他', '健康权益', '其他'], dtype=object)


# 获取字符串长度小于2的行：
df[df['column name'].map(len) < 2]

# 筛选‘专业名称’是物流管理的行
df[df['专业名称']=='物流管理'].values

# 要删除列“score”<50的所有行：
df = df.drop(df[df.score < 50].index)
# 替换版本
df.drop(df[df.score < 50].index, inplace=True)

# 多条件情况：
# 可以使用操作符： | 只需其中一个成立, & 同时成立, ~ 表示取反，它们要用括号括起来。
# 例如删除列“score<50 和>20的所有行
df = df.drop(df[(df.score < 50) & (df.score > 20)].index)

# 对某列进行筛选，判断是否在某个列表中：
df52 = df5[df5['一级分类'].isin(['农学', '医学', '历史学', '工学', '文学', '理学', '管理学', '经济学', '艺术学'])]

# 多字段限定，非空值：
df52 = df5[df5['一级分类'].isin(['农学', '医学', '历史学', '工学', '文学', '理学', '管理学', '经济学', '艺术学']) & df5['专业代码'].notna()]

# 随机取n行，取后顺序会打乱；
df.shape
Out[13]: (152, 11)
df = df.sample(50)
df.shape
Out[15]: (50, 11)

def main():
    pass


if __name__ == '__main__':
    main()
