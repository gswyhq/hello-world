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

# 删除指定列空值异常情况：
有时候采用了空字符串填充nan（df = df.fillna("")）,这个时候采用df.dropna删除空值时候，是无效的；
可以通过下面的方法过滤得到非空值，即达到删除空值的目的，如：
df[df['客户号'] !='']

# 填充空值(对所有的NaN填充为空字符串)：
df = df.fillna("")

# 对指定列空值null进行填充：
dataframe.fillna({'code':'code', 'date':'date'})，第一个code和date分别表示列，后面的表示在该列填充的内容

# 筛选某行或某列有空值，或所有值均为空值
df = pd.DataFrame({'col1':[1, True], 'col2':[False, '是'], 'col3': [32, 0]})
# 筛选哪些列所有值不为空；
df.all()
Out[12]: 
col1     True
col2    False
col3    False
dtype: bool
# 筛选哪些列有一个值不为空；
df.any()
Out[13]: 
col1    True
col2    True
col3    True
dtype: bool
# 在指定列中筛选哪些行，所有值不为空
df[['col1', 'col3']].all(axis=1)
Out[11]: 
0     True
1    False
dtype: bool

# 在指定列中筛选哪些行，所有值不为空：
df[df[['col1', 'col3']].all(axis=1)]
Out[32]: 
  col1   col2  col3
0    1  False    32
# 在指定列中筛选哪些行，至少有一个值为空
df[~df[['col1', 'col3']].all(axis=1)]
Out[33]: 
   col1 col2  col3
1  True    是     0

# 筛选哪些行，至少有一个值不为空：
df.any(axis=1)
Out[14]: 
0    True
1    True
dtype: bool

# 删除指定列均为空值，或者删除指定列任有一个空值：
df = pd.DataFrame({"name": ['A','B','C',np.nan],
                   "age": [np.nan,22,25,np.nan],
                   "gender": ['male','female','male','female'],
                  }) 

# 删除name、age列中,【任意一列】的值为空的行；
df1 = df.dropna(subset=['name', 'age'],
          axis=0, # axis=0表示删除行；
          how='any', # how=any表示若列name、age中，任意一个出现空值，就删掉该行
          inplace=False # inplace=True表示在原df上进行修改；
          )
df1
Out[11]: 
  name   age  gender
1    B  22.0  female
2    C  25.0    male

# 删除name、age列中,二者都为空的行。
# 删除都为空的行，还是删除任意一列值为空的行，使用参数how来控制
df2 = df.dropna(subset=['name', 'age'],
          axis=0,
          how='all', # how='all'表示若指定列的值都为空，就删掉该行
          inplace=False)
df2
Out[12]: 
  name   age  gender
0    A   NaN    male
1    B  22.0  female
2    C  25.0    male

# 删除df中任意字段等于'null'字符串的行：
df=df.astype(str)#把df所有元素转为str类型
df=df[df['A'].isin(['null','NULL'])] #找出df的'A'列值为'null'或'NULL'(注意此处的null是字符串，不是空值)
df=df[~df['A'].isin(['null','NULL'])] #过滤掉A列为'null'或'NULL'的行，~表示取反

# 删除多列均为 null 字符串的行：
from functools import reduce
df = pd.DataFrame({"name": ['A','B','C',np.nan, 'null'],
                   "age": [np.nan,22,25,np.nan, 'null'],
                   "gender": ['male','female','male','female', 0],
                  }) 
df.drop(index=df[reduce(lambda x, y: x&y, [(df[c]=='null') for c in ['name', 'age']])].index)
Out[29]: 
  name  age  gender
0    A  NaN    male
1    B   22  female
2    C   25    male
3  NaN  NaN  female

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

# 获取指定列（这里是word列）字符串长度大于2的行：
df[df.apply(lambda x : len(x['word'])>2,axis=1)]

# 获取字符串长度小于2的行：
df[df['column name'].map(len) < 2]

# 筛选‘专业名称’是物流管理的行
df[df['专业名称']=='物流管理'].values

# 不等于筛选：筛选深圳市，非宝安区的数据：
df[(df['city']=='深圳市') & ~(df['area']=='宝安区')]

# 要删除列“score”<50的所有行：
df = df.drop(df[df.score < 50].index)
# 替换版本
df.drop(df[df.score < 50].index, inplace=True)

# 删除满足条件行列数据
df = df.drop(df.index[df['A'] == 2]) ## 删除A列数值为2的行
df = df.drop(df.index[(df['A'] == 2)|(df['A'] == 3)])  ## 删除A列数值为2或者3的数据所在行
df = df.drop(df.index[(df['A'] >= 2)&(df['A'] < 3)]) ## 删除A列数值大于等于且3的数据所在行## 删除列数据
df = df.drop(df.columns[df.loc['a'] == 3],axis = 1) ## 删除a行数据为3的列
df = df.drop(df.columns[(df.loc['a'] == 3) | (df.loc['a'] == 2)],axis = 1) 
df = df.drop(df.columns[(df.loc['a'] < 3) & (df.loc['a'] >= 2)],axis = 1)

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

# 全部数据打乱，重新洗牌：
方法1：采用pandas中自带的 sample这个方法。
df = df.sample(frac=1) #打乱所有数据
设置frac=0.5表示随机抽取50%的数据
这样对可以对df进行shuffle。其中参数frac是要返回的比例，比如df中有10行数据，我只想返回其中的30%,那么frac=0.3。
有时候，我们可能需要打混后数据集的index（索引）还是按照正常的排序。我们只需要这样操作
df.sample(frac=1).reset_index(drop=True)
方法2：
from sklearn.utils import shuffle
df = shuffle(df)

# 取出多列并去重，根据多列内容去重, 删除某列重复值：
df.drop_duplicates(subset=['province', 'city'],keep='first')
DataFrame中存在重复的行或者几行中某几列的值重复，这时候需要去掉重复行，
示例如下：
data.drop_duplicates(subset=['A','B'],keep='first',inplace=True)
参数含义:
代码中subset对应的值是列名，表示只考虑这两列，将这两列对应值相同的行进行去重。
默认值为subset=None表示考虑所有列。
keep='first'表示保留第一次出现的重复行，是默认值。keep另外两个取值为"last"和False，
分别表示保留最后一次出现的重复行和去除所有重复行。 

# drop_duplicates() 删除重复行
dfLancome = dfLancome.drop_duplicates()

# 筛选出重复行：
df3=pd.DataFrame([[1,2], [2,3], [2,4]], columns=['a', 'b'])
df3
Out[21]: 
   a  b
0  1  2
1  2  3
2  2  4

df3.duplicated(subset=['a'])
Out[22]: 
0    False
1    False
2     True
dtype: bool

df3.duplicated(subset=['a'], keep=False)
Out[26]: 
0    False
1     True
2     True
dtype: bool
df3[df3.duplicated(subset=['a'], keep=False)==True]
Out[27]: 
   a  b
1  2  3
2  2  4

# 删除列，删除指定列，删除字段：
方法一：直接del df['column-name']
方法二：df.drop
通过pandas删除列：
1.del df['columns'] #改变原始数据
2.df.drop('columns',axis=1)#删除不改表原始数据，可以通过重新赋值的方式赋值该数据
3.df.drop('columns',axis=1,inplace='True') #改变原始数据

# 从一个数据框中删除与另一个数据框中，某些重复的行；
或者，从一个数据框中，找出哪些行的指定列与另一个数据框的数据重复了：
dfb = pd.DataFrame([[random.randint(1, 5), t] for t in 'xyz'], columns=['a', 'b'])
dfa
Out[44]: 
   a  b
0  5  a
1  1  b
2  5  c
3  2  d
dfb
Out[45]: 
   a  b
0  1  x
1  4  y
2  2  z
# 找出 dfa中，指定列a，与dfb重复的行；
dfa = dfa.drop_duplicates(subset=['a']) # 先要保证 dfa 自己指定列 a没有重复的；
dfa[pd.concat([dfa, dfb.drop_duplicates(subset=['a'])], axis=0).duplicated(subset=['a'], keep='last')[:dfa.shape[0]]==True]
Out[58]: 
   a  b
0  5  a
1  1  b
3  2  d
# 删除 dfa中，指定列a，与dfb重复的行：
dfa[~pd.concat([dfa, dfb.drop_duplicates(subset=['a'])], axis=0).duplicated(subset=['a'], keep='last')[:dfa.shape[0]]==True]
Out[61]: 
   a  b
2  5  c
说明：
先将dfa,与按指定列去重后的dfb拼接起来，再通过duplicated行数，获取凭借后重复行的行标，再删除或获取对应的行数据；
但获取的数据，也有可能是dfa自身指定列重复的数据；
若要排除自身的重复数据可以这样：
unique_a = dfa.drop_duplicates(subset=['a'])
unique_b = dfb.drop_duplicates(subset=['a'])
df_ab = pd.concat([unique_a, unique_b], axis=0).duplicated(subset=['a'], keep='last')[:unique_a.shape[0]]
获取重复数据的索引，再根据索引在数据框dfa中筛选指定数据：
dfa.loc[df_ab[df_ab==True].index]
Out[100]: 
   a  b
1  1  b
3  2  d
根据索引，在数据框中删除指定索引数据：
dfa.drop(df_ab[df_ab==True].index)
Out[112]: 
   a  b
0  5  a
2  5  c

# 随机采样，Pandas随机删除5%的数据
# 可以使用sample()函数来进行随机选取并删除指定比例的数据。
# 计算要保留的样本数量, 计算保留多少行
keep_count = int(len(df)-int(0.05*len(df)) )
# 随机选取并删除5%的数据
random_indexes = df.sample(n=keep_count).index
new_df = df[df.index.isin(random_indexes)]

# 使用skiprows参数，随机读取1%的数据(按行号读取，每100行读取一次)：
df = pd.read_csv(file_name, skiprows=lambda x: x % 100 != 0)

# 根据索引取行，筛选指定索引行号的数据，如：选取选取索引号为1、3、4的行, 按行号筛选；逗号后的指的列索引，冒号:，代表所有列；
df2.loc[[1,3, 4],:]

# 自定义函数，在某列中筛选符合要求的行：
dft=pd.DataFrame([['5,9'], ['2,3'], ['4,6']], columns=['status']) #若status列字符串分割后在集合{'1', '2', '4'}中，则取出
dft[dft['status'].map(lambda x: True if any(k in {'1', '2', '4'} for k in x.split(',')) else False)]
Out[41]: 
  status
1    2,3
2    4,6

# 字符串匹配筛选：
开头包含某值的模式匹配
cond=df['列名'].str.startswith('值')
中间包含某值的模式匹配
cond=df['列名'].str.contains('值')

# startswith 筛选需注意
merge_df[merge_df['流水号'].str.startswith('TS') & merge_df['投诉类型']=='投诉']
并不等于
merge_df[merge_df['流水号'].str.startswith('TS')] 与 merge_df[merge_df['投诉类型']=='投诉'] 的交集。
需要改成如下形式（即在每个 & 连接部分都添加一个小括号）：
merge_df[(merge_df['流水号'].str.startswith('TS')) & (merge_df['投诉类型']=='投诉')].head()

# 两个数据框DataFrame相互匹配：
一个数据框df1包含用户姓名打卡时间，需要在另一个数据框df2查找每个人上次打卡数据(即人相同，但打卡时间小于df1里打卡时间且取最大值)：
df2 = pd.DataFrame([['A', '2020-11-30 00:00:00'],
       ['B', '2020-12-31 00:00:00'],
       ['C', '2021-01-04 00:00:00']], columns=['uid', 'datetime'])
df1 = pd.DataFrame([['A', '2020-10-12 12:00:00'],
       ['A', '2020-10-13 00:21:00'],
       ['A', '2021-11-02 00:00:03'],
       ['B', '2020-11-06 00:12:00'],
       ['B', '2020-11-06 12:00:00'],
       ['B', '2020-11-06 02:00:00'],
       ['C', '2020-12-18 03:00:00'],
       ['C', '2020-12-01 00:05:00'],
       ['C', '2020-12-04 00:04:00'],
       ['C', '2020-12-04 12:00:55']], columns=['uid', 'datetime'])
df1['datetime'] = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in df1['datetime'].values]
df2['datetime'] = [datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S') for date_str in df2['datetime'].values]
先自定义函数，求解用户相等，日期小于指定日期；再对日期排序获取最大日期：
df1[np.where(df1[['uid', 'datetime']].apply(lambda x: any(x['uid']==uid and x['datetime']<date for uid, date in df2[['uid', 'datetime']].values), axis=1), True, False)].sort_values(by='datetime').groupby('uid').last() 
Out[64]: 
               datetime
uid                    
A   2020-10-13 00:21:00
B   2020-11-06 12:00:00
C   2020-12-18 03:00:00

# 输出某列唯一值：
df['col'].unique()

# 输出多列唯一值：
# 不同列数据类型一致：
np.unique(df[['阈值1', '阈值2']], axis=0)

# 若不同列数据类型不一致时，输出多列唯一值：
np.unique(df[['类型', '阈值1']].values.astype("<U22"), axis=0)

# 两个字段，多列 字符串比较筛选
df[df['落单日期'].apply(lambda x: x.replace("-", "")[:6]) == df['落单日期'].apply(lambda x: x.replace("-", "")[:6])
]

# 自定义函数筛选：
df[
df['names'].apply(lambda x: len(x)>1) &
df['cars'].apply(lambda x: "i" in x) &
df['age'].apply(lambda x: int(x)<2)
]

# 使用自定义函数对指定列进行处理：
df['ExtraScore'] = df['Nationality'].apply(lambda x : 5 if x != '汉' else 0)

# 按某些列进行分组，对组内的数据，自定义函数进行处理：
df3=data_5316429_tag_df2.groupby(['流水号', '标注间隔天数']).apply(lambda x: " [SEP] ".join(x['tag_words'].values)).reset_index()
将相同 流水号+标注间隔天数 的 tag_words列 通过 [SEP] 符号拼接起来；
为了将 groupby 排序列 重新设置为列，我们通常在调用完毕后再次调用reset_index（）方法
而as_index参数也可以起到同样作用，该参数是控制groupby方法是否需要将列作为新的index。默认是True，为了达到上述目的，我们只需要将其设置为False即可
>>> df.groupby(by='grade', as_index=False).sum()

# 将某列大于阈值的替换成固定值：
df = pd.DataFrame({
    'A': [1, 2, 3, 4, 5],
    'B': [10, 20, 30, 40, 50]
})
将A列大于3的替换成-1
df['A'] = df['A'].apply(lambda x: -1 if x > 3 else x)

def main():
    pass


if __name__ == '__main__':
    main()
