
# 读取时指定数据类型
以字符串读取所有字段
df = pd.read_csv(path,encoding='utf-8',sep=',',dtype=object)
自定义
df = pd.read_csv(path,encoding='utf-8',sep=',', dtype={'column1': object, 'column2': np.float64})

# 通过 pd.to_numeric 函数将字符串转换为int或者float
df = pd.DataFrame({'uid':['a', 'b', 'c'], 'data':['1', '2', '3'], 'data2':['8.0', '7.3', None]})
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
uid      3 non-null object
data     3 non-null object
data2    2 non-null object
dtypes: object(3)
memory usage: 152.0+ bytes

for k in ['data', 'data2']:
    df[k] = pd.to_numeric(df[k])
        
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
uid      3 non-null object
data     3 non-null int64
data2    2 non-null float64
dtypes: float64(1), int64(1), object(1)
memory usage: 152.0+ bytes

# 通过 apply 函数将数字转换为字符串：
for k in ['data', 'data2']:
    df[k] = df[k].apply(lambda x: str(x))
        
df.info()
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 3 columns):
uid      3 non-null object
data     3 non-null object
data2    3 non-null object
dtypes: object(3)
memory usage: 152.0+ bytes

# numpy.datetime64 转换为字符串
t, type(t)
(numpy.datetime64('2021-01-01T00:00:00.000000000'), numpy.datetime64)
方法1： str(t)
'2021-01-01T00:00:00.000000000'
方法2：
ts = pd.to_datetime(t) 
d = ts.strftime('%Y.%m.%d')
t, ts, d
(numpy.datetime64('2021-01-01T00:00:00.000000000'), Timestamp('2021-01-01 00:00:00'), '2021.01.01')

# Timestamp 转换为string
t = samp_t['落单日期']
t
Timestamp('2022-05-18 16:10:15')
isinstance(t, pd.Timestamp)
True
t.strftime(format="%Y-%m-%d %H:%M:%S")
'2022-05-18 16:10:15'

# numpy.datetime64 转换为 datetime
datatime = pd.to_datetime(time_data)
或者：
t = numpy.datetime64('2021-04-10T00:00:00.000000000')
pd.Timestamp(t).to_pydatetime().month, pd.Timestamp(t).to_pydatetime().day
Out[22]: (4, 10)

# 批量数据类型转换
merge_df['落单时间'] = pd.to_datetime(merge_df['落单时间'], format='%Y-%m-%d %H:%M:%S')
有时候，求解两个时间列之间的间隔天数耗时太长，to_datetime太慢：
df['间隔天数'] = [(t1-t2).days for t1, t2 in df[['落单时间', '标注时间']].values]
可以改为：
diff_days_list = []
all_data_tag_df['落单时间1'] = pd.to_datetime(all_data_tag_df['落单时间'], format='%Y-%m-%d %H:%M:%S')
all_data_tag_df['标注时间1'] = pd.to_datetime(all_data_tag_df['标注时间'], format='%Y-%m-%d %H:%M:%S')
for t1, t2 in tqdm(all_data_tag_df[['落单时间1', '标注时间1']].values):
    diff_days_list.append(int((t1-t2)/(3600*24*1e9)))
all_data_tag_df['标注间隔天数'] = diff_days_list


