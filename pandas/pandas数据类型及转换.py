
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


