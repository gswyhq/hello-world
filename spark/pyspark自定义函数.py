
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.functions import isnull
from pyspark.sql import functions as F
import pandas as pd
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from datetime import datetime, timedelta
from pyspark.sql.types import BooleanType, TimestampType, DoubleType, StringType
from pyspark.sql.functions import udf, col
from functools import reduce
from pyspark.sql.functions import regexp_extract
import numpy as np

# types.ArrayType(                 types.DataType(                  types.DoubleType(                types.LongType(                  types.StringType(                types.base64                     types.long(
# types.AtomicType(                types.DataTypeSingleton(         types.FloatType(                 types.MapType(                   types.StructField(               types.basestring(                types.re
# types.BinaryType(                types.DateConverter(             types.FractionalType(            types.NullType(                  types.StructType(                types.calendar                   types.register_input_converter(
# types.BooleanType(               types.DateType(                  types.IntegerType(               types.NumericType(               types.TimestampType(             types.datetime                   types.sys
# types.ByteType(                  types.DatetimeConverter(         types.IntegralType(              types.Row(                       types.UserDefinedType(           types.decimal                    types.time
# types.CloudPickleSerializer(     types.DecimalType(               types.JavaClass(                 types.ShortType(                 types.array(

def to_upper(s):
    if s is not None:
        return s.upper()


spark = SparkSession.builder.appName("test").enableHiveSupport().getOrCreate()
df = spark.sql("select * from hive_db.hive_table limit 10")
toDateUDF=udf(to_upper, StringType())  
df.withColumn('mobile',toDateUDF('mobile')).show()


columns = [column for column in df.columns if column not in ['y', 'm', 'd', 'dt']]
val = sum([df.filter(isnull(column)).count() for column in columns]) / (len(columns)*df.count() + 0.01)

def is_null(t):
    null_list = ['暂无数据', '--', '不详', '-', '尚未公开', '未评分', '[]', '暂无资料', '待定', '"', '未知', '无', '0月', '0个', '']
    if t in null_list:
        return udf(True, BooleanType())
    else:
        return udf(False, BooleanType())

# 筛选 mobile 字段以 135 开头的
df.where(col("mobile").startswith("135")).show()

# 用 正则 对data_upd字段进行匹配，匹配结果替换mobile字段；
df.withColumn('mobile', regexp_extract(df.data_upd, r'^(2021)?(.*)', 2)).show()

# 检测缺失值
d1 = spark.createDataFrame([(10,'a', None), (20, 'b', 3.), (30, 'c',4.),
                            (20, 'b',5.), (30, 'd', np.nan), (40, None,7.),
                            (None, 'e',8.), (50, '', 8.)], ['value', 'key','v2'])
d1 = d1.select('key', 'value', 'v2')
d1.show()
d1.where(reduce(lambda x, y: x | y, (F.col(x).isNull() for x in d1.columns))).show()
# 或者：
d1.where(F.col('key').isNull() | F.col('value').isNull() | F.col('v2').isNull()).show()

d1.where(F.col('key').isin(None, "", np.nan) | F.col('key').isNull()).show()

pd.DataFrame(d1.collect())

data = [['Alice', 19, 'blue', '19'],
        ['Jane', 20, 'green', 'green'],
        ['Mary', 21, 'blue', 'Mary'],
        ['Mary1', 22, 'blue', '19'],
        ['Mary2', 23, 'blue', '28'],
        ['Mary3', 24, 'blue', 'True'],
        ['Mary4', 25, 'blue', 'False']
        ]
df3 = spark.createDataFrame(data, schema=["name", "age", "eye_color", "detail"])
# 判断各列的数据类型；
{col : df3.schema[col].dataType for col in df3.columns}

# 正则筛选，并新建列；
df3.withColumn('char', regexp_extract(df3.detail, r'^([a-z]+)$', 1)).show()

c_i=df3.withColumn('char', regexp_extract(df3.detail, r'^([a-z]+)$', 1))

# 过滤到空值，及空字符串；
c_i.filter(F.col('char').isNotNull() & (F.col('char')!="" ) ).show()

# 判断列的数据类型：
isinstance(df3.schema["detail"].dataType, LongType)
# True

# 对 df3的age列求解分位数
percentiles = F.expr('percentile_approx(age, array(0.01, 0.25, 0.75, 0.9))')
df3.agg(percentiles).show()

from decimal import Decimal
# List
data = [{"Category": 'Category A', "ID": 1, "Value": Decimal(12.40)},
        {"Category": 'Category B', "ID": 2, "Value": Decimal(30.10)},
        {"Category": 'Category C', "ID": 3, "Value": None},
        {"Category": 'Category D', "ID": 4, "Value": Decimal(1.0)},
       ]

# 创建DataFrame
df = spark.createDataFrame(data)
df.show(truncate=False)

# 使用标准ANSI-SQL SQL表达式来过滤列
df.filter("Value is not null").show(truncate=False)
df.where("Value is null").show

# 使用type.BooleanType列对象来过滤
# 如果在DataFrame中存在有boolean列，则可以直接将其作为条件传入
df.filter(df['Value'].isNull()).show(truncate=False)
df.where(df.Value.isNotNull()).show(truncate=False)

# 使用col或column函数将列转换为Column对象

from pyspark.sql.functions import col,column

df.filter(col('Value').isNull()).show(truncate=False)
df.filter(column('Value').isNull()).show(truncate=False)

df.where(col('Value').isNotNull()).show(truncate=False)
df.where(column('Value').isNotNull()).show(truncate=False)

# 转换 df4的oldCol列的数据类型；
df4.withColumn("oldCol",df4.oldCol.cast("float"))

# 对 df 的 sex列进行统计计数
df.select(df.sex, df.age).groupby(df.sex).count().show()

# 对 df 的 sex列的长度进行统计计数
df.select(df.sex, df.age).groupby(F.length(F.col('sex'))).count().show()
