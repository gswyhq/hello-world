
网址: http://www.cnblogs.com/chaosimple/p/4153083.html

说明:
首先百度Python pandas DataFrame,下面列出DataFrame该数据结构的部分使用方法,并对其进行说明, DataFrame和Series作为padans两个主要的数据结构.
     如果你经常使用SQL数据库或者做过数据分析等相关工作,可以更快的上手python的pandas库,其pandas库的使用方法跟SQL语句的一些语法类似,只不过语言 变了而已.
正文:
import pandas as pd 引用pandas时使用pd名称就可
使用DataFrame查看数据(类似SQL中的select):
from pandas import DataFrame #从pandas库中引用DataFrame
df_obj = DataFrame() #创建DataFrame对象
df_obj.dtypes #查看各行的数据格式
df_obj.head() #查看前几行的数据,默认前5行
df_obj.tail() #查看后几行的数据,默认后5行
df_obj.index #查看索引
df_obj.columns #查看列名
df_obj.values #查看数据值
df_obj.describe #描述性统计
df_obj.T #转置
df_obj.sort(columns = ‘’)#按列名进行排序
df_obj.sort_index(by=[‘’,’’])#多列排序,使用时报该函数已过时,请用sort_values
df_obj.sort_values(by=['',''])同上
 
使用DataFrame选择数据(类似SQL中的LIMIT):
df_obj[‘客户名称’] #显示列名下的数据
df_obj[1:3] #获取1-3行的数据,该操作叫切片操作,获取行数据
df_obj.loc[:0,['用户号码','产品名称']] #获取选择区域内的数据,逗号前是行范围,逗号后是列范围,注loc通过标签选择数据,iloc通过位置选择数据
df_obj['套餐'].drop_duplicates() #剔除重复行数据
使用DataFrame重置数据:
df_obj.at[df_obj.index,'支局_维护线']='自有厅' #通过标签设置新的值,如果使用iat则是通过位置设置新的值
使用DataFrame筛选数据(类似SQL中的WHERE):
alist = ['023-18996609823']
df_obj['用户号码'].isin(alist) #将要过滤的数据放入字典中,使用isin对数据进行筛选,返回行索引以及每行筛选的结果,若匹配则返回ture
df_obj[df_obj['用户号码'].isin(alist)] #获取匹配结果为ture的行
匹配不在列表中的项，pandas中是没有isnotin这个方法的，但需要求isin的反集即可实现，如：
df_obj[~(df_obj['用户号码'].isin(['13511112345', '18912078935']))] 

# 按列筛选，从d0数据框中，筛选出rs列中包含 '2_161686082’中值的数据框
d1 = d0.loc[d0['rs']=='2_161686082']

# 多条件筛选：
df['Fam_Size'].loc[(df['Fam_Size'] > 1) & (df['Fam_Size'] <= 3)] = 2   
# 只要中间加一个 & 符号 ， 然后把两边的条件小括号起来就行。
# 使用 &（且） 和 |（或） 时每个条件都要用小括号括起来。

# 字符串的模糊筛选，在pandas里我们可以用.str.contains()来实现，筛选包含‘华东’的区域。
df.loc[df['区域'].str.contains('华东')]
# 也可以使用  '|'  来进行多个条件的筛选，筛选包含‘华’或‘东’的区域：
df.loc[df['区域'].str.contains('华|东')]

使用DataFrame模糊筛选数据(类似SQL中的LIKE):
df_obj[df_obj['套餐'].str.contains(r'.*?语音CDMA.*')] #使用正则表达式进行模糊匹配,*匹配0或无限次,?匹配0或1次
使用DataFrame进行数据转换(后期补充说明)
df_obj['支局_维护线'] = df_obj['支局_维护线'].str.replace('巫溪分公司(.{2,})支局','\\1')#可以使用正则表达式
df_obj['支局_维护线'].drop_duplicates() #返回一个移除重复行的数据
可以设置take_last=ture 保留最后一个,或保留开始一个.补充说明:注意take_last=ture已过时,请使用keep='last'
使用pandas中读取文本数据:
read_csv('D:\LQJ.csv',sep=';',nrows=2) #首先输入csv文本地址,然后分割符选择等等
使用pandas聚合数据(类似SQL中的GROUP BY 或HAVING):
data_obj['用户标识'].groupby(data_obj['支局_维护线'])
data_obj.groupby('支局_维护线')['用户标识'] #上面的简单写法
adsl_obj.groupby('支局_维护线')['用户标识'].agg([('ADSL','count')])
#按支局进行汇总对用户标识进行计数,并将计数列的列名命名为ADSL
使用pandas合并数据集(类似SQL中的JOIN):
merge(mxj_obj2, mxj_obj1 ,on='用户标识',how='inner')# mxj_obj1和mxj_obj2将用户标识当成重叠列的键合并两个数据集,inner表示取两个数据集的交集.


