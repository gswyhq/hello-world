
# python 获取excel文件的所有sheet名称

# 当一个excel文件的sheet比较多时候，这时候可能需要获取所有的sheet的名字

xl = pandas.ExcelFile(你的Excel文件路径)

sheet_names = xl.sheet_names  # 所有的sheet名称

df = xl.parse(sheet_name)  # 读取Excel中sheet_name的数据
# 也可以直接读取所有的sheet，将sheetname设置为None，这时候得到的是一个dict结果

df = pandas.read_excel(你的Excel文件路径, None)
# “df”都是作为DataFrames字典的工作表，您可以通过运行以下方法对其进行验证：

df.keys()
