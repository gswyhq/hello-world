#! /usr/lib/python3
# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-
import  xdrlib ,sys
import xlrd
def open_excel(file= 'file.xls'):
    try:
        data = xlrd.open_workbook(file)
        return data
    except Exception,e:
        print str(e)
#根据索引获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_index：表的索引
def excel_table_byindex(file= 'file.xls',colnameindex=0,by_index=0):
    data = open_excel(file)
    table = data.sheets()[by_index]
    nrows = table.nrows #行数
    ncols = table.ncols #列数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(1,nrows):

         row = table.row_values(rownum)
         if row:
             app = {}
             for i in range(len(colnames)):
                app[colnames[i]] = row[i]
             list.append(app)
    return list

#根据名称获取Excel表格中的数据   参数:file：Excel文件路径     colnameindex：表头列名所在行的所以  ，by_name：Sheet1名称
def excel_table_byname(file= 'file.xls',colnameindex=0,by_name=u'Sheet1'):
    data = open_excel(file)
    table = data.sheet_by_name(by_name)
    nrows = table.nrows #行数
    colnames =  table.row_values(colnameindex) #某一行数据
    list =[]
    for rownum in range(1,nrows):
         row = table.row_values(rownum)
         if row:
             app = {}
             for i in range(len(colnames)):
                app[colnames[i]] = row[i]
             list.append(app)
    return list

def main():
   tables = excel_table_byindex()
   for row in tables:
       print row

   tables = excel_table_byname()
   for row in tables:
       print row

if __name__=="__main__":
    main()




  1、导入模块

      import xlrd

   2、打开Excel文件读取数据

       data = xlrd.open_workbook('excelFile.xls')

   3、使用技巧

        获取一个工作表



        table = data.sheets()[0]          #通过索引顺序获取

        table = data.sheet_by_index(0) #通过索引顺序获取


        table = data.sheet_by_name(u'Sheet1')#通过名称获取

        获取整行和整列的值（数组）
 　　
         table.row_values(i)

         table.col_values(i)

        获取行数和列数
　　
        nrows = table.nrows

        ncols = table.ncols

        循环行列表数据
        for i in range(nrows ):
      print table.row_values(i)

单元格
cell_A1 = table.cell(0,0).value

cell_C4 = table.cell(2,3).value

使用行列索引
cell_A1 = table.row(0)[0].value

cell_A2 = table.col(1)[0].value

简单的写入
row = 0

col = 0

# 类型 0 empty,1 string, 2 number, 3 date, 4 boolean, 5 error
ctype = 1 value = '单元格的值'

xf = 0 # 扩展的格式化

table.put_cell(row, col, ctype, value, xf)

table.cell(0,0)  #单元格的值'

table.cell(0,0).value #单元格的值'
