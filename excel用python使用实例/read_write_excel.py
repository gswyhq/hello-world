#!/usr/bin/python3
# coding: utf-8

# 读写excel操作

import xlrd
import xlwt
from datetime import date, datetime


def read_excel(excel_file):
    """
    读取excel文件的内容
    :param excel_file:
    :return:
    """
    # 打开文件
    workbook = xlrd.open_workbook(excel_file)
    # 获取所有sheet
    print(workbook.sheet_names())# [u'sheet1', u'sheet2']
    sheet2_name = workbook.sheet_names()[1]

    # 根据sheet索引或者名称获取sheet内容
    sheet2 = workbook.sheet_by_index(1)  # sheet索引从0开始
    sheet2 = workbook.sheet_by_name('sheet2')

    # sheet的名称，行数，列数
    print(sheet2.name, sheet2.nrows, sheet2.ncols)

    # 获取整行和整列的值（数组）
    rows = sheet2.row_values(3)  # 获取第四行内容
    cols = sheet2.col_values(2)  # 获取第三列内容
    print(rows)
    print(cols)

    # 获取单元格内容
    print(sheet2.cell(1, 0).value)
    print(sheet2.cell_value(1, 0))
    print(sheet2.row(1)[0].value)

    # 获取单元格内容的数据类型
    print(sheet2.cell(1, 0).ctype)
    # python读取excel中单元格的内容返回的有5种类型，即上面例子中的ctype: 0 empty,1 string, 2 number, 3 date, 4 boolean, 5 error

    sheet = workbook.sheet_by_index(1)
    row = 0
    col = 2
    # 判断ctype是否等于3，如果等于3，则用时间格式处理
    if (sheet.cell(row, col).ctype == 3):
        date_value = xlrd.xldate_as_tuple(sheet.cell_value(rows, 3), workbook.datemode)
        date_tmp = date(*date_value[:3]).strftime('%Y/%m/%d')

def merged_excel(excel_file):
    """
    读取excel中合并单元格里头的内容
    :param excel_file:
    :return:
    """
    # 获取合并的单元格
    # 读取文件的时候需要将formatting_info参数设置为True，默认是False，所以上面获取合并的单元格数组为空，
    workbook = xlrd.open_workbook(excel_file, formatting_info=True)
    sheet2 = workbook.sheet_by_name('sheet2')
    print(sheet2.merged_cells)
    # [(7, 8, 2, 5), (1, 3, 4, 5), (3, 6, 4, 5)]
    # merged_cells返回的这四个参数的含义是：(row, row_range, col, col_range), 其中[row, row_range)包括row, 不包括row_range, col也是一样，即
    # (1, 3, 4,5)的含义是：第1到2行（不包括3）合并，
    # (7, 8, 2, 5)的含义是：第2到4列合并。
    # 利用这个，可以分别获取合并的三个单元格的内容：
    print(sheet2.cell_value(1, 4))  # (1, 3, 4, 5)
    print(sheet2.cell_value(3, 4))  # (3, 6, 4, 5)
    print(sheet2.cell_value(7, 2))  # (7, 8, 2, 5)

    merge = []
    for (rlow, rhigh, clow, chigh) in sheet2.merged_cells:
        merge.append([rlow, clow])

    print(merge)  # [[7, 2], [1, 4], [3, 4]]
    for index in merge:
        print(sheet2.cell_value(index[0], index[1]))

def write_excel(excel_file):
    """
    向excel文件写数据
    :param excel_file: 需要写的excel文件名，若原文件已经存在，则覆盖原有的文件内容
    :return:
    """
    import xlwt
    wbk = xlwt.Workbook(encoding='utf8')
    sheet = wbk.add_sheet('sheet1', cell_overwrite_ok=True)  # 使用cell_overwrite_ok=True来创建worksheet, 即可以更改excel里头的内容; 否则若原excel单元格有内容，则不能改写
    sheet.write(2, 0, '写的内容;;')
    sheet.write(2, 1, 78654)
    wbk.save(excel_file)

def change_excel_file(excel_file):
    """
    改写excel文件的内容
    Python中一般使用xlrd（excel read）来读取Excel文件，使用xlwt（excel write）来生成Excel文件（可以控制Excel中单元格的格式），需要注意的是，
    用xlrd读 取excel是不能对其进行操作的：xlrd.open_workbook()方法返回xlrd.Book类型，是只读的，不能对其进行操作。而 xlwt.Workbook()
    返回的xlwt.Workbook类型的save(filepath)方法可以保存excel文件。因此对于读取和生成Excel文件都非常容易处理，但是对于已经存在的Excel文件进行修改就比较麻烦了。
    不过，还有一个xlutils（依赖于xlrd和xlwt）提供复制excel文件内容和修改文件的功能。其实际也只是在xlrd.Book和xlwt.Workbook之间建立了一个管道而已
    :param excel_file: 需要改写的excel文件
    :return:
    """
    from xlrd import open_workbook
    from xlutils.copy import copy
    # xlutils.copy模块的copy();方法实现了这个功能

    rb = open_workbook(excel_file)

    # 通过sheet_by_index()获取的sheet没有write()方法
    rs = rb.sheet_by_index(0)

    wb = copy(rb)

    # 通过get_sheet()获取的sheet有write()方法
    ws = wb.get_sheet(0)
    ws.write(3, 0, '答案!')

    wb.save(excel_file)

def main():
    excel_file = '/home/gswewf/input/auto_test_set.xlsx'
    read_excel(excel_file=excel_file)

if __name__ == '__main__':
    main()
