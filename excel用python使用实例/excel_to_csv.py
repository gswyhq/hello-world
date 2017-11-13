#!/usr/bin/python3
# coding: utf-8

import xlrd
import csv
import sys
import os

def csv_from_excel(excel_file='your_workbook.xls'):
    """将excel文件转换成csv格式的文件，生成的csv文件名：excel文件名 + sheet名
    """
    excel_name, excel_ext = os.path.splitext(excel_file)
    assert_excel_ext = ['.xlsx', '.xls']
    assert excel_ext in assert_excel_ext, '输入的excel文件：{}，的后缀应该是： {}'.format(excel_file, assert_excel_ext)
    assert os.path.isfile(excel_file), "输入的文件有误： {}".format(excel_file)

    wb = xlrd.open_workbook(excel_file)
    # sh = wb.sheet_by_name('Sheet1')
    # sh = wb.sheet_by_index(0)
    sheet_names = wb.sheet_names()
    for sheet_name in sheet_names:
        sh = wb.sheet_by_name(sheet_name)
        csv_file = "{}_{}{}".format(excel_name, sheet_name, '.csv')
        your_csv_file = open(csv_file, 'w')
        wr = csv.writer(your_csv_file, quoting=csv.QUOTE_ALL)

        for rownum in range(sh.nrows):
            wr.writerow(sh.row_values(rownum))

        your_csv_file.close()

def main():
    try:
        excel_file = sys.argv[1]
        csv_from_excel(excel_file=excel_file)
    except Exception as e:
        print(e)

if __name__ == '__main__':
    main()