#!/usr/bin/python3
# coding: utf-8


# 依赖的包：python-docx
#
# 安装：pip install python-docx
#
# 引用：import docx
# .docx文件的结构比较复杂，分为三层，1、Docment对象表示整个文档；2、Docment包含了Paragraph对象的列表，Paragraph对象用来表示文档中的段落；3、一个Paragraph对象包含Run对象的列表，用下面这个图说明Run到底是神马东西。
# Word里面的文本不只是包含了字符串，还有字号、字体、颜色等等属性，都包含在style中。一个Run对象就是style相同的一段文本，新建一个Run就有新的style。

import docx
docx_file = '/home/gswewf/Downloads/数据字典(zy)2018-03-01.docx'
doc = docx.Document(docx_file)

# 遍历每个表格
for table in doc.tables:
    # print(dir(table))
    # print(table.row_cells)
    # 遍历每一行
    for i in range(len(table.rows)):
        # 遍历每一列
        for row in table.row_cells(i):
            print(row.text)
    # print(table.row_cells(0)[0].text)
    # for row in table.rows:
    #     print(dir(row))
    #
    #     print(row.cells)
    #     print(row.part)
    #     print(row.table)
    #     print(dir(row.table))
    #     print(row.table.table_direction)
    #     break
    break


def main():
    pass


if __name__ == '__main__':
    main()