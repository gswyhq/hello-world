#!/usr/bin/env python
# -*- coding: windows-1251 -*-
# Copyright (C) 2005 Kiseliov Roman
__rev_id__ = """$Id$"""


from xlwt import *

w = Workbook()
ws = w.add_sheet('Hey, Dude')

for i in range(6, 80):
    fnt = Font()
    fnt.height = i
    style = XFStyle()
    style.font = fnt
    ws.write(1+i, i, 'Test')
    ws.row(i).height_mismatch = True  # ??????????????????????????height??????? ?????height???????height_mismatch???????1
    if i%2 == 0:
        ws.col(i).width = 0x0d00 + i * 300
        ws.row(i).height = i + 200
    else:
        ws.col(i).width = 0x0d00 + i
        ws.row(i).height = i * 30
w.save('col_width.xls')
