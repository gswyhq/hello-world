
读取csv文件：

import csv
#打开文件，用with打开可以不用去特意关闭file了，python3不支持file()打开文件，只能用open()
with open("XXX.csv","r",encoding="utf-8") as csvfile:
     #读取csv文件，返回的是迭代类型
     read = csv.reader(csvfile)
     for i in read:
          print(i)

存为csv文件：

import csv
with open("XXX.csv","w",newline="") as datacsv:
     #dialect为打开csv文件的方式，默认是excel，delimiter="\t"参数指写入的时候的分隔符
     csvwriter = csv.writer(datacsv,dialect = ("excel"))
     #csv文件插入一行数据，把下面列表中的每一项放入一个单元格（可以用循环插入多行）
     csvwriter.writerow(["A","B","C","D"])

