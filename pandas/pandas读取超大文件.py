#!/usr/bin/python3
# coding: utf-8

# pandas读取大于内存的文件

# 方法1, 设置chunksize, 分块读取
chunksize = 10 ** 6  # # 每块为chunksize条数据(index)
chunks = []
for chunk in pd.read_csv(filename, chunksize=chunksize):
    print(chunk)
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# 方法2, 使用iterator, 但是也需要设置chunksize
chunkSize = 10 ** 6
# iterator参数，默认为False，将其改为True，返回一个可迭代对象TextFileReader，使用它的get_chunk(num)方法可获得前num行的数据
reader = pd.read_csv(filename, iterator=True)
while True:
    try:
        chunk = reader.get_chunk(chunkSize)
        print(chunk)
    except StopIteration:
        break

# 数据写入文件时候，指定换行符及分列符号：
df.to_csv("result.txt", index=False, sep='\001', line_terminator='\n\001\001\001\n') 
# line_terminator：自定义换行符
# sep: 自定义分列符号，分隔符；

# 超大文件，仅仅读取指定行，读取前几行：
df  = pandas.read_csv(filename, header=0, nrows=1000) # nrows: 读取前几行，这里读取前1000行；

def main():
    pass

if __name__ == '__main__':
    main()
