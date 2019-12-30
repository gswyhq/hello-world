#!/usr/bin/python3
# coding: utf-8

# pandas读取大于内存的文件

# 方法1, 设置chunksize, 分块读取
chunksize = 10 ** 6  # # 每块为chunksize条数据(index)
for chunk in pd.read_csv(filename, chunksize=chunksize):
    print(chunk)

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

def main():
    pass

if __name__ == '__main__':
    main()