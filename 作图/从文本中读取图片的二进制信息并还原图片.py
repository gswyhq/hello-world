#!/usr/bin/python3
# coding: utf-8

# 若文本文件是一个单二进制文件，直接二进制读取，再写入为图片文件即可；或者干脆直接将文本文件重命名为图片文件即可
# 这里讨论的文本文件中，包括图片信息的情况，文本文件本身不是二进制文件的情况。

# 读取文本文件，这里读取前五行；
data = pd.read_csv('/home/gswyhq/data/original_tables/Analyst.csv',nrows =5)

# 查看图片信息所在的字段
data['F4919'][0][:30]
# Out[24]: "b'\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF"

# 将图片信息写入文件， 这里用了eval函数，将str,还原为bytes类型
with open('/home/gswyhq/Downloads/O1CN015.jpg', 'wb')as f:
    f.write(eval(data['F4919'][0]))
    
    
def main():
    pass


if __name__ == '__main__':
    main()