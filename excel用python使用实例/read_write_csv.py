#!/usr/bin/python3
# coding: utf-8
import csv
import codecs

def write_answer_csv(csv_file, answer_list, ncol=1, title=''):
    """
    默认将answer_list列表数据，写到csv文件的第二列
    :param csv_file:
    :param answer_list:
    :param ncol: 写入的指定列，默认是第二列
    :return:
    """
    with open(csv_file)as f:
        csv_data = csv.reader(f)
        csv_lines = [t for t in csv_data]

    # 改写数据
    if not csv_lines:
        csv_lines.append([])
    if len(csv_lines[0]) <= ncol:
        csv_lines[0].append([''] * ncol)
    csv_lines[0][ncol] = title

    for _index, d in enumerate(answer_list, 1):
        if len(csv_lines) <= _index:
            csv_lines.append([])
        if len(csv_lines[_index]) <= ncol:
            csv_lines[_index].append(['']*ncol)
        csv_lines[_index][ncol] = d

    # 将数据写入文件
    with open(csv_file, 'w', encoding='utf_8_sig')as f:
        # f.write(codecs.BOM_UTF8.decode('utf8'))  # 防止写入csv文件，在windows中文会乱码, 或者在写文件时定义编码为：utf_8_sig
        spamwriter = csv.writer(f)
        spamwriter.writerows(csv_lines)


def get_auto_test_data_csv(csv_file, ncol=None):
    """
    默认读取csv文件第一个工作表的第一列的文件内容，并返回一个对应的列表数据
    :param csv_file: csv文件名
    :param ncol: 需要读取的列数，如（0,1），表示读取第一列，第二列
    :return:
    """
    # 打开文件
    with open(csv_file) as f:
        reader = csv.reader(f)
        test_set, answer_set, description_set = [], [], []
        for row in reader:
            if not test_set and '问题' in row[0]:
                # 若第一行第一列有问题二字，则跳过
                continue
            test_set.append(row[ncol[0]])
            answer_set.append(row[ncol[1]])
            description_set.append(row[ncol[2]])
        return test_set, answer_set, description_set

def read_t(t_csv_file):
    with open(t_csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            print(row['first_name'], row['last_name'])

def read_t2(csv_file):
    with open(csv_file, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in spamreader:
            print('7777'.join(row))

def write_t(t_csv_file):
    with open(t_csv_file, 'w', newline='', encoding='utf_8_sig') as csvfile:
        csvfile.write(codecs.BOM_UTF8.decode('utf8'))  # 防止写入csv文件，在windows中文会乱码, 或者在写文件时定义编码为：utf_8_sig
        spamwriter = csv.writer(csvfile, delimiter=' ',
                                quotechar=',', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['Spam'] * 5 + ['Baked Beans'])
        spamwriter.writerow(['Spam', 'Lovely Spam', 'Wonderful Spam'])

    with open('names.csv', 'w', newline='', encoding='utf_8_sig') as csvfile:
        csvfile.write(codecs.BOM_UTF8.decode('utf8'))  # 防止写入csv文件，在windows中文会乱码, 或者在写文件时定义编码为：utf_8_sig
        fieldnames = ['first_name', 'last_name']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
        writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
        writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})

def write_t1(out_file):
    with open(out_file, 'w', newline='', encoding='utf_8_sig') as csvfile:
        # csvfile.write(codecs.BOM_UTF8.decode('utf8'))  # 防止写入csv文件，在windows中文会乱码, 或者在写文件时定义编码为：utf_8_sig
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow(['你好'] * 5 + ['Baked, 是否'])
        spamwriter.writerow(['Spam', 'Lovely 地点', 'Wonderful Spam'])
        spamwriter.writerows([['a', 'b'],
                              ['233', 234, '色调']])

def change_csv_file():
    """
    对csv文件进行改写
    :return: 
    """
    # 读取文件数据
    r = csv.reader(open('/tmp/test.csv'))  # Here your csv file
    lines = [l for l in r]
    # lines:
    #
    # [['Ip', 'Sites'],
    #  ['127.0.0.1', '10'],
    #  ['127.0.0.2', '23'],
    #  ['127.0.0.3', '50']]

    # 改写数据
    lines[2][1] = '30'

    # lines:
    # [['Ip', 'Sites'],
    #  ['127.0.0.1', '10'],
    #  ['127.0.0.2', '30'],
    #  ['127.0.0.3', '50']]

    # 将数据写入文件
    writer = csv.writer(open('/tmp/output.csv', 'w'))
    writer.writerows(lines)

def main():
    csv_file = '/home/gswyhq/youhuibao/input/众安航延险知识图谱展示测试集_result.csv'
    write_t1(csv_file)

    read_t2(csv_file)
if __name__ == '__main__':
    main()

# delimiter，最基本的分隔符
# quotechar，如果某个 item 中包含了分隔符，应该用 quotechar 把它包裹起来
# doublequote，如果某个 item 中出现了 quotechar 那么可以把整个内容用 quotechar 包裹，并把 quotechar double 一下用来做区分, 这个标志控制quotechar实例是否成对, 默认值：True
# escapechar，如果不用 doublequote 的方法还可以用 escapechar 来辅助，这个字符用来指示一个转义序列，默认值：None
# lineterminator，每一行的结束符，默认的是 \r\n
# quoting，可以选择任何时候都使用 quotechar 来包裹内容，或者是需要用到的时候再用，或者不用
# skipinitialspace，是否忽略分隔符后面跟着的空格
# strict，这个是 Python 自己的，是否抛要异常

# 有4种不同的引号选项，在csv模块中定义为四个变量：
# QUOTE_ALL不论类型是什么，对所有字段都加引号。
# QUOTE_MINIMAL对包含特殊字符的字段加引号（所谓特殊字符是指，对于一个用相同方言和选项配置的解析器，可能会造成混淆的字符）。这是默认选项。
# QUOTE_NONNUMERIC对所有非整数或浮点数的字段加引号。在阅读器中使用时，不加引号的输入字段会转换为浮点数。
# QUOTE_NONE输出中所有内容都不加引号。在阅读器中使用时，引号字符包含在字段值中（正常情况下，它们会处理为定界符并去除）。