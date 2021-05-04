#!/usr/bin/python3
# coding: utf-8

import csv

def csv_to_markdown(csv_file):
    csv_file = csv_file.lower()
    assert csv_file.endswith('.csv'), "输入的应该是csv文件"
    md_file = csv_file.replace('.csv', '.md')
    with open(csv_file, encoding='utf_8_sig')as f:
        csv_data = csv.reader(f)
        csv_lines = [t for t in csv_data]

    md_datas = ''
    for _index, line in enumerate(csv_lines):
        line = ["{}".format(t) if t else '-' for t in line]

        if _index == 1:
            md_datas += '|' + '--:|' * len(line) + '\n'

        md_datas += "| {} |\n".format(' | '.join(line))

    with open(md_file, 'w', encoding='utf8')as f:
        f.write(md_datas)

    print('markdown文件保存位置： {}'.format(md_file))

def main():
    csv_file = '/home/gswyhq/nlp_request_url/机器人整理20180711.csv'
    csv_to_markdown(csv_file)


if __name__ == '__main__':
    main()