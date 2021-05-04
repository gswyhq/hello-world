#!/usr/bin/python3
# coding: utf-8

import re
import json

cidian_file = '/home/gswewf/data/词典/现代汉语词典第六版电子书.txt'

def parse_version_6(cidian_file):
    cidian_dict = {}
    words_pattent = '^【(?P<name>.+?)】'
    with open(cidian_file)as f:
        line = f.readline()
        while line:
            s = re.search(words_pattent, line)
            if s:
                cidian_dict.setdefault(s.group('name'), line)
            else:
                print(line)
            line = f.readline()

    with open('/home/gswewf/data/词典/现代汉语词典第六版电子书.json', 'w', encoding='utf8')as f:
        json.dump(cidian_dict, f, ensure_ascii=False)

def main():
    parse_version_6(cidian_file)


if __name__ == '__main__':
    main()