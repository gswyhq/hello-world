#!/usr/lib/python3
# -*- coding: utf-8 -*-

import os
import re
import hashlib
from glob import glob
from tqdm import tqdm
import csv
import pandas as pd

def read_csv():
    # https://github.com/kinginsun/Enterprise-Registration-Data-of-Chinese-Mainland
    for csv_file in tqdm(glob("/home/gswyhq/github_project/Enterprise-Registration-Data-of-Chinese-Mainland/csv/2019/*.csv")):
        print(csv_file)
        df = pd.read_csv(csv_file, encoding='utf-8', error_bad_lines=False, sep=',', quoting=csv.QUOTE_MINIMAL, engine='python')
        print(df.shape, df.columns)
        df = df.fillna('')
        for name, nid, date_str, tag, person, mem, chi, province, city, addr in df.values[:25]:
            if not nid:
                continue
            if not re.search("^\d{4}-\d{2}-\d{2}$", date_str):
                date_str = 'null'
            else:
                date_str = f'timestamp("{date_str}")'
            vid = hashlib.md5(('统一社会信用代码'+nid).encode(encoding='utf-8')).hexdigest()
            vid2 = hashlib.md5(('企业法人'+person).encode(encoding='utf-8')).hexdigest()
            vid3 = hashlib.md5(('省份'+province).encode(encoding='utf-8')).hexdigest()
            vid4 = hashlib.md5(('地区'+city).encode(encoding='utf-8')).hexdigest()
            insert_ngql1 = f'''INSERT VERTEX `企业名称` (name, `统一社会信用代码`,  `注册日期` ,  `企业类型` ,  `注册资金` , `经营范围` ,`注册地址` ) VALUES "{vid}":("{name}", "{nid}", {date_str}, "{tag}", "{mem}", "{chi}", "{addr}"); '''
            insert_ngql2 = f'''INSERT VERTEX `企业法人` (name ) VALUES "{vid2}":("{person}"); '''
            insert_ngql3 = f'''INSERT VERTEX `省份` (name ) VALUES "{vid3}":("{province}"); '''
            insert_ngql4 = f'''INSERT VERTEX `地区` (name ) VALUES "{vid4}":("{city}"); '''

            insert_edge1 = f'''INSERT EDGE `法人代表` () VALUES  "{vid}"->"{vid2}":();'''
            insert_edge2 = f'''INSERT EDGE `所属省份` () VALUES  "{vid4}"->"{vid3}":();'''
            insert_edge3 = f'''INSERT EDGE `所在地区` () VALUES  "{vid}"->"{vid4}":();'''

            print('\n'.join([insert_ngql1, insert_ngql2, insert_ngql3, insert_ngql4, insert_edge1, insert_edge2, insert_edge3]))

def main():
    read_csv()


if __name__ == '__main__':
    main()
