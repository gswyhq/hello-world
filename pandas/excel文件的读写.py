#!/usr/bin/python3
# coding: utf-8

import pandas as pd
import numpy as np

excel_file = '/home/gswyhq/保险产品别名表.xls'
save_excel_file = '/home/gswyhq/保险产品实体同义词.xls'

HEAD_TITLES = ['实体类型', '实体标准词']

def transform(excel_file, save_excel_file):
    ds = {}
    # 读取excel文件内容； header=0,代表第0行（也就是excel文件第一行，若不设标题行，默认header=None）,names: 重命名标题列
    data = pd.read_excel(excel_file, header=0, names=['实体标准词', '实体同义词'])
    entity_synonyms = []
    current_standard_word = ''
    # 按行变量excel数据；注意 values 是属性不是函数，后面没有小括号
    for standard_word, entity_synonym in data.values:
        # 注意，用pandas读取excel内容，若单元格内容为空，则读取的值为nan;
        if standard_word is np.nan:
            entity_synonyms.append(entity_synonym)
        else:
            if current_standard_word:
                ds[current_standard_word] = entity_synonyms
                entity_synonyms = []
            current_standard_word = standard_word
            entity_synonyms.append(entity_synonym)
    if current_standard_word:
        ds[current_standard_word] = entity_synonyms

    # 若希望把一个二维列表写入excel，则需要先转换为 DataFrame 格式；
    df = pd.DataFrame([HEAD_TITLES]+[['保险产品', standard_word] + entity_synonyms for standard_word, entity_synonyms in ds.items()])
    # header = False, index = False 分别忽略 DataFrame 数据的序号列，序号行
    df.to_excel(save_excel_file, sheet_name='总表', header=False, index=False)
    
def main():
    transform(excel_file, save_excel_file)


if __name__ == '__main__':
    main()



