#!/usr/bin/python3
# coding: utf-8

import json
import requests

# 先在https://bosonnlp.com/console注册获取API密钥
YOUR_API_TOKEN = "CxfO3nR5.23851.okPx1lOulBHk"

def main():
    NER_URL = 'http://api.bosonnlp.com/ner/analysis'

    s = ['住院了检查出来胃癌能否赔偿重疾险，并且能否获赔重疾保险金，轻症疾病保险金',
         '对于该小孩是不是郑尚金的孩子，目前已做亲子鉴定，结果还没出来，纪检部门仍在调查之中。成都商报记者 姚永忠',
         ]
    data = json.dumps(s)
    headers = {'X-Token': YOUR_API_TOKEN}
    resp = requests.post(NER_URL, headers=headers, data=data.encode('utf-8'))

    for item in resp.json():
        for entity in item['entity']:
            print(entity)
            print(''.join(item['word'][entity[0]:entity[1]]), entity[2])


if __name__ == '__main__':
    main()