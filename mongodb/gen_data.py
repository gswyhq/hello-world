#!/usr/bin/python3
# coding: utf-8

from pymongo import MongoClient

def main():
    client = MongoClient('mongodb://192.168.3.104:27017/')
    db = client.pages
    for i in db['网贷之家页面'].find({"page": "平台页"}).limit(10):
        print(i['url'])


if __name__ == '__main__':
    main()