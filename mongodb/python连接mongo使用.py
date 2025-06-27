#!/usr/bin/python3
# coding: utf-8

from pymongo import MongoClient

def main():
    client = MongoClient('mongodb://192.168.3.104:27017/')
    db = client.pages
    for i in db['网贷之家页面'].find({"page": "平台页"}).limit(10):
        print(i['url'])

# 连接MongoDB，指定用户名、密码和认证数据库(这里数据库是db3)
client = MongoClient('mongodb://username:password@localhost:27017/db3')

# 查看所有数据库
使用list_databases()方法列出所有数据库：
databases = client.list_databases()
for db in databases:
    print(db['name'])

# 选择数据库并查看集合
选择一个数据库，例如your_database，并列出其中的集合：
db = client['your_database']
collections = db.list_collection_names()
print("Collections in your_database:", collections)

# 选择集合并查看索引
选择一个集合，例如your_collection，并查看其索引信息：
collection = db['your_collection']
indexes = collection.index_information()
print("Indexes in your_collection:", indexes)
方法1：查看当前集合有多少文档(没有过滤条件，数据量大时慎用)：
>>> collection.count_documents({})

方法2：查看当前集合有多少文档(数据量大时也可以快速出结果)
>>> collection.estimated_document_count()

# 查看索引的数据量和占用空间
MongoDB没有直接提供索引大小的统计信息，但可以通过以下方法估算：
使用db.collection.stats()
获取集合的统计信息，包括索引的大小：
stats = db.command('collstats', 'your_collection')
print("Index sizes:", stats.get('indexSizes', {}))

# 遍历每个集合并导出数据
for collection_name in collections:
    db = client.get_database()
    collection = db.get_collection(collection_name)
    # 打开文件以写入模式
    with open(f"{collection_name}.json", "w", encoding="utf-8") as file:
        cursor = collection.find(no_cursor_timeout=True, batch_size=1000)
        try:
            # 将所有文档转换为JSON并写入文件
            for document in cursor:
                document['_id'] = str(document['_id'])
                json_line = json.dumps(document, ensure_ascii=False, indent=None)
                file.write(json_line + "\n")
        except Exception as e:
            print("出错：", e)
        finally:
            cursor.close()

# 遍历每个集合并导出10w样例数据
for collection_name in collections:
    db = client.get_database()
    collection = db.get_collection(collection_name)
    # 打开文件以写入模式
    with open(f"{collection_name}-10w.json", "w", encoding="utf-8") as file:
        cursor = collection.find(no_cursor_timeout=True, batch_size=1000).limit(100000)
        try:
            # 将所有文档转换为JSON并写入文件
            for document in cursor:
                _id = str(document['_id'])
                del document['_id']
                document = {k: f"{v}" for k, v in document.items()}
                json_head = { "index" : {"_type": "doc",  "_id" : _id } }
                json_line = json.dumps(document, ensure_ascii=False, indent=None)
                file.write(json.dumps(json_head, ensure_ascii=False, indent=None) + "\n")
                file.write(json_line + "\n")
        except Exception as e:
            print("出错：", e)
        finally:
            cursor.close()

# curl -XPOST http://192.168.3.106:9200/_bulk --data-binary @bulk_import.json


if __name__ == '__main__':
    main()
