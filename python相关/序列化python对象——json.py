#! /usr/lib/python3
# -*- coding: utf-8 -*-

#http://old.sebug.net/paper/books/dive-into-python3/serializing.html


def to_json(python_object):
    if isinstance(python_object, time.struct_time):
        return {'__class__': 'time.asctime',
                '__value__': time.asctime(python_object)}
    if isinstance(python_object, bytes):
        return {'__class__': 'bytes',
                '__value__': list(python_object)}
    raise TypeError(repr(python_object) + ' is not JSON serializable')

with open('entry.json', 'w', encoding='utf-8') as f:
    json.dump(entry, f, default=customserializer.to_json)


def from_json(json_object):
    if '__class__' in json_object:
        if json_object['__class__'] == 'time.asctime':
            return time.strptime(json_object['__value__'])
        if json_object['__class__'] == 'bytes':
            return bytes(json_object['__value__'])
    return json_object

with open('entry.json', 'r', encoding='utf-8') as f:
    entry = json.load(f, object_hook=customserializer.from_json)

#json 并不区分元组和列表；它只有一个类似列表的数据类型，数组，并且json模块在序列化过程中会安静的将元组和列表两个都转换成json 数组。
# 大多数情况下，你可以忽略元组和列表的区别，但是在使用json 模块时应记得有这么一回事。