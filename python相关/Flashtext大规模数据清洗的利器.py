#!/usr/bin/python3
# coding: utf-8

# http://blog.csdn.net/CoderPai/article/details/78574863
# https://github.com/vi3k6i5/flashtext

# 关键词搜索
from flashtext import KeywordProcessor
keyword_processor = KeywordProcessor()
# keyword_processor.add_keyword(<别名>, <标准词>)
keyword_processor.add_keyword('广东', '广东省')
keyword_processor.add_keyword('北京')
keywords_found = keyword_processor.extract_keywords('北京与广东都是中国的')
print(keywords_found)
# ['北京', '广东省']

# 关键词替换
keyword_processor.add_keyword('A卡', '邂逅a卡')
new_sentence = keyword_processor.replace_keywords('a卡与b卡哪个卡好')
print(new_sentence)
# Out[40]: '邂逅a卡与b卡哪个卡好'


# 提取关键词，区分大小写字母
keyword_processor = KeywordProcessor(case_sensitive=True)
keyword_processor.add_keyword('A卡', '邂逅a卡')
keyword_processor.add_keyword('b卡')
keywords_found = keyword_processor.extract_keywords('a卡与b卡哪个卡好？')
print(keywords_found)
# ['b卡']

# 提取关键词，并输出关键词的位置
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('Big Apple', 'New York')
keyword_processor.add_keyword('Bay Area')
keywords_found = keyword_processor.extract_keywords('I love big Apple and Bay Area.', span_info=True)
print(keywords_found)
# Out[5]: [('New York', 7, 16), ('Bay Area', 21, 29)]
# 'I love big Apple and Bay Area.'[7:16]
# Out[6]: 'big Apple'
# 'I love big Apple and Bay Area.'[21:29]
# Out[7]: 'Bay Area'

# 获取关键字提取的额外信息（词性标注）
kp = KeywordProcessor()
kp.add_keyword('红苹果', ('水果', '红苹果'))
kp.add_keyword('北京', ('地名', '北京'))
kp.extract_keywords('你北京的红苹果好吃吗')
# Out[50]: [('地名', '北京'), ('水果', '红苹果')]

# 注意：关键词替换功能将无法正常工作，如下
kp = KeywordProcessor()
kp.add_keyword('苹果', ('水果', '红苹果'))
kp.add_keyword('北京市', ('地名', '北京'))
kp.extract_keywords('你北京的红苹果好吃吗')
# Out[49]: [('水果', '红苹果')]

# 关键字不清晰(可以匹配上多一个标点符号的情况)
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('Big Apple')
keyword_processor.add_keyword('Bay Area')
keywords_found = keyword_processor.extract_keywords('I love big Apple and Bay Area.')
print(keywords_found)
# ['Big Apple', 'Bay Area']

# 同时添加多个关键词
keyword_processor = KeywordProcessor()
# 字典格式的关键词，其对应的key为最终匹配出的词，但key不记入关键词搜索的范围
keyword_dict = {
    "java": ["java_2e", "java programing"],
    "product management": ["PM", "product manager"]
}
# {'clean_name': ['list of unclean names']}
keyword_processor.add_keywords_from_dict(keyword_dict)
# Or add keywords from a list:
keyword_processor.add_keywords_from_list(["java", "python"])
keyword_processor.extract_keywords('I am a product manager for a java_2e platform')
# output ['product management', 'java']

# 删除关键词
keyword_processor = KeywordProcessor()
keyword_dict = {
    "java": ["java_2e", "java programing"],
    "product management": ["PM", "product manager"]
}
keyword_processor.add_keywords_from_dict(keyword_dict)
print(keyword_processor.extract_keywords('I am a product manager for a java_2e platform'))
# output ['product management', 'java']
keyword_processor.remove_keyword('java_2e')
print(keyword_processor.extract_keywords('I am a product manager for a java_2e platform'))
# ['product management']

# you can also remove keywords from a list/ dictionary
keyword_processor.remove_keywords_from_dict({"product management": ["PM"]})
keyword_processor.remove_keywords_from_list(["java programing"])
keyword_processor.extract_keywords('I am a product manager for a java_2e platform')
# output ['product management']

# 查询添加关键词的个数
keyword_processor = KeywordProcessor()
# 字典格式的关键词，其对应的key为最终匹配出的词，但key不记入关键词搜索的范围
keyword_dict = {
    "java": ["java_2e", "java programing"],
    "product management": ["PM", "product manager"]
}
keyword_processor.add_keywords_from_dict(keyword_dict)
print(len(keyword_processor))
# output 4

# 检查关键词是否已经添加
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('j2ee', 'Java')
print('j2ee' in keyword_processor)
# output: True

# 获取某个词的标准词
keyword_processor.get_keyword('j2ee')
# output: Java
keyword_processor['colour'] = 'color'
print(keyword_processor['colour'])
# output: color
keyword_processor.get_keyword('colour')
# Out[31]: 'color'

# 获取字典中的所有关键词
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('j2ee', 'Java')
keyword_processor.add_keyword('colour', 'color')
keyword_processor.get_all_keywords()
# output: {'colour': 'color', 'j2ee': 'Java'}

# 除\w [A-Za-z0-9_]之外的任何字符，都认为是一个单词的边界
keyword_processor = KeywordProcessor()
keyword_processor.add_keyword('Big Apple')
print(keyword_processor.extract_keywords('I love Big Apple/Bay Area.'))
# ['Big Apple']
print(keyword_processor.extract_keywords('I love Big Apple_Bay Area.'))
# []
print(keyword_processor.extract_keywords('I love Big Apple2Bay Area.'))
# []

# 设置或添加字符作为单词字符的一部分
keyword_processor.add_non_word_boundary('/')
print(keyword_processor.extract_keywords('I love Big Apple/Bay Area.'))
# []


def main():
    pass


if __name__ == '__main__':
    main()