#!/usr/bin/python
# -*- coding:utf8 -*- #
# 10 行 Python 代码实现模糊查询
# http://my.oschina.net/leejun2005/blog/486697

from __future__ import  generators
from __future__ import  division
from __future__ import  print_function
from __future__ import  unicode_literals
import sys,os,json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

# pip install fuzzyfinder
# 当我第一次考虑用Python实现“fuzzy matching”的时候，我就知道一个叫做 fuzzywuzzy 的优秀库，但是 fuzzywuzzy 的做法和这里的不太一样，它使用的是 “levenshtein distance” 来从集合中找到最匹配的字符串。”levenshtein distance“是一个非常适合用来做自动更正拼写错误的技术，但在从部分子串匹配长文件名时表现的不太好(所以这里没有使用)。

collection = ['django_migrations.py',
                'django_admin_log.py',
                'main_generator.py',
                'migrations.py',
                'api_user.doc',
                'user_group.doc',
                'accounts.txt',
                ]

# 当用户输入’djm‘字符串时，我们假定是匹配到’django_migrations.py’和’django_admin_log.py’，而最简单的实现方法就是使用正则表达式。
#3、解决方案：
#3.1 常规的正则匹配
#将 "djm" 转换成 "d.*j.*m" 然后用这个正则尝试匹配集合中的每一个字符串，如果匹配到了就被列为候选。

import re
def fuzzyfinder1(user_input, collection):
        suggestions = []
        pattern = '.*'.join(user_input) # Converts 'djm' to 'd.*j.*m'
        regex = re.compile(pattern)     # Compiles a regex.
        for item in collection:
            match = regex.search(item)  # Checks if the current item matches the regex.
            if match:
                suggestions.append(item)
        return suggestions

print (fuzzyfinder1('djm', collection))

print (fuzzyfinder1('mig', collection))



#3.2 带有rank排序的匹配列表
#这里我们对匹配到的结果按照匹配内容第一次出现的起始位置来进行排序。

def fuzzyfinder2(user_input, collection):
        suggestions = []
        pattern = '.*'.join(user_input) # Converts 'djm' to 'd.*j.*m'
        regex = re.compile(pattern)     # Compiles a regex.
        for item in collection:
            match = regex.search(item)  # Checks if the current item matches the regex.
            if match:
                suggestions.append((match.start(), item))
        return [x for _, x in sorted(suggestions)]

print (fuzzyfinder2('mig', collection))

#这次我们生成了一个由二元 tuple 组成的列表，即列表中的每一个元素为一个二元tuple，而该二元tuple的第一个值为匹配到的起始位置、第二个值为对应的文件名，然后使用列表推导式按照匹配到的位置进行排序并返回文件名列表。
#现在我们已经很接近最终的结果了，但还称不上完美——用户想要的是’migration.py’，但我们却把’main_generator.py’作为第一推荐。

#3.3 根据匹配的紧凑程度进行排序

#当用户开始输入一个字符串时，他们倾向于输入连续的字符以进行精确匹配。比如当用户输入’mig‘他们更倾向于找的是’migrations.py’或’django_migrations.py’，而不是’main_generator.py’，所以这里我们所做的改变就是查找匹配到的最紧凑的项目。
#刚才提到的问题对于Python来说不算什么事，因为当我们使用正则表达式进行字符串匹配时，匹配到的字符串就已经被存放在了match.group()中了。下面假设输入为’mig’，对最初定义的’collection’的匹配结果如下：

#这里我们将推荐列表做成了三元tuple的列表的形式，即推荐列表中的每一个元素为一个三元tuple，而该三元tuple的第一个值为匹配到的内容的长度、第二个值为匹配到的起始位置、第三个值为对应的文件名，然后按照匹配长度和起始位置进行排序并返回。

def fuzzyfinder3(user_input, collection):
        suggestions = []
        pattern = '.*'.join(user_input) # Converts 'djm' to 'd.*j.*m'
        regex = re.compile(pattern)     # Compiles a regex.
        for item in collection:
            match = regex.search(item)  # Checks if the current item matches the regex.
            if match:
                suggestions.append((len(match.group()), match.start(), item))
        return [x for _, _, x in sorted(suggestions)]

print (fuzzyfinder3('mig', collection))


#3.4 非贪婪匹配
def fuzzyfinder4(user_input, collection):
    # 对匹配到的结果按照匹配内容第一次出现的起始位置来进行排序。
    suggestions = []
    pattern = '.*?'.join(user_input)    #  'djm' 转换成 'd.*?j.*?m'；然后用这个正则尝试匹配集合中的每一个字符串，如果匹配到了就被列为候选。
    regex = re.compile(pattern)         # Compiles a regex.
    for item in collection:
        match = regex.search(item)      # Checks if the current item matches the regex.
        if match:
            suggestions.append((len(match.group()), match.start(), item))
    # 将推荐列表做成了三元tuple的列表的形式，即推荐列表中的每一个元素为一个三元tuple，而该三元tuple的第一个值为匹配到的内容的长度、第二个值为匹配到的起始位置、第三个值为对应的文件名，然后按照匹配长度和起始位置进行排序并返回。
    return [x for _, _, x in sorted(suggestions)]

print (fuzzyfinder4('mig', collection))

def fuzzyfinder(text, collection):
    """
    Args:
        text (str): A partial string which is typically entered by a user.
        collection (iterable): A collection of strings which will be filtered
                               based on the input `text`.
    Returns:
        suggestions (generator): A generator object that produces a list of
            suggestions narrowed down from `collections` using the `text`
            input.
    """
    suggestions = []
    text = str(text) if not isinstance(text, str) else text
    pat = '.*?'.join(map(re.escape, text))
    regex = re.compile(pat)
    for item in collection:
        r = regex.search(item)
        if r:
            suggestions.append((len(r.group()), r.start(), item))

    return (z for _, _, z in sorted(suggestions))


if __name__ == "__main__":
    main()
