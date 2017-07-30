#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys, os, json, re
from pprint import pprint

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle

    unicode = str
else:
    import cPickle as pickle
    from codecs import open

from logger.logger import logger


# cur_dir = os.getcwd() #os.getcwd()：获取当前工作目录，也就是在哪个目录下运行这个程序。
# cur_path = os.path.join(cur_dir, __file__)
# PATH = os.path.dirname(cur_path)

# https://github.com/kmike/marisa-trie
# Trie树是一种树的数据结构，又被称为字典树，非常适用于Ajax自动补全等场景，因为它通过空间换时间能极大提高特别字符串的查询速度。

class TrieTree(object):
    def __init__(self, tree=None, input_tries=None):
        """参数tree，从字典中导入tries树；
        参数input_tries, 可以是文件，但只限于json文件（tries树结构的dict，或者list）
        或者tries树的dict数据
        或者list数据"""
        # logger.info("初始化tries树")
        if tree is None:
            tree = {}
            self.tree = tree
        if not input_tries:
            pass
        elif isinstance(input_tries, (unicode, str)) and input_tries.startswith("NER_"):
            pass
        elif isinstance(input_tries, (unicode, str)) and input_tries.startswith("RE_PATTERN_"):
            pass
        elif isinstance(input_tries, (unicode, str)) and os.path.isfile(input_tries):
            with open(input_tries, encoding='utf8')as f:
                logger.info("加载文件：{}".format(input_tries))
                tries = json.load(f)
            if isinstance(tries, dict):
                tree = tries
                self.tree = tree
            elif isinstance(tries, (list, set)):
                for word in tries:
                    self.add(word)
            else:
                logger.info("输入数据文件有误")
        elif isinstance(input_tries, dict):
            tree = input_tries
            self.tree = tree
        elif isinstance(input_tries, (list, set)):
            for word in input_tries:
                self.add(word)
        else:
            logger.info("输入参数有误:{}".format(input_tries))

    def add(self, word):
        tree = self.tree

        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                tree[char] = {}
                tree = tree[char]

        tree['exist'] = True

    def add_mark(self, word, mark=True):
        """向tries树中添加词语，并用mark标示"""
        tree = self.tree

        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                tree[char] = {}
                tree = tree[char]

        tree['exist'] = mark

    def get_words(self, word):
        """提取字符串中的子串"""
        data = [word]
        words = word.split('-')  # '多瑞奶奶讲故事-格林童话《白雪公主》'
        for line in words:
            ts = re.findall('[《【（](.*?)[》】）]', line)
            for t in ts:
                data.append(t)
            ts = re.search('(.*)[《【（].*?[》】）](.*)', line)
            if ts:
                for t in ts.groups():
                    if t:
                        data.append(t)
        return list(set(data))

    def gen_trie(self, input_file, output_file='', sep=''):
        """将输入的文件list转换成tries树dict"""
        # trie = self.tree
        with open(input_file) as f:
            data = json.load(f)
        for line in data:
            if not line or not line.strip():
                continue
            line = line.strip()
            # line='多瑞奶奶讲故事-格林童话《白雪公主》'
            # line='【幼师讲故事】白雪公主-格林童话故事（全）'
            if sep:
                for l in line.split(sep):
                    words = self.get_words(l)
                    for word in words:
                        self.add(word)
            else:
                words = self.get_words(line)
                for word in words:
                    self.add(word)

        if output_file:
            with open(output_file, 'w', encoding='utf8') as f:
                json.dump(self.tree, f, ensure_ascii=0)

    def search(self, word):
        """查询单词是否是tries树中收录的词，若是返回真，否则返回假"""
        tree = self.tree

        for char in word:
            if char in tree:
                tree = tree[char]
            else:
                return False

        if "exist" in tree and tree["exist"] == True:
            return True
        else:
            return False

    def search_match(self, word, num=None):
        """查询词开始部分是否是tries树中收录的词，若是返回真，否则返回假
        若num为真，则至少要匹配num个字符，才为真 """
        tree = self.tree

        for char in word:
            if tree.get(char):
                tree = tree[char]
                if tree.get("exist"):
                    if (num is not None) and (word.find(char) + 1 < num):
                        continue
                    return True
            else:
                return False
        return False

    def search_match_mark(self, word, num=None, mark=True):
        """查询词开始部分是否是tries树中收录的词，若是返回标示mark，否则返回假
        若num为真，则至少要匹配num个字符，才返回标示mark """
        tree = self.tree

        for char in word:
            if tree.get(char):
                tree = tree[char]
                if tree.get("exist"):
                    if (num is not None) and (word.find(char) + 1 < num):
                        continue
                    return tree.get("exist")
            else:
                return False
        return False

    def search_word(self, word, num=None):
        """查询词开始部分是否是tries树中收录的词，若是返回查询到的，否则返回空值
        若num为真，则至少要匹配num个字符，才返回 """
        tree = self.tree
        new_words = ''  # 最后返回的单词
        count = ''  # 用于统计的
        for char in word:
            if tree.get(char):
                count += char
                tree = tree[char]
                if tree.get("exist"):
                    if (num is not None) and (word.find(char) + 1 < num):
                        continue
                    new_words = count
            else:
                return new_words
        return new_words

    def search_all_word(self, word, num=None, out_list=False):
        """查询词中是否有tries树中收录的词，若是返回查询到的，否则返回空值
        若num为真，则至少要匹配num个字符，才返回 """
        # tree = self.tree
        # new_words = '' # 最后返回的单词
        # count = '' # 用于统计的
        # for char in word:
        #     if tree.get(char):
        #         count += char
        #         tree = tree[char]
        #         # print(count)
        #         if tree.get("exist"):
        #             if (num is not None) and (word.find(char)+1 < num):
        #                 continue
        #             new_words = count
        #     else:
        #         if new_words:
        #             return new_words
        # return new_words
        results = []
        for index in range(len(word)):
            w = word[index:]
            result = self.search_word(w, num=num)
            if out_list:
                if result:
                    results.append(result)
            else:
                if result:
                    return result
        if out_list:
            return results
        return ''


def test1():
    tree = TrieTree()
    tree.add("abc")
    tree.add("abcefg")
    tree.add("bcd")
    pprint(tree.tree)
    # Print {'a': {'b': {'c': {'exist': True}}}, 'b': {'c': {'d': {'exist': True}}}}
    print(tree.search_match("ab"))
    # Print False
    print(tree.search_match("abc"))
    # Print True
    print(tree.search_match("abcd"))
    # Print False
    print(tree.search_match("dabc"))
    # Print False
    print(tree.search_match("dabec"))
    # print False


def test3():
    RELATIONSHIPS_FILE = '/home/gswyhq/x/data/initiative/relationships.json'
    tree = TrieTree(input_tries=RELATIONSHIPS_FILE)
    # tree = TrieTree(input_tries=['友人','好友','爸爸','朋友', "我妈"])
    # pprint(json.loads(json.dumps(tree.tree,ensure_ascii=0)))
    words = ["朋友", "友人", '爸爸', '你好友人', '我是他朋友']

    for w in words:
        print(w, tree.search_all_word(w))


if __name__ == "__main__":
    # test2()
    # main()
    test3()
