#!/usr/bin/python
# -*- coding:utf8 -*- #
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

>>> from pypinyin import pinyin, lazy_pinyin
>>> import pypinyin
>>> pinyin(u'中心')
[[u'zh\u014dng'], [u'x\u012bn']]
>>> pinyin(u'中心', heteronym=True)  # 启用多音字模式
[[u'zh\u014dng', u'zh\xf2ng'], [u'x\u012bn']]
>>> pinyin(u'中心', style=pypinyin.FIRST_LETTER)  # 设置拼音风格
[['z'], ['x']]
>>> pinyin('中心', style=pypinyin.TONE2, heteronym=True)
[['zho1ng', 'zho4ng'], ['xi1n']]
>>> lazy_pinyin(u'中心')  # 不考虑多音字的情况
['zhong', 'xin']
命令行工具
$ pypinyin 音乐
yīn yuè
$ pypinyin -h
处理不包含拼音的字符
当程序遇到不包含拼音的字符(串)时，会根据 errors 参数的值做相应的处理:

default (默认行为): 不做任何处理，原样返回:

lazy_pinyin(u'你好☆')
[u'ni', u'hao', u'\u2606']
ignore : 忽略该字符

lazy_pinyin(u'你好☆', errors='ignore')
[u'ni', u'hao']
replace : 替换为去掉 \u 的 unicode 编码:

lazy_pinyin(u'你好☆', errors='replace')
[u'ni', u'hao', u'2606']
callable 对象 : 提供一个回调函数，接受无拼音字符(串)作为参数, 支持的返回值类型: unicode 或 list ([unicode, ...]) 或 None 。

可参考 单元测试代码

lazy_pinyin(u'你好☆', errors=lambda x: u'star')
[u'ni', u'hao', u'star']
分词处理(用于处理多音字和非中文字符)
内置了简单的分词功能，对字符串按是否是中文字符进行分词。

>> from pypinyin import lazy_pinyin
>> lazy_pinyin(u'你好abcこんにちは')
[u'ni', u'hao', u'abc\u3053\u3093\u306b\u3061\u306f']
如果需要处理多音字问题，推荐同时安装其他分词模块。

如果安装了 jieba 分词模块，程序会自动调用，
也可以使用经过 jieba 分词处理的 字符串列表 作参数。
使用其他分词模块：

安装分词模块，比如 pip install snownlp ；

使用经过分词处理的 字符串列表 作参数：

>> from pypinyin import lazy_pinyin, TONE2
>> from snownlp import SnowNLP
>> hans = u'音乐123'
>> hans_seg = SnowNLP(hans).words  # 分词处理
>> hans_seg
[u'\u97f3\u4e50', u'123']
>> lazy_pinyin(hans_seg, style=TONE2)
[u'yi1n', u'yue4', u'123']
自定义拼音库
如果对结果不满意，可以通过 load_single_dict() 或 load_phrases_dict() 以自定义拼音库的方式修正结果：

安装了 jieba 分词模块并且支持分词的词组

>> from pypinyin import lazy_pinyin, load_phrases_dict, TONE2
>> hans = u'桔子'
>> lazy_pinyin(hans, style=TONE2)
[u'jie2', u'zi3']
>> load_phrases_dict({u'桔子': [[u'jú'], [u'zǐ']]})
>> lazy_pinyin(hans, style=TONE2)
[u'ju2', u'zi3']
未安装 jieba 分词模块 and/or 不支持分词的词组

>> from pypinyin import lazy_pinyin, load_phrases_dict, TONE2, load_single_dict
>> hans = u'还没'
>> lazy_pinyin(hans, style=TONE2)
['hua2n', 'me2i']
>>>  # 第一种自定义词组的方法
>> load_phrases_dict({u'还没': [[u'hái'], [u'méi']]})
>>> lazy_pinyin(u'还没', style=TONE2)})
['hua2n', 'me2i']
>>> lazy_pinyin([u'还没'], style=TONE2)  # 手动指定 "还没" 为一个词组
['ha2i', 'me2i']
>>>  # 第二种自定义词组的方法
>> load_single_dict({ord(u'还'): u'hái,huán'})  # 调整 "还" 字的拼音顺序
>>> lazy_pinyin(u'还没', style=TONE2)
['ha2i', 'me2i']

In [21]: pinyin('中心')
Building prefix dict from the default dictionary ...
Dumping model to file cache /tmp/jieba.cache
Loading model cost 0.800 seconds.
Prefix dict has been built succesfully.
Out[21]: [['zhōng'], ['xīn']]

In [22]: pinyin('中心',heteronym=1)
Out[22]: [['zhōng', 'zhòng'], ['xīn']]

In [24]: import pypinyin

In [25]: pinyin('中心',style=pypinyin.FIRST_LETTER)
Out[25]: [['z'], ['x']]

In [26]: pinyin('中心',style=pypinyin.TONE2,heteronym=True)
Out[26]: [['zho1ng', 'zho4ng'], ['xi1n']]

In [27]: lazy_pinyin('中心')
Out[27]: ['zhong', 'xin']

In [28]: lazy_pinyin('你好☆')
Out[28]: ['ni', 'hao', '☆']

In [29]: lazy_pinyin('你好☆',errors='ignore')
Out[29]: ['ni', 'hao']

In [30]: lazy_pinyin('你好☆',errors='replace')
Out[30]: ['ni', 'hao', '2606']

In [31]: lazy_pinyin('你好☆',errors=lambda x:'gswyhq')
Out[31]: ['ni', 'hao', 'gswyhq']

In [32]: lazy_pinyin('你好abcこんにちは')
Out[32]: ['ni', 'hao', 'abcこんにちは']

def pinyin_sorted(iterable, reverse=False):
    """对中文按拼音进行排序
    """
    pinyin_iterable = [(word, ''.join(lazy_pinyin(word))) for word in iterable]
    return [word for word, _ in sorted(pinyin_iterable, key=lambda x:x[1], reverse=reverse)]


