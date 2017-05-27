#! /usr/lib/python3
# -*- coding: utf-8 -*-

# pip3 install pynlpir 安装


In[53]: import pynlpir
In[54]: pynlpir.open()
[2016-03-31 18:22:13] Cannot open file /usr/local/lib/python3.5/dist-packages/pynlpir/Data/NewWord.lst
Cannot write log file /usr/local/lib/python3.5/dist-packages/pynlpir/Data/20160331.err!

 重新下载
 替换/usr/local/lib/python3.5/dist-packages/pynlpir/Data/NewWord.lst文件即可。
 
 若报错：RuntimeError: NLPIR function 'NLPIR_Init' failed.
 说明授权文件过期了，重新下载授权文件NLPIR.user，替换/usr/local/lib/python2.7/dist-packages/pynlpir/Data/NLPIR.user即可；
 下载地址：
 https://github.com/NLPIR-team/NLPIR/blob/master/License/license%20for%20a%20month/NLPIR-ICTCLAS%E5%88%86%E8%AF%8D%E7%B3%BB%E7%BB%9F%E6%8E%88%E6%9D%83/NLPIR.user
 
 
In[55]: import pynlpir
In[56]: pynlpir.open()
In[57]: s = '欢迎科研人员、技术工程师、企事业单位与个人参与NLPIR平台的建设工作。'
In[58]: pynlpir.segment(s)

Out[58]:
[('欢迎', 'verb'),
 ('科研', 'noun'),
 ('人员', 'noun'),
 ('、', 'punctuation mark'),
 ('技术', 'noun'),
 ('工程师', 'noun'),
 ('、', 'punctuation mark'),
 ('企事业', 'noun'),
 ('单位', 'noun'),
 ('与', 'conjunction'),
 ('个人', 'noun'),
 ('参与', 'verb'),
 ('NLPIR', 'noun'),
 ('平台', 'noun'),
 ('的', 'particle'),
 ('建设', 'verb'),
 ('工作', 'verb'),
 ('。', 'punctuation mark')]
In[59]: pynlpir.segment(s, pos_tagging=False)
Out[59]:
['欢迎',
 '科研',
 '人员',
 '、',
 '技术',
 '工程师',
 '、',
 '企事业',
 '单位',
 '与',
 '个人',
 '参与',
 'NLPIR',
 '平台',
 '的',
 '建设',
 '工作',
 '。']
In[60]: pynlpir.segment(s, pos_names='child')
Out[60]:
[('欢迎', 'verb'),
 ('科研', 'noun'),
 ('人员', 'noun'),
 ('、', 'enumeration comma'),
 ('技术', 'noun'),
 ('工程师', 'noun'),
 ('、', 'enumeration comma'),
 ('企事业', 'noun'),
 ('单位', 'noun'),
 ('与', 'coordinating conjunction'),
 ('个人', 'noun'),
 ('参与', 'verb'),
 ('NLPIR', 'noun'),
 ('平台', 'noun'),
 ('的', 'particle 的/底'),
 ('建设', 'noun-verb'),
 ('工作', 'noun-verb'),
 ('。', 'period')]
In[61]: pynlpir.segment(s, pos_names='all')
Out[61]:
[('欢迎', 'verb'),
 ('科研', 'noun'),
 ('人员', 'noun'),
 ('、', 'punctuation mark:enumeration comma'),
 ('技术', 'noun'),
 ('工程师', 'noun'),
 ('、', 'punctuation mark:enumeration comma'),
 ('企事业', 'noun'),
 ('单位', 'noun'),
 ('与', 'conjunction:coordinating conjunction'),
 ('个人', 'noun'),
 ('参与', 'verb'),
 ('NLPIR', 'noun'),
 ('平台', 'noun'),
 ('的', 'particle:particle 的/底'),
 ('建设', 'verb:noun-verb'),
 ('工作', 'verb:noun-verb'),
 ('。', 'punctuation mark:period')]
In[62]: pynlpir.segment(s, pos_english=False)
Out[62]:
[('欢迎', '动词'),
 ('科研', '名词'),
 ('人员', '名词'),
 ('、', '标点符号'),
 ('技术', '名词'),
 ('工程师', '名词'),
 ('、', '标点符号'),
 ('企事业', '名词'),
 ('单位', '名词'),
 ('与', '连词'),
 ('个人', '名词'),
 ('参与', '动词'),
 ('NLPIR', '名词'),
 ('平台', '名词'),
 ('的', '助词'),
 ('建设', '动词'),
 ('工作', '动词'),
 ('。', '标点符号')]
In[63]: pynlpir.get_key_words(s, weighted=True)
Out[63]:
[('NLPIR', 2.4),
 ('欢迎', 2.0),
 ('科研', 2.0),
 ('人员', 2.0),
 ('技术', 2.0),
 ('工程师', 2.0),
 ('企事业', 2.0),
 ('单位', 2.0),
 ('个人', 2.0),
 ('参与', 2.0),
 ('平台', 2.0),
 ('建设', 2.0),
 ('工作', 2.0)]
 
 In [10]: pynlpir.segment("我在北京天安门", pos_names=None)
Out[10]: [('我', 'rr'), ('在', 'p'), ('北京', 'ns'), ('天安门', 'ns')]

In[64]: pynlpir.close()
