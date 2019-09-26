#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys
import time

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if PY3:
    import pickle
else:
    import cPickle as pickle
    from codecs import open

def timing(f):
    """函数运行时间的计时器"""
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print ('<函数名称: {0}>[运行耗时: {1} ms]'.format(f.__name__,(time2-time1)*1000.0))
        return ret
    return wrap

class StringPatternt(object):
    def __init__(self,chr,p):
        self.chr = chr
        self.p = p
        self.p_len = len(p)
        self.pi = [0 for i in range(self.p_len)]
    def set_pattern(self,p):
        self.p = p
        self.p_len = len(p)
    def set_chr(self,chr):
        self.chr = chr

    '''KMP
    在此算法中当从前往后搜索时遇到第一个不匹配：A->D时，它将从搜索字符串入手决定移动多少位。在KMP算法的初始阶段会生成一张表，
    例如，上面搜索字符串生成的表为：pi[0,...,4] = {0,0,0,0,1}。这张表决定上面提到的移位。此时的移位为：3-pi[2]
    （因为已经匹配了三个字符）。KMP算法的关键就是：匹配字符的初始生成表，而且是从前往后进行搜索。'''
    def __kmp_partial_match_table__(self):
        """生成初始的移动位数表"""
        k=0
        q = 1
        #self.pi[0] = 0
        while q < self.p_len:
            while (k > 0 and self.p[k] != self.p[q]):
                k = self.pi[k-1]
            if self.p[k] == self.p[q]:
                k = k+1
            self.pi[q] = k
            q = q+1
        return 0

    def string_pattern_kmp(self):
        self.__kmp_partial_match_table__()
        #生成初始的移动位数表
        print(self.pi)
        list_size = len(self.chr)
        pi_len = len(self.pi)
        k=0
        for q in range(list_size):
            while (k > 0) and (self.p[k] != self.chr[q]):
                k = self.pi[k-1]
            if self.p[k] == self.chr[q]:
                k = k+1
            if k == pi_len:
                return q-pi_len+1
            #q = q+1
        return 0

    '''BM
    在此算法中，字符串搜索不是从头开始的，而是从末尾开始的，例如上面的例子中，首先比较的是D和A，因为不相同，
    则在搜索字符中从后往前进行匹配查找，找到最右边的匹配字符后进行移位，如果找不到的话移位长度与匹配字符一样长，如下：
字符串：      ABCADAB ABCDABCDABD
搜索字符串：   ABCDA(移两位)
此时继续进行比较，不过此时的比较要考虑两方面以上上面提到的过程，还有一种情况就是已匹配的字符串中(DA)，包含了搜索字符串的前缀(A)，
我们知道此时移动1~3位是没有意义的。所以BM算法的关键就是找到两种移位中的最大移位，进行以为。
'''
    def __calc_match__(self,num):
        k=num
        j=0
        while k>=0:
            if self.p[-k] == self.p[j]:
                k = k-1
                j=j+1
                if k<=0:
                    self.pi[num-1] = num
                    return 0
            else:
                if num == 1:
                    return 0
                self.pi[num-1] = self.pi[num-2]
                return 0

    def __init_good_table__(self):
        i=1
        while i <= self.p_len:
            self.__calc_match__(i)
            i=i+1
        print (self.pi)
        return 0

    def __check_bad_table__(self,tmp_chr):
        i=1
        while self.p_len-i >= 0:
            if self.p[-i] == tmp_chr:
                return i
            else:
                i = i+1
        return self.p_len+1

    def __check_good_table__(self,num):
        if not num:
            return self.p_len
        else:
            return self.pi[num]

    def string_pettern_bm(self):
        self.__init_good_table__()
        tmp_len = self.p_len
        i = 1
        while tmp_len <= len(self.chr):
            if self.p[-i]==self.chr[tmp_len-i]:
                i = i+1
                if i > self.p_len:
                    return tmp_len-self.p_len
            else:
                tmp_bad = self.__check_bad_table__(self.chr[tmp_len-i])-i
                tmp_good= self.p_len-self.__check_good_table__(i-1)
                tmp_len = tmp_len+ max(tmp_bad,tmp_good)
                print(tmp_bad,tmp_good,tmp_len)
                i=1
        return 0

    '''sunday
    上面的两种字符串匹配算法都涉及到了对搜索字符的预处理，但Sunday算法预期完全不同。同样是上面的例子当搜到不匹配的字符串时，
    Sunday算法采用了一种完全不同的以为确定法。它会先找到字符串的第K+1个字符，K是搜索字符的长度。如果搜索字符串中不包含字符串中第K+1个字符，
    则直接移动K+1位。否则，按着BM算法移动搜索串中最右端的该字符到末尾的距离+1位。'''
    def __check_bad_shift__(self,p):
        i=0
        while i<self.p_len:
            if self.p[i] == p:
                return i
            else:
                i = i+1
        return -1

    def string_pattern(self):
        #self.__init_good_table__()
        tmp_len = 0
        tmp_hop = self.p_len
        i=0
        while tmp_hop <= len(self.chr):
            if self.p[i] == self.chr[tmp_len+i]:
                i = i+1
                if i == self.p_len:
                    return tmp_len
            else:
                tmp_len = tmp_len+self.p_len-self.__check_bad_shift__(self.chr[tmp_hop])
                tmp_hop = tmp_len+self.p_len
                i=0
        return 0

@timing
def pre_process(dst):
    next = [0]
    cur_next = 0
    for i in range(1,len(dst)):
        while cur_next != 0 and dst[i] != dst[cur_next] :
            cur_next = next[cur_next - 1]
        if dst[i] == dst[cur_next] :
            cur_next += 1
        next.append(cur_next)
    # print(next)
    return next

@timing
def SetPrefix( pattern):
    """计算一个字符串的前缀next数组
    前缀数组，也有的叫next数组，每一个子串有一个固定的next数组，它记录着字符串匹配过程中失配情况下可以向前多跳几个字符，
    当然它描述的也是子串的对称程度，程度越高，值越大，当然之前可能出现再匹配的机会就更大。"""
    prefix = [0]
    for i in range(1,len(pattern)):
        k=prefix[i-1] # 前面的字符对称程度是几，就跟pattern中的第几个字符比较；比如，前面的对称程度是0，则与第0个字符比较

        #不断递归判断是否存在子对称，k=0说明不再有子对称，Pattern[i] != Pattern[k]说明虽然对称，但是对称后面的值和当前的字符值不相等，所以继续递推
        while(  k!=0 and pattern[i] != pattern[k]  ) :
            k=prefix[k-1]     #继续递归
        if( pattern[i] == pattern[k]): #找到了这个子对称，或者是直接继承了前面的对称性，这两种都在前面的基础上++
            prefix.append(k+1)
        else:
            prefix.append(0)  #如果遍历了所有子对称都无效，说明这个新字符不具有对称性，清0
    return prefix

def kmp(dst, pattern):

    next = pre_process(dst)
    next = SetPrefix(dst)

    already_match = 0 # 已匹配的字符
    for i in range(0, len(dst)):
        # 当已匹配字符串不为0，或匹配不成功时，考虑在搜索字符串中移位
        while already_match != 0 and dst[i] != pattern[already_match]:
            # 获取在搜索字符串中移动的位数
            already_match = next[already_match]
        # 若匹配上了，就把匹配的字符加1
        if dst[i] == pattern[already_match ]:
            already_match += 1

        # 若已匹配的字符串等于要查找的字符串长度，代表匹配成功，返回匹配到的位置
        if already_match == len(pattern):
            # print ('Matched index is :%d'%(i - already_match + 1))
            # already_match = next[already_match - 1]
            return (i - already_match + 1)
    return 0


def print_pattern(pattern):
    print('[',end='')
    for t in pattern:
        print(t,end=', ')
    print(']')

    print(SetPrefix(pattern))

def find(pattern,test=''):
    """从字符串test中查找字串pattern"""

    prefix = SetPrefix(pattern)

def main():
    # pattern = 'agctagcagctagctg'
    # print_pattern(pattern)
    test = 'babcbabcabcaabcabcabcacabc'
    pattern = 'abcabcacab'

    # test = 'ABCADABABCDABCDABD'
    # pattern = 'ABCDA'

    print_pattern(test)
    s = StringPatternt(pattern,test)
    print(s.string_pattern_kmp())

    print("'{}' 在 '{}' 中的匹配位置是：{}".format(pattern,test,kmp(test, pattern)))
if __name__ == "__main__":
    main()
