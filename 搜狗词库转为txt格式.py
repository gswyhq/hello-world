#/usr/lib/python3.5
# -*- coding: utf-8 -*-

import struct
import os, sys

def read_utf16_str (f, offset=-1, len=2):
    if offset >= 0:
        f.seek(offset)
    str = f.read(len)
    return str.decode('UTF-16LE')

def read_uint16 (f):
    return struct.unpack ('<H', f.read(2))[0]



def get_word_from_sogou_cell_dict (fname):
    f = open (fname, 'rb')
    file_size = os.path.getsize (fname)

    hz_offset = 0
    #mask = struct.unpack ('B', f.read(128)[4])[0]
    mask=f.read(128)[4]
    if mask == 0x44:
        hz_offset = 0x2628#9768
    elif mask == 0x45:
        hz_offset = 0x26c4#9924
    else:
        sys.exit(1)

    title   = read_utf16_str (f, 0x130, 0x338  - 0x130) #词库名 304-824
    types    = read_utf16_str (f, 0x338, 0x540  - 0x338) #词库类型 824-1344
    desc    = read_utf16_str (f, 0x540, 0xd40  - 0x540) #词库描述 1344-3392
    samples = read_utf16_str (f, 0xd40, 0x1540 - 0xd40) #词库示例 3392-5440
    data={}
    data['title']=title.replace('\x00','')
    data['types']=types.replace('\x00','')
    data['desc']=desc.replace('\x00','')
    data['samples']=samples.replace('\x00','')
    py_map = {}
    f.seek(0x1540+4) #5444

    while 1:
        py_code = read_uint16 (f)
        py_len  = read_uint16 (f)
        py_str  = read_utf16_str (f, -1, py_len)

        if py_code not in py_map:
            py_map[py_code] = py_str

        if py_str == 'zuo':
            break

    f.seek(hz_offset)
    #content=''
    while f.tell() != file_size:
        word_count   = read_uint16 (f)
        pinyin_count = read_uint16 (f) / 2

        py_set = []
        for i in range(int(pinyin_count)):
            py_id = read_uint16(f)
            try:
                py_set.append(py_map[py_id])
            except KeyError:
                pass
        py_str = "'".join (py_set)

        for i in range(word_count):
            word_len = read_uint16(f)
            word_str = read_utf16_str (f, -1, word_len)
            f.read(12)
            yield py_str, word_str
            #content+=word_str+' '
    #data['content']=content

    f.close()

    #return data


class MySentences(object):
    def __init__(self, dirname,key=''):
        self.dirname = dirname
        self.key=key

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if self.key and(self.key not in fname):
                continue
            if fname.replace('.scel','.txt') in os.listdir(self.dirname):
                continue
            yield os.path.join(self.dirname, fname)

def main ():
    if len(sys.argv)!=2 and len(sys.argv)!=3:
        print ("参数不对 \n 用法: python3 %s file.scel new.txt" %(sys.argv[0]))
        exit (1)


    #os.path.isdir，判断是否是一个目录
    if os.path.isdir(sys.argv[1]):
        files=MySentences(sys.argv[1],'.scel')
        for fileName in files:
            print (fileName)
            if len(sys.argv)!=3:
                outfile=fileName.replace('.scel','.txt')
            else:
                outfile=sys.argv[2]
            generator = get_word_from_sogou_cell_dict(fileName)
            with open(outfile, "a",encoding='utf-8') as f:
                for k,v in generator:
                    f.write("{}\n".format(v))

    else:
        generator = get_word_from_sogou_cell_dict (sys.argv[1])
        if len(sys.argv)!=3:
            outfile=sys.argv[1].replace('.scel','.txt')
        else:
            outfile=sys.argv[2]
        with open(outfile, "w",encoding='utf-8') as f:
            for k,v in generator:
                f.write("{}\n".format(v))

if __name__ == "__main__":
    main()



"""

"简历专用词库":['办公文教_其它','生活'],
"147种水果的名字":['日常','生活'],
"煎煮、桥梁、道路词库":['建筑_其它','工程与应用科学'],
"动漫片名":['动漫','娱乐'],
"新疆":['城市信息大全','城市信息大全'],
"淘宝词库":['日常','生活'],
"四级行政区划地名词库":['全国_其它','城市信息大全'],
"三级行政区划地名词库":['全国_其它','城市信息大全'],

"人文社科大词库0.01":['人文科学_其它','人文科学'],
"物流货运专业术语":['交通运输物流','城市信息大全'],



import struct
import sys,time
import binascii 
import pdb
#搜狗的scel词库就是保存的文本的unicode编码，每两个字节一个字符（中文汉字或者英文字母）
#找出其每部分的偏移位置即可
#主要两部分
#1.全局拼音表，貌似是所有的拼音组合，字典序
#       格式为(index,len,pinyin)的列表
#       index: 两个字节的整数 代表这个拼音的索引
#       len: 两个字节的整数 拼音的字节长度
#       pinyin: 当前的拼音，每个字符两个字节，总长len
#       
#2.汉语词组表
#       格式为(same,py_table_len,py_table,{word_len,word,ext_len,ext})的一个列表
#       same: 两个字节 整数 同音词数量
#       py_table_len:  两个字节 整数
#       py_table: 整数列表，每个整数两个字节,每个整数代表一个拼音的索引
#
#       word_len:两个字节 整数 代表中文词组字节数长度
#       word: 中文词组,每个中文汉字两个字节，总长度word_len
#       ext_len: 两个字节 整数 代表扩展信息的长度，好像都是10
#       ext: 扩展信息 前两个字节是一个整数(不知道是不是词频) 后八个字节全是0
#
#      {word_len,word,ext_len,ext} 一共重复same次 同音词 相同拼音表

#拼音表偏移，
startPy = 0x1540;#5440


#汉语词组表偏移
startChinese = 0x2628;#9768

In[328]: 0x1540
Out[328]: 5440
In[329]: 0x2628
Out[329]: 9768
In[330]: 9768-5440
Out[330]: 4328

#全局拼音表

GPy_Table ={}

#解析结果
#元组(词频,拼音,中文词组)的列表
GTable = []

def byte2str(data):
    '''将原始字节码转为字符串'''
    i = 0;
    length = len(data)
    ret = u''
    while i < length:
        x = data[i] + data[i+1]
        t = unichr(struct.unpack('H',x)[0])
        if t == u'\r':
            ret += u'\n'
        elif t != u' ':
            ret += t
        i += 2
    return ret
#获取拼音表
def getPyTable(data):

    if data[0:4] != "\x9D\x01\x00\x00":
        return None
    data = data[4:]
    pos = 0
    length = len(data)
    while pos < length:
        index = struct.unpack('H',data[pos]+data[pos+1])[0]
        #print index,
        pos += 2
        l = struct.unpack('H',data[pos]+data[pos+1])[0]
        #print l,
        pos += 2
        py = byte2str(data[pos:pos+l])
        #print py
        GPy_Table[index]=py
        pos += l

#获取一个词组的拼音
def getWordPy(data):
    pos = 0
    length = len(data)
    ret = u''
    while pos < length:
        
        index = struct.unpack('H',data[pos]+data[pos+1])[0]
        ret += GPy_Table[index]
        pos += 2    
    return ret

#获取一个词组
def getWord(data):
    pos = 0
    length = len(data)
    ret = u''
    while pos < length:
        
        index = struct.unpack('H',data[pos]+data[pos+1])[0]
        ret += GPy_Table[index]
        pos += 2    
    return ret

#读取中文表    
def getChinese(data):
    #import pdb
    #pdb.set_trace()
    
    pos = 0
    length = len(data)
    while pos < length:
        #同音词数量
        same = struct.unpack('H',data[pos]+data[pos+1])[0]
        #print '[same]:',same,
        
        #拼音索引表长度
        pos += 2
        py_table_len = struct.unpack('H',data[pos]+data[pos+1])[0]
        #拼音索引表
        pos += 2
        py = getWordPy(data[pos: pos+py_table_len])

        #中文词组
        pos += py_table_len
        for i in range(same):
            #中文词组长度
            c_len = struct.unpack('H',data[pos]+data[pos+1])[0]
            #中文词组
            pos += 2  
            word = byte2str(data[pos: pos + c_len])
            #扩展数据长度
            pos += c_len        
            ext_len = struct.unpack('H',data[pos]+data[pos+1])[0]
            #词频
            pos += 2
            count  = struct.unpack('H',data[pos]+data[pos+1])[0]

            #保存
            GTable.append((count,py,word))
        
            #到下个词的偏移位置
            pos +=  ext_len


def deal(file_name):
    print ('-'*60)
    f = open(file_name,'rb')
    data = f.read()
    f.close()
    
    
    if data[0:12] !="\x40\x15\x00\x00\x44\x43\x53\x01\x01\x00\x00\x00":
        print ("确认你选择的是搜狗(.scel)词库?")
        sys.exit(0)
    #pdb.set_trace()
    
    print ("词库名：" ,byte2str(data[0x130:0x338]))#.encode('GB18030')
    print ("词库类型：" ,byte2str(data[0x338:0x540]))#.encode('GB18030')
    print ("描述信息：" ,byte2str(data[0x540:0xd40]))#.encode('GB18030')
    print ("词库示例：",byte2str(data[0xd40:startPy]))#.encode('GB18030')
    
    getPyTable(data[startPy:startChinese])
    getChinese(data[startChinese:])
    

In[417]: 0x130
Out[417]: 304
In[418]: 0x338
Out[418]: 824
In[419]: 0x540
Out[419]: 1344
In[420]: 0xd40
Out[420]: 3392
startPy = 0x1540;#5440
startChinese = 0x2628;#9768

In[427]: jieri[285:320]
Out[427]: b'I\x84F[\x00\x00\x00[\x00\x00\x00\x1c\x04\x00\x00\x1c\x04\x00\x008\x009\x00*N\x82\x82\xe5e\x00\x00\x00\x00\x00\x00'
In[430]: jieri[820:830]
Out[430]: b'\x00\x00\x00\x00vQ\xd6N\x00\x00'
In[431]: jieri[1340:1350]
Out[431]: b'\x00\x00\x00\x00\xcfk*N\x82\x82'
In[432]: jieri[3390:3400]
Out[432]: b'\x00\x00#W\xde\x8b\x82\x82 \x00'
In[433]: jieri.find(b'#')
Out[433]: 3392

if __name__ == '__main__':
    start=time.time()

    #将要转换的词库添加在这里就可以了
    o = ['计算机词汇大全【官方推荐】.scel',
    'IT计算机.scel',
    '计算机词汇大全【官方推荐】.scel',
    '北京市城市信息精选.scel',
    '常用餐饮词汇.scel',
    '成语.scel',
    '成语俗语【官方推荐】.scel',
    '法律词汇大全【官方推荐】.scel',
    '房地产词汇大全【官方推荐】.scel',
    '手机词汇大全【官方推荐】.scel',
    '网络流行新词【官方推荐】.scel',
    '歇后语集锦【官方推荐】.scel',
    '饮食大全【官方推荐】.scel',
    ]
    o=['/home/gswewf/sougouxibao/细胞词库全收集@AElee/89个节日.scel']
    for f in o:
        deal(f)
        
    #保存结果  
    f = open('/home/gswewf/sougouxibao/sougou.txt','w')
    for count,py,word in GTable:
        #GTable保存着结果，是一个列表，每个元素是一个元组(词频,拼音,中文词组)，有需要的话可以保存成自己需要个格式
        #我没排序，所以结果是按照上面输入文件的顺序
        f.write( unicode('{%(count)s}' %{'count':count}+py+' '+ word).encode('GB18030') )#最终保存文件的编码，可以自给改
        f.write('\n')
    f.close()
    print('运行完毕：',time.time()-start)


In[40]: data1[:50]
Out[40]: b'@\x15\x00\x00DCS\x01\x01\x00\x00\x00\x14-C\x91\x87y\xf7f\xcc\xe1\xad\xba\xb7\xce\xedZ3\x007\x009\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
In[41]: struct.unpack('H',data1[:2])
Out[41]: (5440,)
In[42]: str(5440)
Out[42]: '5440'
In[43]: chr(5440)
Out[43]: 'ᕀ'


[chr(struct.unpack('H',data1[i:i+2])[0])for i in range(int(len(data1)/2))]


In[100]: struct.unpack('H',data[pos:pos+2])
Out[100]: (5440,)
In[101]: data[:20]
Out[101]: b'@\x15\x00\x00DCS\x01\x01\x00\x00\x00z\xd9N\x9e\\yZ\x87'
In[102]: data[5440:5460]
Out[102]: b'\x9d\x01\x00\x00\x00\x00\x02\x00a\x00\x01\x00\x04\x00a\x00i\x00\x02\x00'
In[103]: pos
Out[103]: 0



#获取拼音表
def getPyTable(data,GPy_Table={}):

    if data[0:4] != b"\x9D\x01\x00\x00":
        return None
    data = data[4:]
    pos = 0
    length = len(data)
    while pos < length:
        index = struct.unpack('H',data[pos:pos+2])[0]
        #print index,
        pos += 2
        l = struct.unpack('H',data[pos:pos+2])[0]
        #print l,
        pos += 2
        #py = byte2str(data[pos:pos+l])
        #py=bytes(data[pos]+data[pos+1]).decode('gb18030')
        try:
            py=''.join([chr(data[i]) for i in range(pos,pos+l) if data[i]!=0])
        except :
            print(pos)
        #print (py)
        GPy_Table[index]=py
        pos += l

getPyTable(jieri[5440:0x2628],GPy_Table)

def getWordPy(data):
    pos = 0
    length = len(data)
    ret = ''
    while pos < length:

        index = struct.unpack('H',data[pos:pos+2])[0]
        ret += GPy_Table[index]
        pos += 2
    return ret

In[170]: struct.unpack('H',data5[:2])
Out[170]: (413,)
In[171]: struct.unpack('H',data5[2:4])
Out[171]: (0,)
In[172]: struct.unpack('H',data5[4:6])
Out[172]: (0,)
In[173]: struct.unpack('H',data5[6:8])
Out[173]: (2,)
In[174]: struct.unpack('H',data5[8:10])
Out[174]: (97,)
In[175]: struct.unpack('H',data5[10:12])
Out[175]: (1,)

#1.全局拼音表，貌似是所有的拼音组合，字典序
#       格式为(index,len,pinyin)的列表
#       index: 两个字节的整数 代表这个拼音的索引
#       len: 两个字节的整数 拼音的字节长度
#       pinyin: 当前的拼音，每个字符两个字节，总长len
#
#2.汉语词组表
#       格式为(same,py_table_len,py_table,{word_len,word,ext_len,ext})的一个列表
#       same: 两个字节 整数 同音词数量
#       py_table_len:  两个字节 整数
#       py_table: 整数列表，每个整数两个字节,每个整数代表一个拼音的索引
#
#       word_len:两个字节 整数 代表中文词组字节数长度
#       word: 中文词组,每个中文汉字两个字节，总长度word_len
#       ext_len: 两个字节 整数 代表扩展信息的长度，好像都是10
#       ext: 扩展信息 前两个字节是一个整数(不知道是不是词频) 后八个字节全是0
#
#      {word_len,word,ext_len,ext} 一共重复same次 同音词 相同拼音表

In[328]: 0x1540
Out[328]: 5440
In[329]: 0x2628
Out[329]: 9768
In[330]: 9768-5440
Out[330]: 4328

def by_str(data):
...     return ''.join([chr(data[i]) for i in range(0,len(data)) if data[i]!=0])


def byte2str(data):
    '''将原始字节码转为字符串'''
    i = 0;
    length = len(data)
    ret = ''
    while i < length:
        x = data[i]+data[i+1]
        t = unichr(struct.unpack('H',x)[0])
        if t == u'\r':
            ret += u'\n'
        elif t != u' ':
            ret += t
        i += 2
    return ret

"""
