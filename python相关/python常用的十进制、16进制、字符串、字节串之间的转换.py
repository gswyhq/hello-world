#! /usr/lib/python3
# -*- coding: utf-8 -*-

# python常用的十进制、16进制、字符串、字节串之间的转换

'''

进行协议解析时，总是会遇到各种各样的数据转换的问题，从二进制到十进制，从字节串到整数等等

整数之间的进制转换:
10进制转16进制: hex(16)  ==>  0x10
16进制转10进制: int('0x10', 16)  ==>  16
16进制转10进制: int('f', 16) ==> 15
10进制转2进制:  bin(8) ==> '0b1000'
10进制转16进制: hex(11) ==> '0xb'
10进制转8进制: oct(8) ==> '0o10'
-------------------


字符串转整数:
10进制字符串转换为十进制数字: int('10')  ==>  10
16进制字符串转换为十进制数字: int('10', 16)  ==>  16
16进制字符串转换为十进制数字: int('0x10', 16)  ==>  16


-------------------
unicode、utf-8相互转换：
In[1]: '\u4e00' == '一'
Out[1]:  True
In[2]: u = r'\u{}'.format('4e00')
Out[2]: print('解回字符串: ' + u.encode("utf-8").decode('unicode-escape'))
解回字符串: 一
In[3]: r'\u{}'.format('4e00')
Out[3]: '\\u4e00'
In[65]: '\u4E00-\u9FD5'
Out[65]: '一-鿕'

字节串转整数:
转义为short型整数: struct.unpack('<hh', bytes(b'\x01\x00\x00\x00'))  ==>  (1, 0)
转义为long型整数: struct.unpack('<L', bytes(b'\x01\x00\x00\x00'))  ==>  (1,)


-------------------


整数转字节串:
转为两个字节: struct.pack('<HH', 1,2)  ==>  b'\x01\x00\x02\x00'
转为四个字节: struct.pack('<LL', 1,2)  ==>  b'\x01\x00\x00\x00\x02\x00\x00\x00'


-------------------


字符串转字节串:
字符串编码为字节码: '12abc'.encode('ascii')  ==>  b'12abc'
数字或字符数组: bytes([1,2, ord('1'),ord('2')])  ==>  b'\x01\x0212'
16进制字符串: bytes().fromhex('010210')  ==>  b'\x01\x02\x10'
16进制字符串: bytes(map(ord, '\x01\x02\x31\x32'))  ==>  b'\x01\x0212'
16进制数组: bytes([0x01,0x02,0x31,0x32])  ==>  b'\x01\x0212'


-------------------


字节串转字符串:
字节码解码为字符串: bytes(b'\x31\x32\x61\x62').decode('ascii')  ==>  12ab
字节串转16进制表示,夹带ascii: str(bytes(b'\x01\x0212'))[2:-1]  ==>  \x01\x0212
字节串转16进制表示,固定两个字符表示: str(binascii.b2a_hex(b'\x01\x0212'))[2:-1]  ==>  01023132
字节串转16进制数组: [hex(x) for x in bytes(b'\x01\x0212')]  ==>  ['0x1', '0x2', '0x31', '0x32']


===================


Created on 2014年8月21日

@author: lenovo
'''
import binascii
import struct


def example(express, result=None):
    if result == None:
        result = eval(express)
    print(express, ' ==> ', result)


if __name__ == '__main__':

    print('整数之间的进制转换:')
    print("10进制转16进制", end=': ');example("hex(16)")
    print("16进制转10进制", end=': ');example("int('0x10', 16)")
    print("类似的还有oct()， bin()")

    print('\n-------------------\n')

    print('字符串转整数:')
    print("10进制字符串", end=": ");example("int('10')")
    print("16进制字符串", end=": ");example("int('10', 16)")
    print("16进制字符串", end=": ");example("int('0x10', 16)")

    print('\n-------------------\n')

    print('字节串转整数:')
    print("转义为short型整数", end=": ");example(r"struct.unpack('<hh', bytes(b'\x01\x00\x00\x00'))")
    print("转义为long型整数", end=": ");example(r"struct.unpack('<L', bytes(b'\x01\x00\x00\x00'))")

    print('\n-------------------\n')

    print('整数转字节串:')
    print("转为两个字节", end=": ");example("struct.pack('<HH', 1,2)")
    print("转为四个字节", end=": ");example("struct.pack('<LL', 1,2)")

    print('\n-------------------\n')

    print('字符串转字节串:')
    print('字符串编码为字节码', end=": ");example(r"'12abc'.encode('ascii')")
    print('数字或字符数组', end=": ");example(r"bytes([1,2, ord('1'),ord('2')])")
    print('16进制字符串', end=': ');example(r"bytes().fromhex('010210')")
    print('16进制字符串', end=': ');example(r"bytes(map(ord, '\x01\x02\x31\x32'))")
    print('16进制数组', end =': ');example(r'bytes([0x01,0x02,0x31,0x32])')

    print('\n-------------------\n')

    print('字节串转字符串:')
    print('字节码解码为字符串', end=": ");example(r"bytes(b'\x31\x32\x61\x62').decode('ascii')")
    print('字节串转16进制表示,夹带ascii', end=": ");example(r"str(bytes(b'\x01\x0212'))[2:-1]")
    print('字节串转16进制表示,固定两个字符表示', end=": ");example(r"str(binascii.b2a_hex(b'\x01\x0212'))[2:-1]")
    print('字节串转16进制数组', end=": ");example(r"[hex(x) for x in bytes(b'\x01\x0212')]")


    print('\n===================\n')
    print("以上原理都比较简单，看一下就明白了。这里仅仅是抛砖引玉，有更好更简单的方法，欢迎欢迎")