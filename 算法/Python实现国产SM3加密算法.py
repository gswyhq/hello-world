#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# SM3的根基是哈希，是散列，是单向加密的，用于数字签名之类的，所以不存在解密一说； 但若加密时数据不加盐的话，别人通过数据碰撞也可能获取原文。
# SM4倒是可以加解密的

from math import ceil
import binascii

##############################################################################
#
#                            国产SM3加密算法
#
##############################################################################

IV = "7380166f 4914b2b9 172442d7 da8a0600 a96f30bc 163138aa e38dee4d b0fb0e4e"
IV = int(IV.replace(" ", ""), 16)
a = []
for i in range(0, 8):
    a.append(0)
    a[i] = (IV >> ((7 - i) * 32)) & 0xFFFFFFFF
IV = a


def out_hex(list1):
    for i in list1:
        print("%08x" % i)
    print("\n")


def rotate_left(a, k):
    k = k % 32
    return ((a << k) & 0xFFFFFFFF) | ((a & 0xFFFFFFFF) >> (32 - k))


T_j = []
for i in range(0, 16):
    T_j.append(0)
    T_j[i] = 0x79cc4519
for i in range(16, 64):
    T_j.append(0)
    T_j[i] = 0x7a879d8a


def FF_j(X, Y, Z, j):
    if 0 <= j and j < 16:
        ret = X ^ Y ^ Z
    elif 16 <= j and j < 64:
        ret = (X & Y) | (X & Z) | (Y & Z)
    return ret


def GG_j(X, Y, Z, j):
    if 0 <= j and j < 16:
        ret = X ^ Y ^ Z
    elif 16 <= j and j < 64:
        # ret = (X | Y) & ((2 ** 32 - 1 - X) | Z)
        ret = (X & Y) | ((~ X) & Z)
    return ret


def P_0(X):
    return X ^ (rotate_left(X, 9)) ^ (rotate_left(X, 17))


def P_1(X):
    return X ^ (rotate_left(X, 15)) ^ (rotate_left(X, 23))


def CF(V_i, B_i):
    W = []
    for i in range(16):
        weight = 0x1000000
        data = 0
        for k in range(i * 4, (i + 1) * 4):
            data = data + B_i[k] * weight
            weight = int(weight / 0x100)
        W.append(data)

    for j in range(16, 68):
        W.append(0)
        W[j] = P_1(W[j - 16] ^ W[j - 9] ^ (rotate_left(W[j - 3], 15))) ^ (rotate_left(W[j - 13], 7)) ^ W[j - 6]
        str1 = "%08x" % W[j]
    W_1 = []
    for j in range(0, 64):
        W_1.append(0)
        W_1[j] = W[j] ^ W[j + 4]
        str1 = "%08x" % W_1[j]

    A, B, C, D, E, F, G, H = V_i
    """
    print "00",
    out_hex([A, B, C, D, E, F, G, H])
    """
    for j in range(0, 64):
        SS1 = rotate_left(((rotate_left(A, 12)) + E + (rotate_left(T_j[j], j))) & 0xFFFFFFFF, 7)
        SS2 = SS1 ^ (rotate_left(A, 12))
        TT1 = (FF_j(A, B, C, j) + D + SS2 + W_1[j]) & 0xFFFFFFFF
        TT2 = (GG_j(E, F, G, j) + H + SS1 + W[j]) & 0xFFFFFFFF
        D = C
        C = rotate_left(B, 9)
        B = A
        A = TT1
        H = G
        G = rotate_left(F, 19)
        F = E
        E = P_0(TT2)

        A = A & 0xFFFFFFFF
        B = B & 0xFFFFFFFF
        C = C & 0xFFFFFFFF
        D = D & 0xFFFFFFFF
        E = E & 0xFFFFFFFF
        F = F & 0xFFFFFFFF
        G = G & 0xFFFFFFFF
        H = H & 0xFFFFFFFF

    V_i_1 = []
    V_i_1.append(A ^ V_i[0])
    V_i_1.append(B ^ V_i[1])
    V_i_1.append(C ^ V_i[2])
    V_i_1.append(D ^ V_i[3])
    V_i_1.append(E ^ V_i[4])
    V_i_1.append(F ^ V_i[5])
    V_i_1.append(G ^ V_i[6])
    V_i_1.append(H ^ V_i[7])
    return V_i_1


def hash_msg(msg):
    # print(msg)
    len1 = len(msg)
    reserve1 = len1 % 64
    msg.append(0x80)
    reserve1 = reserve1 + 1
    # 56-64, add 64 byte
    range_end = 56
    if reserve1 > range_end:
        range_end = range_end + 64

    for i in range(reserve1, range_end):
        msg.append(0x00)

    bit_length = (len1) * 8
    bit_length_str = [bit_length % 0x100]
    for i in range(7):
        bit_length = int(bit_length / 0x100)
        bit_length_str.append(bit_length % 0x100)
    for i in range(8):
        msg.append(bit_length_str[7 - i])

    # print(msg)

    group_count = round(len(msg) / 64)

    B = []
    for i in range(0, group_count):
        B.append(msg[i * 64:(i + 1) * 64])

    V = []
    V.append(IV)
    for i in range(0, group_count):
        V.append(CF(V[i], B[i]))

    y = V[i + 1]
    result = ""
    for i in y:
        result = '%s%08x' % (result, i)
    return result


def str2byte(msg):  # 字符串转换成byte数组
    ml = len(msg)
    msg_byte = []
    msg_bytearray = msg  # 如果加密对象是字符串，则在此对msg做encode()编码即可，否则不编码
    for i in range(ml):
        msg_byte.append(msg_bytearray[i])
    return msg_byte


def byte2str(msg):  # byte数组转字符串
    ml = len(msg)
    str1 = b""
    for i in range(ml):
        str1 += b'%c' % msg[i]
    return str1.decode('utf-8')


def hex2byte(msg):  # 16进制字符串转换成byte数组
    ml = len(msg)
    if ml % 2 != 0:
        msg = '0' + msg
    ml = int(len(msg) / 2)
    msg_byte = []
    for i in range(ml):
        msg_byte.append(int(msg[i * 2:i * 2 + 2], 16))
    return msg_byte


def byte2hex(msg):  # byte数组转换成16进制字符串
    ml = len(msg)
    hexstr = ""
    for i in range(ml):
        hexstr = hexstr + ('%02x' % msg[i])
    return hexstr


def KDF(Z, klen):  # Z为16进制表示的比特串（str），klen为密钥长度（单位byte）
    klen = int(klen)
    ct = 0x00000001
    rcnt = ceil(klen / 32)
    Zin = hex2byte(Z)
    Ha = ""
    for i in range(int(rcnt)):
        msg = Zin + hex2byte('%08x' % ct)
        # print(msg)
        Ha = Ha + hash_msg(msg)
        # print(Ha)
        ct += 1
    return Ha[0: klen * 2]


def sm3_hash(msg, Hexstr=0):
    """
    封装方法，外部调用
    :param msg: 二进制流（如若需要传入字符串，则把str2byte方法里msg做encode()编码一下，否则不编码）
    :param Hexstr: 0
    :return: 64位SM3加密结果
    """
    if (Hexstr):
        msg_byte = hex2byte(msg)
    else:
        msg_byte = str2byte(msg)
    return hash_msg(msg_byte)


def main():
    print(sm3_hash(b'SM3Test'))
    # 打印结果：901053b4681483b737dd2dd9f9a7f56805aa1b03337f8c1abb763a96776b8905

    # 用实例进行测试
    # 国标示例
    TEST_LIST = [("国标示例1",
                  "abc",
                  "66c7f0f4 62eeedd9 d1f2d46b dc10e4e2 4167c487 5cf2f7a2 297da02b 8f4ba8e0"
                  ),
                 ("国标示例2",
                  "abcd" * 16,
                  "debe9ff9 2275b8a1 38604889 c18e5a4d 6fdb70e5 387e5765 293dcba3 9c0c5732"
                  ),
                 ]
    for (name, msgStr, ansStr) in TEST_LIST:
        msg = msgStr.encode()  # 输入 ASCII字符串转为字节流
        ansStr = ansStr.replace(" ", "")
        # ans = binascii.a2b_hex(ansStr)  # 答案 十六进制字符串转为字节流
        print(name, sm3_hash(msg) == ansStr)

    # 验证实例 来源于 https://wenku.baidu.com/view/91b57a06a6c30c2259019eb5.htm
    TEST_LIST2 = [("实例 1",

                   "763AFC537F7876C2D0B59FDB68D762E90CEFD222BB358D0D6931867CE26538649BE3579A4004483EA5D84D005063F76FB1CE7E5F2F933B5ED757A718182F383C4D58291A6A5D8D07C081F66806031539093362D854883A8874F7B919925DABC74C173E2162F07E6780E311FF0AEF059AE620303DECB6289E97F72C018723C471",
                   "d3,2b,9d,79,3b,49,66,e1,8c,35,18,92,e6,52,77,75,6e,f0,cc,9d,28,db,95,4c,95,9f,bb,85,e3,94,84,42"
                   ),
                  ("实例 2",
                   "2F917420E702DBA970C071AE4971AD08DE3D7D0D90DC1E334ED20444E54F109BA80DD22F25C24FAA83D5AD58687F1AA68F1B749D0AD999DB9A1AC8E4DC",
                   "5e,11,d3,64,71,23,f4,43,18,7e,f2,5d,9e,fc,33,db,74,29,ce,c0,06,d6,3e,4c,9a,35,db,84,2f,73,4b,a8"
                   ),
                  ("实例 3",
                   "2F917420E702DBA970C071AE4971AD08DE3D7D0D90DC1E334ED20444E54F109BA80DD22F25C24FAA83D5AD58687F1AA68F1B749D0AD999",
                   "d1,e5,44,80,1f,8b,29,ac,fb,c8,cf,69,9f,0d,a3,7b,f3,18,74,eb,66,a0,eb,50,ae,ee,4a,4d,31,58,7c,2f"
                   ),
                  ("实例 4",
                   "2F917420E702DBA970C071AEDE3D7D0D90DC1E334ED20444E54F109BA80DD22F25C24FAA83D5AD58687F1AA68F1B749D0AD999",
                   "06,0f,dc,46,34,e0,70,ac,ab,54,8a,63,08,f7,44,58,68,1d,59,65,07,75,46,79,3c,16,8a,82,c4,a5,d7,ad"
                   ),
                  ("实例 5",
                   "E47F211542C022AC94542DE4EEC6A1B10BF54B6A9F3C439459F4D9779C4BE5326AEA06FF6EEE97F61E66978DFA8543D1520103CDA6AB7655B592BF2D40ECB937",
                   "33,D0,D0,AB,67,A3,11,F4,46,4D,B2,F7,40,45,F0,CE,D7,31,5A,1B,82,FE,5C,03,EA,CA,D3,B1,E8,D4,B7,50"
                   ),
                  ("实例 6",
                   "64055D80810171DCE32D71773ECFDC803203539DB5677401DDD6A2B538D0652978479B5BE524FE809CB35499BDCC4C8FC1081CB2E09BCB4458828C5168BE329D",
                   "B4,21,1F,6B,AB,65,19,B9,9A,30,13,F2,A1,4F,BF,32,04,9F,F2,F0,C4,47,0D,67,DA,32,99,5E,CB,4C,CE,D6"
                   ),
                  ("实例 7",
                   "8BF6F8D53AFBB995FFFC001F76FC8549F67BDDA730B38D3E75301C834FBA67E1395B8D63D07908367B3311E2BD3586DCD4498869994DF0D5006781C880E7C83B",
                   "3E,E8,77,50,BA,8A,B7,07,9C,3C,6A,2C,22,8C,14,84,0C,0C,7D,C2,B9,0E,93,B7,16,06,EC,2B,DB,C3,0D,46"
                   ),
                  ("实例 8",
                   "34523D2F917420E702DBA970C071AE4971AD08DE3D7D0D90DC1E334ED20444E54F109BA80DD22F25C24FAA83D5AD58687F1AA68F1B749D0AD999DB9A1AC8E4DC",
                   "4A,26,7A,B8,23,F7,5B,3B,49,90,F6,0F,49,D1,45,BE,F1,CF,87,B0,2C,86,A9,50,88,E8,FE,02,0D,85,C7,F5"
                   ),
                  ("实例 9",
                   "3366779900",
                   "47,9D,86,3F,D6,74,6F,47,57,0B,81,1F,F6,CF,39,D6,6B,49,57,49,73,BD,C2,AF,A6,DA,4B,23,93,4A,40,6F"
                   ),
                  ("实例10",
                   "3366779900881234567890",
                   "AE,95,25,33,F5,E2,41,97,B6,84,37,57,F5,34,E4,9C,15,C1,F6,5A,46,C0,D0,18,89,45,1C,39,29,4B,79,1B"
                   ),
                  ("实例11",
                   "4EB692FE05A6A6D83FC12B6EDC7F9D877C71F8A5",
                   "39,67,C5,14,1E,D3,45,E6,1B,9A,6D,69,17,39,5F,89,48,93,EA,FD,4F,46,77,ED,5C,4E,33,E9,10,E0,24,D9"
                   ),
                  ("实例12",
                   "3967C5141ED345E61B9A6D6917395F894893EAFD4F4677ED5C4E33E910E024D9",
                   "98,C1,7C,93,6B,46,68,E4,7B,B6,78,0A,47,3E,7F,65,9A,29,D9,65,9F,68,59,E0,1E,02,48,19,67,8C,08,87"
                   ),
                  ("实例13",
                   "69AECD12F42E310465B4301320B277E2E4DE21EA593C377C",
                   "9D,B8,4C,1D,01,24,94,A6,D6,79,EE,A4,5B,55,0B,44,CD,76,BB,F3,78,31,99,85,55,94,38,A1,49,B2,69,0D"
                   ),
                  ]
    for (name, msgHexStr, ansStr) in TEST_LIST2:
        msgHexStr = msgHexStr.replace(",", "").replace(" ", "")
        msg = binascii.a2b_hex(msgHexStr)  # 输入 十六进制字符串转为字节流
        ansStr = ansStr.replace(",", "")
        print(name, sm3_hash(msg) == ansStr.lower())


if __name__ == '__main__':
    main()

# https://www.cnblogs.com/wcwnina/p/13604915.html
# https://blog.csdn.net/weixin_42782939/article/details/106143990