#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 对于62进制，从0数到9以后，10用小写字母a表示，接着数完26个字母，到z为35，然后36为大写字母A，一直到61为大写字母Z。所以，我们可以实现十进制数字base62编码的encode和decode。

import math
ALPHABET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

def base62_encode(num, alphabet=ALPHABET):
    """Encode a number in Base X

    `num`: The number to encode
    `alphabet`: The alphabet to use for encoding
    """
    if (num == 0):
        return alphabet[0]
    arr = []
    base = len(alphabet)
    while num:
        rem = num % base
        num = num // base
        arr.append(alphabet[rem])
    arr.reverse()
    return ''.join(arr)

def base62_decode(string, alphabet=ALPHABET):
    """Decode a Base X encoded string into the number

    Arguments:
    - `string`: The encoded string
    - `alphabet`: The alphabet to use for encoding
    """
    base = len(alphabet)
    strlen = len(string)
    num = 0

    idx = 0
    for char in string:
        power = (strlen - (idx + 1))
        num += alphabet.index(char) * (base ** power)
        idx += 1
    return num

# 对于一个新浪微博url，它是形如：http://weibo.com/2991905905/z579Hz9Wr，中间的数字是用户的uid，重要的是后面的字符串“z579Hz9Wr”。它的计算其实也很简单，从后向前四个字符一组，就得到：
# z
# 579H
# z9Wr
# 将每个字符串用base62编码来decode，就可以得到它们的十进制数字分别为：
# 35
# 1219149
# 8379699
# 将它们拼起来就可以得到mid为：“3512191498379699”。这里要强调的是：对于除了开头的字符串，如果得到的十进制数字不足7位，需要在前面补足0。比如得到的十进制数分别为：35，33040，8906190，则需要在33040前面添上两个0。


def url_to_mid(url):
    '''
    >>> url_to_mid('z0JH2lOMb')
    3501756485200075
    >>> url_to_mid('z0Ijpwgk7')
    3501703397689247
    >>> url_to_mid('z0IgABdSn')
    3501701648871479
    >>> url_to_mid('z08AUBmUe')
    3500330408906190
    >>> url_to_mid('z06qL6b28')
    3500247231472384
    >>> url_to_mid('yCtxn8IXR')
    3491700092079471
    >>> url_to_mid('yAt1n2xRa')
    3486913690606804
    '''

    result_str = ''
    url = str(url)
    len_1 = len(url) % 4
    for i in range(math.ceil(len(url)/4)):
        if i == 0 and len_1 > 0:
            result_str += str(base62_decode(url[:len_1]))
        else:
            result_str += str(base62_decode(url[len_1+4*(i-1):len_1+4*i])).zfill(7)

    return int(result_str)

# mid转为url也就很简单了，对于一个mid，我们从后向前每7位一组，用base62编码来encode，拼起来即可。同样要注意的是，每7个一组的数字，除了开头一组，如果得到的62进制数字不足4位，需要补足0。

def mid_to_url(midint):
    '''
    >>> mid_to_url(3501756485200075)
    'z0JH2lOMb'
    >>> mid_to_url(3501703397689247)
    'z0Ijpwgk7'
    >>> mid_to_url(3501701648871479)
    'z0IgABdSn'
    >>> mid_to_url(3500330408906190)
    'z08AUBmUe'
    >>> mid_to_url(3500247231472384)
    'z06qL6b28'
    >>> mid_to_url(3491700092079471)
    'yCtxn8IXR'
    >>> mid_to_url(3486913690606804)
    'yAt1n2xRa'
    '''
    midint = str(midint)[::-1]
    size = len(midint) // 7 if len(midint) % 7 == 0 else len(midint) // 7 + 1
    result = []
    for i in range(size):
        s = midint[i * 7: (i + 1) * 7][::-1]
        s = base62_encode(int(s))
        s_len = len(s)
        if i < size - 1 and len(s) < 4:
            s = '0' * (4 - s_len) + s
        result.append(s)
    result.reverse()
    return ''.join(result)

def main():
    pass


if __name__ == '__main__':
    main()