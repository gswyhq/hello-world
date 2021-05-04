#!/usr/bin/python3
# coding: utf-8

# 条件(IF - Then - Else)模式
# 正则表达式提供了条件检测的功能。格式如下：
# (?(?=regex)then |else)
#
# 条件可以是一个数字。表示引用前面捕捉到的分组。
#
# 比如我们可以用这个正则表达式来检测打开和闭合的尖括号：

import re
strings = ["<pypix>",  # returns true
           "<foo",  # returns false
           "bar>",  # returns false
           "hello"]  # returns true
for string in strings:
    pattern = re.search(r'^(<)?[a-z]+(?(1)>)$', string)
    if pattern:
        print('True')
    else:
        print('False')

# 在上面的例子中，1表示分组( <)，当然也可以为空因为后面跟着一个问号。当且仅当条件成立时它才匹配关闭的尖括号。
#
# 条件也可以是界定符。

# 注意到每一个替换都有一个共同点，它们都是由一对中括号括起来的。我们可以用一个单独的正则表达式 来捕获它们，并且用一个回调函数来处理具体的替换。

# 所以用回调函数是一个更好的办法：
template = "Hello [first_name] [last_name], Thank you for purchasing [product_name] from [store_name]. The total cost of your purchase was [product_price] plus [ship_price] for shipping. You can expect your product to arrive in [ship_days_min] to [ship_days_max] business days. Sincerely, [store_manager_name]"
# assume dic has all the replacement data
# such as dic['first_name'] dic['product_price'] etc...
dic = {
 "first_name" : "John",
 "last_name" : "Doe",
 "product_name" : "iphone",
 "store_name" : "Walkers",
 "product_price": "$500",
 "ship_price": "$10",
 "ship_days_min": "1",
 "ship_days_max": "5",
 "store_manager_name": "DoeJohn"
}
def multiple_replace(dic, text):
    pattern = "|".join(map(lambda key : re.escape("["+key+"]"), dic.keys()))
    return re.sub(pattern, lambda m: dic[m.group()[1:-1]], text)
print (multiple_replace(dic, template))

# 如何找到表达式中字符串的第n个出现以及如何用正则表达式替换
有以下字符串
txt = "aaa-aaa-aaa-aaa-aaa-aaa-aaa-aaa-aaa-aaa"
想用’|’代替’ – ‘的第五次出现
和“||”的第7次出现
期望结果：
 aaa-aaa-aaa-aaa-aaa|aaa-aaa||aaa-aaa-aaa

方法1，直接一次替换：
re.sub("(^(.*?-){4}.*?)-(.*?-.*?)-", "\\1|\\3||", txt)
Out[20]: 'aaa-aaa-aaa-aaa-aaa|aaa-aaa||aaa-aaa-aaa'

方法2，分步替换：
txt2 = re.sub("(^(.*?-){6}.*?)-", "\\1||", txt)
txt2
Out[25]: 'aaa-aaa-aaa-aaa-aaa-aaa-aaa||aaa-aaa-aaa'
re.sub("(^(.*?-){4}.*?)-", "\\1|", txt2)
Out[26]: 'aaa-aaa-aaa-aaa-aaa|aaa-aaa||aaa-aaa-aaa'


def main():
    pass


if __name__ == '__main__':
    main()
