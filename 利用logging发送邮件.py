#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import sys, os, json

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

import logging, logging.handlers

class EncodingFormatter(logging.Formatter):
    """转换编码

    原生的logging发送邮件时，不支持unincde
    """
    def __init__(self, fmt, datefmt=None, encoding=None):
        logging.Formatter.__init__(self, fmt, datefmt)
        self.encoding = encoding
    def format(self, record):
        result = logging.Formatter.format(self, record)
        if isinstance(result, unicode):
            result = result.encode(self.encoding or 'utf-8')
        return result

def send_log_error(toaddrs='gswewf@126.com', subject=u'邮件标题',contents=u'这是程序发送的测试邮件正文'):
    """利用logging模块发送邮件

    参数：
        toaddrs：收件人邮箱
        subject: 邮件标题
        conenets: 邮件正文
        """
    subject = subject.encode('utf8') if isinstance(subject, unicode) else subject
    root = logging.getLogger()
    sh = logging.handlers.SMTPHandler(mailhost=('smtp.exmail.qq.com', 25),
                                     fromaddr='ai_public@gow.cn',
                                     toaddrs=toaddrs,
                                     subject=subject,
                                     credentials=('ai_public@gow.cn','AI_public123'),
                                      )
    root.addHandler(sh)
    sh.setFormatter(EncodingFormatter('%(message)s', encoding='utf-8'))
    root.error(contents)

def main():
    send_log_error(toaddrs='gswewf@126.com', subject=u'邮件标题',contents=u'这是程序发送的测试邮件正文')

if __name__ == "__main__":
    main()