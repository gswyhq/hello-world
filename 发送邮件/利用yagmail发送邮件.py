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

import yagmail

def send_email(to_email_list=None, subject=None, contents=None, attachments=None, cc=None, bcc=None):
    """发送邮件
    参数：
        to_email_list: 收件人邮箱，如果不指定to_email_list参数，则发送给（'gswewf@126.com', 'zhou_li@gow.cn'）,
                如果to_email_list参数是一个列表，则将该邮件发送给列表中的所有用户
        subject: 邮件标题；
        contents: 邮件正文内容；
            如果它是一个字典，它会假设键是内容，值是一个别名（如： {'/path/to/image.png'：'MyPicture'}
            它将尝试查看内容（字符串）是否可以作为本地文件读取，例如。 '/path/to/image.png'
            如果不可能，它将检查字符串是否是有效的html。
            如果不是，它必须是文本。 例如 “Hi Dorika！
        attachments: 表示附件，该参数可以是str,也可以是列表，表示发送多个附件
        cc: 抄送人的邮箱
        """
    if not to_email_list:
        to_email_list = ['gswewf@126.com', 'zhou_li@gow.cn']

    if PY3:
        user='285264595@qq.com'
        password='xamctomgumpvbggi'
        host='smtp.qq.com'
        port='25'
    else:
        # 若是python2，邮件标题、内容、邮件地址必须是str，不能是unicode；
        subject = subject.encode('utf8') if isinstance(subject, unicode) else subject
        contents = contents.encode('utf8') if isinstance(contents, unicode) else contents
        to_email_list = to_email_list.encode('utf8') if isinstance(to_email_list, unicode) else to_email_list
        to_email_list = [email.encode('utf8') if isinstance(email, unicode) else email for email in to_email_list] if isinstance(to_email_list, (list, tuple)) else to_email_list

        cc = cc.encode('utf8') if isinstance(cc, unicode) else cc
        cc = [email.encode('utf8') if isinstance(email, unicode) else email for email in cc] if isinstance(cc, (list, tuple)) else cc

        user=b'285264595@qq.com'
        password=b'xamctomgumpvbggi'
        host=b'smtp.qq.com'
        port=b'25'

    # 初始化一个SMTP客户端,并发送邮件
    # yagmail.SMTP()默认使用的gmail的SMTP服务
    with yagmail.SMTP(user=user,password=password, host=host, port=port)as yag:
        yag.send(to=to_email_list, subject=subject, contents=contents, attachments=attachments, cc=cc, bcc=bcc)

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
    send_email(to_email_list='gswewf@126.com', subject='测试邮件', contents='无', attachments=['/home/gswewf/gow69/利用yagmail发送邮件.py'], cc='gswewf@126.com')

if __name__ == "__main__":
    main()

# 常用邮箱SMTP服务器地址和端口
# sina.com:
# POP3服务器地址:pop3.sina.com.cn（端口：110）
# SMTP服务器地址:smtp.sina.com.cn（端口：25）
# sinaVIP：
# POP3服务器:pop3.vip.sina.com （端口：110）
# SMTP服务器:smtp.vip.sina.com （端口：25）
# sohu.com:
# POP3服务器地址:pop3.sohu.com（端口：110）
# SMTP服务器地址:smtp.sohu.com（端口：25）
# 126邮箱：
# POP3服务器地址:pop.126.com（端口：110）
# SMTP服务器地址:smtp.126.com（端口：25）
# 139邮箱：
# POP3服务器地址：POP.139.com（端口：110）
# SMTP服务器地址：SMTP.139.com(端口：25)
# 163.com:
# POP3服务器地址:pop.163.com（端口：110）
# SMTP服务器地址:smtp.163.com（端口：25）
# QQ邮箱
# POP3服务器地址：pop.qq.com（端口：110）
# SMTP服务器地址：smtp.qq.com （端口：25）
# QQ企业邮箱
# POP3服务器地址：pop.exmail.qq.com （SSL启用 端口：995）
# SMTP服务器地址：smtp.exmail.qq.com（SSL启用 端口：587/465）
# yahoo.com:
# POP3服务器地址:pop.mail.yahoo.com
# SMTP服务器地址:smtp.mail.yahoo.com
# yahoo.com.cn:
# POP3服务器地址:pop.mail.yahoo.com.cn（端口：995）
# SMTP服务器地址:smtp.mail.yahoo.com.cn（端口：587）
# HotMail
# POP3服务器地址：pop3.live.com （端口：995）
# SMTP服务器地址：smtp.live.com （端口：587）
# gmail(google.com)
# POP3服务器地址:pop.gmail.com（SSL启用 端口：995）
# SMTP服务器地址:smtp.gmail.com（SSL启用 端口：587）
# 263.net:
# POP3服务器地址:pop3.263.net（端口：110）
# SMTP服务器地址:smtp.263.net（端口：25）
# 263.net.cn:
# POP3服务器地址:pop.263.net.cn（端口：110）
# SMTP服务器地址:smtp.263.net.cn（端口：25）
# x263.net:
# POP3服务器地址:pop.x263.net（端口：110）
# SMTP服务器地址:smtp.x263.net（端口：25）
# 21cn.com:
# POP3服务器地址:pop.21cn.com（端口：110）
# SMTP服务器地址:smtp.21cn.com（端口：25）
# Foxmail：
# POP3服务器地址:POP.foxmail.com（端口：110）
# SMTP服务器地址:SMTP.foxmail.com（端口：25）
# china.com:
# POP3服务器地址:pop.china.com（端口：110）
# SMTP服务器地址:smtp.china.com（端口：25）
# tom.com:
# POP3服务器地址:pop.tom.com（端口：110）
# SMTP服务器地址:smtp.tom.com（端口：25）
# etang.com:
# POP3服务器地址:pop.etang.com
# SMTP服务器地址:smtp.etang.com
#
