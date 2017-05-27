#/usr/lib/python3.5
# -*- coding: utf-8 -*-

import configparser

#读取配置文件
config=configparser.ConfigParser()
config.read("/home/gswyhq/有限状态机（FSM）/IpConfig.ini") #可以是一个不存在的文件，意味着准备新建配置文件。

#写入宿舍配置文件
try:
    #configparser.add_section()向配置文件中添加一个Section。
    #如果文件中已经存在相应的项目，则不能再增加同名的节。
    config.add_section("School")
    
    #使用configparser.set()在节School中增加新的参数。
    config.set("School","IP","10.15.40.123")
    config.set("School","Mask","255.255.255.0")
    config.set("School","Gateway","10.15.40.1")
    config.set("School","DNS","211.82.96.1")
except configparser.DuplicateSectionError:
    print("Section 'School' already exists")

#写入比赛配置文件
try:
    config.add_section("Match")
    config.set("Match","IP","172.17.29.120")
    config.set("Match","Mask","255.255.255.0")
    config.set("Match","Gateway","172.17.29.1")
    config.set("Match","DNS","0.0.0.0")
except configparser.DuplicateSectionError:
    print("Section 'Match' already exists")

#写入配置文件
#使用configparser.write()进行写入操作。
config.write(open("IpConfig.ini", "w"))


#使用configparser.get()读取刚才写入配置文件中的参数。读取之前要记得读取ini文件。
ip=config.get("School","IP")
mask=config.get("School","mask")
gateway=config.get("School","Gateway")
dns=config.get("School","DNS")

print((ip,mask+"\n"+gateway,dns))
