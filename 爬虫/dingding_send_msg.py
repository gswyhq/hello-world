#!/usr/bin/python3
# coding: utf-8

import requests

# 在机器人管理页面选择“自定义”机器人，输入机器人名字并选择要发送消息的群。如果需要的话，可以为机器人设置一个头像。点击“完成添加”。
# 
# 点击“复制”按钮，即可获得这个机器人对应的Webhook地址，其格式如下
# https://oapi.dingtalk.com/robot/send?access_token=xxxxxxxx
# 
# 使用自定义机器人:
# 获取到Webhook地址后，用户可以使用任何方式向这个地址发起HTTP POST 请求，即可实现给该群组发送消息。注意，发起POST请求时，必须将字符集编码设置成UTF-8。
# 当前自定义机器人支持文本（text）、连接（link）、markdown（markdown）三种消息类型，大家可以根据自己的使用场景选择合适的消息类型，达到最好的展示样式。具体的消息类型参考下一节内容。
# 自定义机器人发送消息时，可以通过手机号码指定“被@人列表”。在“被@人列表”里面的人员，在收到该消息时，会有@消息提醒（免打扰会话仍然通知提醒，首屏出现“有人@你”）

# curl 'https://oapi.dingtalk.com/robot/send?access_token=d7a0c30704c888c1'    -H 'Content-Type: application/json'    -d '
# {
#     "msgtype": "text",
#     "text": {
#         "content": "我就是我,  @15323** 这是一个测试"
#     },
#     "at": {
#         "atMobiles": [
#             "15323**"
#         ],
#         "isAtAll": false
#     }
# }'

# 更多文档信息：https://open-doc.dingtalk.com/docs/doc.htm?spm=a219a.7629140.0.0.karFPe&treeId=257&articleId=105735&docType=1

def main():
    pass


if __name__ == '__main__':
    main()
