#!/usr/bin/python3
# coding: utf-8

import requests
import json
import datetime

ES_HOST = '192.168.3.145'
# ES_HOST = '52.80.177.148'  # 外网30G内存对应的数据库
# ES_HOST = '52.80.187.77'  # 外网，南迪数据管理关联的es数据库；
# ES_HOST = '10.13.70.57'  # 外网词向量测试服
# ES_HOST = '10.13.70.173'  # 外网词向量正式服
# ES_HOST = None
ES_PORT = '9200'
ES_INDEX = 'intent'  # 必须是小写
ES_USER = 'elastic'
ES_PASSWORD = 'web12008'

intent_entity_dict = {'是否承保某个疾病': [['Jibing'],[]],
'保障项目的责任免除详情': [[],['Baozhangxiangmu']],
 '询问合同变更': [[],[]],
 '保障项目如何申请理赔': [['Baozhangxiangmu'],[]],
 '询问犹豫期': [[],[]],
 '询问保险责任': [[],[]],
 '询问合同恢复': [[],[]],
 '保险事故通知': [[],[]],
 '询问保单借款': [[],[]],
 '询问特别约定': [[],[]],
 '询问合同终止': [[],[]],
 '询问合同生效': [[],[]],
 '宣告死亡': [[],[]],
 '询问如实告知': [[],[]],
 '询问合同解除': [[],[]],
'询问保险期间': [[],[]],
 '询问保费垫缴': [[],[]],
 '询问保险金额': [[],[]],
 '询问受益人': [[],[]],
 '询问体检或验尸': [[],[]],
 '询问未归还款项偿还': [[],[]],
 '询问合同构成': [[],[]],
 '询问争议处理': [[],[]],
 '询问投保年龄': [[],[]],
 '询问适用币种': [[],[]],
 '询问基本保险金额': [[],[]],
 '询问合同种类': [[],[]],
 '询问等待期': [[],[]],
 '询问减额缴清': [[],[]],
 '询问缴费方式': [[],['Jiaofeifangshi']],
 '询问缴费年期': [[],[]],
'服务内容介绍': [[],[]],
 '保哪些疾病': [[],[]],
 '询问公司有哪些产品': [[],['Baozhangxiangmu','Baoxianchanpin']],
 '询问宽限期': [[],[]],
 '询问产品优势': [[],['Baoxianchanpin']],
 '询问免赔额': [[],[]],
 '询问保险费缴纳': [[],[]],
 '减额缴清': [[],[]],
 '询问诉讼时效': [[],[]],
 '询问变更通讯方式': [[],[]],
 '询问保险金给付': [[],[]],
 '某科目排名全国第一的医院': [[],['Yiyuan']],
 '询问区别': [[],[]],
 '咨询时间': [[],[]],
 '患某疾病是否可以投保': [['Jibing'],[]],
 '保单借款的还款期限': [[],[]],
 '某种情景是否可以投保': [['Qingjing'],[]],
 '保险种类的定义': [['Baoxianzhonglei'],[]],
'某情景是否在承保范围内': [['Qingjing'],[]],
'疾病种类的定义': [['Jibingzhonglei'],[]],
'保障项目的定义': [['Baozhangxiangmu'],[]],
 '得了某类疾病怎样申请理赔': [['Jibingzhonglei'],[]],
'询问失踪处理': [[],[]],
 '疾病种类包含哪些疾病': [['Jibingzhonglei'],[]],
 '某疾病是否属于某疾病种类': [['Jibing','Jibingzhonglei'],[]],
 '询问信息误告': [[],[]],
 '询问疾病种类包含哪些': [['Jibingzhonglei'],[]],
 '某种疾病的预防': [['Jibing'],[]],
 '某种疾病的高发区域': [['Jibing'],[]],
 '某种体检异常指标分析': [[],[]],
 '某种体检异常指标定义': [[],[]],
 '身体器官的构成': [[],[]],
 '疾病发病原因': [['Jibing'],[]],
 '某项体检的包含项': [[],[]],
 '某平台使用方法': [[],[]],
 '询问合同领取方式': [[],[]],
 '投保途径': [[],[]],
 '投保人和被保险人关系': [[],[]],
 '区分疾病种类的标准': [[],[]],
 '保障项目的年龄限制': [['Baozhangxiangmu'],[]],
 '公司经营状况': [[],[]],
 '投保的职业要求': [[],[]],
 '购买某项保险产品有无优惠': [['Baoxianchanpin'],[]],
 '某类险种的缴费方式': [['Baoxianzhonglei'],[]],
 '体检的时限要求': [[],[]],
 '某渠道的投保流程': [[],[]],
 '保险费定价': [[],[]],
 '理赔渠道申请流程': [[],[]],
 '某产品的最低保额': [['Baoxianchanpin'],[]],
 '理赔后缓缴保险费的处理办法': [[],[]],
 '申请某保障项目所需资料': [[],['Baozhangxiangmu']],
 '公司股东构成': [[],[]],
 '不同保障项目的重复理赔': [['Baozhangxiangmu'],[]],
 '询问产品价格': [[],['Baoxianchanpin']],
 '公司介绍': [[],[]],
 '产品是否属于某一类险': [['Baoxianchanpin','Baoxianzhonglei'],[]],
 '承保区域': [[],[]],
 '保障项目的赔付次数': [['Baozhangxiangmu'],[]],
 '保费支付渠道': [[],[]],
 '财务问卷针对的对象': [[],[]],
 '疾病种类治疗费用': [['Jibingzhonglei'],[]],
 '特殊人群投保规则': [[],[]],
 '推荐保额': [[],[]],
 '核保不通过解决办法': [[],[]],
 '体检标准': [[],[]],
 '投保所需资料': [[],[]],
 '询问保险条款': [[],[]],
 '理赔医院': [[],['Yiyuan']],
 '保障项目的赔付': [[],['Baozhangxiangmu']],
 '险种介绍': [[],['Baoxianzhonglei']],
 '产品推荐': [[],['Baoxianchanpin']],
 '工具的使用限制': [[],[]],
 '保险类型判断': [[],[]],
 '诊断报告的领取': [[],[]],
 '某人购买产品是否有优惠': [[],['Baoxianchanpin']],
'产品升级介绍': [['Baoxianchanpin'],[]],
 '体检指引': [[],[]],
 '产品保额限制': [[],['Baoxianchanpin']],
 '某情景是否具备购买产品资格': [['Qingjing'],['Baoxianchanpin']],
 '询问疾病种类的生存期': [['Jibingzhonglei'],[]],
 '理赔的地域限制': [[],[]],
 '特殊人群投保限额': [[],[]],
 '权益间的关系': [[],[]],
 '保障项目的保障范围': [['Baozhangxiangmu'],[]],
 '产品介绍': [['Baoxianchanpin'],[]],
 '申请纸质合同': [[],[]],
 '疾病种类的赔付流程': [[],[]],
 '理赔速度': [[],[]],
 '产品保障的地域限制': [[],[]],
 '合同丢失': [[],[]],
'电子合同介绍': [[],[]],
 '保单查询': [[],[]],
 '理赔条件': [[],[]],
 '赔付方式': [[],[]],
 '投保指引': [[],[]], '业务办理提交材料': [[],[]], '投保人与被保险人关系': [[],[]], '投保人条件限制': [[],[]], '红利申请规定': [[],[]], '合同自动终止的情景': [[],['Qingjing']], '理赔报案的目的': [[],[]], '险种关联选择申请资料': [[],[]], '变更投保人': [[],[]], '查询方式': [[],[]], '得了某个疾病怎样申请理赔': [['Jibing'],[]], '疾病是否赔某个保障项目': [['Jibing','Baozhangxiangmu'],[]], '情景是否赔偿某个保障项目': [['Qingjing','Baozhangxiangmu'],[]],
 '情景的定义': [['Qingjing'],[]], '疾病的定义': [['Jibing'],[]], '询问保险事故通知': [[],[]], '疾病包含哪些疾病': [['Jibing'],[]], '如何申请某项服务': [[],[]], '释义项的定义': [['Shiyi'],[]]}


def intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER, es_password=ES_PASSWORD, data=None, pid='all_baoxian'):
    es_index = "{}_{}".format(pid, _index)
    now = datetime.datetime.now()
    index_end = now.strftime('%Y%m%d_%H%M%S')
    current_es_index = "{}_{}".format(es_index, index_end).lower()

    alias_name = '{}_alias'.format(es_index)

    url = "http://{}:{}/{}/{}/_bulk".format(es_host, es_port, current_es_index, _type)

    all_data = ''
    # del_alias_name = {"delete": {"_index": alias_name}}
    # all_data += json.dumps(del_alias_name, ensure_ascii=False) + '\n'
    for template_id, (intent, entity_list) in enumerate(data.items()):
        bixuan, kexuan = entity_list
        doc = {"intent": intent, "必选实体": bixuan, "可选实体": kexuan, '模板id': template_id}
        create_data = {"create": {"_id": template_id}}
        all_data += json.dumps(create_data, ensure_ascii=False) + '\n'
        all_data += json.dumps(doc, ensure_ascii=False) + '\n'
    print(url)
    print(all_data)
    ret = requests.post(url=url, data=all_data.encode('utf8'), auth=(es_user, es_password))
    print(ret.json())

    # 添加别名
    data = {
        "actions": [
            {"remove": {
                "alias": alias_name,
                "index": "_all"
            }},
            {"add": {
                "alias": alias_name,
                "index": current_es_index
            }}
        ]
    }
    url = "http://{}:{}/_aliases".format(es_host, es_port)
    r = requests.post(url, json=data, auth=(es_user, es_password))
    print(r.json())

def main():
    es_user = 'elastic'
    es_password = 'web12008'
    es_host = '192.168.3.145'
    es_port = '9200'
    _index = 'intent'
    _type = 'intent'

    intent_to_es(es_host=ES_HOST, es_port=ES_PORT, _index=ES_INDEX, _type='intent', es_user=ES_USER,
                 es_password=ES_PASSWORD, data=intent_entity_dict, pid='all_baoxian')


if __name__ == '__main__':
    main()