#! /usr/lib/python3
# -*- coding: utf-8 -*-

import os,json,time
import sys,logging
path=sys.path[0]

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%a, %d %b %Y %H:%M:%S',
                filename=os.path.join(path,'{}{}日志.log'.format(sys.argv[0],time.strftime("%Y%m%d%H%M"))),
                filemode='w+')

json_file='/home/gswewf/百度人物关系图谱/数据/音乐人物_演艺人物_演员_导演_歌手_明星.json'
#'/home/gswewf/百科/百度百科title_info_tag_poly/title_info.json'
#'/home/gswewf/百科/RDF/百度百科义项.rdf'
out_file='/home/gswewf/百度人物关系图谱/数据/音乐人物_演艺人物_演员_导演_歌手_明星.rdf'

def load_data():
    path='/home/gswewf/百科/百度百科title_info_tag_poly/json'
    for file in os.listdir(path):
        with open(os.path.join(path,file),encoding='utf8')as f:
            data=json.load(f)

        yield data
        
def property():
    #基本信息，属性
    with open(json_file,'r',encoding='utf8')as f:
        data=json.load(f)
    with open(out_file,'w+',encoding='utf8')as f:
        for k,v in data.items():
            title=v['title']
            for kv,vv in v['info'].items():
                line='<http://zhishi.me/baidubaike/resource/{}> <http://zhishi.me/baidubaike/property/{}> "{}"@zh . '.format(title,kv,vv)
                print(line,file=f)
            for vv in v['tag']:
                line='<http://zhishi.me/baidubaike/resource/{}> <http://zhishi.me/baidubaike/opentag> "{}"@zh . '.format(title,vv)
                print(line,file=f)
            for vv in v['poly']:
                line='<http://zhishi.me/baidubaike/resource/{}> <http://zhishi.me/baidubaike/polysemant> "{}"@zh . '.format(title,vv)
                print(line,file=f)

def open_tag():
    #词条标签
    with open('/home/gswewf/百科/百度百科title_info_tag_poly/title_tag.json','r',encoding='utf8')as f:
        data=json.load(f)
    with open('/home/gswewf/百科/RDF/百度百科词条标签.rdf','w',encoding='utf8')as f:
        for key,vlaue in data.items():
            if not key.split('_')[0]:
                continue
            if vlaue:
                for v in vlaue:
                    line='<http://zhishi.me/baidubaike/resource/{}> <http://zhishi.me/baidubaike/opentag> "{}"@zh . '.format(key.split('_')[0],v)
                    print(line,file=f)

def poly():
    #义项
    with open('/home/gswewf/百科/百度百科title_info_tag_poly/title_poly.json','r',encoding='utf8')as f:
        data=json.load(f)
    with open('/home/gswewf/百科/RDF/百度百科义项.rdf','w',encoding='utf8')as f:
        for key,vlaue in data.items():
            if not key.split('_')[0]:
                continue
            if vlaue:
                for v in vlaue:
                    line='<http://zhishi.me/baidubaike/resource/{}> <http://zhishi.me/baidubaike/polysemant> "{}"@zh . '.format(key.split('_')[0],v)
                    print(line,file=f)


def main():
    path='/home/gswewf/百度人物关系图谱/数据'
    files=[os.path.join(path,f) for f in os.listdir(path) if f.startswith('2016')and f.endswith('.json')]
    data={}
    for fi in files:
        with open(fi,encoding='utf8')as f:
            data.update(json.load(f))
            
    with open('/home/gswewf/百度人物关系图谱/数据/爬取的人物知识图谱.json','w',encoding='utf8')as f:
        json.dump(data,f,ensure_ascii=0)
if __name__ == "__main__":
    start=time.time()
    property()
    #main()
    print('用时：{}'.format(time.time()-start))


'''
{'6186191':
{'info': {'中文名': '孙立军', '出生地': '安徽宿州', '出生日期': '1963年', '性别': '男'},
 'poly': ['同济大学教授',
  '北京电影学院副院长',
  '东北农业大学纪委书记',
  '海拉尔啤酒有限责任公司董事长',
  '河南省淮滨县农场党委书记',
  '西藏自治区公安边防总队政治委员'],
 'tag': ['教育学家', '教授', '学者', '人物'],
 'title': '孙立军'},
 '''
