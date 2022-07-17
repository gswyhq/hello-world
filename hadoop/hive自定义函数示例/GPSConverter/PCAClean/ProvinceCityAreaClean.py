# FileName: Test.py
# 示例代码：将输入的字符串转变为大写


import os
import json
import functools
import pickle as pkl
#import pandas as pd
# from addtopost import ADDTOPOST
# from addtopost import *

'''
省市区数据有好多缺失或有误；
特别是来自百度地图的数据；

省市区名称一致的，一般情况下是有问题的；
比如，对内蒙古自治区，名为“青山区”的城市进行统计：
若仅仅根据统计计数，则结果是有问题的；
故而需要先剔除错误结果（省=城市|城市=区县），再按统计结果取值。
即，内蒙古自治区名为青山区的是哪个城市？
先排除“青山区”、“内蒙古自治区”，再按统计学结果取值“包头市”；
但若有省份，不同城市有相同名称的区。对根据城市代码区分；
但上面清洗也有例外，比如东莞市、中山、三亚、嘉峪关，不设区的市；

故而将地图数据，省市区全部导出，再整理；但也有例外，比如所有数据中，找不到深圳市大鹏新区。

'''

class ProvinceCityAreaClean():
    '''对省市区进行清洗'''
    def __init__(self):
        addr_file = 'addtopost.pkl' if os.path.isfile("addtopost.pkl") else os.path.join('/tmp', 'addtopost.pkl')
        print("文件：{}， 存在与否：{}".format(addr_file, os.path.isfile(addr_file)))
        try:
            with open(addr_file, 'rb')as f:
                addtopost_dict = pkl.load(f)
                print('读取文件 {} 成功！'.format(addr_file))
        except Exception as e:
            print("读取{}文件，出错：{}".format(addr_file, e))
            addtopost_dict = {"NAME_PCA_DICT": {}, "SHORT_LONG_DICT": {}, "NAME_PCA_LIST_DICT": {}, "PCA_ID_DETAIL": {}}

#        addtopost_dict = ADDTOPOST
#        province_city_area_datas = []
#        for province, data in addtopost_dict.items():
#            for city, d in data.items():
#                if city:
#                    province_city_area_datas.extend([[province, city, area] for area in d.keys()])
#        df = pd.DataFrame(province_city_area_datas, columns=['province', 'city', 'area'])
#
#        # 在无歧义时，根据区名，补全省市，如：三水区-> ['广东省', '佛山市', '三水区']；或者根据市名补全省；东莞市-> ['广东省', '东莞市', '']
#        # name_pca_dict
#        # 判断唯一标准：若其他省份没有同名市，则市唯一,但若是直辖市的话，则判断其他城市没有同名区；
#        # 若其他市没有同名区，且直辖市下没有同名区，则区唯一；
#        df_pc = df.drop_duplicates(subset=['province', 'city'],keep='first')
#        unique_city_list = [city for province, city, _ in df_pc.values if province not in ('上海市', '重庆市', '北京市', '天津市') and df_pc[(df_pc['city']==city) & ~(df_pc['province']==province)].shape[0]==0] + \
#                           [city for province, city, _ in df_pc.values if province in ('上海市', '重庆市', '北京市', '天津市') and df[(df['area']==city) & ~(df['province']==province)].shape[0]==0]
#        unique_area_list = [area for province, city, area in df.values if df[(df['area']==area) & ~(df['city']==city)].shape[0]==0 and df[(df['province'].isin(['上海市', '重庆市', '北京市', '天津市'])) & (df['city']==area)].shape[0]==0]
#        name_pca_dict = {}  # 通过无歧义的名称，获取对应的，省市区id;
#        pca_id_detail = {}  # 省市区id对应具体省市区名称；
#        name_pca_list_dict = {}  # 对于有歧义的名称，获取对应的，省市区id列表；
#
#        for pca_id, (province, city, area) in enumerate(df.values):
#            pca_id_detail.setdefault(pca_id, [province, city, area])
#            name_pca_list_dict.setdefault(province, [])
#            name_pca_list_dict.setdefault(city, [])
#            name_pca_list_dict.setdefault(area, [])
#            name_pca_list_dict[province].append(pca_id)
#            name_pca_list_dict[city].append(pca_id)
#            name_pca_list_dict[area].append(pca_id)
#
#            if city in unique_city_list and not area:
#                name_pca_dict.setdefault(city, pca_id)
#            elif area in unique_area_list:
#                name_pca_dict.setdefault(area, pca_id)
#
#        # 无歧义时，通过简称进行补全；
#        # 对省份直接进行处理
#        province_unique_list = df['province'].unique()
#        long_short2_dict = {long_name: long_name[:2] for long_name in province_unique_list if len(long_name) > 2}
#        long_short3_dict = {long_name: long_name[:3] for long_name in province_unique_list if len(long_name) > 3}
#        long_short4_dict = {long_name: long_name[:4] for long_name in province_unique_list if len(long_name) > 4}
#        short_long_list_dict = {}  # 短名，对应可能的长名；
#        for long_short_dict in [long_short2_dict, long_short3_dict, long_short4_dict]:
#            for long_name, short_name in long_short_dict.items():
#                short_long_list_dict.setdefault(short_name, [])
#                short_long_list_dict[short_name].append(long_name)
#
#        # 短名-> 长名映射，如：新疆-> 新疆维吾尔自治区； 虽说，长沙->  ['长沙县', '长沙市'], 但长沙市是上位，故-> 长沙市；
#        short_long_dict = {short_name: long_name_list[0] for short_name, long_name_list in short_long_list_dict.items() if len(long_name_list)==1}
#
#        # 城市简写无歧义判断：除直辖市外，其他城市或区县没有简写相同则视为无歧义；
#        city_other_city_area_dict = {}
#        df_ca = df.drop_duplicates(subset=['area', 'city'],keep='first')
#        for city in df[~df['province'].isin(['上海市', '重庆市', '北京市', '天津市'])]['city'].unique():
#            city_other_city_area_dict[city] = [c for c in df['city'].unique() if c != city and c] + [area for _, c, area in df_ca.values if c != city and c]
#        for city, other_city in city_other_city_area_dict.items():
#            if len(city) > 4:
#                if not any(city[:4]==c[:4] for c in other_city):
#                    short_long_dict.setdefault(city[:4], city)
#                    short_long_list_dict.setdefault(city[:4], [])
#                    short_long_list_dict[city[:4]].append(city)
#                if not any(city[:3]==c[:3] for c in other_city):
#                    short_long_dict.setdefault(city[:3], city)
#                    short_long_list_dict.setdefault(city[:3], [])
#                    short_long_list_dict[city[:3]].append(city)
#                if not any(city[:2]==c[:2] for c in other_city):
#                    short_long_dict.setdefault(city[:2], city)
#                    short_long_list_dict.setdefault(city[:2], [])
#                    short_long_list_dict[city[:2]].append(city)
#            elif len(city) > 3:
#                if not any(city[:3]==c[:3] for c in other_city):
#                    short_long_dict.setdefault(city[:3], city)
#                    short_long_list_dict.setdefault(city[:3], [])
#                    short_long_list_dict[city[:3]].append(city)
#                if not any(city[:2]==c[:2] for c in other_city):
#                    short_long_dict.setdefault(city[:2], city)
#                    short_long_list_dict.setdefault(city[:2], [])
#                    short_long_list_dict[city[:2]].append(city)
#            elif len(city) > 2:
#                if not any(city[:2]==c[:2] for c in other_city):
#                    short_long_dict.setdefault(city[:2], city)
#                    short_long_list_dict.setdefault(city[:2], [])
#                    short_long_list_dict[city[:2]].append(city)
#        # 有些地名有歧义，但补全没有歧义；比如，虽说北京、长春都有朝阳区，但完全可以将 朝阳-> 朝阳区，但若存在一个朝阳市，则不能这么映射；
#        all_unique_list = list(df['province'].unique()) + list(df['city'].unique()) + list(df['area'].unique())
#
#        for name in all_unique_list:
#            if len(name) > 4:
#                for short_name in [name[:4], name[:3], name[:2]]:
#                    short_long_list_dict.setdefault(short_name, [])
#                    short_long_list_dict[short_name].append(name)
#            elif len(name) > 3:
#                for short_name in [name[:3], name[:2]]:
#                    short_long_list_dict.setdefault(short_name, [])
#                    short_long_list_dict[short_name].append(name)
#            elif len(name) > 2:
#                for short_name in [name[:2]]:
#                    short_long_list_dict.setdefault(short_name, [])
#                    short_long_list_dict[short_name].append(name)
#
#        short_long_list_dict = {k: list(set(v)) for k, v in short_long_list_dict.items()}
#        for short_name, long_name_list in short_long_list_dict.items():
#            for long_name in long_name_list:
#                name_pca_list_dict.setdefault(short_name, [])
#                name_pca_list_dict[short_name].extend(name_pca_list_dict[long_name])
#            if len(set(long_name_list)) == 1:
#                short_long_dict.setdefault(short_name, long_name_list[0])
#
#        # pca_id_detail[name_pca_dict.get('宝安')]
#        for short_name, long_name in short_long_dict.items():
#            pca_id = name_pca_dict.get(long_name)
#            if pca_id:
#                name_pca_dict.setdefault(short_name, pca_id)
        NAME_PCA_DICT = addtopost_dict["NAME_PCA_DICT"]
        SHORT_LONG_DICT = addtopost_dict["SHORT_LONG_DICT"]
        NAME_PCA_LIST_DICT = addtopost_dict["NAME_PCA_LIST_DICT"]
        PCA_ID_DETAIL = addtopost_dict["PCA_ID_DETAIL"]
        self.name_pca_dict = NAME_PCA_DICT
        self.short_long_dict = SHORT_LONG_DICT
        self.name_pca_list_dict = NAME_PCA_LIST_DICT
        self.pca_id_detail = {int(k): v for k, v in PCA_ID_DETAIL.items()}

    def clean(self, province, city, area):
        '''对输入的省市区进行清洗
        情形1： 省市有缺失，但区名称无歧义，补全省市区，如: ['', '三水区', ''] -> ['广东省', '佛山市', '三水区']
        清洗2：省份缺失，市区县无歧义，则补全，如：['', '深圳', '南山'] -> ['广东省', '深圳市', '南山区']
        ['', '三水区', ''] -> ['广东省', '佛山市', '三水区']
        ['', '深圳', '南山'] -> ['广东省', '深圳市', '南山区']
        ['广东', '宝安', ''] -> ['广东省', '深圳市', '宝安区']
        '''
        # # 输入数据，仅有一个不为空
        # if sum([1 for t in (province, city, area) if t]) == 1:
        #     pca_id = self.name_pca_dict.get(province or city or area)
        #     if not pca_id is None:
        #         return self.pca_id_detail[pca_id]
        # elif province and city and area:
        province_pca_list = self.name_pca_list_dict.get(province, []) if province else []
        city_pca_list = self.name_pca_list_dict.get(city, []) if city else []
        area_pca_list = self.name_pca_list_dict.get(area, []) if area else []

        # 输入的省市区内容不为空，但无法标准化；
        if (province and not province_pca_list) or (city and not city_pca_list) or (area and not area_pca_list):
            return ['', '', '']
        if any([province_pca_list, city_pca_list, area_pca_list]):
            pca_list = list(functools.reduce(lambda x,y: x&y, [set(t) for t in [p_list for p_list in [province_pca_list, city_pca_list, area_pca_list] if p_list]]))
            if len(pca_list) == 1:
                return self.pca_id_detail[pca_list[0]]

            # 若有歧义项，忽略；若有非歧义项，先确定非歧义项；如： ['', '深圳', ''] -> ['广东省', '深圳市', '']
            elif pca_list:
                # pca_name_list = [self.pca_id_detail[pca_id] for pca_id in pca_list]
                p_name_list, c_name_list, a_name_list = [], [], []
                for pca_id in pca_list:
                    p, c, a = self.pca_id_detail[pca_id]
                    if p:
                        p_name_list.append(p)
                    if c:
                        c_name_list.append(c)
                    if a:
                        a_name_list.append(a)
                return [p_name_list[0] if len(set(p_name_list)) == 1 else '',
                        c_name_list[0] if len(set(c_name_list)) == 1 else '',
                        a_name_list[0] if len(set(a_name_list)) == 1 else ''
                        ]

        # 省市区不为空，但有一项错误，则矫正；如：['安徽', '深圳', '瑶海区'] -> ['安徽省', '合肥市', '瑶海区']
        if province_pca_list and city_pca_list and area_pca_list and not functools.reduce(lambda x, y: x&y, [set(x) for x in [province_pca_list, city_pca_list, area_pca_list]]):
            pc_list = list(set(province_pca_list) & set(city_pca_list))
            pa_list = list(set(province_pca_list) & set(area_pca_list))
            ca_list = list(set(city_pca_list) & set(area_pca_list))
            if len(pc_list) == 1 and not pa_list and not ca_list:
                return self.pca_id_detail[pc_list[0]]
            elif len(pa_list) == 1 and not pc_list and not ca_list:
                return self.pca_id_detail[pa_list[0]]
            elif len(ca_list) == 1 and not pa_list and not pc_list:
                return self.pca_id_detail[ca_list[0]]

            # 存在歧义，且不能进行消歧，但可以确定部分，如：['广东省', '深圳', '惠阳区'] -> ['广东省', '', '']
            p_name_list, c_name_list, a_name_list = [], [], []
            for pca_id in pc_list+pa_list+ca_list:
                p, c, a = self.pca_id_detail[pca_id]
                if p:
                    p_name_list.append(p)
                if c:
                    c_name_list.append(c)
                if a:
                    a_name_list.append(a)
            return [p_name_list[0] if len(set(p_name_list)) == 1 else '',
                    c_name_list[0] if len(set(c_name_list)) == 1 else '',
                    a_name_list[0] if len(set(a_name_list)) == 1 else ''
                    ]

        # elif not province and city and area:
        #     city_pca_list = self.name_pca_list_dict.get(city, [])
        #     area_pca_list = self.name_pca_list_dict.get(area, [])
        #     pca_list = list(set(city_pca_list) & set(area_pca_list))
        #     if len(pca_list) == 1:
        #         return self.pca_id_detail[pca_list[0]]
        return ['', '', '']


    def logic(self, param):
#         print('这是一个python函数')
#         print('python函数参数是： [%s]' % param)
        # return param.upper()
        if not isinstance(param, str) or len(param.split(';')) != 3:
            result =  ['', '', '']
        else:
            result = self.clean(*param.split(';'))
        # result = self.clean('', '深圳', '宝安')
#         print('返回结果：{}'.format(result))
        return ';'.join(result)

pca_clean = ProvinceCityAreaClean()

# 接口函数，导出给Java Native的接口
def JNI_API_TestFunction(param):
#   print("进入 JNI_API_test_function")
  result = pca_clean.logic(param)
#   print("离开 JNI_API_test_function")
  return result

def main():
   test_list = [
       ['', '三水区', ''],
       ['', '深圳', '南山'],
       ['广东', '宝安', ''],
       ['安徽省', '深圳', '瑶海区'],
       ['', '深圳', ''],
       ['广东省', '深圳', '惠阳区']
   ]
#    pca_clean = ProvinceCityAreaClean()
   for province, city, area in test_list:
       province1, city1, area1 = pca_clean.clean(province, city, area)
       print('{} -> {}'.format([province, city, area], [province1, city1, area1]))

if __name__ == '__main__':
   main()

