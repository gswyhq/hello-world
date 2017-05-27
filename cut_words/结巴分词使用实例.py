#!/usr/bin/python
# -*- coding:utf8 -*- #
from __future__ import generators
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# import time
import sys
import re
import jieba,jieba.posseg
import jieba.analyse
import json
import copy
import logging as logger
from itertools import product

MUSIC_ALIASES_DICT = 'aliases_dict.txt'

MUSIC_USERDICT = 'music_userdict.txt'
# 添加的自定义类型
MYSELF_ADD_DICT_TYPE = {
    "01": (10, "aliases"),  # 别名
    "02": (200, "poet"),  # 诗人
    "03": (200, "poetry"),  # 诗名
    "04": (200, "story"),  # ××可以讲的故事
    "05": (200, "myself_music"),  # ××自己的歌曲
    "06": (200, "other_type"),  # 不支持的类型
    "07": (200, "poetry_type"),  # 背诗
    "08": (200, "story_type"),  # 讲故事
    "09": (200, "music_type"),  # 歌曲
    "10": (200, "aaaaa_story_type"),  # ××的故事
    "11": (200, "display_type"),  # 展示的类型
    "12": (200, "sing_type"),  # ××自己的歌曲
    "13": (200, "stop_verb"),  #
    "14": (200, "poetry_verb"),  #
    "15": (200, "display_verb"),  #
    "16": (200, "sing_verb"),  #
    "17": (200, "story_verb"),  #
    "18": (200, "music_verb"),  #
    "19": (200, "album"),  # 歌曲专辑
    "20": (200, "music"),  # 歌曲名
    "21": (200, "author"),  # 歌手
    "22": (200, "tags"),  # 歌曲类型

}
AAAAA_POETRY_POET = 'poet'  # 诗人
AAAAA_POETRY_TITLE = 'title'  # 诗名
MUSIC_MAP_DATA = 'music_map_data.json'
# 分词时候，过滤掉的词
CUT_STOP_WORDS = ['名字', '给我', '一直', '谢谢', '3','还有']
# 其他的类型
OTHER_TYPES = ['小说', '小品', '有声读物', '有声书', '评书', '相声', '传纪']


AAAAA_POETRY_JSON_DATA = 'poetry.json'
AAAAA_STORY_NAMES = ['海的女儿',
                     '丑小鸭',
                     '豌豆上的公主',
                     '冰雪女王']
AAAAA_MYSELF_MUSIC_NAMES = [
    '***咏唱'
]
# 支持的类型
SUPPORT_TYPES = {
    "story_type":['故事','一个故事','故事先'],
    "aaaaa_story_type":[m+n for m,n in product(['你的', '你自己的', '××你自己的', '××你的', '××的', '××自己的', '自己的'],['故事','一个故事','故事先'])], # 两两组合
    "music_type":['歌曲','音乐', '歌', '曲子', '声乐', '一首歌'],
    "sing_type":[m+n for m,n in product(['你的', '你自己的', '××你自己的', '××你的', '××的', '××自己的', '自己的'],['歌曲','音乐', '歌', '曲子', '声乐', '一首歌'])], # 两两组合
    "display_type":['钢琴', '电吉他', '小提琴', '古筝', '木吉他', '吉他'],
    "poetry_type":['诗','诗词'],
}

# 支持类型的播放动词
SUPPORT_TYPES_VERB = {
    "story_verb":['换','播','放','听','讲','说',"说个",'说首','播放','来', '想听'] + [m+n for m,n in product(['不要不', '别不', '不用不','不必不', '不得不', '不准不', '不可不', '不许不'],['播', '放', '讲故事','播放', '听', '放故事', '播放故事'])],
    "music_verb":['换','播','放','听','来','点','播放', '想听'] + [m+n for m,n in product(['不要不', '别不', '不用不','不必不', '不得不', '不准不', '不可不', '不许不'],['唱','播', '放', '听歌','播放', '听', '放歌', '播放歌'])],
    "sing_verb":['唱','唱首歌', '唱一首歌'] + [m+n for m,n in product(['不要不', '别不', '不用不','不必不', '不得不', '不准不', '不可不', '不许不'],['唱', '唱歌'])],
    "display_verb":['弹','演','奏','拉','展示','来','表演','弹奏','演示'],
    "poetry_verb":['背','背诵'],
    "stop_verb":[m+n for m,n in product(['不要', '别', '不用', '不再', '不必', '不得', '不准', '不可', '不许'],['唱','播', '放', '唱歌','播放', '听', '放歌', '播放歌'])], # 两两组合
}

ZH_WORDS_PATTERN = '[\u4e00-\u9fa5]' # 中文字符

if sys.version >= '3':
    PY3 = True
else:
    PY3 = False

if not PY3:
    from codecs import open

sys.path.extend(['.', '../..'])

ZH_WORDS_PATTERN = re.compile(ZH_WORDS_PATTERN)

class MusicSegmentationWord():
    """切词模块，包括加载自定义词典;
    歌手别称词典；
    诗词、故事、××自己的歌曲词典；
    自定义的播放等动词"""
    def __init__(self):
        self.load_myself_userdict()
        self.ADD_MAP_DATA = {}
        self.aliases_map = {}  # 歌手别称与歌手的映射字典
        self.add_aliases()
        self.add_aaaaa_poetry()
        self.add_aaaaa_story()
        self.add_aaaaa_myself_music()
        self.add_other_type()
        self.add_verb_type()
        self.load_map_data()

    def load_myself_userdict(self):
        """加载自定义词典"""
        jieba.load_userdict(MUSIC_USERDICT)

    def add_word(self, key, code):
        """动态向结巴词典添加自定义的词"""
        if code not in MYSELF_ADD_DICT_TYPE:
            logger.info("动态添加字词的类型出错")
            return
        user_weight, user_type = MYSELF_ADD_DICT_TYPE.get(code)
        # 使用 add_word(word, freq=None, tag=None) 和 del_word(word) 可在程序中动态修改词典。
        jieba.add_word(key, user_weight, user_type)
        self.ADD_MAP_DATA.setdefault(key, {})
        self.ADD_MAP_DATA[key][user_type] = user_weight

    def add_aliases(self):
        """读取自定义的别称文件"""
        with open(MUSIC_ALIASES_DICT, encoding='utf8')as f:
            aliases = f.readlines()
        for t in aliases:
            if len(t.strip().split(' ', 2)) == 2:
                key, value = t.strip().split(' ', 2)
                self.add_word(key, '01')
                self.aliases_map.setdefault(key, value)

    def add_aaaaa_poetry(self):
        """添加××诗词背诵部分的数据"""
        with open(AAAAA_POETRY_JSON_DATA, encoding='utf8')as f:
            all_data = json.load(f)
        for line in all_data:
            poet = line.get(AAAAA_POETRY_POET)
            title = line.get(AAAAA_POETRY_TITLE)
            if poet:
                self.add_word(poet, '02')
            if title:
                self.add_word(title, '03')

    def add_aaaaa_story(self):
        """添加××可以讲的故事"""
        for story in AAAAA_STORY_NAMES:
            self.add_word(story, '04')

    def add_aaaaa_myself_music(self):
        """添加××可以讲的故事"""
        for myself_music in AAAAA_MYSELF_MUSIC_NAMES:
            self.add_word(myself_music, '05')

    def add_other_type(self):
        """添加不支持的类型"""
        for _type in OTHER_TYPES:
            # 不支持的类型
            self.add_word(_type, '06')

    def add_verb_type(self):
        """添加播放动词及支持的类型"""
        all_data = {}
        all_data.update(SUPPORT_TYPES)
        all_data.update(SUPPORT_TYPES_VERB)
        for _type, values in all_data.items():
            for code, v in MYSELF_ADD_DICT_TYPE.items():
                if v[1] == _type:
                    for value in values:
                        self.add_word(value, code)
                    break

    def load_map_data(self):
        with open(MUSIC_MAP_DATA, encoding='utf8')as f:
            self.map_data = json.load(f)
        for key, value in self.ADD_MAP_DATA.items():
            self.map_data.setdefault(key, {})
            self.map_data[key].update(value)

    def jieba_cut_words(self, text):
        """对输入的语句进行分词，分词结果如下：
        {"青花瓷": {"album": 11, "music": 670}, "播放": {"music_verb": 200, "story_verb": 200}, "周杰伦": {"album": 12, "music": 200, "author": 1902}}
        {"album": ["青花瓷", "周杰伦"], "music_verb": ["播放"], "author": ["周杰伦"], "music": ["青花瓷", "周杰伦"], "story_verb": ["播放"]}
        """
        text = text.upper()
        map_dict = {} # 获取分词对应的字段，结果如： {"小提琴": {"music": 5}, "筷子兄弟": {"author": 143}, "静夜思": {"album": 2, "music": 550, "poetry": 200}, "李白": {"poet": 200, "music": 90}}
        flag_word_map_dict = {}  # 获取词性及其对应的数据，结果如：{"album": ["静夜思"], "poet": ["李白"], "music": ["小提琴", "静夜思", "李白"], "poetry": ["静夜思"], "author": ["筷子兄弟"]}

        words = jieba.cut(text, cut_all=False, HMM=False)  # HMM 参数用来控制是否使用 HMM 模型

        english_words = [w for w in re.split(ZH_WORDS_PATTERN, text) if w]
        # start = time.time()
        for w in words:
            # print("分词的耗时：",time.time()-start)
            # {"我": {"album": 27, "music": 155}, "兰花草": {"album": 7, "music": 50}, "听": {"music": 20}}
            logger.info("结巴分词：{}".format(w))
            if w in CUT_STOP_WORDS:
                logger.info("屏蔽掉单词：{}".format(w))
                continue
            if not w.strip():
                logger.info("跳过空字符：{}".format(w))
                continue
            if not re.search(ZH_WORDS_PATTERN, w):
                # 若全部是英文字符
                logger.info("跳过非中文字符：{}".format(w))
                continue
            if self.aliases_map.get(w):
                # 别名，重定向
                logger.info("重定向：{}".format(w))
                w = self.aliases_map.get(w)
            if self.map_data.get(w):
                map_value = self.map_data[w]
                wash_map_value = self.del_music_album_tags(map_value)
                map_dict[w] = wash_map_value
            else:
                logger.info("单词映射词典无此词：{}".format(w))

        if english_words:
            # 利用中文字符分词得到的英文单个或多个词
            for w in english_words:
                logger.info("通过中文字符分词：{}".format(w))
                if w in CUT_STOP_WORDS:
                    logger.info("屏蔽掉单词：{}".format(w))
                    continue
                if [word for word in map_dict.keys() if w in word]:
                    logger.info("跳过中文分词已包括此单词：{}".format(w))
                    continue
                if self.aliases_map.get(w):
                    # 别名，重定向
                    logger.info("重定向：{}".format(w))
                    w = self.aliases_map.get(w)
                if self.map_data.get(w):
                    map_value = self.map_data[w]
                    wash_map_value = self.del_music_album_tags(map_value)
                    map_dict[w] = wash_map_value
                else:
                    logger.info("单词映射词典无此词：{}".format(w))

        # logger.info("李谷一169：{}".format(self.map_data.get('李谷一')))
        logger.info("分词处理结果：{}".format(json.dumps(map_dict, ensure_ascii=False)))
        for key, value in map_dict.items():
            for k, v in value.items():
                flag_word_map_dict.setdefault(k, [])
                flag_word_map_dict[k].append(key)
        return map_dict, flag_word_map_dict

    def del_music_album_tags(self, map_value):
        """当播放动词，播放类型词又是故事名、歌曲名时，就删除掉对应的歌曲名，故事名
        {"album": 61, "music": 845, "music_type": 200}
        """
        wash_map_value = copy.deepcopy(map_value)
        key_lists = ['poetry_type', 'sing_type', 'music_type', 'story_type', 'aaaaa_story_type', 'display_type', 'story_verb', 'music_verb', 'display_verb', 'poetry_verb', 'sing_verb']
        del_lists = ['album', 'music', 'tags', 'author']
        logger.info("{}".format(json.dumps(map_value, ensure_ascii=False)))
        for key, value in map_value.items():
            if [k for k in map_value.keys() if k in key_lists] and (key in del_lists):
                logger.info("删除掉{} 映射".format(key))
                del wash_map_value[key]

        return wash_map_value

    def jieba_analyse(self, sentence, top_k=20, allow_pos=None, _type='TF-IDF'):
        """
        基于 TF-IDF 算法的关键词提取
        jieba.analyse.extract_tags(sentence, topK = 20, withWeight = False, allowPOS = ())
        sentence:待提取的文本。
        topK:返回几个 TF/IDF 权重最大的关键词，默认值为20。
        withWeight:是否一并返回关键词权重值，默认值为False。
        allowPOS:仅包括指定词性的词，默认值为空，即不进行筛选。
        jieba.analyse.TFIDF(idf_path=None) 新建 TFIDF 实例，idf_path 为 IDF 频率文件。

        ###基于 TextRank 算法的关键词提取

        基本思想:
        将待抽取关键词的文本进行分词；
        以固定窗口大小(默认为5，通过span属性调整)，词之间的共现关系，构建图；
        计算图中节点的PageRank，注意是无向带权图。
        jieba.analyse.textrank(sentence, topK = 20, withWeight = False, allowPOS = ('ns', 'n', 'v', 'nv')) 注意默认过滤词性。
        jieba.analyse.TextRank() 新建自定义TextRank实例。
        nrfg:张三
        nr:周杰伦
        nz:神龙教、粤语
        n:青花瓷

        """
        if allow_pos is None:
            allow_pos = ()
        if _type == 'TF-IDF':
            return jieba.analyse.extract_tags(sentence, topK=top_k, withWeight=False, allowPOS=allow_pos)
        elif _type == 'TextRank':
            jieba.analyse.textrank(sentence, topK=top_k, withWeight=False, allowPOS=allow_pos)
        else:
            logger.info("输出的提取关键词算法参数有误，不应该为：{}".format(_type))
            return []


def main():
    spw= MusicSegmentationWord()
    map_dict, flag_word_map_dict = spw.jieba_cut_words('播放林海的A Blessed Wind和周杰伦的青花瓷')
    print("{}\n{}".format(json.dumps(map_dict, ensure_ascii=False), json.dumps(flag_word_map_dict, ensure_ascii=False)))

if __name__ == "__main__":
    main()
