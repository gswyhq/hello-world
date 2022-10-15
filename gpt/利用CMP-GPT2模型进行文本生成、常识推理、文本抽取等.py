#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from transformers import BertTokenizer, GPT2LMHeadModel,TextGenerationPipeline
USERNAME = os.getenv("USERNAME")

# 模型来源：https://huggingface.co/uer/gpt2-chinese-poem/tree/main
pretrained_model_name_or_path = rf"D:\Users\{USERNAME}\data\gpt2-chinese-poem"
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path, from_tf=True)
text_generator = TextGenerationPipeline(model, tokenizer)
text_generator("[CLS] 梅 山 如 积 翠 ，", max_length=50, do_sample=True)
    # [{'generated_text': '[CLS] 梅 山 如 积 翠 ， 丛 竹 隠 疏 花 。 水 影 落 寒 濑 ， 竹 声 随 暮 鸦 。 茅 茨 数 间 屋 ， 烟 火 两 三 家 。 安 得 携 琴 酒 ， 相 逢 烟 雨 赊 。 向 湖 边 过 ， 偏 怜 雪 里 看 。 浮 峦 如 画 出 ， 远 树 与 天 连 。 月 上 僧 房 静 ， 风 回 萤 火 寒 。 幽 情 何 可 写 ， 赖 有 子 期 弹 。 棠 真'}]

text_generator("[CLS] 梅 山 如 积 翠 ，", max_length=100, do_sample=True)
    # [{'generated_text': '[CLS] 梅 山 如 积 翠 ， 秀 出 何 其 雄 。 矫 矫 云 间 质 ， 映 日 生 玲 珑 。 根 大 乱 石 结 ， 枝 高 青 云 蒙 。 常 因 风 露 晚 ， 隠 映 瑶 台 中 。 忽 闻 山 石 裂 ， 万 里 吹 天 风 。 又 觉 此 身 高 ， 迥 出 凡 境 空 。 清 影 落 潭 水 ， 暗 香 来 逈 峰 。 却 寻 白 太 白 ， 月 影 摇 江 东 。 [SEP] 而 非'}]


# 但GPT2模型也只能是随机生成一些字词，并不能进行深度内容生成；如：
# 常识推理、单词翻译、主语抽取、三元组抽取
# 测试后，感觉是随机生成
# 模型来源：https://huggingface.co/uer/gpt2-chinese-poem/tree/main
# 模型来源：https://huggingface.co/uer/gpt2-chinese-cluecorpussmall/tree/main

###################################################### 测试 CPM-Generate-distill 模型 ###############################################

from transformers import XLNetTokenizer, TFGPT2LMHeadModel
from transformers import TextGenerationPipeline
import jieba
# add spicel process
class XLNetTokenizer(XLNetTokenizer):
    translator = str.maketrans(" \n", "\u2582\u2583")
    def _tokenize(self, text, *args, **kwargs):
        text = [x.translate(self.translator) for x in jieba.cut(text, cut_all=False)]
        text = " ".join(text)
        return super()._tokenize(text, *args, **kwargs)
    def _decode(self, *args, **kwargs):
        text = super()._decode(*args, **kwargs)
        text = text.replace(' ', '').replace('\u2582', ' ').replace('\u2583', '\n')
        return text

# https://huggingface.co/mymusise/CPM-Generate-distill
pretrained_model_name_or_path = rf"D:\Users\{USERNAME}\data\CPM-Generate-distill"
tokenizer = XLNetTokenizer.from_pretrained(pretrained_model_name_or_path)
model = TFGPT2LMHeadModel.from_pretrained(pretrained_model_name_or_path)

text_generater = TextGenerationPipeline(model, tokenizer)

print(text_generater("天下熙熙，", max_length=15, top_k=1, use_cache=True, prefix=''))

# 常识推理，期望输出：北京
query = u"""
美国的首都是华盛顿
法国的首都是巴黎
日本的首都是东京
中国的首都是
"""
text_generater(query.strip(), max_length=28, top_k=1, use_cache=True, prefix='')
# [{'generated_text': '美国的首都是华盛顿\n法国的首都是巴黎\n日本的首都是东京\n中国的首都是南京\n'}]

# 单词翻译
# 期望输出：bird, 实际输出：pig
query = u"""
狗 dog
猫 cat
猪 pig
鸟 
"""
text_generater(query.strip(), max_length=28, top_k=1, use_cache=True, prefix='')
# [{'generated_text': '狗 dog\n猫 cat\n猪 pig\n鸟 cat\n'}]

# 主语抽取
# 期望输出：杨振宁
query = u"""
从1931年起，华罗庚在清华大学边学习边工作 华罗庚
在一间简陋的房间里，陈景润攻克了“哥德巴赫猜想” 陈景润
在这里，丘成桐得到IBM奖学金 丘成桐
杨振宁在粒子物理学、统计力学和凝聚态物理等领域作出里程碑性贡献 
"""
text_generater(query.strip(), max_length=103, top_k=1, use_cache=True, prefix='')
# Out[20]: [{'generated_text': '从1931年起，华罗庚在清华大学边学习边工作 华罗庚\n在一间简陋的房间里，陈景润攻克了“哥德巴赫猜想” 陈景润\n在这里，丘成桐得到IBM奖学金 丘成桐\n杨振宁在粒子物理学、统计力学和凝聚态物理等领域作出里程碑性贡献 杨振宁\n'}]

# 三元组抽取
# 期望输出：张红,体重,140斤
query = u"""
姚明的身高是211cm，是很多人心目中的偶像。 ->姚明，身高，211cm
毛泽东是绍兴人，早年在长沙读书。->毛泽东，出生地，绍兴
虽然周杰伦在欧洲办的婚礼，但是他是土生土长的中国人->周杰伦，国籍，中国
小明出生于武汉，但是却不喜欢在武汉生成，长大后去了北京。->小明，出生地，武汉
吴亦凡是很多人的偶像，但是他却是加拿大人，另很多人失望->吴亦凡，国籍，加拿大
武耀的生日在5月8号，这一天，大家都为他庆祝了生日->武耀，生日，5月8号
《青花瓷》是周杰伦最得意的一首歌。->周杰伦，作品，《青花瓷》
北京是中国的首都。->中国，首都，北京
蒋碧的家乡在盘龙城，毕业后去了深圳工作。->蒋碧，籍贯，盘龙城
上周我们和王立一起去了他的家乡云南玩昨天才回到了武汉。->王立，籍贯，云南
昨天11月17号，我和朋友一起去了海底捞，期间服务员为我的朋友刘章庆祝了生日。->刘章，生日，11月17号
张红的体重达到了140斤，她很苦恼。->
"""
text_generater(query.strip(), max_length=402, top_k=1, use_cache=True, prefix='')
# [{'generated_text': '姚明的身高是211cm，是很多人心目中的偶像。 ->姚明，身高，211cm\n毛泽东是绍兴人，早年在长沙读书。->毛泽东，出生地，绍兴\n虽然周杰伦在欧洲办的婚礼，但是他是土生土长的中国人->周杰伦，国籍，中国\n小明出生于武汉，但是却不喜欢在武汉生成，长大后去了北京。->小明，出生地，武汉\n吴亦凡是很多人的偶像，但是他却是加拿大人，另很多人失望->吴亦凡，国籍，加拿大\n武耀的生日在5月8号，这一天，大家都为他庆祝了生日->武耀，生日，5月8号\n《青花瓷》是周杰伦最得意的一首歌。->周杰伦，作品，《青花瓷》\n北京是中国的首都。->中国，首都，北京\n蒋碧的家乡在盘龙城，毕业后去了深圳工作。->蒋碧，籍贯，盘龙城\n上周我们和王立一起去了他的家乡云南玩昨天才回到了武汉。->王立，籍贯，云南\n昨天11月17号，我和朋友一起去了海底捞，期间服务员为我的朋友刘章庆祝了生日。->刘章，生日，11月17号\n张红的体重达到了140斤，她很苦恼。->张红,籍贯,云南\n'}]

from transformers import TextGenerationPipeline
import jieba

text_generater = TextGenerationPipeline(model, tokenizer)

texts = [
    '今天天气不错',
    '天下武功, 唯快不',
    """
    我们在火星上发现了大量的神奇物种。有神奇的海星兽，身上是粉色的，有5条腿；有胆小的猫猫兽，橘色，有4条腿；有令人恐惧的蜈蚣兽，全身漆黑，36条腿；有纯洁的天使兽，全身洁白无瑕，有3条腿；有贪吃的汪汪兽，银色的毛发，有5条腿；有蛋蛋兽，紫色，8条腿。

    请根据上文，列出一个表格，包含物种名、颜色、腿数量。
    |物种名|颜色|腿数量|
    |亚古兽|金黄|2|
    |海星兽|粉色|5|
    |猫猫兽|橘色|4|
    |蜈蚣兽|漆黑|36|
    """
]

for text in texts:
    token_len = len(tokenizer._tokenize(text))
    print(text_generater(text, max_length=token_len + 15, top_k=1, use_cache=True, prefix='')[0]['generated_text'])
    print(text_generater(text, max_length=token_len + 15, do_sample=True, top_k=5)[0]['generated_text'])

# 今天天气不错,我就去了。 我在院子
# 今天天气不错,但是我觉得我的家人还没有到那个时候
# 天下武功, 唯快不破。 ”
# 天下武功, 唯快不破!” “ 你的武功,
#     我们在火星上发现了大量的神奇物种。有神奇的海星兽，身上是粉色的，有5条腿；有胆小的猫猫兽，橘色，有4条腿；有令人恐惧的蜈蚣兽，全身漆黑，36条腿；有纯洁的天使兽，全身洁白无瑕，有3条腿；有贪吃的汪汪兽，银色的毛发，有5条腿；有蛋蛋兽，紫色，8条腿。
#     请根据上文，列出一个表格，包含物种名、颜色、腿数量。
#     |物种名|颜色|腿数量|
#     |亚古兽|金黄|2|
#     |海星兽|粉色|5|
#     |猫猫兽|橘色|4|
#     |蜈蚣兽|漆黑|36|
#     |蜈蚣兽|白色|3|
#     我们在火星上发现了大量的神奇物种。有神奇的海星兽，身上是粉色的，有5条腿；有胆小的猫猫兽，橘色，有4条腿；有令人恐惧的蜈蚣兽，全身漆黑，36条腿；有纯洁的天使兽，全身洁白无瑕，有3条腿；有贪吃的汪汪兽，银色的毛发，有5条腿；有蛋蛋兽，紫色，8条腿。
#     请根据上文，列出一个表格，包含物种名、颜色、腿数量。
#     |物种名|颜色|腿数量|
#     |亚古兽|金黄|2|
#     |海星兽|粉色|5|
#     |猫猫兽|橘色|4|
#     |蜈蚣兽|漆黑|36|
#     |蜈蚣兽|紫色|6|

from transformers import XLNetTokenizer, TFGPT2LMHeadModel, TextGenerationPipeline
text_generater = TextGenerationPipeline(model, tokenizer)
text_generater('美国首都是华盛顿，中国首都是', max_length=11)
# Out[81]: [{'generated_text': '美国首都是华盛顿，中国首都是北京'}]

# CPM-Generate-distill, 在“常识推理、单词翻译、主语抽取、三元组抽取”上虽说没有像一般的GPT2模型一样随机生成，但生成的结果也是不正确的。
# CPM(Chinese Pretrained Models)
# 若机器条件允许，可以尝试更多模型：
# https://huggingface.co/mymusise/CPM-GPT2-FP16
# https://huggingface.co/TsinghuaAI/CPM-Generate/tree/main


def main():
    pass


if __name__ == '__main__':
    main()
