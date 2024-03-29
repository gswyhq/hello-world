#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# PromptCLUE：支持最多中文任务的开源预训练模型
# 这个模型是基于1000亿token中文语料上预训练，并且在数百种任务上进行Prompt任务式训练。针对理解类任务，如分类、情感分析、抽取等，可以自定义标签体系；针对多种生成任务，可以进行采样自由生成。
# 模型来源：
# https://huggingface.co/ClueAI/PromptCLUE-base/tree/main
# https://github.com/clue-ai/PromptCLUE

import os 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

USERNAME = os.getenv("USERNAME")

pretrained_model_name_or_path = rf"D:\Users\{USERNAME}\data\PromptCLUE-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
# 使用模型进行预测推理方法：

import torch
device = torch.device('cpu')
# device = torch.device('cuda')
model.to(device)
def preprocess(text):
    return text.replace("\n", "_")

def postprocess(text):
    return text.replace("_", "\n")

def answer(text, sample=False, top_p=0.6):
    '''sample：是否抽样。生成任务，可以设置为True;
    top_p：0-1之间，生成的内容越多样'''
    text = preprocess(text)
    encoding = tokenizer(text=[text], truncation=True, padding=True, max_length=768, return_tensors="pt").to(device)
    if not sample:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=128, num_beams=4, length_penalty=0.6)
    else:
        out = model.generate(**encoding, return_dict_in_generate=True, output_scores=False, max_length=64, do_sample=True, top_p=top_p)
    out_text = tokenizer.batch_decode(out["sequences"], skip_special_tokens=True)
    return postprocess(out_text[0])


test_inputs_list = [('意图分类', '''意图分类：
帮我定一个周日上海浦东的房间
选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
答案：'''),
('新闻分类', '''新闻分类：
本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
选项：财经，法律，国际，军事
答案：'''),
('情感分析', '''情感分析：
这个看上去还可以，但其实我不喜欢
选项：积极，消极
答案：'''),
('推理', '''推理关系判断：
前提：小明今天在北京
假设：小明在深圳旅游
选项：矛盾，蕴含，中立
答案：'''),
('阅读理解', '''阅读理解：
段落：本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
问题：北溪管道是什么原因泄露的？
答案：'''),
('阅读理解-自由式', '''阅读以下对话并回答问题。
男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
问题：女的怎么样？
选项：正在工作，感冒了，在打电话，要出差。
答案：'''),
('摘要', '''摘要生成：
本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
答案：'''),
('翻译-中英', '''翻译成英文：
前国务卿去了台湾，中国人民很愤怒。
答案：'''),
('翻译-英中', '''翻译成中文：
This is a dialogue robot that can talk to people.
答案：'''),
('通用信息抽取', '''信息抽取：
据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
问题：机构名，人名，职位
答案：'''),
('简历信息抽取', '''阅读文本抽取关键信息：
张玄武1990年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
答案：'''),
('医疗信息抽取', '''找出指定的信息：
我今天胃不舒服，手有点麻木，浑身不舒服，前几天去医院医生说我有胃炎
问题：症状，治疗，检查，身体部位，疾病
答案：'''),
('电商客户需求分析', '''电商客户诉求分类：
收到但不太合身，可以退换吗
选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
答案：'''),
('医疗语义相似度', '''下面句子是否表示了相同的语义：
文本1：糖尿病腿麻木怎么办？
文本2：糖尿病怎样控制生活方式
选项：相似，不相似
答案：'''),
('问题生成', '''问题生成：
本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的领导人表示，这是一起蓄意破坏行为。
相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
答案：'''),
('指代消解', '''代词指向哪个名词短语：
段落：
我回来，咱们两个人劳动，一定要把少平和兰香的书供成。只要他两个有本事，能考到哪里，咱们就把他们（代词）供到哪里。
问题：代词“ 他们”指代的是？
答案：'''),
('关键词抽取', ''' 抽取关键词：
当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
关键词：'''),
('情感倾向', '''文字中包含了怎样的情感：
超可爱的帅哥，爱了。。。
选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
答案：'''),
('标题生成文章', '''根据标题生成文章：
标题：俄罗斯天然气管道泄漏爆炸
答案：'''),
('中心词提取', '''中心词提取：
现在有京东国际太方便了，可以安全的买到正经的电子设备，游戏机完全没问题，发货非常快，第二天就到
答案：'''),
('改写', '''生成与下列文字相同意思的句子：
长期通胀前景在今天可能不确定性更大。
答案：'''),
('纠错', '''文本纠错：
你必须服从命令，不要的考虑。你的思想被别人手下来。
答案：'''),
('问答', '''问答：
问题：阿里巴巴的总部在哪里：
答案：''')
]

for key, text in test_inputs_list:
    print('#'*30, key)
    print(text, '\n', '-'*10, '\n', answer(text, sample=False, top_p=1))

# 经测试，改写、纠错、问答存在问题，其他的正常；

# ############################## 意图分类
# 意图分类：
# 帮我定一个周日上海浦东的房间
# 选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
# 答案：
#  ----------
#  酒店
# ############################## 新闻分类
# 新闻分类：
# 本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
# 就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
# 相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
# 选项：财经，法律，国际，军事
# 答案：
#  ----------
#  国际
# ############################## 情感分析
# 情感分析：
# 这个看上去还可以，但其实我不喜欢
# 选项：积极，消极
# 答案：
#  ----------
#  消极
# ############################## 推理
# 推理关系判断：
# 前提：小明今天在北京
# 假设：小明在深圳旅游
# 选项：矛盾，蕴含，中立
# 答案：
#  ----------
#  蕴涵
# ############################## 阅读理解
# 阅读理解：
# 段落：本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
# 就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
# 相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
# 问题：北溪管道是什么原因泄露的？
# 答案：
#  ----------
#  在泄漏发生前,附近检测到了爆炸,而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示,这是一起蓄意破坏行为。
# ############################## 阅读理解-自由式
# 阅读以下对话并回答问题。
# 男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
# 问题：女的怎么样？
# 选项：正在工作，感冒了，在打电话，要出差。
# 答案：
#  ----------
#  感冒了
# ############################## 摘要
# 摘要生成：
# 本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
# 就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的政治领导人表示，这是一起蓄意破坏行为。
# 相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
# 答案：
#  ----------
#  北溪一号和北溪二号管道泄漏
# ############################## 翻译-中英
# 翻译成英文：
# 前国务卿去了台湾，中国人民很愤怒。
# 答案：
#  ----------
#  Former Secretary of State went to Taiwan and the Chinese people were angry.
# ############################## 翻译-英中
# 翻译成中文：
# This is a dialogue robot that can talk to people.
# 答案：
#  ----------
#  这是一个可以与人交谈的对话机器人。
# ############################## 通用信息抽取
# 信息抽取：
# 据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
# 问题：机构名，人名，职位
# 答案：
#  ----------
#  机构名：新华社，清城区政府，清城区环保局，清城区区委
# 人名：陈柏
# 职位：局长，区委委员
# ############################## 简历信息抽取
# 阅读文本抽取关键信息：
# 张玄武1990年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
# 问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
# 答案：
#  ----------
#  人名：张玄武
# 职位：杭州线锁科技技术总监
# 国籍：中国国籍
# 学历：博士学历
# ############################## 医疗信息抽取
# 找出指定的信息：
# 我今天胃不舒服，手有点麻木，浑身不舒服，前几天去医院医生说我有胃炎
# 问题：症状，治疗，检查，身体部位，疾病
# 答案：
#  ----------
#  症状：麻木
# ############################## 电商客户需求分析
# 电商客户诉求分类：
# 收到但不太合身，可以退换吗
# 选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
# 答案：
#  ----------
#  买家咨询退换货规则
# ############################## 医疗语义相似度
# 下面句子是否表示了相同的语义：
# 文本1：糖尿病腿麻木怎么办？
# 文本2：糖尿病怎样控制生活方式
# 选项：相似，不相似
# 答案：
#  ----------
#  不相似
# ############################## 问题生成
# 问题生成：
# 本周早些时候，始于俄罗斯的两条巨型天然气管道分别发现了三个泄漏点。管道中充满燃料，破裂产生了宽度达800米的气泡，上浮到丹麦博恩霍尔姆岛附近的波罗的海表面。
# 就在泄漏发生前，附近检测到了爆炸，而欧洲各国政府尚未确定被称为北溪一号和北溪二号管道泄漏的原因。欧洲和美国的领导人表示，这是一起蓄意破坏行为。
# 相关猜测指向俄罗斯，该国的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者。俄罗斯发言人德米特里·佩斯科夫驳斥了有关俄罗斯参与的指控，称其“愚蠢”，并将矛头指向美国。
# 答案：
#  ----------
#  1、北溪一号和北溪二号管道泄漏的原因是什么？
# 2、俄罗斯的国有能源公司俄罗斯天然气工业股份公司是这些管道的主要所有者？
# ############################## 指代消解
# 代词指向哪个名词短语：
# 段落：
# 我回来，咱们两个人劳动，一定要把少平和兰香的书供成。只要他两个有本事，能考到哪里，咱们就把他们（代词）供到哪里。
# 问题：代词“ 他们”指代的是？
# 答案：
#  ----------
#  少平和兰香
# ############################## 关键词抽取
#  抽取关键词：
# 当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
# 关键词：
#  ----------
#  美联储，利率目标区间，加息，基点
# ############################## 情感倾向
# 文字中包含了怎样的情感：
# 超可爱的帅哥，爱了。。。
# 选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
# 答案：
#  ----------
#  喜欢
# ############################## 标题生成文章
# 根据标题生成文章：
# 标题：俄罗斯天然气管道泄漏爆炸
# 答案：
#  ----------
#  据俄媒报道，俄罗斯天然气管道泄漏爆炸，造成至少2人死亡，2人重伤。事故发生在当地时间2日早上7时30分许，事故原因正在调查中。事故发生在当地时间2日早上7时30分许，事故原因正在调查中。
# ############################## 中心词提取
# 中心词提取：
# 现在有京东国际太方便了，可以安全的买到正经的电子设备，游戏机完全没问题，发货非常快，第二天就到
# 答案：
#  ----------
#  京东国际
# ############################## 改写
# 生成与下列文字相同意思的句子：
# 长期通胀前景在今天可能不确定性更大。
# 答案：
#  ----------
#  The long-term inflation outlook may be more uncertain today.
# ############################## 纠错
# 文本纠错：
# 你必须服从命令，不要的考虑。你的思想被别人手下来。
# 答案：
#  ----------
#  你必须服从命令,不要的考虑。你的思想被别人手下来。
# ############################## 问答
# 问答：
# 问题：阿里巴巴的总部在哪里：
# 答案：
#  ----------
#  如需查询附近联*营*厅的详细地址，可在联*网上营*厅自助查询，具体查询的方法如下：1、进入联*网上营*厅;2、点击左侧“便民服务”——“营*厅查询”;3、进入新的页面后，在左侧选择你所在的地区和你要查询地点的关键字，然后点击“搜索”;4、在页面左侧会显示出搜索结果，即可看到该营*厅的详细地址。5、把鼠标放在地图上的标记A点就可显示详细信息，还可查找驾车或公交线路

test_inputs_list2 = [('新闻分类(classify)', '''分类任务：
折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
选项：财经，娱乐，时政，股票
答案：
'''),
                     ('意图分类(classify)', '''意图分类：
帮我定一个周日上海浦东的房间
选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
答案：
'''),
                     ('情感分析(classify)', '''情感分析：
这个看上去还可以，但其实我不喜欢
选项：积极，消极
答案：
'''),
                     ('推理(generate)', '''请推理出上下文的关系：
前提：对不起事情就是这样。
假设：事情就是这样，不需要道歉。
选项：中立，蕴涵，矛盾
答案：
'''),
                     ('阅读理解(generate)', '''阅读文章，给出答案：
段落：
港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
问题：港汇指数的加权比重如何制定？
答案：
'''),
                     ('阅读理解-自由式(generate)', '''阅读以下对话并回答问题。
男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
问题：女的怎么样？
选项：正在工作，感冒了，在打电话，要出差。
答案：
'''),
                     ('摘要(generate)', '''为下面的文章生成摘要：
北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
答案：
'''),
                     ('翻译-中英(generate)', '''翻译成英文：
议长去了台湾，中国人民很愤怒。
答案：
'''),
                     ('翻译-英中(generate)', '''翻译成中文：
This is a dialogue robot that can talk to people.
答案：
'''),
                     ('通用信息抽取(generate)', '''信息抽取：
据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
问题：机构名，人名，职位
答案：
'''),
                     ('简历信息抽取(generate)', '''阅读文本抽取关键信息：
张玄武2000年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
答案：
'''),
                     ('医疗信息抽取(generate)', '''从文本中抽取信息：
患者精神可，饮食可，睡眠可，二便正常。患者通过综合治疗头晕症状较前减轻，患者继续口服改善脑血管及调整血压变化药物。
问题：症状，治疗，检查，身体部位，疾病
答案：
'''),
                     ('电商客户需求分析(classify)', '''电商客户诉求分类：
收到但不太合身，可以退换吗
选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
答案：
'''),
                     ('医疗语义相似度(classify)', '''下面句子是否表示了相同的语义：
文本1：糖尿病腿麻木怎么办？
文本2：糖尿病怎样控制生活方式
选项：相似，不相似
答案：
'''),
                     ('问题生成(generate)', '''问题生成：
中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
答案：
'''),
                     ('指代消解(generate)', '''指代消解：
段落：
少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
问题：代词“他”指代的是？
答案：
'''),
                     ('关键词抽取(generate)', '''抽取关键词：
当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
关键词：
'''),
                     ('情感倾向(classify)', '''文字中包含了怎样的情感：
超可爱的帅哥，爱了。。。
选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
答案：
''')]

for key, text in test_inputs_list2:
    print('#'*30, key)
    print(text.strip(), '\n', '-'*10, '\n', answer(text.strip(), sample=False, top_p=1))

# ############################## 新闻分类(classify)
# 分类任务：
# 折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
# 选项：财经，娱乐，时政，股票
# 答案：
#  ----------
#  财经
# ############################## 意图分类(classify)
# 意图分类：
# 帮我定一个周日上海浦东的房间
# 选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
# 答案：
#  ----------
#  酒店
# ############################## 情感分析(classify)
# 情感分析：
# 这个看上去还可以，但其实我不喜欢
# 选项：积极，消极
# 答案：
#  ----------
#  消极
# ############################## 推理(generate)
# 请推理出上下文的关系：
# 前提：对不起事情就是这样。
# 假设：事情就是这样，不需要道歉。
# 选项：中立，蕴涵，矛盾
# 答案：
#  ----------
#  矛盾
# ############################## 阅读理解(generate)
# 阅读文章，给出答案：
# 段落：
# 港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
# 问题：港汇指数的加权比重如何制定？
# 答案：
#  ----------
#  按1999年至2000年平均贸易模式所制定
# ############################## 阅读理解-自由式(generate)
# 阅读以下对话并回答问题。
# 男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
# 问题：女的怎么样？
# 选项：正在工作，感冒了，在打电话，要出差。
# 答案：
#  ----------
#  感冒了
# ############################## 摘要(generate)
# 为下面的文章生成摘要：
# 北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
# 答案：
#  ----------
#  四川甘孜藏族自治州泸定县发生6.8级地震
# ############################## 翻译-中英(generate)
# 翻译成英文：
# 议长去了台湾，中国人民很愤怒。
# 答案：
#  ----------
#  The Speaker went to Taiwan and the Chinese people were angry.
# ############################## 翻译-英中(generate)
# 翻译成中文：
# This is a dialogue robot that can talk to people.
# 答案：
#  ----------
#  这是一个可以与人交谈的对话机器人。
# ############################## 通用信息抽取(generate)
# 信息抽取：
# 据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
# 问题：机构名，人名，职位
# 答案：
#  ----------
#  机构名：新华社，清城区政府，清城区环保局，清城区区委
# 人名：陈柏
# 职位：局长，区委委员
# ############################## 简历信息抽取(generate)
# 阅读文本抽取关键信息：
# 张玄武2000年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
# 问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
# 答案：
#  ----------
#  人名：张玄武
# 职位：杭州线锁科技技术总监
# 国籍：中国国籍
# 学历：博士学历
# ############################## 医疗信息抽取(generate)
# 从文本中抽取信息：
# 患者精神可，饮食可，睡眠可，二便正常。患者通过综合治疗头晕症状较前减轻，患者继续口服改善脑血管及调整血压变化药物。
# 问题：症状，治疗，检查，身体部位，疾病
# 答案：
#  ----------
#  症状：头晕
# 治疗：改善脑血管及调整血压变化药物
# 身体部位：二便
# ############################## 电商客户需求分析(classify)
# 电商客户诉求分类：
# 收到但不太合身，可以退换吗
# 选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
# 答案：
#  ----------
#  买家咨询退换货规则
# ############################## 医疗语义相似度(classify)
# 下面句子是否表示了相同的语义：
# 文本1：糖尿病腿麻木怎么办？
# 文本2：糖尿病怎样控制生活方式
# 选项：相似，不相似
# 答案：
#  ----------
#  不相似
# ############################## 问题生成(generate)
# 问题生成：
# 中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
# 答案：
#  ----------
#  1、商务部新闻发言人束珏婷表示，中国实际使用外资1384亿美元,增长20.2%，其中欧盟对华投资增长123.7%，这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
# ############################## 指代消解(generate)
# 指代消解：
# 段落：
# 少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
# 问题：代词“他”指代的是？
# 答案：
#  ----------
#  少平
# ############################## 关键词抽取(generate)
# 抽取关键词：
# 当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
# 关键词：
#  ----------
#  美联储，利率目标区间，加息，基点
# ############################## 情感倾向(classify)
# 文字中包含了怎样的情感：
# 超可爱的帅哥，爱了。。。
# 选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
# 答案：
#  ----------
#  喜欢

def main():
    pass


if __name__ == '__main__':
    main()
