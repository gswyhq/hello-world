#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
USERNAME = os.getenv("USERNAME")
# 模型来源：https://huggingface.co/ClueAI/PromptCLUE-base-v1-5
model_dir = rf"D:\Users\{USERNAME}\data\PromptCLUE-base-v1-5"

# 加载模型
from transformers import T5Tokenizer, T5ForConditionalGeneration, convert_graph_to_onnx
tokenizer = T5Tokenizer.from_pretrained(model_dir)
model = T5ForConditionalGeneration.from_pretrained(model_dir)

# 使用模型进行预测推理方法：

import torch
device = torch.device('cpu')
# device = torch.device('cuda')
model.to(device)
def preprocess(text):
    return text.replace("\n", "_")

def postprocess(text):
    return text.replace("_", "\n")

def answer(text, sample=False, top_p=0.8):
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

examples_list = [
    ['改写(paraphrase)', True, ['生成与下列文字相同意思的句子： 白云遍地无人扫 答案：',
                              '用另外的话复述下面的文字： 怎么到至今还不回来，这满地的白云幸好没人打扫。 答案：',
                              '改写下面的文字，确保意思相同： 一个如此藐视本国人民民主权利的人，怎么可能捍卫外国人的民权？ 答案：']],
    ['知识问答（knowledge_qa)', False, [
        '根据问题给出答案： 问题：手指发麻的主要可能病因是： 答案：',
        '问答： 问题：黄果悬钩子的目是： 答案：',
        '问答： 问题：今天是星期三，明天星期几：选项：一，二，三，四，五，六，七 答案：'
    ]],
    ['新闻分类(classify)', False, [
        '''分类任务：
        折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
        选项：财经，娱乐，时政，股票
        答案：''',
                '''分类任务：\nOpenAI近期发布聊天机器人模型ChatGPT，迅速出圈全网。它以对话方式进行交互。以更贴近人的对话方式与使用者互动，可以回答问题、承认错误、挑战不正确的前提、拒绝不适当的请求。高质量的回答、上瘾式的交互体验，圈内外都纷纷惊呼。
        选项：财经，娱乐，时政，体育，互联网，金融\n答案：'''
    ]],
    ['意图分类(classify)', False, [
        '''意图分类：
        帮我定一个周日上海浦东的房间
        选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
        答案：'''
    ]],
    ['情感分析(classify)', False, [
        '''情感分析：
        这个看上去还可以，但其实我不喜欢
        选项：积极，消极
        答案：''',
        '''情感分析：
        周末的时候可以去市民广场的椅子上坐坐，其实也不错；
        选项：积极，消极
        答案：'''
    ]],
    ['推理(generate)', False, [
        '''请推理出上下文的关系：
        前提：对不起事情就是这样。
        假设：事情就是这样，不需要道歉。
        选项：中立，蕴涵，矛盾
        答案：'''
    ]],
    ['阅读理解(generate)', True, [
        '''阅读文章，给出答案：
        段落：
        港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
        问题：港汇指数的加权比重如何制定？
        答案：''',
        '''阅读文章，给出答案：
        段落：
        OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
        问题：ChatGPT可以做什么？
        答案：''',
        '''阅读文章，给出答案：
        段落：
        OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
        问题：ChatGPT是由哪家公司开发的？
        答案：'''
    ]],
    ['阅读理解-自由式(generate)', False, [
        '''阅读以下对话并回答问题。
        男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
        问题：女的怎么样？
        选项：正在工作，感冒了，在打电话，要出差。
        答案：'''
    ]],
    ['摘要(generate)', True, [
        '''为下面的文章生成摘要：
        北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
        答案：''',
        '''为下面的文章生成摘要：
        OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
        答案：'''
    ]],
    ['翻译-中英(generate)', True, [
        '''翻译成英文：
        议长去了台湾，中国人民很愤怒。
        答案：''',
        '''翻译成英文：
        我们应该好好学习，天天向上。
        答案：''',
        '''翻译成中文：
        This is a dialogue robot that can talk to people.
        答案：'''
    ]],
    ['通用信息抽取(generate)', True, [
        '''信息抽取：
        据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
        问题：机构名，人名，职位
        答案：''',
        '''阅读文本抽取关键信息：
        张玄武2000年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
        问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
        答案：''',
        '''从文本中抽取信息：
        患者精神可，饮食可，睡眠可，二便正常。患者通过综合治疗头晕症状较前减轻，患者继续口服改善脑血管及调整血压变化药物。
        问题：症状，治疗，检查，身体部位，疾病
        答案：'''
    ]],
    ['电商客户需求分析(classify)', True, [
        '''电商客户诉求分类：
收到但不太合身，可以退换吗
选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
答案：'''
    ]],
    ['相似度(classify)', True, [
        '''下面句子是否表示了相同的语义：
        文本1：糖尿病腿麻木怎么办？
        文本2：糖尿病怎样控制生活方式
        选项：相似，不相似
        答案：''',
        '''下面句子是否表示了相同的语义：
        文本1：这款保险的等待期是多久？
        文本2：这款保险的犹豫期是多久？
        选项：相似，不相似
        答案：'''
    ]],
    ['问题生成(generate)', True, [
        '''问题生成：
中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
答案：'''
    ]],
    ['指代消解(generate)', False, [
        '''指代消解：
        段落：
        少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
        问题：代词“他”指代的是？
        答案：''',
        '''指代消解：
        段落：
        早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
        问题：代词“他”指代的是？
        答案：''',
        '''指代消解：
        段落：
        早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
        问题：代词“她”指代的是？
        答案：''',
        '''指代消解：
        段落：
        早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
        问题：代词“姐姐”指代的是？
        答案：'''
    ]],
    ['关键词抽取(generate)', False, [
        '''抽取关键词：
        当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
        关键词：''',
        '''抽取关键词：
        OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
        关键词：'''
    ]],
    ['情感倾向(classify)', False, [
        '''文字中包含了怎样的情感：
        超可爱的帅哥，爱了。。。
        选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
        答案：''',
        '''文字中包含了怎样的情感：
        因南京的案子，导致现在好多人摔倒了，路人都不敢去扶。
        选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕，担忧
        答案：'''
    ]],
    ['文章生成(generate)', True, [
        '''根据标题生成文章：
        标题：俄罗斯天然气管道泄漏爆炸
        答案：''',
        '''根据标题生成文章：
        标题：深圳两小学生放学后走丢
        答案：'''
    ]],
    ['中心词提取(center-word-extract)', False, [
        '''中心词提取：
        现在有京东国际太方便了，可以安全的买到正经的电子设备，游戏机完全没问题，发货非常快，第二天就到
        答案：'''
    ]],
    ['文本纠错(corrrect)', False, [
        '''文本纠错：
        你必须服从命令，不要的考虑。你的思想被别人手下来。
        答案：''',
        '''文本纠错：
        不以物喜不以己悲，是来自范仲淹的越阳楼记。
        答案：'''
    ]],
    ['问答(qa)', False, [
        '''问答：
问题：阿里巴巴的总部在哪里：
答案：''',
        '''问答：
问题：深圳市是属于中国的哪个省份：
答案：'''
    ]],
    ['纠错', False], [
        '''纠错：
新增10个社区养老服务一站，就近为有需求的居家老年人提供生活照料、陪伴护理等多样化服务，提升老年人生活质量。
答案：答案'''
    ]
]





def main():
    for task, sample, input_list in examples_list:
        print(task, '*' * 80)
        for text in input_list:
            output = answer(text, sample=sample, top_p=0.8)
            print(text, "\n->\n", output)
            print('-' * 30)

if __name__ == '__main__':
    main()


# 模型测试结果：
# 改写(paraphrase) ********************************************************************************
# 生成与下列文字相同意思的句子： 白云遍地无人扫 答案：
# ->
#  白云一片没有一个人扫过。
# ------------------------------
# 用另外的话复述下面的文字： 怎么到至今还不回来，这满地的白云幸好没人打扫。 答案：
# ->
#  满地白云无清扫
# ------------------------------
# 改写下面的文字，确保意思相同： 一个如此藐视本国人民民主权利的人，怎么可能捍卫外国人的民权？ 答案：
# ->
#  这样藐视本国人民民主权利的人怎么可能捍卫外国人的民权？
# ------------------------------
# 知识问答（knowledge_qa) ********************************************************************************
# 根据问题给出答案： 问题：手指发麻的主要可能病因是： 答案：
# ->
#  手指发麻
# ------------------------------
# 问答： 问题：黄果悬钩子的目是： 答案：
# ->
#  蔷薇目
# ------------------------------
# 问答： 问题：今天是星期三，明天星期几：选项：一，二，三，四，五，六，七 答案：
# ->
#  七
# ------------------------------
# 新闻分类(classify) ********************************************************************************
# 分类任务：
#         折价率过低遭抛售基金泰和跌7.15%，证券时报记者 朱景锋本报讯 由于折价率在大盘封基中处于最低水平，基金泰和昨日遭到投资者大举抛售，跌幅达到7.15%，远超大盘。盘面显示，基金泰和随大盘高开，之后开始震荡走低，午后开始加速下行，几乎没有像样反弹。截至收盘时，在沪深300指数仅下跌2.56%的情况下，基金泰和收盘跌幅高达7.15%，在所有封基中跌幅最大，而昨日多数封基跌幅在2%左右。
#         选项：财经，娱乐，时政，股票
#         答案：
# ->
#  财经
# ------------------------------
# 分类任务：
# OpenAI近期发布聊天机器人模型ChatGPT，迅速出圈全网。它以对话方式进行交互。以更贴近人的对话方式与使用者互动，可以回答问题、承认错误、挑战不正确的前提、拒绝不适当的请求。高质量的回答、上瘾式的交互体验，圈内外都纷纷惊呼。
#         选项：财经，娱乐，时政，体育，互联网，金融
# 答案：
# ->
#  互联网
# ------------------------------
# 意图分类(classify) ********************************************************************************
# 意图分类：
#         帮我定一个周日上海浦东的房间
#         选项：闹钟，文学，酒店，艺术，体育，健康，天气，其他
#         答案：
# ->
#  酒店
# ------------------------------
# 情感分析(classify) ********************************************************************************
# 情感分析：
#         这个看上去还可以，但其实我不喜欢
#         选项：积极，消极
#         答案：
# ->
#  消极
# ------------------------------
# 情感分析：
#         周末的时候可以去市民广场的椅子上坐坐，其实也不错；
#         选项：积极，消极
#         答案：
# ->
#  积极
# ------------------------------
# 推理(generate) ********************************************************************************
# 请推理出上下文的关系：
#         前提：对不起事情就是这样。
#         假设：事情就是这样，不需要道歉。
#         选项：中立，蕴涵，矛盾
#         答案：
# ->
#  矛盾
# ------------------------------
# 阅读理解(generate) ********************************************************************************
# 阅读文章，给出答案：
#         段落：
#         港汇指数，全称港元实际汇兑指数（Effective Exchange Rate Index for the Hong Kong Dollar）是由香港政府统计处编制的一项指数，以反映港元与香港主要贸易伙伴之货币的名义有效汇率加权平均数的变动情况。加权比重是按1999年至2000年平均贸易模式所制定，但政府并未有公布详细的计算公式。旧港汇指数基准日为2000年1月1日，基数为100点。由2012年1月3日起，新系列港汇指数 (包括15种货币及以2010年1月 = 100) 已取代旧港汇指数系列。港汇指数的作用，主要是用于反映香港的货品及服务的价格相对于其主要贸易伙伴的变动，并通常被视作反映香港价格竞争力的指标。
#         问题：港汇指数的加权比重如何制定？
#         答案：
# ->
#  按1999年至2000年平均贸易模式所制定
# ------------------------------
# 阅读文章，给出答案：
#         段落：
#         OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
#         问题：ChatGPT可以做什么？
#         答案：
# ->
#  在全球用户争相晒出自己极具创意的与ChatGPT交流的成果。
# ------------------------------
# 阅读文章，给出答案：
#         段落：
#         OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
#         问题：ChatGPT是由哪家公司开发的？
#         答案：
# ->
#  谷歌
# ------------------------------
# 阅读理解-自由式(generate) ********************************************************************************
# 阅读以下对话并回答问题。
#         男：今天怎么这么晚才来上班啊？女：昨天工作到很晚，而且我还感冒了。男：那你回去休息吧，我帮你请假。女：谢谢你。
#         问题：女的怎么样？
#         选项：正在工作，感冒了，在打电话，要出差。
#         答案：
# ->
#  感冒了
# ------------------------------
# 摘要(generate) ********************************************************************************
# 为下面的文章生成摘要：
#         北京时间9月5日12时52分，四川甘孜藏族自治州泸定县发生6.8级地震。地震发生后，领导高度重视并作出重要指示，要求把抢救生命作为首要任务，全力救援受灾群众，最大限度减少人员伤亡
#         答案：
# ->
#  四川泸定县6.6级地震 领导批示全力救援
# ------------------------------
# 为下面的文章生成摘要：
#         OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
#         答案：
# ->
#  OpenAI 11月30号发布
# ------------------------------
# 翻译-中英(generate) ********************************************************************************
# 翻译成英文：
#         议长去了台湾，中国人民很愤怒。
#         答案：
# ->
#  He went to Taiwan and the Chinese were angry.
# ------------------------------
# 翻译成英文：
#         我们应该好好学习，天天向上。
#         答案：
# ->
#  We should be able to go to get ready.
# ------------------------------
# 翻译成中文：
#         This is a dialogue robot that can talk to people.
#         答案：
# ->
#  这是个对话机器人，能够与人们说话。
# ------------------------------
# 通用信息抽取(generate) ********************************************************************************
# 信息抽取：
#         据新华社电广东省清远市清城区政府昨日对外发布信息称,日前被实名举报涉嫌勒索企业、说“分分钟可以搞垮一间厂”的清城区环保局局长陈柏,已被免去清城区区委委员
#         问题：机构名，人名，职位
#         答案：
# ->
#  人名：陈柏
# 机构名：新华社，清城区政府，清城区环保局，清城区区委委员
# 职位：清城区环保局局长
# ------------------------------
# 阅读文本抽取关键信息：
#         张玄武2000年出生中国国籍无境外居留权博士学历现任杭州线锁科技技术总监。
#         问题：机构，人名，职位，籍贯，专业，国籍，学历，种族
#         答案：
# ->
#  人名：张玄武
# 职位：杭州线锁科技技术总监
# 国籍：中国国籍
# 学历：博士学历
# ------------------------------
# 从文本中抽取信息：
#         患者精神可，饮食可，睡眠可，二便正常。患者通过综合治疗头晕症状较前减轻，患者继续口服改善脑血管及调整血压变化药物。
#         问题：症状，治疗，检查，身体部位，疾病
#         答案：
# ->
#  症状：头晕
# 治疗：改善脑血管及调整血压变化药物
# ------------------------------
# 电商客户需求分析(classify) ********************************************************************************
# 电商客户诉求分类：
# 收到但不太合身，可以退换吗
# 选项：买家咨询商品是否支持花呗付款，买家表示收藏关注店铺，买家咨询退换货规则，买家需要商品推荐
# 答案：
# ->
#  买家咨询退换货规则
# ------------------------------
# 相似度(classify) ********************************************************************************
# 下面句子是否表示了相同的语义：
#         文本1：糖尿病腿麻木怎么办？
#         文本2：糖尿病怎样控制生活方式
#         选项：相似，不相似
#         答案：
# ->
#  不相似
# ------------------------------
# 下面句子是否表示了相同的语义：
#         文本1：这款保险的等待期是多久？
#         文本2：这款保险的犹豫期是多久？
#         选项：相似，不相似
#         答案：
# ->
#  相似
# ------------------------------
# 问题生成(generate) ********************************************************************************
# 问题生成：
# 中新网2022年9月22日电 22日，商务部召开例行新闻发布会，商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%；其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华投资合作。
# 答案：
# ->
#  商务部新闻发言人束珏婷表示，今年1-8月，中国实际使用外资1384亿美元，增长20.2%;其中，欧盟对华投资增长123.7%(含通过自由港投资数据)。这充分表明，包括欧盟在内的外国投资者持续看好中国市场，希望继续深化对华
# ------------------------------
# 指代消解(generate) ********************************************************************************
# 指代消解：
#         段落：
#         少平跟润叶进了她二爸家的院子，润生走过来对他（代词）说：“我到宿舍找了你两回，你到哪里去了？”
#         问题：代词“他”指代的是？
#         答案：
# ->
#  少平
# ------------------------------
# 指代消解：
#         段落：
#         早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
#         问题：代词“他”指代的是？
#         答案：
# ->
#  小明
# ------------------------------
# 指代消解：
#         段落：
#         早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
#         问题：代词“她”指代的是？
#         答案：
# ->
#  小明
# ------------------------------
# 指代消解：
#         段落：
#         早晨，小明吃了5个包子，妈妈说她吃得太多了，爸爸却认为不是很多，因为弟弟小舒也吃了4个，他比姐姐仅仅少一个包子。
#         问题：代词“姐姐”指代的是？
#         答案：
# ->
#  小明
# ------------------------------
# 关键词抽取(generate) ********************************************************************************
# 抽取关键词：
#         当地时间21日，美国联邦储备委员会宣布加息75个基点，将联邦基金利率目标区间上调到3.00%至3.25%之间，符合市场预期。这是美联储今年以来第五次加息，也是连续第三次加息，创自1981年以来的最大密集加息幅度。
#         关键词：
# ->
#  联邦基金利率，美联储，加息
# ------------------------------
# 抽取关键词：
#         OpenAI 11月30号发布，首先在北美、欧洲等已经引发了热烈的讨论。随后在国内开始火起来。全球用户争相晒出自己极具创意的与ChatGPT交流的成果。ChatGPT在大量网友的疯狂测试中表现出各种惊人的能力，如流畅对答、写代码、写剧本、纠错等，甚至让记者编辑、程序员等从业者都感受到了威胁，更不乏其将取代谷歌搜索引擎之说。继AlphaGo击败李世石、AI绘画大火之后，ChatGPT成为又一新晋网红。
#         关键词：
# ->
#  OpenAI，ChatGPT，疯狂测试
# ------------------------------
# 情感倾向(classify) ********************************************************************************
# 文字中包含了怎样的情感：
#         超可爱的帅哥，爱了。。。
#         选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕
#         答案：
# ->
#  喜欢
# ------------------------------
# 文字中包含了怎样的情感：
#         因南京的案子，导致现在好多人摔倒了，路人都不敢去扶。
#         选项：厌恶，喜欢，开心，悲伤，惊讶，生气，害怕，担忧
#         答案：
# ->
#  悲伤
# ------------------------------
# 文章生成(generate) ********************************************************************************
# 根据标题生成文章：
#         标题：俄罗斯天然气管道泄漏爆炸
#         答案：
# ->
#  中新网7月27日电“据外电报道，俄罗斯国家天然气管道泄漏爆炸事故的调查小组25日说，当地时间27日凌晨1时左右，位于俄罗斯莫斯科附近的俄罗斯天然气管道泄漏爆炸事故附近，造成12人死亡。莫斯科当地天然气管道泄漏爆炸事故调查小组说，事故发生在
# ------------------------------
# 根据标题生成文章：
#         标题：深圳两小学生放学后走丢
#         答案：
# ->
#  深圳晚报记者“马明“实习生“杨晓颖“实习生“徐明“实习生“王燕“深圳报道“5月21日，深圳龙岗区坪田小学二年级一班学生小明(化名)在放学后走丢。目前，警方已介入调查，将对此事
# ------------------------------
# 中心词提取(center-word-extract) ********************************************************************************
# 中心词提取：
#         现在有京东国际太方便了，可以安全的买到正经的电子设备，游戏机完全没问题，发货非常快，第二天就到
#         答案：
# ->
#  京东国际
# ------------------------------
# 文本纠错(corrrect) ********************************************************************************
# 文本纠错：
#         你必须服从命令，不要的考虑。你的思想被别人手下来。
#         答案：
# ->
#  你必须服从命令，不要的考虑。你的思想被别人掌握。
# ------------------------------
# 文本纠错：
#         不以物喜不以己悲，是来自范仲淹的越阳楼记。
#         答案：
# ->
#  不以物喜不以己悲，是来自范仲淹的越阳楼记。
# ------------------------------
# 问答(qa) ********************************************************************************
# 问答：
# 问题：阿里巴巴的总部在哪里：
# 答案：
# ->
#  香港
# ------------------------------
# 问答：
# 问题：深圳市是属于中国的哪个省份：
# 答案：
# ->
#  广东省
# ------------------------------
# ------------------------------
# 纠错：
# 新增10个社区养老服务一站，就近为有需求的居家老年人提供生活照料、陪伴护理等多样化服务，提升老年人生活质量。
# 答案：
# ->
#  新增10个社区养老服务站，就近为有需求的居家老年人提供生活照料、陪伴护理等多样化服务，提升老年人生活质量。
# ------------------------------
