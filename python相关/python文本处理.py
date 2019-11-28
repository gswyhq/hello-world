
# 预处理，文本标准化
import unicodedata
unicodedata.normalize('NFKD', '，')  # 将文本标准化
# 会将中文逗号变化英文逗号；但是不会将繁体字转换为简体字
# 此方法不能将“2E80-2EFF, 中日韩汉字部首补充” 转换为简体中文；
中日韩字根(\u2EE2) -> 中日韩象形文字(\u9A6C); 这两个字符都是简体中文‘马’字，但却不能转换；

列表去重，但保持顺序不变：
recommend_question = list('abdwqfxsfwxwqe')
temp_recommend_question = list(set(recommend_question))
temp_recommend_question.sort(key=recommend_question.index)
temp_recommend_question
Out[7]: ['a', 'b', 'd', 'w', 'q', 'f', 'x', 's', 'e']

