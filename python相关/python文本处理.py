
# 预处理，文本标准化
import unicodedata
unicodedata.normalize('NFKD', '，')  # 将文本标准化
# 会将中文逗号变化英文逗号；但是不会将繁体字转换为简体字

列表去重，但保持顺序不变：
recommend_question = list('abdwqfxsfwxwqe')
temp_recommend_question = list(set(recommend_question))
temp_recommend_question.sort(key=recommend_question.index)
temp_recommend_question
Out[7]: ['a', 'b', 'd', 'w', 'q', 'f', 'x', 's', 'e']

