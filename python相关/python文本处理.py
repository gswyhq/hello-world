
# 预处理，文本标准化
import unicodedata
unicodedata.normalize('NFKD', '，')  # 将文本标准化
# 会将中文逗号变化英文逗号；但是不会将繁体字转换为简体字

