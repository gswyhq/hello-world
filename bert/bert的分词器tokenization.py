#! encoding=utf-8

# BERT 源码中 tokenization.py 就是预处理进行分词的程序，主要有两个分词器：BasicTokenizer 和 WordpieceTokenizer，另外一个 FullTokenizer 是这两个的结合：先进行 BasicTokenizer 得到一个分得比较粗的 token 列表，然后再对每个 token 进行一次 WordpieceTokenizer，得到最终的分词结果。

tokenizer = BertTokenizer.from_pretrained(rf'D:\Users\{USERNAME}\data\bert_base_pytorch\bert-base-chinese') # Bert的分词器

# BasicTokenizer
BasicTokenizer（以下简称 BT）是一个初步的分词器。对于一个待分词字符串，流程大致就是转成 unicode -> 去除各种奇怪字符 -> 处理中文 -> 空格分词 -> 去除多余字符和标点分词 -> 再次空格分词，结束。
tokenizer.basic_tokenizer.tokenize('图书馆 的四楼')
Out[59]: ['图', '书', '馆', '的', '四', '楼']

清理特殊字符：去除控制符以及替换空白字符为空格，这个地方需要注意，一些序列标注任务输入需要保持输入长度不变，否则标注无法对上，一般要么使用BasicTokenizer把原数据过一遍，再把标注对上，要么使用list方法代替bert的tokenizer
中文分割：将中文每个汉字单独作为一个token
小写：可选，如果小写，还会对字符做做NFD标准化
标点符号分割：除中文汉字外，其它字符通过标点符号分割开

# WordpieceTokenizer
按照从左到右的顺序，将一个词拆分成多个子词，每个子词尽可能长。 贪婪最长优先匹配算法。
对BasicTokenizer分出的每个token再进行WordPieceTokenizer处理，得到一些词典中有的词片段，非词首的词片要变成”##词片“形式，如：
tokenizer.wordpiece_tokenizer.tokenize('图书馆的四楼')
Out[55]: ['图', '##书', '##馆', '##的', '##四', '##楼']
tokenizer.wordpiece_tokenizer.tokenize('图书馆 的 四楼')
Out[56]: ['图', '##书', '##馆', '的', '四', '##楼']
所以，中文的话，可以先自行分词，再用此分词器进行处理；


