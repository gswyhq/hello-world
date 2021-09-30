
Rouge(Recall-Oriented Understudy for Gisting Evaluation)，是评估自动文摘以及机器翻译的一组指标。它通过将自动生成的摘要或翻译与一组参考摘要（通常是人工生成的）进行比较计算，得出相应的分值，以衡量自动生成的摘要或翻译与参考摘要之间的“相似度”。

普遍使用Rouge评价体系如 Rouge-1、Rouge-2、Rouge-L 
Rouge-1、Rouge-2、Rouge-L分别是：生成的摘要的1gram-2gram在真实摘要的1gram-2gram的准确率召回率和f1值，还有最长公共子序列在预测摘要中所占比例是准确率，在真实摘要中所占比例是召回率，然后可以计算出f1值。


步骤1：使用pip安装rouge

pip install rouge
步骤二 计算生成文本与 参考文本的Rouge值

# coding:utf8
from rouge import Rouge
a = ["i am a student from china"]  # 预测摘要 （可以是列表也可以是句子）
b = ["i am student from school on japan"] #真实摘要
 
'''
f:F1值  p：查准率  R：召回率
'''
rouge = Rouge()
rouge_score = rouge.get_scores(a, b)
print(rouge_score[0]["rouge-1"])
print(rouge_score[0]["rouge-2"])
print(rouge_score[0]["rouge-l"])
以上就是计算一对数据的Rouge的值如下

{'p': 1.0, 'f': 0.7272727226446282, 'r': 0.5714285714285714}
{'p': 1.0, 'f': 0.6666666622222223, 'r': 0.5}
{'p': 1.0, 'f': 0.6388206388206618, 'r': 0.5714285714285714}


