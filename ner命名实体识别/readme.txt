
来源：http://x-algo.cn/index.php/2016/02/29/crf-name-entity-recognition/
代码：https://pan.baidu.com/s/1gemKdoR#list/path=%2F

运行过程见： start.sh


BIO标注：将每个元素标注为“B-X”、“I-X”或者“O”。其中，“B-X”表示此元素所在的片段属于X类型并且此元素在此片段的开头，“I-X”表示此元素所在的片段属于X类型并且此元素在此片段的中间位置，“O”表示不属于任何类型。
      比如，我们将 X 表示为名词短语（Noun Phrase, NP），则BIO的三个标记为：
（1）B-NP：名词短语的开头
（2）I-NP：名词短语的中间
（3）O：不是名词短语
    因此可以将一段话划分为如下结果;
 我们可以进一步将BIO应用到NER中，来定义所有的命名实体（人名、组织名、地点、时间等），那么我们会有许多 B 和 I 的类别，如 B-PERS、I-PERS、B-ORG、I-ORG等。然后可以得到以下结果：

BIOES标注：
B表示这个词处于一个实体的开始(Begin), I 表示内部(inside), O 表示外部(outside), E 表示这个词处于一个实体的结束为止， S 表示，这个词是自己就可以组成一个实体(Single)
BIOES   (B-begin，I-inside，O-outside，E-end，S-single)
B 表示开始，I表示内部， O表示非实体 ，E实体尾部，S表示改词本身就是一个实体。

BMES  四位序列标注法
B表示一个词的词首位值，M表示一个词的中间位置，E表示一个词的末尾位置，S表示一个单独的字词。
我/S 是/S 广/B 东/M 人/E    （符号标注，‘东’是‘广’和‘人’的中间部分，凑成‘广东人’这个实体）
我/ 是/ 广东人/        （标注上分出来的实体块）


