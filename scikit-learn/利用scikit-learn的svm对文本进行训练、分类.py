#!/usr/bin/python3
# coding: utf-8
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import jieba
from jieba import analyse
import MySQLdb
import numpy as np
from sklearn import metrics
from sklearn.svm import SVC

conn = MySQLdb.connect(host="127.0.0.1",user="myuser",passwd="mypasswd",db="mydatabase",charset="utf8")

"""python代码利用了scikit-learn机器学习库的svm（支持向量机）算法，先计算出tf-idf矩阵，这个矩阵包括已经标注好分类的前5行，
和未进行分类5行以后的行，以前5行为样本交给svm做训练，然后用训练出来的模型对5行以后的部分做预测，并写回数据库

来源： http://www.shareditor.com/blogshow/?blogId=51

准备工作
1. 准备一张mysql数据库表，至少包含这些列：id、title(文章标题)、content(文章内容)、segment(中文切词)、isTec(技术类)、isSoup(鸡汤类)、
isMR(机器学习类)、isMath(数学类)、isNews(新闻类)
2. 根据你关注的微信公众号，把更新的文章的title和content自动写入数据库中
3. 收集一部分文章做人工标注，按照你自己的判断找几百篇文章标记出属于那种类别
"""

def get_segment(all=False):
    """利用了jieba中文分词器对文章内容做分词，并保存到数据库的segment列中，如果已经做过分词，则不再重复做
    """
    cursor = conn.cursor()

    if True == all:
        # 找到全部文章的标题和内容
        sql = "select id, title, content from CrawlPage"
    else:
        # 找到尚未切词的文章的标题和内容
        sql = "select id, title, content from CrawlPage where segment is null"
    cursor.execute(sql)

    for row in cursor.fetchall():
        print("cutting %s" % row[1])
        # 对标题和内容切词
        seg_list = jieba.cut(row[1]+' '+row[2])
        line = ""
        for str in seg_list:
            line = line + " " + str
        line = line.replace('\'', ' ')

        # 把切词按空格分隔并去特殊字符后重新写到数据库的segment字段里
        sql = "update CrawlPage set segment='%s' where id=%d" % (line, row[0])
        try:
            cursor.execute(sql)
            conn.commit()
        except Exception as e:
            print(line, e)
            sys.exit(-1)

def classify():
    """第一部分先取出同一类文章的segment数据拼成一行作为特征向量
    第二部分取出未分类的文章的segment数据追加到后面，每篇文章一行
    第三部分以第一部分为输入做SVM模型训练
    第四部分用训练好的SVM模型对第二部分的行做预测
    第五部分把预测出来的分类信息写到数据库中
    """
    cursor = conn.cursor()

    # 一共分成5类，并且类别的标识定为0，1，3，4，5
    category_ids = range(0, 5)
    category={}
    category[0] = 'isTec'
    category[1] = 'isSoup'
    category[2] = 'isMR'
    category[3] = 'isMath'
    category[4] = 'isNews'

    corpus = []
    for category_id in category_ids:
        # 找到这一类的所有已分类的文章并把所有切词拼成一行，加到语料库中
        sql = "select segment from CrawlPage where " + category[category_id] + "=1"
        cursor.execute(sql)
        line = ""
        for result in cursor.fetchall():
            segment = result[0]
            line = line + " " + segment
        corpus.append(line)

    # 把所有未分类的文章追加到语料库末尾行
    sql = "select id, title, segment from CrawlPage where isTec=0 and isSoup=0 and isMR=0 and isMath=0 and isNews=0"
    cursor.execute(sql)
    line = ""
    update_ids = []
    update_titles = []
    need_predict = False
    for result in cursor.fetchall():
        id = result[0]
        update_ids.append(id)
        title = result[1]
        update_titles.append(title)
        segment = result[2]
        corpus.append(segment)
        need_predict = True

    if False == need_predict:
        return

    # 计算tf-idf
    vectorizer=CountVectorizer()
    csr_mat = vectorizer.fit_transform(corpus)
    transformer=TfidfTransformer()
    tfidf=transformer.fit_transform(csr_mat)
    y = np.array(category_ids)

    # 用前5行已标分类的数据做模型训练
    model = SVC()
    model.fit(tfidf[0:5], y)

    # 对5行以后未标注分类的数据做分类预测
    predicted = model.predict(tfidf[5:])

    # 把机器学习得出的分类信息写到数据库
    for index in range(0, len(update_ids)):
        update_id = update_ids[index]
        predicted_category = category[predicted[index]]
        print("predict title: %s <==============> category: %s" % (update_titles[index], predicted_category))
        sql = "update CrawlPage set %s=1 where id=%d" % (predicted_category, update_id)
        cursor.execute(sql)

    conn.commit()

def main():
    # 切词
    get_segment()
    # 分类
    classify()
    conn.close()


if __name__ == '__main__':
    main()