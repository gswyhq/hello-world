# -*- coding: utf-8 -*-
#计算相似度
#来源：《智慧编程 python算法应用》，p9
from math import sqrt
class xiangsidu:

# 欧几里得距离评价
# 以经过人们一致评价的物品为坐标轴，然后将参入的人绘制到图上，并考察他们彼此间距离的远近。
# 距离（每一坐标轴上的差值，求平方后再相加，最后对总和取平方根。）越近，
# 表示相似度（将近似距离值加1,避免遇到被零整除的错误，并取倒数）越大。

    #返回person1和person2的基于欧几里得距离的相似度
    def sim_distance(prefs,person1,person2):
        
        si={}
        #获得两个人都评价了的项目
        for item in prefs[person1]:
            if item in prefs[person2]:
                si[item]=1
        #如果两者没有共同之处，则返回0
        if len(si)==0:
            return 0

        #计算所有差值的平方和
        sum_of_squares=sum(
            [pow(prefs[person1][item]-prefs[person2][item],2)
                            for item in si]
            )
        return 1/(1+sqrt(sum_of_squares))
    
    
# 皮尔逊相关系数评价
# 皮尔逊相关系数是判断两组数据与某一直线拟合程度的量度。
# 比欧几里得距离评价的计算公式复杂，但是它在数据不规范的时候，会倾向于给出更好的结果。
# 以参评人为坐标轴，参评人的评价分数为数据点，作图。
# 若两个人对所有项目的评分情况相同，那么最佳拟合线，将是对角线。并且与图上的所有点都相交。
# 从而得出一个结果为1的理想相关度评价。

#皮尔逊相关系数：可以看做将两组数据首先做Z分数处理之后, 然后两组数据的乘积和除以样本数。
#z分数（z-score）,也叫标准分数（standard score）是一个数与平均数的差再除以标准差的过程。
#标准差则等于变量减掉平均数的平方和,再除以样本数,最后再开方.

#皮尔逊相关系数比较复杂一点,可以看做是两组数据的向量夹角的余弦

    #返回p1和p2的皮尔逊相关系数
    def sim_pearson(prefs,p1,p2):
        #获得双方都评价过的物品列表
        si={}
        for item in prefs[p1]:
            if item in prefs[p2]:
                si[item]=1

        n=len(si)

        #如果两者没有共同之处，则返回1
        if n==0:
            return 1

        #对所有的偏好求和
        sum1=sum([prefs[p1][it] for it in si])
        sum2=sum([prefs[p2][it] for it in si])

        #求平方和
        sumsq1=sum([pow(prefs[p1][it],2)for it in si])
        sumsq2=sum([pow(prefs[p2][it],2)for it in si])

        #求乘积之和
        psum=sum([prefs[p1][it]*prefs[p2][it]for it in si])

        #求皮尔逊评价值
        num=psum-(sum1*sum2/n)
        den=sqrt(
            (sumsq1-pow(sum1,2)/n)
            *(sumsq2-pow(sum2,2)/n)
            )

        if den==0:
            return 0
        r=num/den
        return r
    #该函数将返回介于-1到1之间的值，值为1，表示两个人对每一样物品均有完全一致的评价
    

if __name__=="__main__":
    critics={
        'Lisa Rose': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.5,
                      'Just My Luck': 3.0, 'Superman Returns': 3.5,
                      'You, Me and Dupree': 2.5, 'The Night Listener': 3.0
                      },
        'Gene Seymour': {'Lady in the Water': 3.0, 'Snakes on a Plane': 3.5,
                         'Just My Luck': 1.5, 'Superman Returns': 5.0,
                         'The Night Listener': 3.0, 'You, Me and Dupree': 3.5
                         },
        'Michael Phillips': {'Lady in the Water': 2.5, 'Snakes on a Plane': 3.0,
                             'Superman Returns': 3.5, 'The Night Listener': 4.0
                             },
        'Claudia Puig': {'Snakes on a Plane': 3.5, 'Just My Luck': 3.0,
                         'The Night Listener': 4.5, 'Superman Returns': 4.0,
                         'You, Me and Dupree': 2.5
                         },
        'Mick LaSalle': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                         'Just My Luck': 2.0, 'Superman Returns': 3.0,
                         'The Night Listener': 3.0,'You, Me and Dupree': 2.0
                         },
        'Jack Matthews': {'Lady in the Water': 3.0, 'Snakes on a Plane': 4.0,
                          'The Night Listener': 3.0, 'Superman Returns': 5.0,
                          'You, Me and Dupree': 3.5
                          },
        'Toby': {'Snakes on a Plane':4.5,'You, Me and Dupree':1.0,
                 'Superman Returns':4.0
                 }
        }
    xs=xiangsidu
    o=xs.sim_distance(critics,'Lisa Rose','Jack Matthews')
    print("欧几里得相似度:",o)
    p=xs.sim_pearson(critics,'Lisa Rose','Jack Matthews')
    print("皮尔逊相似度:",p)
        




