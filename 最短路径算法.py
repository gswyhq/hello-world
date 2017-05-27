#/usr/lib/python3.5
# -*- coding: utf-8 -*-


from collections import defaultdict
from heapq import heapify, heappop, heappush
 
def prim( nodes, edges ):
    conn = defaultdict( list )
    for n1,n2,c in edges:
        conn[ n1 ].append( (c, n1, n2) )
        conn[ n2 ].append( (c, n2, n1) )
 
    mst = []
    used = set( nodes[ 0 ] )
    usable_edges = conn[ nodes[0] ][:]
    #heapq.heapify(x):x必须是list，此函数将list变成堆，实时操作。从而能够在任何情况下使用堆的函数。
    heapify( usable_edges )
 
    while usable_edges:
        #heapq.heappop(heap):从堆数组heap中取出最小的值，并返回。
        cost, n1, n2 = heappop( usable_edges )
        if n2 not in used:
            used.add( n2 )
            mst.append( ( n1, n2, cost ) )
 
            for e in conn[ n2 ]:
                if e[ 2 ] not in used:
                    #heapq.heappush(heap, item):将item压入到堆数组heap中。
                    heappush( usable_edges, e )
    return mst
#heapq.nlargest(n, iterable[, key]),heapq.nsmallest(n, iterable[, key])
#获取列表中最大、最小的几个值。 
#In[230]: nlargest(2,a)
#Out[228]: [9, 6]
#In[232]: nsmallest(3,a)
#Out[230]: [1, 1, 2]

#test
nodes = list("ABCDEFG")
edges = [ ("A", "B", 7), ("A", "D", 5),
          ("B", "C", 8), ("B", "D", 9), ("B", "E", 7),
      ("C", "E", 5),
      ("D", "E", 15), ("D", "F", 6),
      ("E", "F", 8), ("E", "G", 9),
      ("F", "G", 11)]
 
print ("prim:", prim( nodes, edges ))


'''
heapq.heapreplace(heap, item)

是heappop(heap)和heappush(heap,item)的联合操作。
注意，与heappushpop(heap,item)的区别在于，顺序不同，这里是先进行删除，后压入堆
>>> a=[]
>>> heapreplace(a,3)            #如果list空，则报错
Traceback (most recent call last):
File "<stdin>", line 1, in <module>
IndexError: index out of range
>>> heappush(a,3)
>>> a
[3]
>>> heapreplace(a,2)            #先执行删除（heappop(a)->3),再执行加入（heappush(a,2))
3
>>> a
[2]
>>> heappush(a,5)  
>>> heappush(a,9)
>>> heappush(a,4)
>>> a
[2, 4, 9, 5]
>>> heapreplace(a,6)            #先从堆a中找出最小值并返回，然后加入6
2
>>> a
[4, 5, 9, 6]
>>> heapreplace(a,1)            #1是后来加入的，在1加入之前，a中的最小值是4
4
>>> a
[1, 5, 9, 6]
>>> a=[3,6,1]
>>> heapify(a)                  #将a变成堆之后，可以对其操作
>>> heappop(a)
1
>>> b=[4,2,5]                   #b不是堆，如果对其进行操作，显示结果如下
>>> heappop(b)                  #按照顺序，删除第一个数值并返回,不会从中挑选出最小的
4
>>> heapify(b)                  #变成堆之后，再操作
>>> heappop(b)
2

heapq.heappop(heap):从堆数组heap中取出最小的值，并返回。

>>> h=[]                    #定义一个list
>>> from heapq import *     #引入heapq模块
>>> h
[]
>>> heappush(h,5)               #向堆中依次增加数值
>>> heappush(h,2)
>>> heappush(h,3)
>>> heappush(h,9)
>>> h                           #h的值
[2, 5, 3, 9]
>>> heappop(h)                  #从h中删除最小的，并返回该值
2
>>> h
[3, 5, 9]
>>> h.append(1)                 #注意，如果不是压入堆中，而是通过append追加一个数值
>>> h                           #堆的函数并不能操作这个增加的数值，或者说它堆对来讲是不存在的
[3, 5, 9, 1]
>>> heappop(h)                  #从h中能够找到的最小值是3,而不是1
3
>>> heappush(h,2)               #这时，不仅将2压入到堆内，而且1也进入了堆。
>>> h
[1, 2, 9, 5]
>>> heappop(h)                  #操作对象已经包含了1
1
heapq.heappushpop(heap, item)

是上述heappush和heappop的合体，同时完成两者的功能.注意：相当于先操作了heappush(heap,item),然后操作heappop(heap)

>>> h
[1, 2, 9, 5]
>>> heappop(h)
1
>>> heappushpop(h,4)            #增加4同时删除最小值2并返回该最小值，与下列操作等同：
2                               #heappush(h,4),heappop(h)
>>> h
[4, 5, 9]


'''
