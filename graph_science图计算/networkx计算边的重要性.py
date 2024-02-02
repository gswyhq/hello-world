#!/usr/bin/env python
# coding=utf-8

# networkx计算边的重要性：边介数或者中介中心性 edge_betweenness

# 需求：有向图中删除某一些边的子集，但是又需要尽量减少对图的弱连通性的影响。
# 解决方案，先将有向图转为无向图，计算边的betweenness，有时也被翻译成中介中心性，然后删除中介中心性较低的边。

# 定义
# betweenness顾名思义，是它作为中介的一种度量。具体是在所有最短路径中，此边通过的最短路径所占的比例。因此betweenness越高，其中介性越高。

# 我认为删除中介性更高的边，对图的连通性影响性更大。

import networkx as nx

G = nx.Graph()
G.add_edges_from([[0, 1], [0, 2], [1, 2], [2, 3], [3, 4], [4, 5], [3, 5]])
nx.draw(G, with_labels=True)


nx.edge_betweenness_centrality(G, k=None)
>>> {(0, 1): 0.06666666666666667,
      (0, 2): 0.26666666666666666,
      (1, 2): 0.26666666666666666,
      (2, 3): 0.6,
      (3, 4): 0.26666666666666666,
      (3, 5): 0.26666666666666666,
      (4, 5): 0.06666666666666667}


# 可以看到(2, 3)
# 的中介性最高，(0, 1)(4, 5)
# 的中介性最低，其余几条边的中介性相同，跟预期相同。
#
# 大规模图上的边介数
# 既然edge_betweenness的计算涉及了最短路径，因此计算复杂度一定不低，因此在大规模图上有实现难度。
# 不过networkx的edge_betweenness提供了一个k参数，选择sample的节点数目。k越大，其计算的介数准确度越高。
# 在我的大规模图上，我选择了10，计算时长在一个小时左右。根据返回的edge_betweenness删除边后，图的连通性的影响在可接受范围内。
# 另外需要注意的是，当设置了k后，返回的edge_betweenness不一定包括所有的边。

# 问题：
# TypeError: '_AxesStack' object is not callable networkx
#
# 可能是matplotlib 版本问题，再使用pip3 install -U matplotlib==3.5.1 降级安装

def main():
    pass


if __name__ == "__main__":
    main()
