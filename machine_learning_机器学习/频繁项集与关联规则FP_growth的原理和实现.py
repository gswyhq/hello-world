#author: xiaolinhan_daisy
#date: 2017/11/09
#site: XAJTU

# https://developer.ibm.com/zh/articles/machine-learning-hands-on2-fp-growth/
# 频繁项集与关联规则 FP-growth 的原理和实现

# FP-growth（Frequent Pattern Growth）模型，它是目前业界经典的频繁项集和关联规则挖掘的算法。

# 关联规则简介
# 关联规则 是在频繁项集的基础上得到的。关联规则指由集合 A，可以在某置信度下推出集合 B。通俗来说，就是如果 A 发生了，那么 B 也很有可能会发生。举个例子，有关联规则如：{‘鸡蛋’, ‘面包’} -> {‘牛奶’}，该规则的置信度是 0.9，意味着在所有买了鸡蛋和面包的客户中，有 90%的客户还买了牛奶。关联规则可以用来发现很多有趣的规律。这其中需要先阐明两个概念：支持度和置信度。
#
# 支持度 Support
# 支持度 指某频繁项集在整个数据集中的比例。假设数据集有 10 条记录，包含{‘鸡蛋’, ‘面包’}的有 5 条记录，那么{‘鸡蛋’, ‘面包’}的支持度就是 5/10 = 0.5。
#
# 置信度 Confidence
# 置信度 是针对某个关联规则定义的。有关联规则如{‘鸡蛋’, ‘面包’} -> {‘牛奶’}，它的置信度计算公式为{‘鸡蛋’, ‘面包’, ‘牛奶’}的支持度/{‘鸡蛋’, ‘面包’}的支持度。假设{‘鸡蛋’, ‘面包’, ‘牛奶’}的支持度为 0.45，{‘鸡蛋’, ‘面包’}的支持度为 0.5，则{‘鸡蛋’, ‘面包’} -> {‘牛奶’}的置信度为 0.45 / 0.5 = 0.9。

def loadDataSet():
    dataSet = [['面包', '牛奶', '蔬菜', '水果', '鸡蛋'],
             ['面条', '牛肉', '猪肉', '水', '袜子', '手套', '鞋', '米'],
             ['袜子', '手套'],
             ['面包', '牛奶', '鞋', '袜子', '鸡蛋'],
             ['袜子', '鞋', '毛衣', '帽', '牛奶', '蔬菜', '手套'],
             ['鸡蛋', '面包', '牛奶', '鱼', '蟹', '虾', '米']]
    return dataSet

def transfer2FrozenDataSet(dataSet):
    frozenDataSet = {}
    for elem in dataSet:
        frozenDataSet[frozenset(elem)] = 1 # frozenset() 返回一个冻结的集合，冻结后集合不能再添加或删除任何元素。
    return frozenDataSet

# 首先，需要创建一个树形的数据结构，叫做 FP 树。
# 该树结构包含结点名称 nodeName，结点元素出现频数 count，父节点 nodeParent，指向下一个相同元素的指针 nextSimilarItem，子节点集合 children。
class TreeNode:
    def __init__(self, nodeName, count, nodeParent):
        self.nodeName = nodeName
        self.count = count
        self.nodeParent = nodeParent
        self.nextSimilarItem = None
        self.children = {}

    def increaseC(self, count):
        self.count += count

# 用第一步构造出的数据结构来创建 FP 树。
# 代码主要分为两层。第一层，扫描数据库，统计出各个元素的出现频数；
# 第二层，扫描数据库，对每一条数据记录，将数据记录中不包含在频繁元素中的元素删除，然后将数据记录中的元素按出现频数排序。
# 将数据记录逐条插入 FP 树中，不断更新 FP 树。

def createFPTree(frozenDataSet, minSupport):
    #scan dataset at the first time, filter out items which are less than minSupport
    headPointTable = {}
    for items in frozenDataSet:
        for item in items:
            headPointTable[item] = headPointTable.get(item, 0) + frozenDataSet[items]
    headPointTable = {k:v for k,v in headPointTable.items() if v >= minSupport}
    frequentItems = set(headPointTable.keys())
    if len(frequentItems) == 0: return None, None

    for k in headPointTable:
        headPointTable[k] = [headPointTable[k], None]
    fptree = TreeNode("null", 1, None)
    #scan dataset at the second time, filter out items for each record
    for items,count in frozenDataSet.items():
        frequentItemsInRecord = {}
        for item in items:
            if item in frequentItems:
                frequentItemsInRecord[item] = headPointTable[item][0]
        if len(frequentItemsInRecord) > 0:
            orderedFrequentItems = [v[0] for v in sorted(frequentItemsInRecord.items(), key=lambda v:v[1], reverse = True)]
            updateFPTree(fptree, orderedFrequentItems, headPointTable, count)

    return fptree, headPointTable

# 更新 FP 树，这里用到了递归的技巧。
# 每次递归迭代中，处理数据记录中的第一个元素处理，如果该元素是 fptree 节点的子节点，则只增加该子节点的 count 树，
# 否则，需要新创建一个 TreeNode 节点，然后将其赋给 fptree 节点的子节点，并更新头指针表关于下一个相同元素指针的信息。
# 迭代的停止条件是当前迭代的数据记录长度小于等于 1。
def updateFPTree(fptree, orderedFrequentItems, headPointTable, count):
    #handle the first item
    if orderedFrequentItems[0] in fptree.children:
        fptree.children[orderedFrequentItems[0]].increaseC(count)
    else:
        fptree.children[orderedFrequentItems[0]] = TreeNode(orderedFrequentItems[0], count, fptree)

        #update headPointTable
        if headPointTable[orderedFrequentItems[0]][1] == None:
            headPointTable[orderedFrequentItems[0]][1] = fptree.children[orderedFrequentItems[0]]
        else:
            updateHeadPointTable(headPointTable[orderedFrequentItems[0]][1], fptree.children[orderedFrequentItems[0]])
    #handle other items except the first item
    if(len(orderedFrequentItems) > 1):
        updateFPTree(fptree.children[orderedFrequentItems[0]], orderedFrequentItems[1::], headPointTable, count)

def updateHeadPointTable(headPointBeginNode, targetNode):
    while(headPointBeginNode.nextSimilarItem != None):
        headPointBeginNode = headPointBeginNode.nextSimilarItem
    headPointBeginNode.nextSimilarItem = targetNode

# 开始挖掘频繁项集，这里也是递归迭代的思路。
# 对于头指针表中的每一个元素，首先获取该元素结尾的所有前缀路径，然后将所有前缀路径作为新的数据集传入 createFPTree 函数中以创建条件 FP 树。
# 然后对条件 FP 树对应的头指针表中的每一个元素，开始获取前缀路径，并创建新的条件 FP 树。这两步不断重复，直到条件 FP 树中只有一个元素为止。
def mineFPTree(headPointTable, prefix, frequentPatterns, minSupport):
    #for each item in headPointTable, find conditional prefix path, create conditional fptree, then iterate until there is only one element in conditional fptree
    headPointItems = [v[0] for v in sorted(headPointTable.items(), key = lambda v:v[1][0])]
    if(len(headPointItems) == 0): return

    for headPointItem in headPointItems:
        newPrefix = prefix.copy()
        newPrefix.add(headPointItem)
        support = headPointTable[headPointItem][0]
        frequentPatterns[frozenset(newPrefix)] = support

        prefixPath = getPrefixPath(headPointTable, headPointItem)
        if(prefixPath != {}):
            conditionalFPtree, conditionalHeadPointTable = createFPTree(prefixPath, minSupport)
            if conditionalHeadPointTable != None:
                mineFPTree(conditionalHeadPointTable, newPrefix, frequentPatterns, minSupport)

# 获取前缀路径的步骤。对于每一个相同元素，通过父节点指针不断向上遍历，所得的路径就是该元素的前缀路径。
def getPrefixPath(headPointTable, headPointItem):
    prefixPath = {}
    beginNode = headPointTable[headPointItem][1]
    prefixs = ascendTree(beginNode)
    if((prefixs != [])):
        prefixPath[frozenset(prefixs)] = beginNode.count

    while(beginNode.nextSimilarItem != None):
        beginNode = beginNode.nextSimilarItem
        prefixs = ascendTree(beginNode)
        if (prefixs != []):
            prefixPath[frozenset(prefixs)] = beginNode.count
    return prefixPath

def ascendTree(treeNode):
    prefixs = []
    while((treeNode.nodeParent != None) and (treeNode.nodeParent.nodeName != 'null')):
        treeNode = treeNode.nodeParent
        prefixs.append(treeNode.nodeName)
    return prefixs

# 挖掘关联规则的代码，这里也用到了递归迭代的技巧。对于每一个频繁项集，构造所有可能的关联规则，然后对每一个关联规则计算置信度，输出置信度大于阈值的关联规则。
def rulesGenerator(frequentPatterns, minConf, rules):
    for frequentset in frequentPatterns:
        if(len(frequentset) > 1):
            getRules(frequentset,frequentset, rules, frequentPatterns, minConf)

def removeStr(set, str):
    tempSet = []
    for elem in set:
        if(elem != str):
            tempSet.append(elem)
    tempFrozenSet = frozenset(tempSet)
    return tempFrozenSet


def getRules(frequentset,currentset, rules, frequentPatterns, minConf):
    for frequentElem in currentset:
        subSet = removeStr(currentset, frequentElem)
        confidence = frequentPatterns[frequentset] / frequentPatterns[subSet]
        if (confidence >= minConf):
            flag = False
            for rule in rules:
                if(rule[0] == subSet and rule[1] == frequentset - subSet):
                    flag = True
            if(flag == False):
                rules.append((subSet, frequentset - subSet, confidence))

            if(len(subSet) >= 2):
                getRules(frequentset, subSet, rules, frequentPatterns, minConf)

if __name__=='__main__':
    print("fptree:")
    dataSet = loadDataSet()
    frozenDataSet = transfer2FrozenDataSet(dataSet)
    minSupport = 3
    fptree, headPointTable = createFPTree(frozenDataSet, minSupport)
    # fptree.disp()
    frequentPatterns = {}
    prefix = set([])
    mineFPTree(headPointTable, prefix, frequentPatterns, minSupport)
    print("频繁模式(frequent pattern):")
    print(frequentPatterns)
    minConf = 0.6
    rules = []
    rulesGenerator(frequentPatterns, minConf, rules)
    print("关联规则(AssociationRules):")
    print(rules)