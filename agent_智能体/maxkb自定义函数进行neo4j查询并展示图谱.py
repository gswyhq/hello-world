
import json 
import traceback

import requests
import json

# 测试数据构建
"""
MERGE (s:Node { name: 'A' }) MERGE (o:Node { name: 'H' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'A' }) MERGE (o:Node { name: 'I' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'I' }) MERGE (o:Node { name: 'D' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'I' }) MERGE (o:Node { name: 'C' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'G' }) MERGE (o:Node { name: 'E' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'G' }) MERGE (o:Node { name: 'I' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'E' }) MERGE (o:Node { name: 'F' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'E' }) MERGE (o:Node { name: 'B' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'I' }) MERGE (o:Node { name: 'D' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
MERGE (s:Node { name: 'I' }) MERGE (o:Node { name: 'C' })
MERGE (s)-[r:下游]->(o)
RETURN s, r, o;
"""

def batch_match(cypher="match (n: Person{name:\"Tom Hanks\"}) -[r]-> (m) return n,r, m limit 3;  ", statements=None, url='http://192.168.3.105:7474/db/data/transaction/commit', user='neo4j', password='123456'):
    try:
        if statements is None:
            body = {
                "statements": [{
                    "statement": cypher
                }]
            }
        else:
            body = {
                "statements": statements
            }
        r = requests.post(url, auth=(user, password), json=body,
                          headers={"Content-Type": "application/json; charset=UTF-8"}, timeout=(2, 2))
        return r.json()
    except Exception as e:
        print(f"查询出错， {e}, 错误详情：{traceback.format_exc()}")
        return {"errors": [f"{e}"]}

def show_graph(name):
    '''根据节点名称查询neo4j数据，并转换为可展示数据结构'''
    query = f"""MATCH p = (n:Node)-[r:下游*]->(m) 
            WHERE n.name = "{name}" 
            UNWIND relationships(p) AS rel
            RETURN distinct
              startNode(rel) AS source,
              endNode(rel) AS target
    LIMIT 100"""
    ret = batch_match(cypher=query)
    if not ret.get('results', []) or not ret['results'][0].get('data'):
        answer =  f"查询`{name}`的结果为空"
        query="MATCH (n:Node)-->() return distinct n.name limit 10"
        ret2 = batch_match(cypher=query)
        if ret2.get("results") and ret2['results'][0].get('data'):
            answer += "\n你可以查询如下节点："
            for row in ret2['results'][0]['data']:
                answer += f"\n<quick_question>{row['row'][0]}</quick_question>"
        return answer 
        
    result = ret['results'][0]['data']
    nodes = set()
    links = []
    for record in result:
        source, target = record["row"]
        source, target = source['name'], target['name']
        nodes.add(source)
        nodes.add(target)
        links.append({"source": source, "target": target})
    # 构建 ECharts 可用的图数据格式
    nodes = [{"name": node} for node in nodes]

    colorPalette = [
        '#FF6666', '#66CCFF', '#99FF99', '#FFD700', '#FFA07A',
        '#87CEFA', '#FFB6C1', '#DDA0DD', '#00CED1', '#FF69B4',
        '#8A2BE2', '#00BFFF', '#7FFF00', '#FF4500', '#DA70D6',
        '#3CB371', '#00FA9A', '#FF6347', '#4682B4', '#F0E68C'
    ]
    
    for idx, n in enumerate(nodes):
        n['symbolSize'] = 25 
        n['itemStyle'] = {'color': colorPalette[idx % len(colorPalette)]}
    option = """option = {
                    title: {
                        text: ''
                    },
                    tooltip: {},
                    legend: [{
                        data: ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
                    }],
                    series: [{
                        type: 'graph',
                        layout: 'force',
                        data: graph_nodes,
                        links: graph_links,
                        label: {
                            show: true,
                            position: 'right'
                        },
                        force: {
                            repulsion: 1000
                        },
                        roam: true,
                        edgeLabel: {show: false, position: 'middle'},
                        
                        // 设置连接线箭头
                        edgeSymbol: ['none', 'arrow'], // 只在终点显示箭头
                        edgeSymbolSize: [8, 14], // 箭头大小
                        
                        // 设置节点形状为圆形
                        symbol: 'circle', // circle为圆形， arrow 为方块
                        
                        // 添加箭头
                        lineStyle: {
                            color: 'source',
                            width: 2,
                            curveness: 0, // 0表示直线
                            type: 'arrow' // 添加箭头
                        },
                        symbolSize: [20, 20] // 调整箭头大小
                    }]
                };
    """
    option = option.replace('graph_nodes', json.dumps(nodes, ensure_ascii=0)).replace('graph_links', json.dumps(links, ensure_ascii=0))
    return f"查询`{name}`的下游节点如下\n<echarts_rander>{json.dumps({'actionType': 'EVAL', 'option': option, 'style': {'height': '600px', 'width': '100%'}}, ensure_ascii=0)}</echarts_rander>"



    

