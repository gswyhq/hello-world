
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法17：
# 图增强检索增强生成：Graph RAG

# Graph RAG——一种通过将知识组织成一个连通图而不是扁平的文档集合来增强传统RAG系统的技术。

### Graph RAG的关键优势
# - 保留信息片段之间的关系
# - 通过连接的概念进行遍历以找到相关上下文
# - 提高处理复杂多部分查询的能力
# - 通过可视化知识路径提供更好的可解释性

## 环境设置

import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
from typing import List, Dict, Tuple, Any
import networkx as nx
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import re
from PIL import Image
import io

# 知识图谱构建
# 从文本中提取概念
def extract_concepts(text):
    """
    使用OpenAI的API从文本中提取关键概念。

    Args:
        text (str): 要提取概念的文本

    Returns:
        List[str]: 概念列表
    """
    # 系统消息以指示模型执行操作
    system_message = """提取文本中的关键概念和实体。
返回文本中最重要的5-10个关键术语、实体或概念。
将响应格式化为字符串数组的JSON数组。"""
    # 发送请求到OpenAI API
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"从以下文本中提取关键概念:\n\n{text[:3000]}"}  # 限制API调用
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    try:
        # 从响应中解析概念
        concepts_json = json.loads(response.choices[0].message.content)
        concepts = concepts_json.get("concepts", [])
        if not concepts and "concepts" not in concepts_json:
            # 尝试获取响应中的任何数组
            for key, value in concepts_json.items():
                if isinstance(value, list):
                    concepts = value
                    break
        return concepts
    except (json.JSONDecodeError, AttributeError):
        # 如果JSON解析失败，使用备用方法
        content = response.choices[0].message.content
        # 尝试提取看起来像列表的内容
        matches = re.findall(r'\[(.*?)\]', content, re.DOTALL)
        if matches:
            items = re.findall(r'"([^"]*)"', matches[0])
            return items
        return []

# 构建知识图谱
def build_knowledge_graph(chunks):
    """
    从文本块构建知识图谱。

    Args:
        chunks (List[Dict]): 包含元数据的文本块列表

    Returns:
        Tuple[nx.Graph, List[np.ndarray]]: 知识图谱和块嵌入
    """
    print("构建知识图谱...")
    # 创建一个图
    graph = nx.Graph()
    # 提取块文本
    texts = [chunk["text"] for chunk in chunks]
    # 为所有块创建嵌入
    print("为块创建嵌入...")
    embeddings = create_embeddings(texts)
    # 将节点添加到图中
    print("将节点添加到图中...")
    for i, chunk in enumerate(chunks):
        # 从块中提取概念
        print(f"提取块{i+1}/{len(chunks)}的概念...")
        concepts = extract_concepts(chunk["text"])
        # 添加节点及其属性
        graph.add_node(i,
                      text=chunk["text"],
                      concepts=concepts,
                      embedding=embeddings[i])
    # 基于共享概念连接节点
    print("在节点之间创建边...")
    for i in range(len(chunks)):
        node_concepts = set(graph.nodes[i]["concepts"])
        for j in range(i + 1, len(chunks)):
            # 计算概念重叠
            other_concepts = set(graph.nodes[j]["concepts"])
            shared_concepts = node_concepts.intersection(other_concepts)
            # 如果有共享概念，则添加边
            if shared_concepts:
                # 使用嵌入计算语义相似性
                similarity = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
                # 基于概念重叠和语义相似性计算边权重
                concept_score = len(shared_concepts) / min(len(node_concepts), len(other_concepts))
                edge_weight = 0.7 * similarity + 0.3 * concept_score
                # 仅添加具有显著关系的边
                if edge_weight > 0.6:
                    graph.add_edge(i, j,
                                  weight=edge_weight,
                                  similarity=similarity,
                                  shared_concepts=list(shared_concepts))
    print(f"知识图谱构建完成，包含{graph.number_of_nodes()}个节点和{graph.number_of_edges()}条边")
    return graph, embeddings

# 图遍历和查询处理
# 图遍历
def traverse_graph(query, graph, embeddings, top_k=5, max_depth=3):
    """
    遍历知识图谱以查找与查询相关的信息。

    Args:
        query (str): 用户的问题
        graph (nx.Graph): 知识图谱
        embeddings (List): 节点嵌入列表
        top_k (int): 要考虑的初始节点数
        max_depth (int): 遍历的最大深度

    Returns:
        List[Dict]: 图遍历找到的相关信息
    """
    print(f"为查询遍历图谱：{query}")
    # 获取查询嵌入
    query_embedding = create_embeddings(query)
    # 计算查询与所有节点的相似性
    similarities = []
    for i, node_embedding in enumerate(embeddings):
        similarity = np.dot(query_embedding, node_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding))
        similarities.append((i, similarity))
    # 按相似性降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    # 获取最相似的top_k节点作为起点
    starting_nodes = [node for node, _ in similarities[:top_k]]
    print(f"从{len(starting_nodes)}个节点开始遍历")
    # 初始化遍历
    visited = set()  # 用于跟踪已访问节点的集合
    traversal_path = []  # 用于存储遍历路径的列表
    results = []  # 用于存储结果的列表
    # 使用优先队列进行遍历
    queue = []
    for node in starting_nodes:
        heapq.heappush(queue, (-similarities[node][1], node))  # 使用负数以实现最大堆
    # 使用修改后的广度优先搜索进行遍历
    while queue and len(results) < (top_k * 3):  # 限制结果为top_k * 3
        _, node = heapq.heappop(queue)
        if node in visited:
            continue
        # 标记为已访问
        visited.add(node)
        traversal_path.append(node)
        # 将当前节点的文本添加到结果
        results.append({
            "text": graph.nodes[node]["text"],
            "concepts": graph.nodes[node]["concepts"],
            "node_id": node
        })
        # 如果尚未达到最大深度，则探索邻居
        if len(traversal_path) < max_depth:
            neighbors = [(neighbor, graph[node][neighbor]["weight"])
                        for neighbor in graph.neighbors(node)
                        if neighbor not in visited]
            # 将邻居按权重添加到队列
            for neighbor, weight in sorted(neighbors, key=lambda x: x[1], reverse=True):
                heapq.heappush(queue, (-weight, neighbor))
    print(f"图遍历找到{len(results)}个相关块")
    return results, traversal_path

# 响应生成
def generate_response(query, context_chunks):
    """
    基于检索到的上下文生成响应。

    Args:
        query (str): 用户的问题
        context_chunks (List[Dict]): 图遍历找到的相关块

    Returns:
        str: 生成的响应
    """
    # 从上下文块中提取文本
    context_texts = [chunk["text"] for chunk in context_chunks]
    # 将提取的文本组合成一个上下文字符串，用“---”分隔
    combined_context = "\n\n---\n\n".join(context_texts)
    # 定义允许的上下文最大长度（OpenAI限制）
    max_context = 14000
    # 如果上下文超过最大长度，则截断
    if len(combined_context) > max_context:
        combined_context = combined_context[:max_context] + "... [已截断]"
    # 定义系统消息以指导AI助手
    system_message = """你是一个有帮助的AI助手。根据提供的上下文回答用户的问题。
如果信息不在上下文中，请说明。在回答时尽量引用上下文的具体部分。"""
    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定要使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 系统消息以指导助手
            {"role": "user", "content": f"上下文:\n{combined_context}\n\n问题: {query}"}  # 包含上下文和查询的用户消息
        ],
        temperature=0.2  # 设置响应生成的温度
    )
    # 返回生成的响应内容
    return response.choices[0].message.content

# 可视化
# 可视化图遍历
def visualize_graph_traversal(graph, traversal_path):
    """
    可视化知识图谱和遍历路径。

    Args:
        graph (nx.Graph): 知识图谱
        traversal_path (List): 遍历顺序的节点列表
    """
    plt.figure(figsize=(12, 10))  # 设置图形大小
    # 定义节点颜色，默认为浅蓝色
    node_color = ['lightblue'] * graph.number_of_nodes()
    # 将遍历路径节点突出显示为浅绿色
    for node in traversal_path:
        node_color[node] = 'lightgreen'
    # 将起始节点突出显示为绿色，结束节点为红色
    if traversal_path:
        node_color[traversal_path[0]] = 'green'
        node_color[traversal_path[-1]] = 'red'
    # 使用弹簧布局创建所有节点的位置
    pos = nx.spring_layout(graph, k=0.5, iterations=50, seed=42)
    # 绘制图节点
    nx.draw_networkx_nodes(graph, pos, node_color=node_color, node_size=500, alpha=0.8)
    # 绘制边，宽度与权重成正比
    for u, v, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], width=weight*2, alpha=0.6)
    # 用红色虚线绘制遍历路径
    traversal_edges = [(traversal_path[i], traversal_path[i+1])
                      for i in range(len(traversal_path)-1)]
    nx.draw_networkx_edges(graph, pos, edgelist=traversal_edges,
                          width=3, alpha=0.8, edge_color='red',
                          style='dashed', arrows=True)
    # 使用节点的第一个概念作为标签
    labels = {}
    for node in graph.nodes():
        concepts = graph.nodes[node]['concepts']
        label = concepts[0] if concepts else f"节点{node}"
        labels[node] = f"{node}: {label}"
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=8)
    plt.title("知识图谱与遍历路径")  # 设置图形标题
    plt.axis('off')  # 关闭坐标轴
    plt.tight_layout()  # 调整布局
    plt.show()  # 显示图形

# 完整的Graph RAG管道
def graph_rag_pipeline(pdf_path, query, chunk_size=1000, chunk_overlap=200, top_k=3):
    """
    从文档到答案的完整Graph RAG管道。

    Args:
        pdf_path (str): PDF文档的路径
        query (str): 用户的问题
        chunk_size (int): 文本块的大小
        chunk_overlap (int): 块之间的重叠
        top_k (int): 要考虑的初始节点数

    Returns:
        Dict: 包含答案和图可视化数据的结果
    """
    # 从PDF文档中提取文本
    text = extract_text_from_pdf(pdf_path)
    # 将提取的文本分割为重叠块
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    # 从文本块构建知识图谱
    graph, embeddings = build_knowledge_graph(chunks)
    # 遍历知识图谱以查找与查询相关的信息
    relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings, top_k)
    # 基于查询和相关块生成响应
    response = generate_response(query, relevant_chunks)
    # 可视化图遍历路径
    visualize_graph_traversal(graph, traversal_path)
    # 返回查询、响应、相关块、遍历路径和图
    return {
        "query": query,
        "response": response,
        "relevant_chunks": relevant_chunks,
        "traversal_path": traversal_path,
        "graph": graph
    }

# 评估函数
# 评估Graph RAG
def evaluate_graph_rag(pdf_path, test_queries, reference_answers=None):
    """
    在多个测试查询上评估Graph RAG。

    Args:
        pdf_path (str): PDF文档的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 参考答案用于比较

    Returns:
        Dict: 评估结果
    """
    # 从PDF中提取文本
    text = extract_text_from_pdf(pdf_path)
    # 将文本分割为块
    chunks = chunk_text(text)
    # 构建知识图谱（所有查询共用一次）
    graph, embeddings = build_knowledge_graph(chunks)
    results = []
    for i, query in enumerate(test_queries):
        print(f"\n\n=== 评估查询 {i+1}/{len(test_queries)} ===")
        print(f"查询: {query}")
        # 遍历图以查找相关信息
        relevant_chunks, traversal_path = traverse_graph(query, graph, embeddings)
        # 生成响应
        response = generate_response(query, relevant_chunks)
        # 如果有参考答案，则进行比较
        reference = None
        comparison = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
            comparison = compare_with_reference(response, reference, query)
        # 为当前查询附加结果
        results.append({
            "query": query,
            "response": response,
            "reference_answer": reference,
            "comparison": comparison,
            "traversal_path_length": len(traversal_path),
            "relevant_chunks_count": len(relevant_chunks)
        })
        # 显示结果
        print(f"\n响应: {response}\n")
        if comparison:
            print(f"比较: {comparison}\n")
    # 返回评估结果和图统计信息
    return {
        "results": results,
        "graph_stats": {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "avg_degree": sum(dict(graph.degree()).values()) / graph.number_of_nodes()
        }
    }

# 比较与参考答案
def compare_with_reference(response, reference, query):
    """
    比较生成的响应与参考答案。

    Args:
        response (str): 生成的响应
        reference (str): 参考答案
        query (str): 原始查询

    Returns:
        str: 比较分析
    """
    # 系统消息以指示模型如何比较响应
    system_message = """比较AI生成的响应与参考答案。
根据正确性、完整性和与查询的相关性进行评估。
提供一个简短的分析（2-3句话），说明AI响应与参考答案的匹配程度。"""
    # 构造包含查询、AI生成响应和参考答案的提示
    prompt = f"""查询: {query}

AI生成的响应:
{response}

参考答案:
{reference}

AI响应与参考答案的匹配程度如何？
"""
    # 发送请求到OpenAI API以生成比较分析
    comparison = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_message},  # 系统消息以指导助手
            {"role": "user", "content": prompt}  # 包含提示的用户消息
        ],
        temperature=0.0  # 设置响应生成的温度
    )
    # 返回生成的比较分析
    return comparison.choices[0].message.content

# 在示例PDF文档上评估Graph RAG
# PDF文档的路径，包含AI信息
pdf_path = "data/AI_Information.pdf"

# 定义一个用于测试Graph RAG的AI相关查询
query = "什么是Transformer在自然语言处理中的关键应用？"

# 执行Graph RAG管道以处理文档并回答查询
results = graph_rag_pipeline(pdf_path, query)

# 打印Graph RAG系统生成的响应
print("\n=== 答案 ===")
print(results["response"])

# 定义测试查询和参考答案用于正式评估
test_queries = [
    "与RNN相比，Transformer如何处理顺序数据？"
]

# 用于评估的参考答案
reference_answers = [
    "与RNN不同，Transformer使用自注意力机制而不是递归连接来处理顺序数据。这使得Transformer能够并行处理所有标记，而不是逐个处理，从而更有效地捕获长距离依赖关系，并在训练过程中实现更好的并行化。与RNN不同，Transformer不会在长序列中遇到梯度消失问题。"
]

# 在测试查询上正式评估Graph RAG系统
evaluation = evaluate_graph_rag(pdf_path, test_queries, reference_answers)

# 打印评估摘要统计信息
print("\n=== 评估摘要 ===")
print(f"图节点数: {evaluation['graph_stats']['nodes']}")
print(f"图边数: {evaluation['graph_stats']['edges']}")
for i, result in enumerate(evaluation['results']):
    print(f"\n查询 {i+1}: {result['query']}")
    print(f"路径长度: {result['traversal_path_length']}")
    print(f"使用的块数: {result['relevant_chunks_count']}")


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


