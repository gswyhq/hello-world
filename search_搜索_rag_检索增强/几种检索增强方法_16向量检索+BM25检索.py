
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法16：
# 融合检索系统，结合了语义向量搜索和基于关键词的BM25检索的优点。这种方法通过捕捉概念相似性和精确关键词匹配，提高了检索质量。
# 融合检索通过以下方式结合了两者的优点：
#
# - 执行基于向量和关键词的检索
# - 对每种方法的分数进行标准化
# - 使用加权公式组合它们
# - 根据组合分数对文档进行排名

## 设置环境

# 我们从导入必要的库开始。

import os
import numpy as np
from rank_bm25 import BM25Okapi
import fitz
from openai import OpenAI
import re
import json
import time
from sklearn.metrics.pairwise import cosine_similarity

# 清理文本
def clean_text(text):
    """
    通过移除多余的空白字符和特殊字符来清理文本。

    Args:
        text (str): 输入文本

    Returns:
        str: 清理后的文本
    """
    # 使用正则表达式将多个空白字符替换为单个空格
    text = re.sub(r'\s+', ' ', text)

    # 修复常见的OCR问题，将制表符和换行符替换为空格
    text = text.replace('\\t', ' ')
    text = text.replace('\\n', ' ')

    # 移除任何前导或尾随空白，并确保单词之间只有一个空格
    text = ' '.join(text.split())

    return text

# BM25实现
# 创建BM25索引
def create_bm25_index(chunks):
    """
    从给定的块创建BM25索引。

    Args:
        chunks (List[Dict]): 文本块列表

    Returns:
        BM25Okapi: BM25索引
    """
    # 从每个块中提取文本
    texts = [chunk["text"] for chunk in chunks]

    # 通过空格分割每个文档进行分词
    tokenized_docs = [text.split() for text in texts]

    # 使用分词后的文档创建BM25索引
    bm25 = BM25Okapi(tokenized_docs)

    # 打印BM25索引中的文档数
    print(f"创建了包含{len(texts)}个文档的BM25索引")
    return bm25

# BM25搜索
def bm25_search(bm25, chunks, query, k=5):
    """
    使用查询搜索BM25索引。

    Args:
        bm25 (BM25Okapi): BM25索引
        chunks (List[Dict]): 文本块列表
        query (str): 查询字符串
        k (int): 要返回的结果数

    Returns:
        List[Dict]: 带有分数的前k个结果
    """
    # 通过空格分割查询进行分词
    query_tokens = query.split()

    # 获取查询分词在索引文档上的BM25分数
    scores = bm25.get_scores(query_tokens)

    # 初始化一个空列表以存储结果及其分数
    results = []

    # 遍历分数和对应的块
    for i, score in enumerate(scores):
        # 复制元数据以避免修改原始数据
        metadata = chunks[i].get("metadata", {}).copy()
        # 添加索引到元数据
        metadata["index"] = i

        results.append({
            "text": chunks[i]["text"],
            "metadata": metadata,  # 带有索引的元数据
            "bm25_score": float(score)
        })

    # 按BM25分数降序排序结果
    results.sort(key=lambda x: x["bm25_score"], reverse=True)

    # 返回前k个结果
    return results[:k]
# 融合检索函数
def fusion_retrieval(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    执行结合向量和BM25搜索的融合检索。

    Args:
        query (str): 查询字符串
        chunks (List[Dict]): 原始文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要返回的结果数
        alpha (float): 向量分数的权重（0-1），其中1-alpha是BM25权重

    Returns:
        List[Dict]: 根据组合分数排名的前k个结果
    """
    print(f"为查询执行融合检索：{query}")
    # 定义一个很小的epsilon以避免除零错误
    epsilon = 1e-8

    # 获取向量搜索结果
    query_embedding = create_embeddings(query)  # 为查询创建嵌入
    vector_results = vector_store.similarity_search_with_scores(query_embedding, k=len(chunks))  # 执行向量搜索

    # 获取BM25搜索结果
    bm25_results = bm25_search(bm25_index, chunks, query, k=len(chunks))  # 执行BM25搜索

    # 创建字典以映射文档索引到分数
    vector_scores_dict = {result["metadata"]["index"]: result["similarity"] for result in vector_results}
    bm25_scores_dict = {result["metadata"]["index"]: result["bm25_score"] for result in bm25_results}

    # 确保所有文档都有两种方法的分数
    all_docs = vector_store.get_all_documents()
    combined_results = []

    for i, doc in enumerate(all_docs):
        vector_score = vector_scores_dict.get(i, 0.0)  # 获取向量分数或0如果未找到
        bm25_score = bm25_scores_dict.get(i, 0.0)  # 获取BM25分数或0如果未找到
        combined_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "vector_score": vector_score,
            "bm25_score": bm25_score,
            "index": i
        })

    # 提取分数作为数组
    vector_scores = np.array([doc["vector_score"] for doc in combined_results])
    bm25_scores = np.array([doc["bm25_score"] for doc in combined_results])

    # 标准化分数
    norm_vector_scores = (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)
    norm_bm25_scores = (bm25_scores - np.min(bm25_scores)) / (np.max(bm25_scores) - np.min(bm25_scores) + epsilon)

    # 计算组合分数
    combined_scores = alpha * norm_vector_scores + (1 - alpha) * norm_bm25_scores

    # 将组合分数添加到结果中
    for i, score in enumerate(combined_scores):
        combined_results[i]["combined_score"] = float(score)

    # 按组合分数降序排序
    combined_results.sort(key=lambda x: x["combined_score"], reverse=True)

    # 返回前k个结果
    top_results = combined_results[:k]

    print(f"通过融合检索检索到{len(top_results)}个文档")
    return top_results

# 文档处理管道

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为融合检索处理文档。

    Args:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠（以字符为单位）

    Returns:
        Tuple[List[Dict], SimpleVectorStore, BM25Okapi]: 块、向量存储和BM25索引
    """
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    # 清理提取的文本以移除多余的空白字符和特殊字符
    cleaned_text = clean_text(text)

    # 将清理后的文本分割为重叠的块
    chunks = chunk_text(cleaned_text, chunk_size, chunk_overlap)

    # 从每个块中提取文本内容以创建嵌入
    chunk_texts = [chunk["text"] for chunk in chunks]
    print("创建块的嵌入...")
    embeddings = create_embeddings(chunk_texts)

    # 初始化向量存储
    vector_store = SimpleVectorStore()

    # 将块及其嵌入添加到向量存储
    vector_store.add_items(chunks, embeddings)
    print(f"将{len(chunks)}个项目添加到向量存储")

    # 从块创建BM25索引
    bm25_index = create_bm25_index(chunks)

    # 返回块、向量存储和BM25索引
    return chunks, vector_store, bm25_index

# 响应生成
def generate_response(query, context):
    """
    根据查询和上下文生成响应。

    Args:
        query (str): 用户查询
        context (str): 来自检索文档的上下文

    Returns:
        str: 生成的响应
    """
    # 定义系统提示以指导AI助手
    system_prompt = """你是一个有帮助的AI助手。根据提供的上下文回答用户的问题。
    如果上下文中没有包含足够的相关信息来完整回答问题，请指出这一限制。"""

    # 格式化用户提示，包含上下文和查询
    user_prompt = f"""上下文：
    {context}

    问题：{query}

    请根据提供的上下文回答问题。"""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": user_prompt}  # 用户消息，包含上下文和查询
        ],
        temperature=0.1  # 设置响应生成的温度
    )

    # 返回生成的响应
    return response.choices[0].message.content

# 主检索函数
def answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k=5, alpha=0.5):
    """
    使用融合RAG回答查询。

    Args:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数
        alpha (float): 向量分数的权重

    Returns:
        Dict: 查询结果，包括检索到的文档和响应
    """
    # 使用融合检索方法检索文档
    retrieved_docs = fusion_retrieval(query, chunks, vector_store, bm25_index, k=k, alpha=alpha)

    # 通过连接检索到的文档的文本，使用分隔符格式化上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # 根据查询和格式化的上下文生成响应
    response = generate_response(query, context)

    # 返回查询、检索到的文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

# 比较检索方法
# 仅向量RAG
def vector_only_rag(query, vector_store, k=5):
    """
    使用仅向量的RAG回答查询。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要检索的文档数

    Returns:
        Dict: 查询结果
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    # 使用向量相似性搜索检索文档
    retrieved_docs = vector_store.similarity_search_with_scores(query_embedding, k=k)

    # 通过连接检索到的文档的文本，使用分隔符格式化上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # 根据查询和格式化的上下文生成响应
    response = generate_response(query, context)

    # 返回查询、检索到的文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

# 仅BM25 RAG
def bm25_only_rag(query, chunks, bm25_index, k=5):
    """
    使用仅BM25的RAG回答查询。

    Args:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数

    Returns:
        Dict: 查询结果
    """
    # 检索文档使用BM25搜索
    retrieved_docs = bm25_search(bm25_index, chunks, query, k=k)

    # 通过连接检索到的文档的文本，使用分隔符格式化上下文
    context = "\n\n---\n\n".join([doc["text"] for doc in retrieved_docs])

    # 根据查询和格式化的上下文生成响应
    response = generate_response(query, context)

    # 返回查询、检索到的文档和生成的响应
    return {
        "query": query,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

# 评估函数
# 比较检索方法
def compare_retrieval_methods(query, chunks, vector_store, bm25_index, k=5, alpha=0.5, reference_answer=None):
    """
    比较不同检索方法的查询结果。

    Args:
        query (str): 用户查询
        chunks (List[Dict]): 文本块
        vector_store (SimpleVectorStore): 向量存储
        bm25_index (BM25Okapi): BM25索引
        k (int): 要检索的文档数
        alpha (float): 融合检索中向量分数的权重
        reference_answer (str, 可选): 参考答案用于比较

    Returns:
        Dict: 比较结果
    """
    print(f"\n=== 比较检索方法的查询：{query} ===\n")

    # 运行仅向量的RAG
    print("\n运行仅向量的RAG...")
    vector_result = vector_only_rag(query, vector_store, k)

    # 运行仅BM25的RAG
    print("\n运行仅BM25的RAG...")
    bm25_result = bm25_only_rag(query, chunks, bm25_index, k)

    # 运行融合RAG
    print("\n运行融合RAG...")
    fusion_result = answer_with_fusion_rag(query, chunks, vector_store, bm25_index, k, alpha)

    # 比较不同检索方法的响应
    print("\n比较响应...")
    comparison = evaluate_responses(
        query,
        vector_result["response"],
        bm25_result["response"],
        fusion_result["response"],
        reference_answer
    )

    # 返回比较结果
    return {
        "query": query,
        "vector_result": vector_result,
        "bm25_result": bm25_result,
        "fusion_result": fusion_result,
        "comparison": comparison
    }

# 评估响应
def evaluate_responses(query, vector_response, bm25_response, fusion_response, reference_answer=None):
    """
    评估来自不同检索方法的响应。

    Args:
        query (str): 用户查询
        vector_response (str): 仅向量RAG的响应
        bm25_response (str): 仅BM25 RAG的响应
        fusion_response (str): 融合RAG的响应
        reference_answer (str, 可选): 参考答案

    Returns:
        str: 响应的评估
    """
    # 系统提示以指导评估器
    system_prompt = """你是一位RAG系统的专家评估者。比较三种不同检索方法的响应：
    1. 基于向量的检索：使用语义相似性进行文档检索
    2. BM25关键词检索：使用关键词匹配进行文档检索
    3. 融合检索：结合了基于向量和关键词的方法

    根据以下标准评估响应：
    - 与查询的相关性
    - 事实正确性
    - 全面性
    - 清晰度和连贯性"""

    # 用户提示，包含查询和响应
    user_prompt = f"""查询：{query}

    基于向量的响应：
    {vector_response}

    BM25关键词响应：
    {bm25_response}

    融合响应：
    {fusion_response}
    """

    # 如果提供参考答案，将其添加到提示中
    if reference_answer:
        user_prompt += f"""参考答案：
        {reference_answer}
        """

    # 将详细比较指令添加到用户提示
    user_prompt += """请提供这三种响应的详细比较。哪种方法对此查询表现最好，为什么？
    具体说明每种方法在该特定查询中的优缺点。"""

    # 使用meta-llama/Llama-3.2-3B-Instruct生成评估
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导评估器
            {"role": "user", "content": user_prompt}  # 用户消息，包含查询和响应
        ],
        temperature=0  # 设置响应生成的温度
    )

    # 返回生成的评估内容
    return response.choices[0].message.content

# 完整评估管道
# 评估融合检索
def evaluate_fusion_retrieval(pdf_path, test_queries, reference_answers=None, k=5, alpha=0.5):
    """
    评估融合检索与其他方法的比较。

    Args:
        pdf_path (str): PDF文件的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], 可选): 参考答案
        k (int): 要检索的文档数
        alpha (float): 融合检索中向量分数的权重

    Returns:
        Dict: 评估结果
    """
    print("=== 评估融合检索 ===\n")

    # 处理文档以提取文本、创建块、构建向量和BM25索引
    chunks, vector_store, bm25_index = process_document(pdf_path)

    # 初始化一个列表以存储每个查询的结果
    results = []

    # 遍历每个测试查询
    for i, query in enumerate(test_queries):
        print(f"\n\n=== 评估查询 {i+1}/{len(test_queries)} ===")
        print(f"查询：{query}")

        # 获取参考答案如果可用
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # 比较检索方法的当前查询
        comparison = compare_retrieval_methods(
            query,
            chunks,
            vector_store,
            bm25_index,
            k=k,
            alpha=alpha,
            reference_answer=reference
        )

        # 将比较结果附加到结果列表
        results.append(comparison)

        # 打印不同检索方法的响应
        print("\n=== 基于向量的响应 ===")
        print(comparison["vector_result"]["response"])

        print("\n=== BM25响应 ===")
        print(comparison["bm25_result"]["response"])

        print("\n=== 融合响应 ===")
        print(comparison["fusion_result"]["response"])

        print("\n=== 比较 ===")
        print(comparison["comparison"])

    # 生成融合检索性能的总体分析
    overall_analysis = generate_overall_analysis(results)

    # 返回结果和总体分析
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }

# 生成总体分析
def generate_overall_analysis(results):
    """
    生成融合检索的总体分析。

    Args:
        results (List[Dict]): 评估查询的结果

    Returns:
        str: 总体分析
    """
    # 系统提示以指导评估过程
    system_prompt = """你是一位评估信息检索系统的专家。根据多个测试查询，提供对三种检索方法的总体分析：
    1. 基于向量的检索（语义相似性）
    2. BM25关键词检索（关键词匹配）
    3. 融合检索（两者的结合）

    重点分析：
    1. 每种方法在不同类型的查询中表现最佳的情况
    2. 每种方法的总体优缺点
    3. 融合检索如何平衡权衡
    4. 推荐在什么情况下使用每种方法"""

    # 创建每个查询评估的摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}: {result['query']}\n"
        evaluations_summary += f"比较摘要：{result['comparison'][:200]}...\n\n"

    # 用户提示，包含评估摘要
    user_prompt = f"""根据以下对不同检索方法在{len(results)}个查询上的评估，提供对这三种方法的总体分析：

    {evaluations_summary}

    请根据这些评估结果，全面分析基于向量、BM25和融合检索方法，突出融合检索在个体方法上的优势。"""

    # 使用meta-llama/Llama-3.2-3B-Instruct生成总体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 返回生成的分析内容
    return response.choices[0].message.content

# 评估融合检索
# PDF文档的路径
# 包含AI信息的PDF文件路径，用于知识检索测试
pdf_path = "data/AI_Information.pdf"

# 定义一个与AI相关的测试查询
test_queries = [
    "什么是自然语言处理中变压器模型的主要应用？"  # 与AI相关的查询
]

# 可选参考答案
reference_answers = [
    "变压器模型在自然语言处理中彻底改变了多个领域，包括机器翻译、文本摘要、问答系统、情感分析和文本生成。它们特别擅长捕捉文本中的长距离依赖关系，并已成为BERT、GPT和T5等模型的基础。",
]

# 设置参数
k = 5  # 要检索的文档数
alpha = 0.5  # 向量分数的权重（0.5表示向量和BM25的权重相等）

# 运行评估
evaluation_results = evaluate_fusion_retrieval(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers,
    k=k,
    alpha=alpha
)

# 打印总体分析
print("\n\n=== 总体分析 ===\n")
print(evaluation_results["overall_analysis"])

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


