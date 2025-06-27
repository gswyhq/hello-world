
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法20：
# 校正 RAG (CRAG) 实现

 # Corrective RAG（CRAG）——一种高级方法，它能够动态评估检索到的信息，并在必要时使用网络搜索作为补充。

# CRAG 相对于传统 RAG 的改进包括：
#
# - 在使用之前评估检索到的内容
# - 动态切换到更相关的知识来源
# - 当本地知识不足时，使用网络搜索进行检索纠正
# - 在适当的情况下结合多个来源的信息

## 环境设置

import os
import numpy as np
import json
import fitz  # PyMuPDF
from openai import OpenAI
import requests
from typing import List, Dict, Tuple, Any
import re
from urllib.parse import quote_plus
import time

# 初始化 OpenAI API 客户端
# 我们初始化 OpenAI 客户端以生成嵌入和响应。
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.getenv("OPENAI_API_KEY")  # 从环境变量中获取 API 密钥
)

# 文档处理函数
# 从 PDF 中提取文本
def extract_text_from_pdf(pdf_path):
    """
    从 PDF 文件中提取文本内容。

    Args:
        pdf_path (str): PDF 文件的路径

    Returns:
        str: 提取的文本内容
    """
    print(f"从 {pdf_path} 中提取文本...")
    pdf = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text += page.get_text()
    return text

# 文本分块
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块，以便高效检索和处理。

    Args:
        text (str): 输入文本
        chunk_size (int): 每个块的最大字符数
        overlap (int): 相邻块之间的重叠字符数

    Returns:
        List[Dict]: 文本块列表，每个块包含：
                   - text: 块内容
                   - metadata: 包含位置信息和来源类型的字典
    """
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]
        if chunk_text:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "start_pos": i,
                    "end_pos": i + len(chunk_text),
                    "source_type": "document"
                }
            })
    print(f"创建了 {len(chunks)} 个文本块")
    return chunks

# 简单向量存储实现
class SimpleVectorStore:
    """
    使用 NumPy 实现的简单向量存储。
    """
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        将一个项添加到向量存储中。

        Args:
            text (str): 文本内容
            embedding (List[float]): 嵌入向量
            metadata (Dict, optional): 额外元数据
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def add_items(self, items, embeddings):
        """
        将多个项添加到向量存储中。

        Args:
            items (List[Dict]): 包含文本和元数据的项列表
            embeddings (List[List[float]]): 嵌入向量列表
        """
        for i, (item, embedding) in enumerate(zip(items, embeddings)):
            self.add_item(
                text=item["text"],
                embedding=embedding,
                metadata=item.get("metadata", {})
            )

    def similarity_search(self, query_embedding, k=5):
        """
        根据查询嵌入找到最相似的项。

        Args:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回的结果数量

        Returns:
            List[Dict]: 最相似的 top k 项
        """
        if not self.vectors:
            return []
        query_vector = np.array(query_embedding)
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))
        similarities.sort(key=lambda x: x[1], reverse=True)
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": float(score)
            })
        return results

# 创建嵌入
def create_embeddings(texts, model="text-embedding-3-small"):
    """
    使用 OpenAI 的嵌入模型为文本创建向量嵌入。

    Args:
        texts (str 或 List[str]): 输入文本，可以是单个字符串或字符串列表
        model (str): 要使用的嵌入模型名称，默认为 "text-embedding-3-small"

    Returns:
        List[List[float]]: 如果输入是列表，返回嵌入向量列表；
                          如果输入是单个字符串，返回单个嵌入向量
    """
    input_texts = texts if isinstance(texts, list) else [texts]
    batch_size = 100
    all_embeddings = []
    for i in range(0, len(input_texts), batch_size):
        batch = input_texts[i:i + batch_size]
        response = client.embeddings.create(
            model=model,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)
    if isinstance(texts, str):
        return all_embeddings[0]
    return all_embeddings

# 文档处理管道
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理成向量存储。

    Args:
        pdf_path (str): PDF 文件的路径
        chunk_size (int): 每个块的大小（字符数）
        chunk_overlap (int): 块之间的重叠字符数

    Returns:
        SimpleVectorStore: 包含文档块的向量存储
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print("为块创建嵌入...")
    chunk_texts = [chunk["text"] for chunk in chunks]
    chunk_embeddings = create_embeddings(chunk_texts)
    vector_store = SimpleVectorStore()
    vector_store.add_items(chunks, chunk_embeddings)
    print(f"创建了包含 {len(chunks)} 个块的向量存储")
    return vector_store

# 相关性评估函数
def evaluate_document_relevance(query, document):
    """
    评估文档与查询的相关性。

    Args:
        query (str): 用户查询
        document (str): 文档文本

    Returns:
        float: 相关性评分（0-1）
    """
    system_prompt = """
    你是一位擅长评估文档相关性的专家。
    根据给定的查询，对文档的相关性进行评分，评分范围为 0 到 1。
    0 表示完全不相关，1 表示完全相关。
    仅返回评分，格式为 0 到 1 之间的浮点数。
    """
    user_prompt = f"查询: {query}\n\n文档: {document}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            max_tokens=5
        )
        score_text = response.choices[0].message.content.strip()
        score_match = re.search(r'(\d+(?:\.\d+)?)', score_text)
        if score_match:
            return float(score_match.group(1))
        return 0.5
    except Exception as e:
        print(f"评估文档相关性时出错: {e}")
        return 0.5

# 网络搜索函数
# 使用 DuckDuckGo 进行搜索
def duck_duck_go_search(query, num_results=3):
    """
    使用 DuckDuckGo 进行网络搜索。

    Args:
        query (str): 搜索查询
        num_results (int): 返回的结果数量

    Returns:
        Tuple[str, List[Dict]]: 搜索结果文本和来源元数据
    """
    encoded_query = quote_plus(query)
    url = f"https://api.duckduckgo.com/?q={encoded_query}&format=json"
    try:
        response = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })
        data = response.json()
        results_text = ""
        sources = []
        if data.get("AbstractText"):
            results_text += f"{data['AbstractText']}\n\n"
            sources.append({
                "title": data.get("AbstractSource", "维基百科"),
                "url": data.get("AbstractURL", "")
            })
        for topic in data.get("RelatedTopics", [])[:num_results]:
            if "Text" in topic and "FirstURL" in topic:
                results_text += f"{topic['Text']}\n\n"
                sources.append({
                    "title": topic.get("Text", "").split(" - ")[0],
                    "url": topic.get("FirstURL", "")
                })
        return results_text, sources
    except Exception as e:
        print(f"执行网络搜索时出错: {e}")
        try:
            backup_url = f"https://serpapi.com/search.json?q={encoded_query}&engine=duckduckgo"
            response = requests.get(backup_url)
            data = response.json()
            results_text = ""
            sources = []
            for result in data.get("organic_results", [])[:num_results]:
                results_text += f"{result.get('title', '')}: {result.get('snippet', '')}\n\n"
                sources.append({
                    "title": result.get("title", ""),
                    "url": result.get("link", "")
                })
            return results_text, sources
        except Exception as backup_error:
            print(f"备用搜索也失败: {backup_error}")
            return "无法检索到搜索结果。", []

# 重写搜索查询
def rewrite_search_query(query):
    """
    将查询重写为更适合网络搜索的形式。

    Args:
        query (str): 原始查询

    Returns:
        str: 重写后的查询
    """
    system_prompt = """
    你是一位擅长创建有效搜索查询的专家。
    将给定的查询重写为更适合网络搜索引擎的形式。
    着重于关键词和事实，去除不必要的词语，并使其简洁。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"原始查询: {query}\n\n重写后的查询:"}
            ],
            temperature=0.3,
            max_tokens=50
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"重写搜索查询时出错: {e}")
        return query

# 执行网络搜索
def perform_web_search(query):
    """
    执行带有查询重写的网络搜索。

    Args:
        query (str): 用户查询

    Returns:
        Tuple[str, List[Dict]]: 搜索结果文本和来源元数据
    """
    rewritten_query = rewrite_search_query(query)
    print(f"重写后的搜索查询: {rewritten_query}")
    results_text, sources = duck_duck_go_search(rewritten_query)
    return results_text, sources

# 知识精炼函数
def refine_knowledge(text):
    """
    从文本中提取和精炼关键信息。

    Args:
        text (str): 输入文本

    Returns:
        str: 精炼后的关键点
    """
    system_prompt = """
    从以下文本中提取关键信息，以清晰简洁的 bullet points 形式呈现。
    着重于最重要的事实和细节。
    请以每个要点开头为 "• " 的方式返回。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"待精炼的文本:\n\n{text}"}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"精炼知识时出错: {e}")
        return text

# 核心 CRAG 过程
def crag_process(query, vector_store, k=3):
    """
    运行 Corrective RAG 过程。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 初始检索的文档数量

    Returns:
        Dict: 处理结果，包括响应和调试信息
    """
    print(f"\n=== 正在处理查询: {query} ===\n")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=k)
    relevance_scores = []
    for doc in retrieved_docs:
        score = evaluate_document_relevance(query, doc["text"])
        relevance_scores.append(score)
        doc["relevance"] = score
        print(f"文档评分: {score:.2f} 相关性")
    max_score = max(relevance_scores) if relevance_scores else 0
    best_doc_idx = relevance_scores.index(max_score) if relevance_scores else -1
    sources = []
    final_knowledge = ""
    if max_score > 0.7:
        print(f"高相关性 ({max_score:.2f}) - 直接使用文档")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        final_knowledge = best_doc
        sources.append({
            "title": "文档",
            "url": ""
        })
    elif max_score < 0.3:
        print(f"低相关性 ({max_score:.2f}) - 执行网络搜索")
        web_results, web_sources = perform_web_search(query)
        final_knowledge = refine_knowledge(web_results)
        sources.extend(web_sources)
    else:
        print(f"中等相关性 ({max_score:.2f}) - 结合文档和网络搜索")
        best_doc = retrieved_docs[best_doc_idx]["text"]
        refined_doc = refine_knowledge(best_doc)
        web_results, web_sources = perform_web_search(query)
        refined_web = refine_knowledge(web_results)
        final_knowledge = f"来自文档:\n{refined_doc}\n\n来自网络搜索:\n{refined_web}"
        sources.append({
            "title": "文档",
            "url": ""
        })
        sources.extend(web_sources)
    response = generate_response(query, final_knowledge, sources)
    return {
        "query": query,
        "response": response,
        "retrieved_docs": retrieved_docs,
        "relevance_scores": relevance_scores,
        "max_relevance": max_score,
        "final_knowledge": final_knowledge,
        "sources": sources
    }

# 响应生成
def generate_response(query, knowledge, sources):
    """
    根据查询和知识生成响应。

    Args:
        query (str): 用户查询
        knowledge (str): 基础知识
        sources (List[Dict]): 来源列表，包含标题和 URL

    Returns:
        str: 生成的响应
    """
    sources_text = ""
    for source in sources:
        title = source.get("title", "未知来源")
        url = source.get("url", "")
        if url:
            sources_text += f"- {title}: {url}\n"
        else:
            sources_text += f"- {title}\n"
    system_prompt = """
    你是一位乐于助人的 AI 助手。根据提供的知识，生成一个全面、信息丰富的查询响应。
    包括所有相关的信息，同时保持回答清晰简洁。
    如果知识无法完全回答查询，请承认这一限制。
    在响应末尾包含来源归属。
    """
    user_prompt = f"""
    查询: {query}

    知识:
    {knowledge}

    来源:
    {sources_text}

    请根据此信息提供对查询的详细回答。
    在响应末尾包含来源。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.2
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成响应时出错: {e}")
        return f"我为您的查询 '{query}' 遇到了一个错误: '{str(e)}'"

# 评估函数
# 评估 CRAG 响应
def evaluate_crag_response(query, response, reference_answer=None):
    """
    评估 CRAG 响应的质量。

    Args:
        query (str): 用户查询
        response (str): 生成的响应
        reference_answer (str, optional): 参考答案

    Returns:
        Dict: 评估指标
    """
    system_prompt = """
    你是一位擅长评估问题回答质量的专家。
    根据以下标准评估提供的响应:
    1. 相关性 (0-10): 响应是否直接回答查询？
    2. 准确性 (0-10): 信息是否事实正确？
    3. 完整性 (0-10): 响应是否全面回答查询的所有方面？
    4. 清晰度 (0-10): 响应是否清晰易懂？
    5. 来源质量 (0-10): 响应是否恰当引用来源？
    返回包含每个标准评分和简要解释的 JSON 对象。
    同时包含 "overall_score" (0-10) 和 "summary"。
    """
    user_prompt = f"""
    查询: {query}

    待评估的响应:
    {response}
    """
    if reference_answer:
        user_prompt += f"""
    参考答案（用于比较）:
    {reference_answer}
    """
    try:
        evaluation_response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0
        )
        evaluation = json.loads(evaluation_response.choices[0].message.content)
        return evaluation
    except Exception as e:
        print(f"评估响应时出错: {e}")
        return {
            "error": str(e),
            "overall_score": 0,
            "summary": "评估失败，出现错误。"
        }

# 比较 CRAG 和标准 RAG
def compare_crag_vs_standard_rag(query, vector_store, reference_answer=None):
    """
    比较 CRAG 和标准 RAG 对查询的处理。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        reference_answer (str, optional): 参考答案

    Returns:
        Dict: 比较结果
    """
    print("\n=== 正在运行 CRAG ===")
    crag_result = crag_process(query, vector_store)
    crag_response = crag_result["response"]
    print("\n=== 正在运行标准 RAG ===")
    query_embedding = create_embeddings(query)
    retrieved_docs = vector_store.similarity_search(query_embedding, k=3)
    combined_text = "\n\n".join([doc["text"] for doc in retrieved_docs])
    standard_sources = [{"title": "文档", "url": ""}]
    standard_response = generate_response(query, combined_text, standard_sources)
    print("\n=== 正在评估 CRAG 响应 ===")
    crag_eval = evaluate_crag_response(query, crag_response, reference_answer)
    print("\n=== 正在评估标准 RAG 响应 ===")
    standard_eval = evaluate_crag_response(query, standard_response, reference_answer)
    print("\n=== 正在比较方法 ===")
    comparison = compare_responses(query, crag_response, standard_response, reference_answer)
    return {
        "query": query,
        "crag_response": crag_response,
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "crag_evaluation": crag_eval,
        "standard_evaluation": standard_eval,
        "comparison": comparison
    }

# 比较响应
def compare_responses(query, crag_response, standard_response, reference_answer=None):
    """
    比较 CRAG 和标准 RAG 的响应。

    Args:
        query (str): 用户查询
        crag_response (str): CRAG 响应
        standard_response (str): 标准 RAG 响应
        reference_answer (str, optional): 参考答案

    Returns:
        str: 比较分析
    """
    system_prompt = """
    你是一位比较两种响应生成方法的专家:
    1. CRAG (Corrective RAG): 一种评估文档相关性并在需要时切换到网络搜索的系统。
    2. 标准 RAG: 一种直接根据嵌入相似性检索文档并用于响应生成的系统。
    根据以下标准比较两种方法:
    - 准确性和事实正确性
    - 查询相关性
    - 回答完整性
    - 清晰度和组织性
    - 来源归属质量
    解释哪种方法在特定查询中表现更好，以及原因。
    """
    user_prompt = f"""
    查询: {query}

    CRAG 响应:
    {crag_response}

    标准 RAG 响应:
    {standard_response}
    """
    if reference_answer:
        user_prompt += f"""
    参考答案:
    {reference_answer}
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"比较响应时出错: {e}")
        return f"比较响应时出错: {str(e)}"

# 完整评估管道
# 运行 CRAG 评估
def run_crag_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    运行 CRAG 的完整评估，使用多个测试查询。

    Args:
        pdf_path (str): PDF 文档的路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 参考答案列表

    Returns:
        Dict: 完整评估结果
    """
    vector_store = process_document(pdf_path)
    results = []
    for i, query in enumerate(test_queries):
        print(f"\n\n===== 评估查询 {i+1}/{len(test_queries)} =====")
        print(f"查询: {query}")
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]
        result = compare_crag_vs_standard_rag(query, vector_store, reference)
        results.append(result)
        print("\n=== 比较 ===")
        print(result["comparison"])
    overall_analysis = generate_overall_analysis(results)
    return {
        "results": results,
        "overall_analysis": overall_analysis
    }

# 生成总体分析
def generate_overall_analysis(results):
    """
    根据评估结果生成总体分析。

    Args:
        results (List[Dict]): 单个查询评估结果列表

    Returns:
        str: 总体分析
    """
    system_prompt = """
    你是一位擅长评估信息检索和响应生成系统的专家。
    根据多个测试查询，提供 CRAG (Corrective RAG) 和标准 RAG 的总体分析。
    着重于:
    1. CRAG 表现更好的情况及原因
    2. 标准 RAG 表现更好的情况及原因
    3. 每种方法的优缺点
    4. 推荐使用每种方法的场景
    """
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i+1}: {result['query']}\n"
        if 'crag_evaluation' in result and 'overall_score' in result['crag_evaluation']:
            crag_score = result['crag_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"CRAG 评分: {crag_score}\n"
        if 'standard_evaluation' in result and 'overall_score' in result['standard_evaluation']:
            std_score = result['standard_evaluation'].get('overall_score', 'N/A')
            evaluations_summary += f"标准 RAG 评分: {std_score}\n"
        evaluations_summary += f"比较摘要: {result['comparison'][:200]}...\n\n"
    user_prompt = f"""
    根据以下评估结果，比较 CRAG 和标准 RAG 在 {len(results)} 个查询中的表现:
    {evaluations_summary}
    请提供对这两种方法相对优缺点的全面分析，重点在于哪种方法在什么情况下表现更好，以及原因。
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"生成总体分析时出错: {e}")
        return f"生成总体分析时出错: {str(e)}"

# 使用测试查询评估 CRAG
# PDF 文档路径
pdf_path = "data/AI_Information.pdf"

# 运行包含多个 AI 相关查询的全面评估
test_queries = [
    "机器学习与传统编程有何不同？",
]

# 可选参考答案以提高评估质量
reference_answers = [
    "机器学习与传统编程的不同之处在于，机器学习使计算机通过数据中的模式进行学习，而不是遵循显式的指令。在传统编程中，开发人员编写特定的规则供计算机遵循，而在机器学习中，",
]

# 运行完整评估，比较 CRAG 和标准 RAG
evaluation_results = run_crag_evaluation(pdf_path, test_queries, reference_answers)
print("\n=== CRAG 与标准 RAG 的总体分析 ===")
print(evaluation_results["overall_analysis"])



# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


