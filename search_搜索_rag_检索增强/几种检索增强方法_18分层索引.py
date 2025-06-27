
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法18：
# 分层索引用于RAG系统
# 用于RAG系统的分层索引方法。该技术通过使用两层检索方法来提高检索效果：首先通过摘要识别相关文档部分，然后从这些部分中检索具体细节。

# 传统的RAG方法将所有文本块同等对待，这可能导致以下问题：
#
# - 当文本块过小时丢失上下文
# - 当文档集合较大时出现不相关结果
# - 在整个语料库中进行低效搜索
#
# 分层检索通过以下方式解决这些问题：
#
# - 为较大的文档部分创建简洁的摘要
# - 首先搜索这些摘要以识别相关部分
# - 然后仅从这些部分中检索详细信息
# - 在保持上下文的同时保留具体细节

## 设置环境


import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
import pickle


# 文档处理函数
# 从PDF提取文本

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取带分页的文本内容。

    Args:
        pdf_path (str): PDF文件路径

    Returns:
        List[Dict]: 包含文本内容和元数据的页面列表
    """
    print(f"从{pdf_path}提取文本...")  # 打印正在处理的PDF路径
    pdf = fitz.open(pdf_path)  # 使用PyMuPDF打开PDF文件
    pages = []  # 初始化一个空列表以存储包含文本内容的页面

    # 遍历PDF中的每一页
    for page_num in range(len(pdf)):
        page = pdf[page_num]  # 获取当前页面
        text = page.get_text()  # 从当前页面提取文本

        # 跳过文本量非常小的页面（少于50个字符）
        if len(text.strip()) > 50:
            # 将页面文本和元数据附加到列表中
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页面编号（从1开始）
                }
            })

    print(f"提取了{len(pages)}个包含内容的页面")  # 打印提取的页面数量
    return pages  # 返回包含文本内容和元数据的页面列表


class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """
    def init(self):
        self.vectors = []  # 用于存储向量嵌入的列表
        self.texts = []  # 用于存储文本内容的列表
        self.metadata = []  # 用于存储元数据的列表

    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目。

        参数:
            text (str): 文本内容
            embedding (List[float]): 向量嵌入
            metadata (Dict, 可选): 额外的元数据
        """
        self.vectors.append(np.array(embedding))  # 将嵌入作为NumPy数组追加
        self.texts.append(text)  # 追加文本内容
        self.metadata.append(metadata or {})  # 追加元数据或如果为None则追加空字典


    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        查找与查询嵌入最相似的项目。

        参数:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回的结果数量
            filter_func (callable, 可选): 用于过滤结果的函数

        返回:
            List[Dict]: 最相似的前k个项目
        """
        if not self.vectors:
            return []  # 如果没有向量，则返回空列表

        # 将查询嵌入转换为NumPy数组
        query_vector = np.array(query_embedding)

        # 使用余弦相似度计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 如果不通过过滤器，则跳过
            if filter_func and not filter_func(self.metadata[i]):
                continue

            # 计算余弦相似度
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 追加索引和相似度分数

        # 按相似度排序（降序）
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加文本内容
                "metadata": self.metadata[idx],  # 添加元数据
                "similarity": float(score)  # 添加相似度分数
            })

        return results  # 返回最相似的前k个结果列表

# 文本分块
def chunk_text(text, metadata, chunk_size=1000, overlap=200):
    """
    在保留元数据的同时将文本分成重叠块。

    Args:
        text (str): 输入文本
        metadata (Dict): 元数据
        chunk_size (int): 每个块的大小（以字符为单位）
        overlap (int): 块之间的重叠部分（以字符为单位）

    Returns:
        List[Dict]: 包含文本块和元数据的列表
    """
    chunks = []  # 初始化一个空列表以存储块

    # 以指定的块大小和重叠部分遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 提取文本块

        # 跳过非常小的块（少于50个字符）
        if chunk_text and len(chunk_text.strip()) > 50:
            # 复制元数据并添加块特定信息
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": len(chunks),  # 块索引
                "start_char": i,  # 块起始字符索引
                "end_char": i + len(chunk_text),  # 块结束字符索引
                "is_summary": False  # 标记表示这不是摘要
            })

            # 将块及其元数据附加到列表中
            chunks.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks  # 返回包含文本块和元数据的列表


# 摘要生成函数
def generate_page_summary(page_text):
    """
    生成页面的简洁摘要。

    Args:
        page_text (str): 页面文本内容

    Returns:
        str: 生成的摘要
    """
    # 定义系统提示以指导摘要模型
    system_prompt = """你是一个专业的摘要系统。
    为提供的文本创建详细摘要。专注于捕捉主要主题、关键信息和重要事实。
    你的摘要应足够全面以理解页面内容，但比原文更简洁。"""

    # 如果文本超过最大标记限制，进行截断
    max_tokens = 6000
    truncated_text = page_text[:max_tokens] if len(page_text) > max_tokens else page_text

    # 请求OpenAI API生成摘要
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": f"请总结以下文本:\n\n{truncated_text}"}  # 用户消息包含要总结的文本
        ],
        temperature=0.3  # 设置响应生成的温度
    )

    # 返回生成的摘要内容
    return response.choices[0].message.content


# 分层文档处理
def process_document_hierarchically(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    将文档处理为分层索引。

    Args:
        pdf_path (str): PDF文件路径
        chunk_size (int): 详细块的大小
        chunk_overlap (int): 块之间的重叠部分

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 摘要和详细向量存储
    """
    # 从PDF提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 为每个页面创建摘要
    print("生成页面摘要...")
    summaries = []
    for i, page in enumerate(pages):
        print(f"总结第{i + 1}/{len(pages)}页...")
        summary_text = generate_page_summary(page["text"])

        # 创建摘要元数据
        summary_metadata = page["metadata"].copy()
        summary_metadata.update({"is_summary": True})

        # 将摘要文本和元数据附加到摘要列表中
        summaries.append({
            "text": summary_text,
            "metadata": summary_metadata
        })

    # 为每个页面创建详细块
    detailed_chunks = []
    for page in pages:
        # 将页面文本分块
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # 将当前页面的块扩展到详细块列表中
        detailed_chunks.extend(page_chunks)

    print(f"创建了{len(detailed_chunks)}个详细块")

    # 为摘要创建嵌入
    print("为摘要创建嵌入...")
    summary_texts = [summary["text"] for summary in summaries]
    summary_embeddings = create_embeddings(summary_texts)

    # 为详细块创建嵌入
    print("为详细块创建嵌入...")
    chunk_texts = [chunk["text"] for chunk in detailed_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # 创建向量存储
    summary_store = SimpleVectorStore()
    detailed_store = SimpleVectorStore()

    # 将摘要添加到摘要存储
    for i, summary in enumerate(summaries):
        summary_store.add_item(
            text=summary["text"],
            embedding=summary_embeddings[i],
            metadata=summary["metadata"]
        )

    # 将块添加到详细存储
    for i, chunk in enumerate(detailed_chunks):
        detailed_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    print(f"创建了包含{len(summaries)}个摘要和{len(detailed_chunks)}个块的向量存储")
    return summary_store, detailed_store


# 分层检索
def retrieve_hierarchically(query, summary_store, detailed_store, k_summaries=3, k_chunks=5):
    """
    使用分层索引进行检索。

    Args:
        query (str): 用户查询
        summary_store (SimpleVectorStore): 摘要文档存储
        detailed_store (SimpleVectorStore): 详细块存储
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 要检索的块数量（每个摘要）

    Returns:
        List[Dict]: 检索到的块及其相关性分数
    """
    print(f"为查询执行分层检索：{query}")
    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 首先检索相关摘要
    summary_results = summary_store.similarity_search(
        query_embedding,
        k=k_summaries
    )

    print(f"检索到{len(summary_results)}个相关摘要")

    # 收集来自相关摘要的页面
    relevant_pages = [result["metadata"]["page"] for result in summary_results]

    # 创建过滤函数以仅保留相关页面的块
    def page_filter(metadata):
        return metadata["page"] in relevant_pages

    # 然后从相关页面中检索详细块
    detailed_results = detailed_store.similarity_search(
        query_embedding,
        k=k_chunks * len(relevant_pages),
        filter_func=page_filter
    )

    print(f"从相关页面检索到{len(detailed_results)}个详细块")

    # 对于每个结果，添加其来自哪个摘要/页面
    for result in detailed_results:
        page = result["metadata"]["page"]
        matching_summaries = [s for s in summary_results if s["metadata"]["page"] == page]
        if matching_summaries:
            result["summary"] = matching_summaries[0]["text"]

    return detailed_results


# 带上下文的响应生成
def generate_response(query, retrieved_chunks):
    """
    基于查询和检索到的块生成响应。

    Args:
        query (str): 用户查询
        retrieved_chunks (List[Dict]): 分层搜索检索到的块

    Returns:
        str: 生成的响应
    """
    # 从块中提取文本并准备上下文部分
    context_parts = []

    for i, chunk in enumerate(retrieved_chunks):
        page_num = chunk["metadata"]["page"]  # 从元数据中获取页面编号
        context_parts.append(f"[第{page_num}页]: {chunk['text']}")  # 格式化块文本并附加页面编号

    # 将所有上下文部分组合成一个上下文字符串
    context = "\n\n".join(context_parts)

    # 定义系统消息以指导AI助手
    system_message = """你是一个乐于助人的AI助手，根据提供的上下文回答用户的问题。
使用上下文中的信息准确回答用户的问题。
如果上下文中没有相关信息，请说明。
在引用具体信息时包含页面编号。"""

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_message},  # 系统消息以指导助手
            {"role": "user", "content": f"上下文:\n\n{context}\n\n问题: {query}"}  # 用户消息包含上下文和查询
        ],
        temperature=0.2  # 设置响应生成的温度
    )

    # 返回生成的响应内容
    return response.choices[0].message.content


# 带分层检索的完整RAG管道
def hierarchical_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200,
                     k_summaries=3, k_chunks=5, regenerate=False):
    """
    完整的分层RAG管道。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        chunk_size (int): 详细块的大小
        chunk_overlap (int): 块之间的重叠部分
        k_summaries (int): 要检索的摘要数量
        k_chunks (int): 要检索的块数量（每个摘要）
        regenerate (bool): 是否重新生成向量存储

    Returns:
        Dict: 包括响应和检索到的块的结果
    """
    # 创建存储文件名用于缓存
    summary_store_file = f"{os.path.basename(pdf_path)}_summary_store.pkl"
    detailed_store_file = f"{os.path.basename(pdf_path)}_detailed_store.pkl"

    # 处理文档并创建存储（如果需要）
    if regenerate or not os.path.exists(summary_store_file) or not os.path.exists(detailed_store_file):
        print("处理文档并创建向量存储...")
        # 处理文档以创建分层索引和向量存储
        summary_store, detailed_store = process_document_hierarchically(
            pdf_path, chunk_size, chunk_overlap
        )

        # 将摘要存储保存到文件以备将来使用
        with open(summary_store_file, 'wb') as f:
            pickle.dump(summary_store, f)

        # 将详细存储保存到文件以备将来使用
        with open(detailed_store_file, 'wb') as f:
            pickle.dump(detailed_store, f)
    else:
        # 从文件加载现有的摘要存储
        print("加载现有的向量存储...")
        with open(summary_store_file, 'rb') as f:
            summary_store = pickle.load(f)

        # 从文件加载现有的详细存储
        with open(detailed_store_file, 'rb') as f:
            detailed_store = pickle.load(f)

    # 使用查询分层检索相关块
    retrieved_chunks = retrieve_hierarchically(
        query, summary_store, detailed_store, k_summaries, k_chunks
    )

    # 基于检索到的块生成响应
    response = generate_response(query, retrieved_chunks)

    # 返回包括查询、响应、检索到的块以及摘要和详细块数量的结果
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks,
        "summary_count": len(summary_store.texts),
        "detailed_count": len(detailed_store.texts)
    }


# 标准（非分层）RAG用于比较
def standard_rag(query, pdf_path, chunk_size=1000, chunk_overlap=200, k=15):
    """
    不带分层检索的标准RAG管道。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        chunk_size (int): 每个块的大小
        chunk_overlap (int): 块之间的重叠部分
        k (int): 要检索的块数量

    Returns:
        Dict: 包括响应和检索到的块的结果
    """
    # 从PDF提取页面
    pages = extract_text_from_pdf(pdf_path)

    # 直接从所有页面创建块
    chunks = []
    for page in pages:
        # 将页面文本分块
        page_chunks = chunk_text(
            page["text"],
            page["metadata"],
            chunk_size,
            chunk_overlap
        )
        # 将当前页面的块扩展到块列表中
        chunks.extend(page_chunks)

    print(f"为标准RAG创建了{len(chunks)}个块")

    # 创建向量存储以保存块
    store = SimpleVectorStore()

    # 为块创建嵌入
    print("为块创建嵌入...")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = create_embeddings(texts)

    # 将块添加到向量存储
    for i, chunk in enumerate(chunks):
        store.add_item(
            text=chunk["text"],
            embedding=embeddings[i],
            metadata=chunk["metadata"]
        )

    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 基于查询嵌入检索最相关的块
    retrieved_chunks = store.similarity_search(query_embedding, k=k)
    print(f"通过标准RAG检索到{len(retrieved_chunks)}个块")

    # 基于检索到的块生成响应
    response = generate_response(query, retrieved_chunks)

    # 返回包括查询、响应和检索到的块的结果
    return {
        "query": query,
        "response": response,
        "retrieved_chunks": retrieved_chunks
    }


# 评估函数,比较方法
def compare_approaches(query, pdf_path, reference_answer=None):
    """
    比较分层和标准RAG方法。

    Args:
        query (str): 用户查询
        pdf_path (str): PDF文档路径
        reference_answer (str, optional): 参考答案用于评估

    Returns:
        Dict: 比较结果
    """
    print(f"\n=== 为查询比较RAG方法：{query} ===")

    # 运行分层RAG
    print("\n运行分层RAG...")
    hierarchical_result = hierarchical_rag(query, pdf_path)
    hier_response = hierarchical_result["response"]

    # 运行标准RAG
    print("\n运行标准RAG...")
    standard_result = standard_rag(query, pdf_path)
    std_response = standard_result["response"]

    # 比较分层和标准RAG的结果
    comparison = compare_responses(query, hier_response, std_response, reference_answer)

    # 返回包含比较结果的字典
    return {
        "query": query,  # 原始查询
        "hierarchical_response": hier_response,  # 分层RAG的响应
        "standard_response": std_response,  # 标准RAG的响应
        "reference_answer": reference_answer,  # 用于评估的参考答案
        "comparison": comparison,  # 比较分析
        "hierarchical_chunks_count": len(hierarchical_result["retrieved_chunks"]),  # 分层RAG检索到的块数量
        "standard_chunks_count": len(standard_result["retrieved_chunks"])  # 标准RAG检索到的块数量
    }


# 比较响应
def compare_responses(query, hierarchical_response, standard_response, reference=None):
    """
    比较分层和标准RAG生成的响应。

    Args:
        query (str): 用户查询
        hierarchical_response (str): 分层RAG的响应
        standard_response (str): 标准RAG的响应
        reference (str, optional): 参考答案

    Returns:
        str: 比较分析
    """
    # 定义系统提示以指导模型评估响应
    system_prompt = """你是一位信息检索系统的专家评估员。
    比较针对同一查询生成的两个响应，一个使用分层检索，另一个使用标准检索。

    根据以下标准进行评估：
    1. 准确性：哪个响应提供了更多事实上的正确信息？
    2. 全面性：哪个响应更好地涵盖了查询的所有方面？
    3. 连贯性：哪个响应具有更好的逻辑流程和组织？
    4. 页面引用：哪个响应更好地使用了页面引用？

    在分析中具体说明每种方法的优缺点。"""

    # 创建包含查询和两个响应的用户提示
    user_prompt = f"""查询：{query}

分层RAG生成的响应：
{hierarchical_response}

标准RAG生成的响应：
{standard_response}"""

    # 如果提供了参考答案，将其包含在用户提示中
    if reference:
        user_prompt += f"""参考答案：
{reference}"""

    # 添加最终的指令到用户提示中
    user_prompt += """请详细比较这两个响应，突出说明哪种方法表现更好以及原因。"""

    # 请求OpenAI API生成比较分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": user_prompt}  # 用户消息包含查询和响应
        ],
        temperature=0  # 设置响应生成的温度
    )

    # 返回生成的比较分析
    return response.choices[0].message.content


# 运行评估
def run_evaluation(pdf_path, test_queries, reference_answers=None):
    """
    运行完整的评估，使用多个测试查询。

    Args:
        pdf_path (str): PDF文档路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 查询的参考答案

    Returns:
        Dict: 评估结果
    """
    results = []  # 初始化一个空列表以存储结果

    # 遍历每个测试查询
    for i, query in enumerate(test_queries):
        print(f"查询：{query}")  # 打印当前查询

        # 获取参考答案（如果可用）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]  # 检索当前查询的参考答案

        # 比较分层和标准RAG方法
        result = compare_approaches(query, pdf_path, reference)
        results.append(result)  # 将结果附加到结果列表中

    # 生成评估结果的总体分析
    overall_analysis = generate_overall_analysis(results)

    return {
        "results": results,  # 返回单个查询结果
        "overall_analysis": overall_analysis  # 返回总体分析
    }


# 生成总体分析
def generate_overall_analysis(results):
    """
    生成评估结果的总体分析。

    Args:
        results (List[Dict]): 单个查询评估结果列表

    Returns:
        str: 总体分析
    """
    # 定义系统提示以指导模型评估结果
    system_prompt = """你是一位信息检索系统的专家评估员。
    基于多个测试查询，提供分层RAG与标准RAG的总体分析。

    重点：
    1. 分层检索表现更好的情况及原因
    2. 标准检索表现更好的情况及原因
    3. 每种方法的总体优缺点
    4. 推荐每种方法的使用场景"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询{i + 1}：{result['query']}\n"
        evaluations_summary += f"分层块：{result['hierarchical_chunks_count']}，标准块：{result['standard_chunks_count']}\n"
        evaluations_summary += f"比较摘要：{result['comparison'][:200]}...\n\n"

    # 定义包含评估摘要的用户提示
    user_prompt = f"""基于以下评估，比较分层与标准RAG在{len(results)}个查询上的表现：

{evaluations_summary}

请根据检索质量和响应生成，提供分层RAG与标准RAG的综合分析。"""

    # 请求OpenAI API生成总体分析
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": user_prompt}  # 用户消息包含评估摘要
        ],
        temperature=0  # 设置响应生成的温度
    )

    # 返回生成的总体分析
    return response.choices[0].message.content


# 分层与标准RAG方法的评估
# AI信息PDF文件的路径
pdf_path = "data/AI_Information.pdf"

# 用于测试分层RAG方法的AI相关示例查询
query = "什么是Transformer模型在自然语言处理中的关键应用？"
result = hierarchical_rag(query, pdf_path)

print("\n=== 响应 ===")
print(result["response"])

# 用于正式评估的测试查询（仅使用一个查询以符合要求）
test_queries = [
    "与RNN相比，Transformer如何处理序列数据？"
]

# 用于测试查询的参考答案以进行比较
reference_answers = [
    "Transformer通过自注意力机制而不是递归连接来处理序列数据，这允许它们并行处理所有标记，而不是按顺序处理，从而更高效地捕获长距离依赖关系，并在训练过程中实现更好的并行化。与RNN不同，Transformer不会因长序列而遭受梯度消失问题。"
]

# 运行比较分层和标准RAG方法的评估
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印比较的总体分析
print("\n=== 总体分析 ===")
print(evaluation_results["overall_analysis"])

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


