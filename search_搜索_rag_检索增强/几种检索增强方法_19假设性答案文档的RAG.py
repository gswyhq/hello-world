
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法19：
# 假设性文档嵌入（HyDE）用于RAG
# HyDE（Hypothetical Document Embedding）——一种创新的检索技术，它通过将用户查询转换为假设性答案文档后再进行检索，从而弥合了短查询与长文档之间的语义差距。
# 传统RAG系统直接嵌入用户的短查询，但这通常无法捕捉到最优检索所需的语义丰富性。HyDE通过以下方式解决了这一问题：
# - 生成一个回答查询的假设性文档
# - 嵌入这个扩展后的文档，而不是原始查询
# - 检索与该假设性文档相似的文档
# - 创建更具上下文相关性的答案

## 环境设置
import os
import numpy as np
import json
import fitz
from openai import OpenAI
import re
import matplotlib.pyplot as plt

# 文档处理函数
# 从PDF提取文本
def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容，按页面分开。

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

        # 跳过文本很少的页面（少于50个字符）
        if len(text.strip()) > 50:
            # 将页面文本和元数据添加到列表中
            pages.append({
                "text": text,
                "metadata": {
                    "source": pdf_path,  # 源文件路径
                    "page": page_num + 1  # 页面编号（从1开始）
                }
            })

    print(f"提取了{len(pages)}个包含内容的页面")  # 打印提取的页面数量
    return pages  # 返回包含文本内容和元数据的页面列表


# 文本分块
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    将文本分割成重叠的块。

    Args:
        text (str): 输入文本
        chunk_size (int): 每个块的大小（以字符为单位）
        overlap (int): 块之间的重叠（以字符为单位）

    Returns:
        List[Dict]: 包含块和元数据的列表
    """
    chunks = []  # 初始化一个空列表以存储块

    # 以(chunk_size - overlap)的步长遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk_text = text[i:i + chunk_size]  # 提取文本块
        if chunk_text:  # 确保不添加空块
            chunks.append({
                "text": chunk_text,  # 添加文本块
                "metadata": {
                    "start_pos": i,  # 原始文本中的块起始位置
                    "end_pos": i + len(chunk_text)  # 原始文本中的块结束位置
                }
            })

    print(f"创建了{len(chunks)}个文本块")  # 打印创建的块数量
    return chunks  # 返回包含块和元数据的列表


# 简单向量存储实现
class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """

    def __init__(self):
        self.vectors = []  # 存储向量嵌入的列表
        self.texts = []  # 存储文本内容的列表
        self.metadata = []  # 存储元数据的列表

    def add_item(self, text, embedding, metadata=None):
        """
        将项目添加到向量存储中。

        Args:
            text (str): 文本内容
            embedding (List[float]): 向量嵌入
            metadata (Dict, optional): 额外的元数据
        """
        self.vectors.append(np.array(embedding))  # 将嵌入作为numpy数组添加
        self.texts.append(text)  # 添加文本内容
        self.metadata.append(metadata or {})  # 添加元数据或空字典（如果为None）

    def similarity_search(self, query_embedding, k=5, filter_func=None):
        """
        找到与查询嵌入最相似的项目。

        Args:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回的结果数量
            filter_func (callable, optional): 过滤结果的函数

        Returns:
            List[Dict]: 最相似的前k个项目
        """
        if not self.vectors:
            return []  # 如果没有向量，返回空列表

        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)

        # 计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 如果不通过过滤器，则跳过
            if filter_func and not filter_func(self.metadata[i]):
                continue

            # 计算余弦相似性
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))  # 添加索引和相似性分数

        # 按相似性降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],  # 添加文本内容
                "metadata": self.metadata[idx],  # 添加元数据
                "similarity": float(score)  # 添加相似性分数
            })

        return results  # 返回前k个结果的列表


# 创建嵌入
def create_embeddings(texts, model="BAAI/bge-en-icl"):
    """
    为给定的文本创建嵌入。

    Args:
        texts (List[str]): 输入文本
        model (str): 嵌入模型名称

    Returns:
        List[List[float]]: 嵌入向量
    """
    # 处理空输入
    if not texts:
        return []

    # 如果需要，按批次处理（OpenAI API限制）
    batch_size = 100
    all_embeddings = []

    # 遍历输入文本的批次
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]  # 获取当前批次的文本

        # 为当前批次创建嵌入
        response = client.embeddings.create(
            model=model,
            input=batch
        )

        # 从响应中提取嵌入
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)  # 将批次嵌入添加到列表中

    return all_embeddings  # 返回所有嵌入


# 文档处理管道
def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG处理文档。

    Args:
        pdf_path (str): PDF文件路径
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠（以字符为单位）

    Returns:
        SimpleVectorStore: 包含文档块的向量存储
    """
    # 从PDF文件中提取文本
    pages = extract_text_from_pdf(pdf_path)

    # 处理每一页并创建块
    all_chunks = []
    for page in pages:
        # 将文本内容（字符串）传递给chunk_text，而不是字典
        page_chunks = chunk_text(page["text"], chunk_size, chunk_overlap)

        # 更新块的元数据以包含页面的元数据
        for chunk in page_chunks:
            chunk["metadata"].update(page["metadata"])

        all_chunks.extend(page_chunks)

    # 为文本块创建嵌入
    print("为块创建嵌入...")
    chunk_texts = [chunk["text"] for chunk in all_chunks]
    chunk_embeddings = create_embeddings(chunk_texts)

    # 创建一个向量存储以保存块及其嵌入
    vector_store = SimpleVectorStore()
    for i, chunk in enumerate(all_chunks):
        vector_store.add_item(
            text=chunk["text"],
            embedding=chunk_embeddings[i],
            metadata=chunk["metadata"]
        )

    print(f"创建了包含{len(all_chunks)}个块的向量存储")
    return vector_store


# 假设性文档生成
def generate_hypothetical_document(query, desired_length=1000):
    """
    生成一个回答查询的假设性文档。

    Args:
        query (str): 用户查询
        desired_length (int): 假设性文档的目标长度

    Returns:
        str: 生成的假设性文档
    """
    # 定义系统提示，指示模型如何生成文档
    system_prompt = f"""你是一位专家文档创建者。
给定一个问题，生成一个详细回答该问题的文档。
该文档应大约{desired_length}个字符长，并提供对该问题的深入、
信息丰富的回答。写作风格將如同该文档来自该主题的权威来源。
包括具体细节、事实和解释。
不要提到这是一个假设性文档——直接写出内容。"""

    # 定义包含查询的用户提示
    user_prompt = f"问题：{query}\n\n生成一个完全回答此问题的文档："

    # 请求OpenAI API生成假设性文档
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 指定使用的模型
        messages=[
            {"role": "system", "content": system_prompt},  # 系统消息以指导助手
            {"role": "user", "content": user_prompt}  # 包含查询的用户消息
        ],
        temperature=0.1  # 设置响应生成的温度
    )

    # 返回生成的文档内容
    return response.choices[0].message.content


# 完整的HyDE RAG实现
def hyde_rag(query, vector_store, k=5, should_generate_response=True):
    """
    使用假设性文档嵌入执行RAG。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 要检索的块数量
        should_generate_response (bool): 是否生成最终响应

    Returns:
        Dict: 包含假设性文档和检索到的块的结果
    """
    print(f"\n=== 使用HyDE处理查询：{query} ===\n")

    # 步骤1：生成一个回答查询的假设性文档
    print("生成假设性文档...")
    hypothetical_doc = generate_hypothetical_document(query)
    print(f"生成了{len(hypothetical_doc)}个字符的假设性文档")

    # 步骤2：为假设性文档创建嵌入
    print("为假设性文档创建嵌入...")
    hypothetical_embedding = create_embeddings([hypothetical_doc])[0]

    # 步骤3：基于假设性文档检索相似的块
    print(f"检索前{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(hypothetical_embedding, k=k)

    # 准备结果字典
    results = {
        "query": query,
        "hypothetical_document": hypothetical_doc,
        "retrieved_chunks": retrieved_chunks
    }

    # 步骤4：如果需要，生成最终响应
    if should_generate_response:
        print("生成最终响应...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response

    return results


# 标准（直接）RAG实现以供比较
def standard_rag(query, vector_store, k=5, should_generate_response=True):
    """
    使用直接查询嵌入执行标准RAG。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        k (int): 要检索的块数量
        should_generate_response (bool): 是否生成最终响应

    Returns:
        Dict: 包含检索到的块的结果
    """
    print(f"\n=== 使用标准RAG处理查询：{query} ===\n")

    # 步骤1：为查询创建嵌入
    print("为查询创建嵌入...")
    query_embedding = create_embeddings([query])[0]

    # 步骤2：基于查询嵌入检索相似的块
    print(f"检索前{k}个最相似的块...")
    retrieved_chunks = vector_store.similarity_search(query_embedding, k=k)

    # 准备结果字典
    results = {
        "query": query,
        "retrieved_chunks": retrieved_chunks
    }

    # 步骤3：如果需要，生成最终响应
    if should_generate_response:
        print("生成最终响应...")
        response = generate_response(query, retrieved_chunks)
        results["response"] = response

    return results


# 响应生成
def generate_response(query, relevant_chunks):
    """
    基于查询和相关块生成最终响应。

    Args:
        query (str): 用户查询
        relevant_chunks (List[Dict]): 检索到的相关块

    Returns:
        str: 生成的响应
    """
    # 将块中的文本连接起来以创建上下文
    context = "\n\n".join([chunk["text"] for chunk in relevant_chunks])

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": "你是一位乐于助人的助手。根据提供的上下文回答问题."},
            {"role": "user", "content": f"上下文：\n{context}\n\n问题：{query}"}
        ],
        temperature=0.5,
        max_tokens=500
    )

    return response.choices[0].message.content


# 评估函数-比较方法
def compare_approaches(query, vector_store, reference_answer=None):
    """
    比较HyDE和标准RAG方法的查询。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        reference_answer (str, optional): 参考答案以供评估

    Returns:
        Dict: 比较结果
    """
    # 运行HyDE RAG
    hyde_result = hyde_rag(query, vector_store)
    hyde_response = hyde_result["response"]

    # 运行标准RAG
    standard_result = standard_rag(query, vector_store)
    standard_response = standard_result["response"]

    # 比较结果
    comparison = compare_responses(query, hyde_response, standard_response, reference_answer)

    return {
        "query": query,
        "hyde_response": hyde_response,
        "hyde_hypothetical_doc": hyde_result["hypothetical_document"],
        "standard_response": standard_response,
        "reference_answer": reference_answer,
        "comparison": comparison
    }


# 比较响应
def compare_responses(query, hyde_response, standard_response, reference=None):
    """
    比较来自HyDE和标准RAG的响应。

    Args:
        query (str): 用户查询
        hyde_response (str): HyDE RAG的响应
        standard_response (str): 标准RAG的响应
        reference (str, optional): 参考答案

    Returns:
        str: 比较分析
    """
    system_prompt = """你是一位信息检索系统的专家评估者。
比较针对同一查询的两个响应，一个使用HyDE（假设性文档嵌入）生成，另一个使用标准RAG和直接查询嵌入生成。

根据以下标准进行评估：
1. 准确性：哪个响应提供了更多事实上的正确信息？
2. 相关性：哪个响应更好地解决了查询？
3. 完整性：哪个响应提供了更全面的主题覆盖？
4. 清晰度：哪个响应组织得更好，更易于理解？

具体说明每种方法的优缺点。"""

    user_prompt = f"""查询：{query}\n
响应来自HyDE RAG：
{hyde_response}\n
响应来自标准RAG：
{standard_response}"""

    if reference:
        user_prompt += f"""\n
参考答案：
{reference}"""

    user_prompt += """\n
请详细比较这两个响应，突出说明哪种方法表现更好以及原因。"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# 运行评估
def run_evaluation(pdf_path, test_queries, reference_answers=None, chunk_size=1000, chunk_overlap=200):
    """
    运行完整的评估，使用多个测试查询。

    Args:
        pdf_path (str): PDF文档路径
        test_queries (List[str]): 测试查询列表
        reference_answers (List[str], optional): 查询的参考答案
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠（以字符为单位）

    Returns:
        Dict: 评估结果
    """
    # 处理文档并创建向量存储
    vector_store = process_document(pdf_path, chunk_size, chunk_overlap)

    results = []

    for i, query in enumerate(test_queries):
        print(f"\n\n===== 评估查询 {i + 1}/{len(test_queries)} =====")
        print(f"查询：{query}")

        # 获取参考答案（如果可用）
        reference = None
        if reference_answers and i < len(reference_answers):
            reference = reference_answers[i]

        # 比较方法
        result = compare_approaches(query, vector_store, reference)
        results.append(result)

        # 生成总体分析
        overall_analysis = generate_overall_analysis(results)

        return {
            "results": results,
            "overall_analysis": overall_analysis
        }


# 生成总体分析
def generate_overall_analysis(results):
    """
    生成评估结果的总体分析。

    Args:
        results (List[Dict]): 单个查询评估的结果列表

    Returns:
        str: 总体分析
    """
    system_prompt = """你是一位信息检索系统的专家评估者。
基于多个测试查询，提供HyDE RAG（使用假设性文档嵌入）与标准RAG（使用直接查询嵌入）的总体分析。

重点：
1. HyDE表现更好的情况及原因
2. 标准RAG表现更好的情况及原因
3. 最适合使用HyDE的查询类型
4. 每种方法的总体优缺点
5. 推荐使用每种方法的场景"""

    # 创建评估摘要
    evaluations_summary = ""
    for i, result in enumerate(results):
        evaluations_summary += f"查询 {i + 1}：{result['query']}\n"
        evaluations_summary += f"比较摘要：{result['comparison'][:200]}...\n\n"

    user_prompt = f"""基于以下针对{len(results)}个查询的评估，比较HyDE与标准RAG，
提供这两种方法的总体分析：
{evaluations_summary}
请根据HyDE与标准RAG在不同查询上的表现，提供一个全面的分析，重点说明哪种方法在什么情况下表现更好以及原因。"""

    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


# 可视化函数
def visualize_results(query, hyde_result, standard_result):
    """
    可视化HyDE和标准RAG方法的结果。

    Args:
        query (str): 用户查询
        hyde_result (Dict): HyDE RAG的结果
        standard_result (Dict): 标准RAG的结果
    """
    # 创建包含3个子图的图形
    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    # 在第一个子图中绘制查询
    axs[0].text(0.5, 0.5, f"查询：\n\n{query}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=12, wrap=True)
    axs[0].axis('off')  # 隐藏查询图的坐标轴

    # 在第二个子图中绘制假设性文档
    hypothetical_doc = hyde_result["hypothetical_document"]
    # 如果假设性文档太长，则缩短
    shortened_doc = hypothetical_doc[:500] + "..." if len(hypothetical_doc) > 500 else hypothetical_doc
    axs[1].text(0.5, 0.5, f"假设性文档：\n\n{shortened_doc}",
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
    axs[1].axis('off')  # 隐藏假设性文档图的坐标轴

    # 在第三个子图中绘制检索到的块的比较
    # 缩短每个块文本以供更好的可视化
    hyde_chunks = [chunk["text"][:100] + "..." for chunk in hyde_result["retrieved_chunks"]]
    std_chunks = [chunk["text"][:100] + "..." for chunk in standard_result["retrieved_chunks"]]

    # 准备比较文本
    comparison_text = "由HyDE检索到的块：\n\n"
    for i, chunk in enumerate(hyde_chunks):
        comparison_text += f"{i + 1}. {chunk}\n\n"

    comparison_text += "\n由标准RAG检索到的块：\n\n"
    for i, chunk in enumerate(std_chunks):
        comparison_text += f"{i + 1}. {chunk}\n\n"

    # 在第三个子图中绘制比较文本
    axs[2].text(0.5, 0.5, comparison_text,
                horizontalalignment='center', verticalalignment='center',
                fontsize=8, wrap=True)
    axs[2].axis('off')  # 隐藏比较图的坐标轴

    # 调整布局以防止重叠
    plt.tight_layout()
    # 显示图形
    plt.show()


# 假设性文档嵌入（HyDE）与标准RAG的评估
# PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 处理文档并创建向量存储
# 这会加载文档，提取文本，分块，并创建嵌入
vector_store = process_document(pdf_path)

# 示例1：与AI相关查询的直接比较
query = "人工智能开发的主要伦理考虑是什么？"

# 运行HyDE RAG方法
# 这会生成一个回答查询的假设性文档，对其进行嵌入，
# 并使用该嵌入检索相关块
hyde_result = hyde_rag(query, vector_store)
print("\n=== HyDE响应 ===")
print(hyde_result["response"])

# 运行标准RAG方法以供比较
# 这会直接对查询进行嵌入并使用其检索相关块
standard_result = standard_rag(query, vector_store)
print("\n=== 标准RAG响应 ===")
print(standard_result["response"])

# 可视化HyDE和标准RAG方法之间的差异
# 显示查询、假设性文档和检索到的块
visualize_results(query, hyde_result, standard_result)

# 示例2：使用多个AI相关查询运行全面评估
test_queries = [
    "神经网络架构如何影响人工智能性能？"
]

# 可选的参考答案以供更好评估
reference_answers = [
    "神经网络架构通过深度（层数）、宽度（每层神经元数量）、连接模式和激活函数等因素对人工智能性能产生重大影响。不同的架构如CNN、RNN和Transformer分别针对图像识别、序列处理和自然语言理解等特定任务进行了优化。"
]

# 运行全面评估以比较HyDE和标准RAG方法
evaluation_results = run_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印哪种方法在查询上表现更好的总体分析
print("\n=== 总体分析 ===")
print(evaluation_results["overall_analysis"])

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


