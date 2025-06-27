
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。


####################################################################################################################################
# 方法9：
# 相关段落提取（RSE）用于增强RAG
# 相关段落提取（RSE）技术，以提高RAG系统中的上下文质量。我们通过识别和重构文本中的连续段落，提供更连贯的上下文供语言模型使用。
# 相关文本块通常在文档中聚集在一起。通过识别这些聚集并保持其连续性，我们为LLM提供了更连贯的上下文。

# 分块提取的文本
# 将文本分成较小的、重叠的块，以提高检索准确性。
def chunk_text(text, chunk_size=800, overlap=0):
    """
    将文本分成非重叠块。
    对于RSE，我们通常希望非重叠块以便正确重构段落。

    Args:
        text (str): 输入文本
        chunk_size (int): 每个块的大小（字符数）
        overlap (int): 块之间的重叠（字符数）

    Returns:
        List[str]: 文本块列表
    """
    chunks = []

    # 简单的基于字符的分块
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk:  # 确保不添加空块
            chunks.append(chunk)

    return chunks

# 使用RSE处理文档
def process_document(pdf_path, chunk_size=800):
    """
    使用RSE处理文档。

    Args:
        pdf_path (str): PDF文档的路径
        chunk_size (int): 每个块的大小（字符数）

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: 块、向量存储和文档信息
    """
    print("从文档中提取文本...")
    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    print("将文本分块为非重叠段落...")
    # 将提取的文本分块为非重叠段落
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    print(f"创建了{len(chunks)}个块")

    print("为块生成嵌入...")
    # 为文本块生成嵌入
    chunk_embeddings = create_embeddings(chunks)

    # 创建SimpleVectorStore的实例
    vector_store = SimpleVectorStore()

    # 添加文档（包括块索引以便后续重构）
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    # 跟踪原始文档结构以便段落重构
    doc_info = {
        "chunks": chunks,
        "source": pdf_path,
    }

    return chunks, vector_store, doc_info

# RSE核心算法：计算块值和找到最佳段落
# 现在我们有了处理文档和为块生成嵌入的必要函数，可以实现RSE的核心算法。


def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    通过结合相关性和位置计算块值。

    Args:
        query (str): 查询文本
        chunks (List[str]): 文档块列表
        vector_store (SimpleVectorStore): 包含块的向量存储
        irrelevant_chunk_penalty (float): 无关块的惩罚

    Returns:
        List[float]: 块值列表
    """
    # 创建查询嵌入
    query_embedding = create_embeddings([query])[0]

    # 获取所有块及其相似性得分
    num_chunks = len(chunks)
    results = vector_store.search(query_embedding, top_k=num_chunks)

    # 创建块索引到相关性得分的映射
    relevance_scores = {result["metadata"]["chunk_index"]: result["score"] for result in results}

    # 计算块值（相关性得分减去惩罚）
    chunk_values = []
    for i in range(num_chunks):
        # 获取相关性得分或默认为0（如果不在结果中）
        score = relevance_scores.get(i, 0.0)
        # 应用惩罚以将无关块的值转换为负值
        value = score - irrelevant_chunk_penalty
        chunk_values.append(value)

    return chunk_values

def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    使用最大和子数组算法的变体查找最佳段落。

    Args:
        chunk_values (List[float]): 每个块的值
        max_segment_length (int): 单个段落的最大长度
        total_max_length (int): 所有段落的最大总长度
        min_segment_value (float): 段落被视为相关的最小值

    Returns:
        List[Tuple[int, int]]: 最佳段落的（开始，结束）索引列表
    """
    print("查找最优连续文本段落...")
    best_segments = []
    segment_scores = []
    total_included_chunks = 0

    # 继续查找段落直到达到限制
    while total_included_chunks < total_max_length:
        best_score = min_segment_value  # 段落的最小阈值
        best_segment = None

        # 尝试每个可能的起始位置
        for start in range(len(chunk_values)):
            # 如果起始位置已经在选定的段落中，则跳过
            if any(start >= s[0] and start < s[1] for s in best_segments):
                continue

            # 尝试每个可能的段落长度
            for length in range(1, min(max_segment_length, len(chunk_values) - start) + 1):
                end = start + length

                # 如果结束位置已经在选定的段落中，则跳过
                if any(end > s[0] and end <= s[1] for s in best_segments):
                    continue

                # 计算段落值为块值的总和
                segment_value = sum(chunk_values[start:end])

                # 如果当前段落更好，则更新最佳段落
                if segment_value > best_score:
                    best_score = segment_value
                    best_segment = (start, end)

        # 如果找到一个好的段落，则添加它
        if best_segment:
            best_segments.append(best_segment)
            segment_scores.append(best_score)
            total_included_chunks += best_segment[1] - best_segment[0]
            print(f"找到段落{best_segment}，得分为{best_score:.4f}")
        else:
            # 没有更多好的段落可找
            break

    # 按起始位置对段落进行排序以便阅读
    best_segments = sorted(best_segments, key=lambda x: x[0])

    return best_segments, segment_scores

# 重构段落用于RAG

def reconstruct_segments(chunks, best_segments):
    """
    根据块索引重构文本段落。

    Args:
        chunks (List[str]): 所有文档块列表
        best_segments (List[Tuple[int, int]]): 段落的（开始，结束）索引列表

    Returns:
        List[str]: 重构的文本段落列表
    """
    reconstructed_segments = []  # 初始化一个空列表存储重构的段落

    for start, end in best_segments:
        # 将段落中的块连接起来形成完整的段落文本
        segment_text = " ".join(chunks[start:end])
        # 将段落文本及其范围追加到reconstructed_segments列表中
        reconstructed_segments.append({
            "text": segment_text,
            "segment_range": (start, end),
        })

    return reconstructed_segments  # 返回重构的文本段落列表

def format_segments_for_context(segments):
    """
    将段落格式化为LLM使用的上下文字符串。

    Args:
        segments (List[Dict]): 段落字典列表

    Returns:
        str: 格式化的上下文文本
    """
    context = []  # 初始化一个空列表存储格式化的上下文

    for i, segment in enumerate(segments):
        # 为每个段落创建一个包含索引和块范围的标题
        segment_header = f"段落{i+1}（块{segment['segment_range'][0]}-{segment['segment_range'][1]-1}）:"
        context.append(segment_header)  # 将段落标题添加到上下文列表中
        context.append(segment['text'])  # 将段落文本添加到上下文列表中
        context.append("-" * 80)  # 添加分隔线以便阅读

    # 将上下文列表中的所有元素用双换行符连接并返回结果
    return "\n\n".join(context)

# 完整的RSE管道函数

def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    带有相关段落提取的完整RAG管道。

    Args:
        pdf_path (str): 文档的路径
        query (str): 用户查询
        chunk_size (int): 块的大小
        irrelevant_chunk_penalty (float): 无关块的惩罚

    Returns:
        Dict: 包含查询、段落和响应的结果
    """
    print("\n=== 开始使用相关段落提取的RAG ===")
    print(f"查询: {query}")

    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    # 计算相关性得分和块值基于查询
    print("\n计算相关性得分和块值...")
    chunk_values = calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty)

    # 根据块值查找最佳段落
    best_segments, scores = find_best_segments(
        chunk_values,
        max_segment_length=20,
        total_max_length=30,
        min_segment_value=0.2
    )

    # 从最佳块重构文本段落
    print("\n从块重构文本段落...")
    segments = reconstruct_segments(chunks, best_segments)

    # 将段落格式化为语言模型的上下文字符串
    context = format_segments_for_context(segments)

    # 使用上下文生成语言模型的响应
    response = generate_response(query, context)

    # 将结果编入字典
    result = {
        "query": query,
        "segments": segments,
        "response": response
    }

    print("\n=== 最终响应 ===")
    print(response)

    return result

# 与标准检索比较
# 让我们实现一个标准检索方法来与RSE比较：
def standard_top_k_retrieval(pdf_path, query, k=10, chunk_size=800):
    """
    带有标准top-k检索的RAG。

    Args:
        pdf_path (str): 文档的路径
        query (str): 用户查询
        k (int): 要检索的块数量
        chunk_size (int): 块的大小

    Returns:
        Dict: 包含查询、块和响应的结果
    """
    print("\n=== 开始标准top-k检索 ===")
    print(f"查询: {query}")

    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    # 创建查询嵌入
    print("创建查询嵌入并检索块...")
    query_embedding = create_embeddings([query])[0]

    # 根据查询嵌入检索最相关的top-k块
    results = vector_store.search(query_embedding, top_k=k)
    retrieved_chunks = [result["document"] for result in results]

    # 将检索到的块格式化为上下文字符串
    context = "\n\n".join([
        f"块{i+1}:\n{chunk}"
        for i, chunk in enumerate(retrieved_chunks)
    ])

    # 使用上下文生成语言模型的响应
    response = generate_response(query, context)

    # 将结果编入字典
    result = {
        "query": query,
        "chunks": retrieved_chunks,
        "response": response
    }

    print("\n=== 最终响应 ===")
    print(response)

    return result


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


