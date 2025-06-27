
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法4：
# 上下文增强检索，检索的信息包含相邻片段，从而提高连贯性：
# 提取文本后，我们将文本分割为较小的、带重叠的片段，以提高检索准确性。
# 从PDF文件中提取文本
extracted_text = extract_text_from_pdf(pdf_path)

# 将提取的文本分割为1000个字符的片段，重叠200个字符
text_chunks = chunk_text(extracted_text, 1000, 200)

def create_embeddings(text, model="BAAI/bge-en-icl", emb=False):
    """
    使用指定的OpenAI模型为给定文本创建嵌入。

    Args:
        text (str): 需要创建嵌入的输入文本。
        model (str): 用于创建嵌入的模型。默认为"BAAI/bge-en-icl"。

    Returns:
        dict: 包含嵌入的OpenAI API响应。
    """
    # 使用指定模型为输入文本创建嵌入
    response = client.embeddings.create(
        model=model,
        input=text
    )
    if emb:
        return response.data[0].embedding
    return response  # 返回包含嵌入的响应

# 上下文感知语义搜索：
def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    检索最相关的片段及其相邻片段。

    Args:
        query (str): 搜索查询。
        text_chunks (List[str]): 文本片段列表。
        embeddings (List[dict]): 文本片段的嵌入列表。
        k (int): 要检索的相关片段数量。
        context_size (int): 要包含的相邻片段数量。

    Returns:
        List[str]: 包含上下文信息的相关文本片段。
    """
    # 将查询转换为嵌入向量
    query_embedding = create_embeddings(query).data[0].embedding
    similarity_scores = []

    # 计算查询与每个文本片段嵌入之间的相似性分数
    for i, chunk_embedding in enumerate(embeddings):
        # 计算查询嵌入与当前片段嵌入之间的余弦相似性
        similarity_score = cosine_similarity(np.array(query_embedding), np.array(chunk_embedding.embedding))
        # 将索引和相似性分数存储为元组
        similarity_scores.append((i, similarity_score))

    # 按相似性分数降序排序片段（最高相似性优先）
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关片段的索引
    top_index = similarity_scores[0][0]

    # 定义上下文包含的范围
    # 确保不超出文本片段的范围
    start = max(0, top_index - context_size)
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回最相关片段及其上下文片段
    return [text_chunks[i] for i in range(start, end)]

# 检索最相关的片段及其相邻片段以获取上下文
top_chunks = context_enriched_search(query, text_chunks, response.data, k=1, context_size=1)

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


