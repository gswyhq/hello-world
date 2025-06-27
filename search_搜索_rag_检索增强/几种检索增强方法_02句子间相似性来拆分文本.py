
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法2：
# 文本分块是检索增强生成 （RAG） 中的一个重要步骤，其中大型文本正文被划分为有意义的段以提高检索准确性。与固定长度分块不同，语义分块根据句子之间的内容相似性来拆分文本。
# 将文本拆分为句子：
sentences = extracted_text.split("。")
# Generate embeddings for each sentence
embeddings = [get_embedding(sentence) for sentence in sentences]

# 计算连续句子的余弦相似度：
def cosine_similarity(vec1, vec2):
    """
    计算两个向量之间的余弦相似度。

    参数：
    vec1 （np.ndarray）：第一个向量。
    vec2 （np.ndarray）：第二个向量。

    返回：
    float：余弦相似度。
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 计算连续句子之间的相似度
similarities = [cosine_similarity(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

# 实现语义分块
# 我们实施了三种不同的方法来查找断点。
def compute_breakpoints(similarities, method="percentile", threshold=90):
    """
    根据相似度下降计算分块断点

    Args:
    similarities (List[float]): 句子之间相似度列表.
    method (str): 'percentile', 'standard_deviation', or 'interquartile'.
    threshold (float): 阈值 value (percentile: 百分位数, std：标准差).

    Returns:
    List[int]: 应该发生分块的索引.
    """
    # 根据所选择的方法，确定阈值
    if method == "percentile":
        # 计算相似度的，x分位数
        threshold_value = np.percentile(similarities, threshold)
    elif method == "standard_deviation":
        # 计算相似度的均值和标准差
        mean = np.mean(similarities)
        std_dev = np.std(similarities)
        # 涉及的阈值为均值减去 x倍标准差
        threshold_value = mean - (threshold * std_dev)
    elif method == "interquartile":
        # 计算Q1 和 Q3 分位数
        q1, q3 = np.percentile(similarities, [25, 75])
        #使用IQR规则，为异常值设置阈值
        threshold_value = q1 - 1.5 * (q3 - q1)
    else:
        raise ValueError("Invalid method. Choose 'percentile', 'standard_deviation', or 'interquartile'.")

    # 找出相似度低于阈值的索引
    return [i for i, sim in enumerate(similarities) if sim < threshold_value]

# 使用90%分位数作为阈值计算断点
breakpoints = compute_breakpoints(similarities, method="percentile", threshold=90)

# 根据计算出的断点来分割文本。
def split_into_chunks(sentences, breakpoints):
    """
    将句子拆分为语义块；

    Args:
    sentences (List[str]): 句子列表.
    breakpoints (List[int]): 应该发生分块的索引

    Returns:
    List[str]: 文本块列表
    """
    chunks = []
    start = 0

    # 遍历每个断点以创建分块
    for bp in breakpoints:
        # 将开始到当前断点的句子添加到分块；
        chunks.append(". ".join(sentences[start:bp + 1]) + ".")
        start = bp + 1  # 将开始索引更新到当前断点的下一句；

    # 将剩余的句子作为最后一个分块添加；
    chunks.append(". ".join(sentences[start:]))
    return chunks  # Return the list of chunks

# 使用分块函数创建分块；
text_chunks = split_into_chunks(sentences, breakpoints)

# 为每个分块创建向量，以便后面检索：
chunk_embeddings = [get_embedding(chunk) for chunk in text_chunks]

def semantic_search(query, text_chunks, chunk_embeddings, k=5):
    """
    查找与query最相关的k个文本块

    Args:
    query (str): 待查询的句子.
    text_chunks (List[str]): 文本块列表.
    chunk_embeddings (List[np.ndarray]): 文本块对应的向量列表.
    k (int): 要返回的结果数.

    Returns:
    List[str]: 返回 Top-k个文本块.
    """
    # 为查询的句子生成向量；
    query_embedding = get_embedding(query)

    # 计算查询向量与每个文本块的余弦相似度
    similarities = [cosine_similarity(query_embedding, emb) for emb in chunk_embeddings]

    # 获取top-k分块索引
    top_indices = np.argsort(similarities)[-k:][::-1]

    # 返回top-k文本块；
    return [text_chunks[i] for i in top_indices]

# 定义系统提示词
system_prompt = "您是一个AI助手，严格根据给定的上下文进行回答。如果无法直接从提供的上下文中得出答案，请回答：‘我没有足够的信息来回答这个问题’"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据系统提示词及用户问题，生成答案；

    Args:
    system_prompt (str): 系统提示词
    user_message (str): 用户问题的提示词
    model (str): 用于生成相应的模型

    Returns:
    dict: 来自模型的答案；
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 根据top-k分开文本创建用户问题的提示词；
user_prompt = "\n".join([f"Context {i + 1}:\n{chunk}\n=====================================\n" for i, chunk in enumerate(top_chunks)])
user_prompt = f"{user_prompt}\nQuestion: {query}"

# 生成模型的答案；
ai_response = generate_response(system_prompt, user_prompt)

# 评估 AI 响应
# 我们将 AI 响应与预期答案进行比较并分配分数。
# 定义评估系统的提示词
evaluate_system_prompt = "您是一个智能评估系统，负责评估 AI 助手的响应。如果 AI 助手的响应非常接近真实响应，则分配分数 1。如果响应相对于真实响应不正确或不满意，则分配分数 0。如果响应与真实响应部分一致，则分配 0.5 分。"

# 结合用户的查询，模型的答案，真实答案生成评估提示词；
evaluation_prompt = f"User Query: {query}\nAI Response:\n{ai_response.choices[0].message.content}\nTrue Response: {data[0]['ideal_answer']}\n{evaluate_system_prompt}"

# 使用评估系统提示词和评估提示，生成评估分数
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# 打印评估结果；
print(evaluation_response.choices[0].message.content)

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


