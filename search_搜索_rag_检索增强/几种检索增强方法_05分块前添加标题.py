
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法5：
# 通过在分块前添加高级上下文（如文档标题或章节标题）来增强RAG，从而提高检索质量和防止出现脱离上下文的响应。
# 为了提高检索效果，我们使用LLM模型为每个块生成描述性标题。


def generate_chunk_header(chunk, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    使用LLM模型为给定的文本块生成标题/头。

    Args:
        chunk (str): 需要总结为标题的文本块。
        model (str): 用于生成标题的模型，默认为"meta-llama/Llama-3.2-3B-Instruct"。

    Returns:
        str: 生成的标题/头。
    """
    # 定义系统提示以指导AI行为
    system_prompt = "为给定的文本生成一个简洁且信息丰富的标题。"

    # 使用AI模型根据系统提示和文本块生成响应
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    # 返回生成的标题/头，去除首尾空格
    return response.choices[0].message.content.strip()

def chunk_text_with_headers(text, n, overlap):
    """
    将文本分割为较小的块，并为每个块生成标题。

    Args:
        text (str): 需要分块的完整文本。
        n (int): 每个块的大小（字符数）。
        overlap (int): 块之间的重叠字符数。

    Returns:
        List[dict]: 包含'header'和'text'键的字典列表。
    """
    chunks = []  # 初始化空列表存储块

    # 以指定的块大小和重叠量遍历文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]  # 提取文本块
        header = generate_chunk_header(chunk)  # 为块生成标题
        chunks.append({"header": header, "text": chunk})  # 将标题和块追加到列表中

    return chunks  # 返回带标题的块列表

# 将提取的文本分割为块（块大小为1000字符，重叠200字符）
text_chunks = chunk_text_with_headers(extracted_text, 1000, 200)

# 打印一个样本块及其生成的标题
print("样本块:")
print("标题:", text_chunks[0]['header'])
print("内容:", text_chunks[0]['text'])

# 为每个块生成嵌入
embeddings = []  # 初始化空列表存储嵌入

# 使用进度条遍历每个文本块
for chunk in tqdm(text_chunks, desc="生成嵌入"):
    # 为块的文本生成嵌入
    text_embedding = create_embeddings(chunk["text"], emb=True)
    # 为块的标题生成嵌入
    header_embedding = create_embeddings(chunk["header"], emb=True)
    # 将标题、文本及其嵌入追加到列表中
    embeddings.append({
        "header": chunk["header"],
        "text": chunk["text"],
        "embedding": text_embedding,
        "header_embedding": header_embedding
    })

# 语义搜索：
def semantic_search(query, chunks, k=5):
    """
    根据查询查找最相关的块。

    Args:
        query (str): 用户查询。
        chunks (List[dict]): 包含嵌入的文本块列表。
        k (int): 返回的最相关块数量。

    Returns:
        List[dict]: 最相关的块列表。
    """
    # 为查询生成嵌入
    query_embedding = create_embeddings(query, emb=True)

    similarities = []  # 初始化空列表存储相似度分数

    # 遍历每个块以计算相似度分数
    for chunk in chunks:
        # 计算查询嵌入与块文本嵌入之间的余弦相似度
        sim_text = cosine_similarity(np.array(query_embedding), np.array(chunk["embedding"]))
        # 计算查询嵌入与块标题嵌入之间的余弦相似度
        sim_header = cosine_similarity(np.array(query_embedding), np.array(chunk["header_embedding"]))
        # 计算平均相似度分数
        avg_similarity = (sim_text + sim_header) / 2
        # 将块及其平均相似度分数追加到列表中
        similarities.append((chunk, avg_similarity))

    # 按相似度分数降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    # 返回最相关的k个块
    return [x[0] for x in similarities[:k]]

# 检索最相关的文本块
top_chunks = semantic_search(query, embeddings, k=2)

# 定义AI助手的系统提示
system_prompt = "你是一个严格基于给定上下文回答的AI助手。如果无法从提供的上下文中直接推导出答案，请回答：'我没有足够的信息来回答这个问题。'"

def generate_response(system_prompt, user_message, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据系统提示和用户消息生成AI响应。

    Args:
        system_prompt (str): 指导AI行为的系统提示。
        user_message (str): 用户的消息或查询。
        model (str): 用于生成响应的模型，默认为"meta-llama/Llama-3.2-3B-Instruct"。

    Returns:
        dict: AI模型的响应。
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

# 根据最相关的块创建用户提示
user_prompt = "\n".join([f"标题: {chunk['header']}\n内容:\n{chunk['text']}" for chunk in top_chunks])
user_prompt = f"{user_prompt}\n问题: {query}"

# 生成AI响应
ai_response = generate_response(system_prompt, user_prompt)

# 比较AI响应与预期答案，并分配一个评分。
# 定义评估系统提示
evaluate_system_prompt = """你是一个智能评估系统。
根据提供的上下文评估AI助手的响应。
- 如果响应非常接近正确答案，得分为1。
- 如果响应部分正确，得分为0.5。
- 如果响应不正确，得分为0。
只返回分数（0、0.5或1）。"""

# 从验证数据中提取真实答案
true_answer = data[0]['ideal_answer']

# 构建评估提示
evaluation_prompt = f"""
用户查询: {query}
AI响应: {ai_response}
正确答案: {true_answer}
{evaluate_system_prompt}
"""

# 生成评估分数
evaluation_response = generate_response(evaluate_system_prompt, evaluation_prompt)

# 打印评估分数
print("评估分数:", evaluation_response.choices[0].message.content)


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


