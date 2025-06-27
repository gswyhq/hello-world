
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法6：
# 通过文档增强（问题生成）来提升检索效果。通过为每个文本块生成相关问题，从而使得语言模型能够生成更好的响应。
# 1. **数据摄入**：从PDF文件中提取文本。
# 2. **分块**：将文本分割成可管理的块。
# 3. **问题生成**：为每个块生成相关问题。
# 4. **嵌入创建**：为文本块和生成的问题创建嵌入。
# 5. **向量存储构建**：使用NumPy构建一个简单的向量存储。
# 6. **语义搜索**：检索与用户查询相关的文本块和问题。
# 7. **响应生成**：基于检索到的内容生成答案。
# 8. **评估**：评估生成响应的质量。

# 为文本块生成问题。我们为每个文本块生成可以由其回答的问题。

def generate_questions(text_chunk, num_questions=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    生成可以从给定文本块中回答的相关问题。

    Args:
    text_chunk (str): 要生成问题的文本块。
    num_questions (int): 要生成的问题数量。
    model (str): 用于问题生成的模型。

    Returns:
    List[str]: 生成的问题列表。
    """
    # 定义系统提示以指导AI的行为
    system_prompt = "你是一位擅长从文本中生成相关问题的专家。创建简洁的问题，这些问题只能使用提供的文本回答。关注关键信息和概念。"

    # 定义用户提示，包含文本块和要生成的问题数量
    user_prompt = f"""基于以下文本，生成{num_questions}个不同的问题，这些问题只能使用该文本回答：

{text_chunk}

请以编号列表的形式仅返回问题，不要添加其他文本。
    """

    # 使用OpenAI API生成问题
    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 从响应中提取并清理问题
    questions_text = response.choices[0].message.content.strip()
    questions = []

    # 使用正则表达式模式匹配提取问题
    for line in questions_text.split('\n'):
        # 移除编号并清理空白字符
        cleaned_line = re.sub(r'^\d+\. ', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)

    return questions


# 构建简单向量存储
# 使用NumPy实现一个简单的向量存储。

class SimpleVectorStore:
    """
    使用NumPy实现的简单向量存储。
    """

    def __init__(self):
        """
        初始化向量存储。
        """
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        将一个项目添加到向量存储中。

        Args:
        text (str): 原始文本。
        embedding (List[float]): 嵌入向量。
        metadata (dict, optional): 额外的元数据。
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        """
        找到与查询嵌入最相似的项目。

        Args:
        query_embedding (List[float]): 查询嵌入向量。
        k (int): 要返回的结果数量。

        Returns:
        List[Dict]: 最相似的前k个项目，包含文本和元数据。
        """
        if not self.vectors:
            return []

        # 将查询嵌入转换为numpy数组
        query_vector = np.array(query_embedding)

        # 使用余弦相似度计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vector))
            similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results


# 处理文档并进行问题增强
# 将所有内容整合在一起，处理文档，生成问题，并构建增强的向量存储。

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    处理文档并进行问题增强。

    Args:
    pdf_path (str): PDF文件的路径。
    chunk_size (int): 每个文本块的字符数。
    chunk_overlap (int): 块之间的重叠字符数。
    questions_per_chunk (int): 每个块要生成的问题数量。

    Returns:
    Tuple[List[str], SimpleVectorStore]: 文本块和向量存储。
    """
    print("从PDF中提取文本...")
    extracted_text = extract_text_from_pdf(pdf_path)

    print("分块文本...")
    text_chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"创建了{len(text_chunks)}个文本块")

    vector_store = SimpleVectorStore()

    print("处理块并生成问题...")
    for i, chunk in enumerate(tqdm(text_chunks, desc="处理块")):
        # 为块本身创建嵌入
        chunk_embedding_response = create_embeddings(chunk)
        chunk_embedding = chunk_embedding_response.data[0].embedding

        # 将块添加到向量存储中
        vector_store.add_item(
            text=chunk,
            embedding=chunk_embedding,
            metadata={"type": "chunk", "index": i}
        )

        # 为该块生成问题
        questions = generate_questions(chunk, num_questions=questions_per_chunk)

        # 为每个问题创建嵌入并添加到向量存储中
        for j, question in enumerate(questions):
            question_embedding_response = create_embeddings(question)
            question_embedding = question_embedding_response.data[0].embedding

            # 将问题添加到向量存储中
            vector_store.add_item(
                text=question,
                embedding=question_embedding,
                metadata={"type": "question", "chunk_index": i, "original_chunk": chunk}
            )

    return text_chunks, vector_store


# 处理文档（提取文本、创建块、生成问题、构建向量存储）
text_chunks, vector_store = process_document(
    pdf_path,
    chunk_size=1000,
    chunk_overlap=200,
    questions_per_chunk=3
)

print(f"向量存储包含{len(vector_store.texts)}个文档")

# 语义搜索函数，但适应了增强的向量存储。

def semantic_search(query, vector_store, k=5):
    """
    使用查询和向量存储执行语义搜索。

    Args:
    query (str): 搜索查询。
    vector_store (SimpleVectorStore): 要搜索的向量存储。
    k (int): 要返回的结果数量。

    Returns:
    List[Dict]: 最相关的前k个项目。
    """
    # 为查询创建嵌入
    query_embedding_response = create_embeddings(query)
    query_embedding = query_embedding_response.data[0].embedding

    # 搜索向量存储
    results = vector_store.similarity_search(query_embedding, k=k)

    return results


# 执行语义搜索以找到相关内容
search_results = semantic_search(query, vector_store, k=5)

print("查询:", query)
print("\n搜索结果:")

# 按类型组织结果
chunk_results = []
question_results = []

for result in search_results:
    if result["metadata"]["type"] == "chunk":
        chunk_results.append(result)
    else:
        question_results.append(result)

# 打印块结果
print("\n相关文档块:")

for i, result in enumerate(chunk_results):
    print(f"上下文 {i + 1} (相似度: {result['similarity']:.4f}):")
    print(result["text"][:300] + "...")
    print("=====================================")

# 然后打印匹配的问题
print("\n匹配的问题:")

for i, result in enumerate(question_results):
    print(f"问题 {i + 1} (相似度: {result['similarity']:.4f}):")
    print(result["text"])
    chunk_idx = result["metadata"]["chunk_index"]
    print(f"来自块 {chunk_idx}")
    print("=====================================")

# 通过结合相关块和问题来准备上下文。

def prepare_context(search_results):
    """
    从搜索结果中准备用于响应生成的统一上下文。

    Args:
    search_results (List[Dict]): 语义搜索的结果。

    Returns:
    str: 结合后的上下文字符串。
    """
    # 提取结果中引用的唯一块
    chunk_indices = set()
    context_chunks = []

    # 首先添加直接的块匹配
    for result in search_results:
        if result["metadata"]["type"] == "chunk":
            chunk_indices.add(result["metadata"]["index"])
            context_chunks.append(f"块 {result['metadata']['index']}:\n{result['text']}")

    # 然后添加问题引用的块
    for result in search_results:
        if result["metadata"]["type"] == "question":
            chunk_idx = result["metadata"]["chunk_index"]
            if chunk_idx not in chunk_indices:
                chunk_indices.add(chunk_idx)
                context_chunks.append(
                    f"块 {chunk_idx}（由问题'{result['text']}'引用）:\n{result['metadata']['original_chunk']}")

    # 结合所有上下文块
    full_context = "\n\n".join(context_chunks)
    return full_context

# 基于检索到的块生成响应

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    基于查询和上下文生成响应。

    Args:
    query (str): 用户的问题。
    context (str): 从向量存储中检索到的上下文信息。
    model (str): 用于响应生成的模型。

    Returns:
    str: 生成的响应。
    """
    system_prompt = "你是一位严格基于给定上下文回答的AI助手。如果无法直接从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    user_prompt = f"""上下文：

{context}

问题：{query}

请仅基于上述提供的上下文回答问题。请简洁且准确。
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

# 生成并显示响应

# 从搜索结果中准备上下文
context = prepare_context(search_results)

# 生成响应
response_text = generate_response(query, context)

print("\n查询:", query)
print("\n响应:")
print(response_text)

# 评估AI响应
# 将AI响应与预期答案进行比较并分配一个分数。

def evaluate_response(query, response, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将AI响应与参考答案进行比较并进行评估。

    Args:
    query (str): 用户的问题。
    response (str): AI生成的响应。
    reference_answer (str): 参考/理想答案。
    model (str): 用于评估的模型。

    Returns:
    str: 评估反馈。
    """
    # 定义评估系统的系统提示
    evaluate_system_prompt = """你是一位智能评估系统，负责评估AI响应。

    将AI助手的响应与真实/参考答案进行比较，并根据以下标准进行评估：
    1. 事实正确性 - 响应是否包含准确的信息？
    2. 完整性 - 是否涵盖了参考答案中的所有重要方面？
    3. 相关性 - 是否直接回答了问题？

    从0到1分配一个分数：
    - 1.0：内容和含义完全匹配
    - 0.8：非常好，只有 minor omissions/differences
    - 0.6：良好，涵盖主要点但缺少一些细节
    - 0.4：部分答案，存在重大遗漏
    - 0.2：少量相关信息
    - 0.0：不正确或不相关

    提供你的评分理由。
    """

    # 创建评估提示
    evaluation_prompt = f"""用户查询：{query}

AI响应：
{response}

参考答案：
{reference_answer}

请评估AI响应与参考答案的对比。
    """

    # 生成评估
    eval_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": evaluate_system_prompt},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    return eval_response.choices[0].message.content

# 运行评估

# 从验证数据中获取参考答案
reference_answer = data[0]['ideal_answer']

# 评估响应
evaluation = evaluate_response(query, response_text, reference_answer)

print("\n评估:")
print(evaluation)



####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


