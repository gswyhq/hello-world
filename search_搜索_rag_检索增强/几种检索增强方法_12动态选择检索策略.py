
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法12：
# 根据查询类型动态选择最合适的检索策略。这种方法显著提升了RAG系统在各种问题类型下提供准确相关回答的能力。
# 不同的问题需要不同的检索策略。我们的系统：
# 1. 分类查询类型（事实型、分析型、观点型或上下文型）
# 2. 选择合适的检索策略
# 3. 执行专门的检索技术
# 4. 生成定制化响应

# 查询分类

def classify_query(query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将查询分类为四个类别之一：事实型、分析型、观点型或上下文型。

    参数：
        query (str): 用户查询
        model (str): LLM模型

    返回：
        str: 查询类别
    """
    # 定义系统提示以指导AI的分类
    system_prompt = """你是一位擅长分类问题的专家。
        将给定查询分类到以下四个类别之一：
        - 事实型：寻求具体、可验证信息的查询。
        - 分析型：需要全面分析或解释的查询。
        - 观点型：涉及主观事项或寻求多种观点的查询。
        - 上下文型：依赖用户特定上下文的查询。

        仅返回类别名称，不带任何解释或额外文本。
    """

    # 创建包含要分类查询的用户提示
    user_prompt = f"分类此查询：{query}"

    # 从AI模型生成分类响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取并去除空白的类别
    category = response.choices[0].message.content.strip()

    # 定义有效类别列表
    valid_categories = ["事实型", "分析型", "观点型", "上下文型"]

    # 确保返回的类别有效
    for valid in valid_categories:
        if valid in category:
            return valid

    # 如果分类失败，默认返回“事实型”
    return "事实型"

# 实现专门的检索策略
# 1. 事实型策略 - 精确性
def factual_retrieval_strategy(query, vector_store, k=4):
    """
    用于事实型查询的检索策略，侧重于精确性。

    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要返回的文档数

    返回：
        List[Dict]: 检索到的文档
    """
    print(f"执行事实型检索策略：'{query}'")

    # 使用LLM增强查询以提高精确性
    system_prompt = """你是一位擅长增强搜索查询的专家。
        你的任务是重新表述给定的事实型查询，使其更精确和具体，以提高信息检索效果。专注于关键实体及其关系。

        仅返回增强后的查询，不带任何解释。
    """

    user_prompt = f"增强此事实型查询：{query}"

    # 使用LLM生成增强后的查询
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取并打印增强后的查询
    enhanced_query = response.choices[0].message.content.strip()
    print(f"增强后的查询：{enhanced_query}")

    # 为增强后的查询创建嵌入
    query_embedding = create_embeddings(enhanced_query)

    # 执行初始相似性搜索以检索文档
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)

    # 初始化一个列表以存储排名结果
    ranked_results = []

    # 使用LLM对文档进行评分和排名
    for doc in initial_results:
        relevance_score = score_document_relevance(enhanced_query, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "relevance_score": relevance_score
        })

    # 按相关性分数降序排序
    ranked_results.sort(key=lambda x: x["relevance_score"], reverse=True)

    # 返回前k个结果
    return ranked_results[:k]
# 2. 分析型策略 - 综合覆盖

def analytical_retrieval_strategy(query, vector_store, k=4):
    """
    用于分析型查询的检索策略，侧重于综合覆盖。

    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要返回的文档数

    返回：
        List[Dict]: 检索到的文档
    """
    print(f"执行分析型检索策略：'{query}'")

    # 定义系统提示以指导AI生成子查询
    system_prompt = """你是一位擅长分解复杂问题的专家。
    为给定的分析型查询生成子查询，这些子查询应探索主题的不同方面，以帮助检索全面信息。

    返回一个包含三个子查询的列表，每个子查询占一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"为这个分析型查询生成子查询：{query}"

    # 使用LLM生成子查询
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    # 从响应中提取并清理子查询
    sub_queries = response.choices[0].message.content.strip().split('\n')
    sub_queries = [q.strip() for q in sub_queries if q.strip()]
    print(f"生成的子查询：{sub_queries}")

    # 为每个子查询检索文档
    all_results = []
    for sub_query in sub_queries:
        # 为子查询创建嵌入
        sub_query_embedding = create_embeddings(sub_query)
        # 执行相似性搜索
        results = vector_store.similarity_search(sub_query_embedding, k=2)
        all_results.extend(results)

    # 确保多样性，从不同子查询结果中选择
    # 去除重复内容（相同文本内容）
    unique_texts = set()
    diverse_results = []

    for result in all_results:
        if result["text"] not in unique_texts:
            unique_texts.add(result["text"])
            diverse_results.append(result)

    # 如果需要更多结果以达到k个，直接从初始结果中添加
    if len(diverse_results) < k:
        # 主查询的直接检索
        main_query_embedding = create_embeddings(query)
        main_results = vector_store.similarity_search(main_query_embedding, k=k)

        for result in main_results:
            if result["text"] not in unique_texts and len(diverse_results) < k:
                unique_texts.add(result["text"])
                diverse_results.append(result)

    # 返回前k个多样化结果
    return diverse_results[:k]

# 3. 观点型策略 - 多样化视角
def opinion_retrieval_strategy(query, vector_store, k=4):
    """
    用于观点型查询的检索策略，侧重于多样化视角。

    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要返回的文档数

    返回：
        List[Dict]: 检索到的文档
    """
    print(f"执行观点型检索策略：'{query}'")

    # 定义系统提示以指导AI识别不同视角
    system_prompt = """你是一位擅长识别主题不同观点的专家。
        对于给定的意见或观点查询，识别人们可能对此主题的不同看法。

        返回一个包含三个不同观点角度的列表，每个观点占一行。
    """

    # 创建包含主查询的用户提示
    user_prompt = f"识别以下主题的不同观点：{query}"

    # 使用LLM生成不同观点
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3
    )

    # 从响应中提取并清理观点
    viewpoints = response.choices[0].message.content.strip().split('\n')
    viewpoints = [v.strip() for v in viewpoints if v.strip()]
    print(f"识别的观点：{viewpoints}")

    # 为每个观点检索文档
    all_results = []
    for viewpoint in viewpoints:
        # 将主查询与观点结合
        combined_query = f"{query} {viewpoint}"
        # 为组合查询创建嵌入
        viewpoint_embedding = create_embeddings(combined_query)
        # 执行相似性搜索
        results = vector_store.similarity_search(viewpoint_embedding, k=2)

        # 在结果中标记其代表的观点
        for result in results:
            result["viewpoint"] = viewpoint

        # 将结果添加到所有结果列表中
        all_results.extend(results)

    # 选择多样化观点
    # 确保尽可能从每个观点中至少选择一个文档
    selected_results = []
    for viewpoint in viewpoints:
        # 过滤属于该观点的文档
        viewpoint_docs = [r for r in all_results if r.get("viewpoint") == viewpoint]
        if viewpoint_docs:
            selected_results.append(viewpoint_docs[0])

    # 填充剩余插槽，使用相似度最高的文档
    remaining_slots = k - len(selected_results)
    if remaining_slots > 0:
        # 按相似度排序剩余文档
        remaining_docs = [r for r in all_results if r not in selected_results]
        remaining_docs.sort(key=lambda x: x["similarity"], reverse=True)
        selected_results.extend(remaining_docs[:remaining_slots])

    # 返回前k个结果
    return selected_results[:k]

# 4. 上下文型策略 - 用户上下文整合
def contextual_retrieval_strategy(query, vector_store, k=4, user_context=None):
    """
    用于上下文型查询的检索策略，侧重于用户上下文整合。

    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要返回的文档数
        user_context (str): 额外的用户上下文

    返回：
        List[Dict]: 检索到的文档
    """
    print(f"执行上下文型检索策略：'{query}'")

    # 如果没有提供用户上下文，则尝试从查询中推断
    if not user_context:
        system_prompt = """你是一位擅长理解问题中隐含上下文的专家。
对于给定的查询，推断可能相关但未明确说明的上下文信息。专注于背景信息，这些背景信息有助于回答查询。

返回隐含上下文的简要描述。"""
        user_prompt = f"推断此查询中的隐含上下文：{query}"

        # 使用LLM推断上下文
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-3B-Instruct",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )

        # 从响应中提取并打印推断的上下文
        user_context = response.choices[0].message.content.strip()
        print(f"推断的上下文：{user_context}")

    # 将查询与上下文结合以生成更具体的查询
    system_prompt = """你是一位擅长结合上下文重新表述查询的专家。
    给定一个查询和一些上下文信息，生成一个更具体的查询，以结合上下文获取更相关的信息。

    仅返回重新表述后的查询，不带任何解释。"""
    user_prompt = f"""查询：{query}
上下文：{user_context}

重新表述查询以结合此上下文："""

    # 使用LLM生成结合上下文的查询
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取并打印结合上下文的查询
    contextualized_query = response.choices[0].message.content.strip()
    print(f"结合上下文的查询：{contextualized_query}")

    # 为结合上下文的查询创建嵌入
    query_embedding = create_embeddings(contextualized_query)
    initial_results = vector_store.similarity_search(query_embedding, k=k*2)

    # 根据相关性和用户上下文对文档进行排名
    ranked_results = []

    for doc in initial_results:
        # 计算文档与查询和上下文的相关性分数
        context_relevance = score_document_context_relevance(query, user_context, doc["text"])
        ranked_results.append({
            "text": doc["text"],
            "metadata": doc["metadata"],
            "similarity": doc["similarity"],
            "context_relevance": context_relevance
        })

    # 按上下文相关性降序排序并返回前k个结果
    ranked_results.sort(key=lambda x: x["context_relevance"], reverse=True)
    return ranked_results[:k]

# 核心自适应检索器

def adaptive_retrieval(query, vector_store, k=4, user_context=None):
    """
    通过选择并执行适当的策略执行自适应检索。

    参数：
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 要检索的文档数
        user_context (str): 可选用户上下文

    返回：
        List[Dict]: 检索到的文档
    """
    # 分类查询以确定其类型
    query_type = classify_query(query)
    print(f"查询分类为：{query_type}")

    # 根据查询类型选择并执行适当的检索策略
    if query_type == "事实型":
        # 使用事实型检索策略以获取精确信息
        results = factual_retrieval_strategy(query, vector_store, k)
    elif query_type == "分析型":
        # 使用分析型检索策略以获取全面覆盖
        results = analytical_retrieval_strategy(query, vector_store, k)
    elif query_type == "观点型":
        # 使用观点型检索策略以获取多样化视角
        results = opinion_retrieval_strategy(query, vector_store, k)
    elif query_type == "上下文型":
        # 使用上下文型检索策略，结合用户上下文
        results = contextual_retrieval_strategy(query, vector_store, k, user_context)
    else:
        # 如果分类失败，默认使用事实型检索策略
        results = factual_retrieval_strategy(query, vector_store, k)

    return results  # 返回检索到的文档

# 响应生成

def generate_response(query, results, query_type, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据查询、检索到的文档和查询类型生成响应。

    参数：
        query (str): 用户查询
        results (List[Dict]): 检索到的文档
        query_type (str): 查询类型
        model (str): LLM模型

    返回：
        str: 生成的响应
    """
    # 从检索到的文档中准备上下文，通过连接文本并用分隔符分隔
    context = "\n\n---\n\n".join([r["text"] for r in results])

    # 根据查询类型创建自定义系统提示
    if query_type == "事实型":
        system_prompt = """你是一位提供事实信息的有帮助的助手。
    根据提供的上下文回答问题，专注于准确性和精确性。
    如果上下文中不包含所需信息，请承认限制。"""
    elif query_type == "分析型":
        system_prompt = """你是一位提供分析见解的有帮助的助手。
    根据提供的上下文，对主题进行全面分析。
    覆盖不同方面和视角，提供全面的解释。
    如果上下文存在空白，请在分析中承认。"""
    elif query_type == "观点型":
        system_prompt = """你是一位讨论主题多种观点的有帮助的助手。
    根据提供的上下文，呈现主题的不同观点。
    确保公平地代表多种观点，不带偏见。
    承认上下文中呈现的有限观点。"""
    elif query_type == "上下文型":
        system_prompt = """你是一位提供上下文相关信息的有帮助的助手。
    根据查询及其上下文回答问题。
    在上下文和查询之间建立联系。
    如果上下文无法完全涵盖具体情况，请承认限制。"""
    else:
        system_prompt = """你是一位有帮助的助手。根据提供的上下文回答问题。如果无法从上下文中回答，请承认限制。"""

    # 创建包含上下文和查询的用户提示
    user_prompt = f"""上下文：
{context}

问题：{query}

请根据上下文提供一个有帮助的响应。
"""

    # 使用OpenAI客户端生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 返回生成的响应内容
    return response.choices[0].message.content

# 完整的RAG管道，带自适应检索

def rag_with_adaptive_retrieval(pdf_path, query, k=4, user_context=None):
    """
    带自适应检索的完整RAG管道。

    参数：
        pdf_path (str): PDF文件路径
        query (str): 用户查询
        k (int): 要检索的文档数
        user_context (str): 可选用户上下文

    返回：
        Dict: 结果，包括查询、检索到的文档、查询类型和响应
    """
    print("\n=== RAG WITH ADAPTIVE RETRIEVAL ===")
    print(f"查询：{query}")

    # 处理文档以提取文本、分块并创建嵌入
    chunks, vector_store = process_document(pdf_path)

    # 分类查询以确定其类型
    query_type = classify_query(query)
    print(f"查询分类为：{query_type}")

    # 使用自适应检索策略根据查询类型检索文档
    retrieved_docs = adaptive_retrieval(query, vector_store, k, user_context)

    # 根据查询、检索到的文档和查询类型生成响应
    response = generate_response(query, retrieved_docs, query_type)

    # 将结果编入字典
    result = {
        "query": query,
        "query_type": query_type,
        "retrieved_documents": retrieved_docs,
        "response": response
    }

    print("\n=== 响应 ===")
    print(response)

    return result


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


