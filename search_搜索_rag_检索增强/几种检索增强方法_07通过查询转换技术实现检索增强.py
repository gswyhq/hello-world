
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法7：
# 通过查询转换技术实现检索增强
# 1、查询改写：使查询更具针对性和详细性，以提高检索精度。
# 2、回退提示：生成更广泛的查询，以检索有用的背景信息。通过生成更宽泛的查询语句来实现更好的上下文检索。这种方法不是直接回答问题,而是通过考虑更高层次的概念和原则进行推理。
# 3、子查询分解：将复杂查询分解为更简单的组件，以实现全面检索。

# 查询改写
# 该技术使查询更具针对性和详细性，以提高检索精度。
def rewrite_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    重写查询以使其更具针对性和详细性，从而提高检索效果。

    参数:
        original_query (str): 原始用户查询
        model (str): 用于查询改写的模型

    返回:
        str: 重写的查询
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = "你是一个专门用于改进搜索查询的AI助手。你的任务是将用户查询重写得更具针对性、详细性，并更有可能检索到相关信息。"

    # 定义用户提示，包含待重写的原始查询
    user_prompt = f"""
    将以下查询重写得更具针对性和详细性。包含相关术语和概念，以帮助检索到准确信息。

    原始查询: {original_query}

    重写后的查询:
    """

    # 使用指定模型生成重写后的查询
    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # 低温度以获得确定性输出
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回重写的查询，去除首尾空格
    return response.choices[0].message.content.strip()

# 回退提示
# 该技术生成更广泛的查询，以检索背景信息。

def generate_step_back_query(original_query, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    生成更广泛的“回退”查询以检索背景信息。

    参数:
        original_query (str): 原始用户查询
        model (str): 用于生成回退查询的模型

    返回:
        str: 回退查询
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = "你是一个专门用于搜索策略的AI助手。你的任务是将具体查询生成更广泛、更通用的版本，以检索相关背景信息。"

    # 定义用户提示，包含待广化的原始查询
    user_prompt = f"""
    生成一个更广泛、更通用的版本，以帮助检索有用背景信息。

    原始查询: {original_query}

    回退查询:
    """

    # 使用指定模型生成回退查询
    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # 稍高的温度以获得一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回回退查询，去除首尾空格
    return response.choices[0].message.content.strip()

# 子查询分解
# 该技术将复杂查询分解为更简单的组件，以实现全面检索。
def decompose_query(original_query, num_subqueries=4, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    将复杂查询分解为更简单的子查询。

    参数:
        original_query (str): 原始复杂查询
        num_subqueries (int): 要生成的子查询数量
        model (str): 用于查询分解的模型

    返回:
        List[str]: 简单子查询的列表
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = "你是一个专门用于分解复杂问题的AI助手。你的任务是将复杂查询分解为更简单的子查询，这些子查询的回答共同解决原始查询。"

    # 定义用户提示，包含待分解的原始查询
    user_prompt = f"""
    将以下复杂查询分解为 {num_subqueries} 个更简单的子查询。每个子查询应关注原始问题的不同方面。

    原始查询: {original_query}

    生成 {num_subqueries} 个子查询，每个子查询占一行，格式如下:
    1. [第一个子查询]
    2. [第二个子查询]
    依此类推...
    """

    # 使用指定模型生成子查询
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # 稍高的温度以获得一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 处理响应以提取子查询
    content = response.choices[0].message.content.strip()

    # 使用简单解析提取编号查询
    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # 去除编号和前导空格
            query = line.strip()
            query = query[query.find(".") + 1:].strip()
            sub_queries.append(query)

    return sub_queries

# 演示查询转换技术
# 示例查询
original_query = "AI 对工作自动化和就业有什么影响？"

# 应用查询转换
print("Original Query:", original_query)

# 查询改写
rewritten_query = rewrite_query(original_query)
print("\n1. Rewritten Query:")
print(rewritten_query)

# 回退提示
step_back_query = generate_step_back_query(original_query)
print("\n2. Step-back Query:")
print(step_back_query)

# 子查询分解
sub_queries = decompose_query(original_query, num_subqueries=4)
print("\n3. Sub-queries:")
for i, query in enumerate(sub_queries, 1):
    print(f"   {i}. {query}")

# 实现RAG与查询转换

def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    使用转换后的查询进行搜索。

    参数:
        query (str): 原始查询
        vector_store (SimpleVectorStore): 向量存储
        transformation_type (str): 转换类型 ('rewrite', 'step_back', 或 'decompose')
        top_k (int): 要返回的结果数量

    返回:
        List[Dict]: 搜索结果
    """
    print(f"Transformation type: {transformation_type}")
    print(f"Original query: {query}")

    results = []

    if transformation_type == "rewrite":
        # 查询改写
        transformed_query = rewrite_query(query)
        print(f"Rewritten query: {transformed_query}")

        # 为转换后的查询创建嵌入
        query_embedding = create_embeddings(transformed_query)

        # 使用重写后的查询进行搜索
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "step_back":
        # 回退提示
        transformed_query = generate_step_back_query(query)
        print(f"Step-back query: {transformed_query}")

        # 为转换后的查询创建嵌入
        query_embedding = create_embeddings(transformed_query)

        # 使用回退查询进行搜索
        results = vector_store.similarity_search(query_embedding, k=top_k)

    elif transformation_type == "decompose":
        # 子查询分解
        sub_queries = decompose_query(query)
        print("Decomposed into sub-queries:")
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"{i}. {sub_q}")

        # 为所有子查询创建嵌入
        sub_query_embeddings = create_embeddings(sub_queries)

        # 使用每个子查询进行搜索并合并结果
        all_results = []
        for i, embedding in enumerate(sub_query_embeddings):
            sub_results = vector_store.similarity_search(embedding, k=2)  # 每个子查询获取较少结果
            all_results.extend(sub_results)

        # 去重（保留相似度最高的结果）
        seen_texts = {}
        for result in all_results:
            text = result["text"]
            if text not in seen_texts or result["similarity"] > seen_texts[text]["similarity"]:
                seen_texts[text] = result

        # 按相似度排序并取前top_k
        results = sorted(seen_texts.values(), key=lambda x: x["similarity"], reverse=True)[:top_k]

    else:
        # 无转换的常规搜索
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)

    return results

# 生成带转换查询的响应

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据查询和检索到的上下文生成响应。

    参数:
        query (str): 用户查询
        context (str): 检索到的上下文
        model (str): 用于响应生成的模型

    返回:
        str: 生成的响应
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = "你是一个乐于助人的AI助手。根据提供的上下文回答用户的问题。如果无法在上下文中找到答案，请说明你没有足够的信息。"

    # 定义用户提示，包含上下文和查询
    user_prompt = f"""
        上下文:
        {context}

        问题: {query}

        请根据上述上下文提供一个全面的回答。
    """

    # 使用指定模型生成响应
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # 低温度以获得确定性输出
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 返回生成的响应，去除首尾空格
    return response.choices[0].message.content.strip()

# 运行完整的RAG管道与查询转换

def rag_with_query_transformation(pdf_path, query, transformation_type=None):
    """
    运行完整的RAG管道，可选查询转换。

    参数:
        pdf_path (str): PDF文档路径
        query (str): 用户查询
        transformation_type (str): 转换类型（None, 'rewrite', 'step_back', 或 'decompose'）

    返回:
        Dict: 结果，包括查询、转换后的查询、上下文和响应
    """
    # 处理文档以创建向量存储
    vector_store = process_document(pdf_path)

    # 应用查询转换并搜索
    if transformation_type:
        # 使用转换后的查询进行搜索
        results = transformed_search(query, vector_store, transformation_type)
    else:
        # 无转换的常规搜索
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=3)

    # 从搜索结果中组合上下文
    context = "\n\n".join([f"PASSAGE {i+1}:\n{result['text']}" for i, result in enumerate(results)])

    # 根据查询和组合上下文生成响应
    response = generate_response(query, context)

    # 返回结果，包括原始查询、转换类型、上下文和响应
    return {
        "original_query": query,
        "transformation_type": transformation_type,
        "context": context,
        "response": response
    }
# 评估转换技术

def compare_responses(results, reference_answer, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    比较不同查询转换技术生成的响应。

    参数:
        results (Dict): 不同转换技术的结果
        reference_answer (str): 参考答案
        model (str): 用于评估的模型
    """
    # 定义系统提示以指导AI助手的行为
    system_prompt = """你是一个RAG系统的专家评估员。
    你的任务是比较不同查询转换技术生成的响应与参考答案。"""

    # 准备比较文本，包含参考答案和每种技术的响应
    comparison_text = f"Reference Answer: {reference_answer}\n\n"

    for technique, result in results.items():
        comparison_text += f"{technique.capitalize()} Query Response:\n{result['response']}\n\n"

    # 定义用户提示，包含比较文本
    user_prompt = f"""
    {comparison_text}

    比较不同查询转换技术生成的响应与参考答案。

    对于每种技术（original, rewrite, step_back, decompose）:
    1. 根据准确性、完整性和相关性进行评分（1-10分）
    2. 识别优缺点

    然后按效果从好到差排序，并解释哪种技术表现最好及其原因。
    """

    # 使用指定模型生成评估响应
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 打印评估结果
    print("\n===== 评估结果 =====")
    print(response.choices[0].message.content)
    print("===================")

# 评估转换技术
def evaluate_transformations(pdf_path, query, reference_answer=None):
    """
    评估同一查询的不同转换技术。

    参数:
        pdf_path (str): PDF文档路径
        query (str): 查询
        reference_answer (str): 可选参考答案

    返回:
        Dict: 评估结果
    """
    # 定义要评估的转换技术
    transformation_types = [None, "rewrite", "step_back", "decompose"]
    results = {}

    # 使用每种转换技术运行RAG
    for transformation_type in transformation_types:
        type_name = transformation_type if transformation_type else "original"
        print(f"\n===== 运行 {type_name} 查询的RAG =====")

        # 获取当前转换类型的RAG结果
        result = rag_with_query_transformation(pdf_path, query, transformation_type)
        results[type_name] = result

        # 打印当前转换类型的响应
        print(f"{type_name}查询的响应:")
        print(result["response"])
        print("=" * 50)

    # 如果提供参考答案，比较结果
    if reference_answer:
        compare_responses(results, reference_answer)

    return results

# 运行评估
evaluation_results = evaluate_transformations(pdf_path, query, reference_answer)


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


