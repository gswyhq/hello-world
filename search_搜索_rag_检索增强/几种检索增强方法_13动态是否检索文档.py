
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法13：
# 动态RAG方法实现
# ## Self-RAG的核心组件
# 1. **检索决策**：确定是否需要对给定查询进行检索
# 2. **文档检索**：在需要时获取相关文档
# 3. **相关性评估**：评估每个检索到的文档的相关性
# 4. **响应生成**：基于相关上下文生成响应
# 5. **支持评估**：评估响应是否充分基于上下文
# 6. **效用评估**：对生成响应的整体有用性进行评分

# 1. 检索决策

def determine_if_retrieval_needed(query):
    """
    确定是否需要检索以回答给定查询。

    Args:
        query (str): 用户查询

    Returns:
        bool: 如果需要检索，返回True，否则返回False
    """
    # 系统提示，指示AI如何确定是否需要检索
    system_prompt = """你是一个AI助手，负责确定是否需要检索来准确回答查询。
    对于事实性问题、特定信息请求或关于事件、人物或概念的问题，回答“是”。
    对于意见、假设性情景或常识性简单查询，回答“否”。
    只回答“是”或“否”。"""

    # 用户提示，包含查询
    user_prompt = f"查询：{query}\n\n是否需要检索来准确回答此查询？"

    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    # 如果答案包含“是”，则返回True，否则返回False
    return "yes" in answer


# 2. 相关性评估

def evaluate_relevance(query, context):
    """
    评估上下文与查询的相关性。

    Args:
        query (str): 用户查询
        context (str): 上下文文本

    Returns:
        str: 'relevant' 或 'irrelevant'
    """
    # 系统提示，指示AI如何确定文档相关性
    system_prompt = """你是一个AI助手，负责确定文档是否与查询相关。
    考虑文档是否包含有助于回答查询的信息。
    只回答“相关”或“不相关”。"""

    # 如果上下文太长，截断以避免超过令牌限制
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [已截断]"

    # 用户提示，包含查询和文档内容
    user_prompt = f"""查询：{query}
    文档内容：
    {context}

    该文档是否与查询相关？只回答“相关”或“不相关”。"""

    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    return answer  # 返回相关性评估


# 3. 支持评估

def assess_support(response, context):
    """
    评估响应是否得到上下文的支持。

    Args:
        response (str): 生成的响应
        context (str): 上下文文本

    Returns:
        str: 'fully supported'、'partially supported' 或 'no support'
    """
    # 系统提示，指示AI如何评估支持
    system_prompt = """你是一个AI助手，负责确定响应是否得到给定上下文的支持。
    评估响应中的事实、声明和信息是否得到上下文的支持。
    只回答以下三个选项之一：
    - “完全支持”：响应中的所有信息都直接得到上下文支持。
    - “部分支持”：响应中部分信息得到上下文支持，但部分没有。
    - “无支持”：响应包含大量未在上下文中找到或与上下文矛盾的信息。"""

    # 如果上下文太长，截断以避免超过令牌限制
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "... [已截断]"

    # 用户提示，包含上下文和要评估的响应
    user_prompt = f"""上下文：
    {context}

    响应：
    {response}

    该响应在多大程度上得到上下文的支持？只回答“完全支持”、“部分支持”或“无支持”。"""

    # 从模型生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取答案并转换为小写
    answer = response.choices[0].message.content.strip().lower()

    return answer  # 返回支持评估


# 4. 效用评估

def rate_utility(query, response):
    """
    对响应对查询的有用性进行评分。

    Args:
        query (str): 用户查询
        response (str): 生成的响应

    Returns:
        int: 有用性评分（1到5分）
    """
    # 系统提示，指示AI如何对响应的有用性进行评分
    system_prompt = """你是一个AI助手，负责对查询响应的有用性进行评分。
    考虑响应如何回答查询、其完整性和正确性以及有用性。
    在1到5的评分标准下进行评分：
    - 1：完全无用
    - 2：稍微有用
    - 3：中等有用
    - 4：非常有用
    - 5：极其有用
    只回答1到5之间的单个数字。"""

    # 用户提示，包含查询和要评分的响应
    user_prompt = f"""查询：{query}
    响应：
    {response}

    对该响应的有用性进行评分（1到5）："""

    # 使用OpenAI客户端生成有用性评分
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从模型响应中提取评分
    rating = response.choices[0].message.content.strip()

    # 从评分中提取数字
    rating_match = re.search(r'[1-5]', rating)
    if rating_match:
        return int(rating_match.group())  # 返回提取的评分作为整数

    return 3  # 如果解析失败，默认返回中间评分


# 响应生成

def generate_response(query, context=None):
    """
    基于查询和可选上下文生成响应。

    Args:
        query (str): 用户查询
        context (str, optional): 上下文文本

    Returns:
        str: 生成的响应
    """
    # 系统提示，指示AI如何生成有用响应
    system_prompt = """你是一个乐于助人的AI助手。请根据查询提供清晰、准确且信息丰富的响应。"""

    # 根据是否提供上下文创建用户提示
    if context:
        user_prompt = f"""上下文：
        {context}

        查询：{query}

        请根据提供的上下文回答查询。"""
    else:
        user_prompt = f"""查询：{query}

        请尽你所能回答查询。"""

    # 使用OpenAI客户端生成响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2
    )

    # 返回生成的响应文本
    return response.choices[0].message.content.strip()


# 完整的Self - RAG实现

def self_rag(query, vector_store, top_k=3):
    """
    实现完整的Self-RAG管道。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 包含文档块的向量存储
        top_k (int): 要检索的初始文档数

    Returns:
        dict: 包含查询、响应和Self-RAG过程指标的结果
    """
    print(f"\n=== 开始Self-RAG处理查询：{query} ===\n")

    # 步骤1：确定是否需要检索
    print("步骤1：确定是否需要检索...")
    retrieval_needed = determine_if_retrieval_needed(query)
    print(f"需要检索：{retrieval_needed}")

    # 初始化指标以跟踪Self-RAG过程
    metrics = {
        "retrieval_needed": retrieval_needed,
        "documents_retrieved": 0,
        "relevant_documents": 0,
        "response_support_ratings": [],
        "utility_ratings": []
    }

    best_response = None
    best_score = -1

    if retrieval_needed:
        # 步骤2：检索相关文档
        print("\n步骤2：检索相关文档...")
        query_embedding = create_embeddings(query)
        results = vector_store.similarity_search(query_embedding, k=top_k)
        metrics["documents_retrieved"] = len(results)
        print(f"检索到{len(results)}个文档")

        # 步骤3：评估每个文档的相关性
        print("\n步骤3：评估文档相关性...")
        relevant_contexts = []

        for i, result in enumerate(results):
            context = result["text"]
            relevance = evaluate_relevance(query, context)
            print(f"文档{i + 1}相关性：{relevance}")

            if relevance == "relevant":
                relevant_contexts.append(context)

        metrics["relevant_documents"] = len(relevant_contexts)
        print(f"找到{len(relevant_contexts)}个相关文档")

        if relevant_contexts:
            # 步骤4：处理每个相关上下文
            print("\n步骤4：处理相关上下文...")
            for i, context in enumerate(relevant_contexts):
                print(f"\n处理上下文{i + 1}/{len(relevant_contexts)}...")

                # 基于上下文生成响应
                print("生成响应...")
                response = generate_response(query, context)

                # 评估响应是否得到上下文支持
                print("评估支持...")
                support_rating = assess_support(response, context)
                print(f"支持评分：{support_rating}")
                metrics["response_support_ratings"].append(support_rating)

                # 对响应的有用性进行评分
                print("评分有用性...")
                utility_rating = rate_utility(query, response)
                print(f"有用性评分：{utility_rating}/5")
                metrics["utility_ratings"].append(utility_rating)

                # 计算整体评分（支持和有用性越高，评分越高）
                support_score = {
                    "fully supported": 3,
                    "partially supported": 1,
                    "no support": 0
                }.get(support_rating, 0)

                overall_score = support_score * 5 + utility_rating
                print(f"整体评分：{overall_score}")

                # 保留最佳响应
                if overall_score > best_score:
                    best_response = response
                    best_score = overall_score
                    print("找到新的最佳响应！")
        else:
            # 如果没有找到合适的上下文或所有响应评分较低
            print("\n未找到合适的上下文或响应评分较低，正在生成不使用检索的响应...")
            best_response = generate_response(query)
    else:
        # 不需要检索，直接生成响应
        print("\n不需要检索，正在直接生成响应...")
        best_response = generate_response(query)

    # 最终指标
    metrics["best_score"] = best_score
    metrics["used_retrieval"] = retrieval_needed and best_score > 0

    print("\n=== Self-RAG完成 ===")

    return {
        "query": query,
        "response": best_response,
        "metrics": metrics
    }


# 运行完整的Self - RAG系统

def run_self_rag_example():
    """
    展示完整的Self-RAG系统的示例。
    """
    # 处理文档
    pdf_path = "data/AI_Information.pdf"  # PDF文档的路径
    print(f"处理文档：{pdf_path}")
    vector_store = process_document(pdf_path)  # 处理文档并创建向量存储

    # 示例1：可能需要检索的查询
    query1 = "什么是人工智能开发的主要伦理问题？"
    print("\n" + "=" * 80)
    print(f"示例1：{query1}")
    result1 = self_rag(query1, vector_store)  # 运行Self-RAG处理第一个查询
    print("\n最终响应：")
    print(result1["response"])  # 打印第一个查询的最终响应
    print("\n指标：")
    print(json.dumps(result1["metrics"], indent=2))  # 打印第一个查询的指标

    # 示例2：可能不需要检索的查询
    query2 = "你能写一首关于人工智能的短诗吗？"
    print("\n" + "=" * 80)
    print(f"示例2：{query2}")
    result2 = self_rag(query2, vector_store)  # 运行Self-RAG处理第二个查询
    print("\n最终响应：")
    print(result2["response"])  # 打印第二个查询的最终响应
    print("\n指标：")
    print(json.dumps(result2["metrics"], indent=2))  # 打印第二个查询的指标

    # 示例3：与文档相关但需要额外知识的查询
    query3 = "人工智能如何影响发展中国家的经济增长？"
    print("\n" + "=" * 80)
    print(f"示例3：{query3}")
    result3 = self_rag(query3, vector_store)  # 运行Self-RAG处理第三个查询
    print("\n最终响应：")
    print(result3["response"])  # 打印第三个查询的最终响应
    print("\n指标：")
    print(json.dumps(result3["metrics"], indent=2))  # 打印第三个查询的指标

    return {
        "example1": result1,
        "example2": result2,
        "example3": result3
    }

####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


