
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法10：
# 上下文压缩技术来提高 RAG 系统的效率。我们将过滤和压缩检索到的文本块，以仅保留最相关的部分，从而减少噪音并提高响应质量。
# 在为 RAG 检索文档时，我们通常会得到同时包含相关和不相关信息的块。上下文压缩可以帮助我们：
# 删除不相关的句子和段落
# 仅关注与查询相关的信息
# 最大化上下文窗口中的有用信号

# 使用LLM过滤和压缩检索到的内容。
def compress_chunk(chunk, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    压缩检索到的块，仅保留与查询相关的内容。

    参数:
        chunk (str): 要压缩的文本块
        query (str): 用户查询
        compression_type (str): 压缩类型（"selective"，"summary"，或"extraction"）
        model (str): 要使用的LLM模型

    返回:
        str: 压缩后的文本块
    """
    # 定义不同压缩方法的系统提示
    if compression_type == "selective":
        system_prompt = """你是一位信息过滤专家。
        你的任务是分析文档块并提取仅与用户查询直接相关的句子或段落。删除所有不相关的内容。

        你的输出应该：
        1. 仅包含有助于回答查询的文本
        2. 保留相关句子的原始措辞（不要进行改写）
        3. 保持原文的顺序
        4. 包含所有相关的内容，即使看起来重复
        5. 排除任何与查询无关的文本

        将你的回答格式化为纯文本，不添加任何额外的评论。"""
    elif compression_type == "summary":
        system_prompt = """你是一位摘要专家。
        你的任务是为提供的块创建一个简洁的摘要，专注于与用户查询相关的信息。

        你的输出应该：
        1. 简洁但全面地涵盖与查询相关的信息
        2. 专注于与查询相关的信息
        3. 省略不相关信息
        4. 使用中立、事实性的语气

        将你的回答格式化为纯文本，不添加任何额外的评论。"""
    else:  # extraction
        system_prompt = """你是一位信息提取专家。
        你的任务是从文档块中提取仅包含与回答用户查询直接相关的句子。

        你的输出应该：
        1. 仅包含原始文本中与查询相关的句子的直接引用
        2. 保留原始措辞（不要修改文本）
        3. 仅包含直接与查询相关的句子
        4. 用换行符分隔提取的句子
        5. 不添加任何评论或额外文本

        将你的回答格式化为纯文本，不添加任何额外的评论。"""

    # 定义包含查询和文档块的用户提示
    user_prompt = f"""查询：{query}

    文档块：
    {chunk}

    提取仅与回答此查询相关的内容。
    """

    # 使用OpenAI API生成响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取压缩后的块
    compressed_chunk = response.choices[0].message.content.strip()

    # 计算压缩率
    original_length = len(chunk)
    compressed_length = len(compressed_chunk)
    compression_ratio = (original_length - compressed_length) / original_length * 100

    return compressed_chunk, compression_ratio

# 批量压缩
# 为了提高效率，我们一次性压缩多个块。
def batch_compress_chunks(chunks, query, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    批量压缩多个块。

    参数:
        chunks (List[str]): 要压缩的文本块列表
        query (str): 用户查询
        compression_type (str): 压缩类型（"selective"，"summary"，或"extraction"）
        model (str): 要使用的LLM模型

    返回:
        List[Tuple[str, float]]: 压缩后的块列表及其压缩率
    """
    print(f"压缩{len(chunks)}个块...")  # 打印要压缩的块数
    results = []  # 初始化一个空列表存储结果
    total_original_length = 0  # 初始化变量存储块的总原始长度
    total_compressed_length = 0  # 初始化变量存储块的总压缩长度

    # 遍历每个块
    for i, chunk in enumerate(chunks):
        print(f"压缩第{i+1}/{len(chunks)}个块...")  # 打印压缩进度
        # 压缩块并获取压缩后的块和压缩率
        compressed_chunk, compression_ratio = compress_chunk(chunk, query, compression_type, model)
        results.append((compressed_chunk, compression_ratio))  # 将结果追加到results列表中

        total_original_length += len(chunk)  # 将块的原始长度加到总原始长度中
        total_compressed_length += len(compressed_chunk)  # 将块的压缩长度加到总压缩长度中

    # 计算总体压缩率
    overall_ratio = (total_original_length - total_compressed_length) / total_original_length * 100
    print(f"总体压缩率：{overall_ratio:.2f}%")  # 打印总体压缩率

    return results  # 返回压缩后的块列表及其压缩率

# 完整的RAG管道带上下文压缩
def rag_with_compression(pdf_path, query, k=10, compression_type="selective", model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    带上下文压缩的完整RAG管道。

    参数:
        pdf_path (str): PDF文档的路径
        query (str): 用户查询
        k (int): 要检索的初始块数
        compression_type (str): 压缩类型
        model (str): 要使用的LLM模型

    返回:
        dict: 包含查询、压缩块和响应的结果
    """
    print("\n=== 带上下文压缩的RAG ===")
    print(f"查询：{query}")
    print(f"压缩类型：{compression_type}")

    # 处理文档以提取文本、分块并创建嵌入
    vector_store = process_document(pdf_path)

    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    # 检索与查询嵌入最相似的前k个块
    print(f"检索前{k}个块...")
    results = vector_store.similarity_search(query_embedding, k=k)
    retrieved_chunks = [result["text"] for result in results]

    # 对检索到的块应用压缩
    compressed_results = batch_compress_chunks(retrieved_chunks, query, compression_type, model)
    compressed_chunks = [result[0] for result in compressed_results]
    compression_ratios = [result[1] for result in compressed_results]

    # 过滤掉任何空的压缩块
    filtered_chunks = [(chunk, ratio) for chunk, ratio in zip(compressed_chunks, compression_ratios) if chunk.strip()]

    if not filtered_chunks:
        # 如果所有块都被压缩为空字符串，则使用原始块
        print("警告：所有块都被压缩为空字符串。使用原始块。")
        filtered_chunks = [(chunk, 0.0) for chunk in retrieved_chunks]
    else:
        compressed_chunks, compression_ratios = zip(*filtered_chunks)

    # 从压缩块生成上下文
    context = "\n\n---\n\n".join(compressed_chunks)

    # 基于压缩块生成响应
    print("基于压缩块生成响应...")
    response = generate_response(query, context, model)

    # 准备结果字典
    result = {
        "query": query,
        "original_chunks": retrieved_chunks,
        "compressed_chunks": compressed_chunks,
        "compression_ratios": compression_ratios,
        "context_length_reduction": f"{sum(compression_ratios)/len(compression_ratios):.2f}%",
        "response": response
    }

    print("\n=== 响应 ===")
    print(response)

    return result


####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


