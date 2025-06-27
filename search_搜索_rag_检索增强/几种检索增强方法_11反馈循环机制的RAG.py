
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法11：
# 带有反馈循环机制的RAG系统，该系统能够随着时间的推移不断改进。通过收集和整合用户反馈，我们的系统能够在每次交互中提供更相关、更高质量的响应。
# 传统的RAG系统是静态的，它们仅基于嵌入相似性进行信息检索。通过引入反馈循环，我们创建了一个动态系统，该系统能够：
# - 记住哪些内容有效（哪些无效）
# - 随时间调整文档的相关性评分
# - 将成功的问答对整合到其知识库中
# - 随着每次用户交互而变得更加智能

# 实现反馈系统的核心组件。


def get_user_feedback(query, response, relevance, quality, comments=""):
    """
    将用户反馈格式化为字典。

    参数：
        query (str)：用户的查询
        response (str)：系统的响应
        relevance (int)：相关性评分（1-5）
        quality (int)：质量评分（1-5）
        comments (str)：可选的反馈评论
        timestamp (str)：反馈的时间戳

    返回：
        Dict：格式化的反馈
    """
    return {
        "query": query,
        "response": response,
        "relevance": int(relevance),
        "quality": int(quality),
        "comments": comments,
        "timestamp": datetime.now().isoformat()
    }

def store_feedback(feedback, feedback_file="feedback_data.json"):
    """
    将反馈存储在JSON文件中。

    参数：
        feedback (Dict)：反馈数据
        feedback_file (str)：反馈文件的路径
    """
    with open(feedback_file, "a") as f:
        json.dump(feedback, f)
        f.write("\n")

def load_feedback_data(feedback_file="feedback_data.json"):
    """
    从文件加载反馈数据。

    参数：
        feedback_file (str)：反馈文件的路径

    返回：
        List[Dict]：反馈条目的列表
    """
    feedback_data = []
    try:
        with open(feedback_file, "r") as f:
            for line in f:
                if line.strip():
                    feedback_data.append(json.loads(line.strip()))
    except FileNotFoundError:
        print("未找到反馈数据文件。从空反馈开始。")

    return feedback_data

# 文档处理（带反馈感知）

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    为RAG（检索增强生成）处理文档，带反馈循环。
    该函数处理完整的文档处理管道：
    1. 从PDF中提取文本
    2. 将文本分成具有重叠的块以提高检索准确性
    3. 为块创建嵌入
    4. 将其存储在向量数据库中，带元数据

    参数：
    pdf_path (str)：PDF文件的路径
    chunk_size (int)：每个文本块的字符数
    chunk_overlap (int)：连续块之间的重叠字符数

    返回：
    Tuple[List[str], SimpleVectorStore]：包含以下内容的元组：
        - 文档块的列表
        - 填充了嵌入和元数据的向量存储
    """
    # 步骤1：从PDF文档中提取原始文本内容
    print("从PDF中提取文本...")
    extracted_text = extract_text_from_pdf(pdf_path)

    # 步骤2：将文本分成具有重叠的块以保留上下文
    print("分块文本...")
    chunks = chunk_text(extracted_text, chunk_size, chunk_overlap)
    print(f"创建了{len(chunks)}个文本块")

    # 步骤3：为每个文本块生成向量嵌入
    print("为块创建嵌入...")
    chunk_embeddings = create_embeddings(chunks)

    # 步骤4：初始化向量数据库以存储块及其嵌入
    store = SimpleVectorStore()

    # 步骤5：将每个块及其嵌入添加到向量存储中
    # 包括用于反馈改进的元数据
    for i, (chunk, embedding) in enumerate(zip(chunks, chunk_embeddings)):
        store.add_item(
            text=chunk,
            embedding=embedding,
            metadata={
                "index": i,                # 在原始文档中的位置
                "source": pdf_path,        # 源文档路径
                "relevance_score": 1.0,    # 初始相关性评分（将根据反馈更新）
                "feedback_count": 0        # 接收到的反馈计数
            }
        )

    print(f"将{len(chunks)}个块添加到向量存储中")
    return chunks, store

# 基于反馈的相关性调整
def assess_feedback_relevance(query, doc_text, feedback):
    """
    使用LLM评估过去反馈条目与当前查询和文档的相关性。
    该函数帮助确定哪些过去的反馈应该影响当前检索，
    通过将当前查询、过去的查询+反馈和文档内容发送到LLM
    进行相关性评估。

    参数：
        query (str)：需要信息检索的当前用户查询
        doc_text (str)：正在评估的文档内容
        feedback (Dict)：包含'query'和'response'键的过去反馈数据

    返回：
        bool：如果反馈被认为与当前查询/文档相关，则返回True，否则返回False
    """
    # 定义系统提示，指示LLM仅进行二进制相关性判断
    system_prompt = """你是一个确定过去反馈是否与当前查询和文档相关的AI系统。
    只回答'yes'或'no'。你的任务是严格判断相关性，而不是提供解释。"""

    # 构建用户提示，包含当前查询、过去的反馈数据和文档内容
    user_prompt = f"""当前查询：{query}
    接收过反馈的过去查询：{feedback['query']}
    文档内容：{doc_text[:500]}... [已截断]
    接收过反馈的过去响应：{feedback['response'][:500]}... [已截断]

    这个过去的反馈是否与当前查询和文档相关？（yes/no）
    """

    # 调用LLM API，使用零温度以获得确定性响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用温度=0以获得一致的、确定性的响应
    )

    # 提取并规范化响应以确定相关性
    answer = response.choices[0].message.content.strip().lower()
    return 'yes' in answer  # 如果答案包含'yes'，则返回True

def adjust_relevance_scores(query, results, feedback_data):
    """
    根据历史反馈动态调整文档的相关性评分，以提高检索质量。
    该函数分析过去的用户反馈，动态调整检索文档的相关性评分，
    识别与当前查询上下文相关的反馈，计算评分调整，
    并根据新的评分重新排序结果。

    参数：
        query (str)：当前用户查询
        results (List[Dict])：检索到的文档及其原始相似性评分
        feedback_data (List[Dict])：包含用户评分的历史反馈

    返回：
        List[Dict]：根据反馈历史调整相关性评分后的结果，按新评分排序
    """
    # 如果没有反馈数据，返回原始结果不变
    if not feedback_data:
        return results

    print("根据反馈历史调整相关性评分...")

    # 处理每个检索到的文档
    for i, result in enumerate(results):
        document_text = result["text"]
        relevant_feedback = []

        # 找出与当前查询和文档相关的过去反馈
        # 通过调用LLM评估每个历史反馈条目的相关性
        for feedback in feedback_data:
            is_relevant = assess_feedback_relevance(query, document_text, feedback)
            if is_relevant:
                relevant_feedback.append(feedback)

        # 如果有相关反馈，应用评分调整
        if relevant_feedback:
            # 计算所有适用反馈条目的平均相关性评分
            # 反馈相关性评分范围为1-5（1=不相关，5=高度相关）
            avg_relevance = sum(f['relevance'] for f in relevant_feedback) / len(relevant_feedback)

            # 将平均相关性评分转换为评分调整因子（范围0.5-1.5）
            # - 低于3/5的评分将降低原始相似性（调整因子<1.0）
            # - 高于3/5的评分将提高原始相似性（调整因子>1.0）
            modifier = 0.5 + (avg_relevance / 5.0)

            # 应用调整因子到原始相似性评分
            original_score = result["similarity"]
            adjusted_score = original_score * modifier

            # 更新结果字典，包含新的评分
            result["original_similarity"] = original_score  # 保留原始评分
            result["similarity"] = adjusted_score           # 更新主要评分
            result["relevance_score"] = adjusted_score      # 更新相关性评分
            result["feedback_applied"] = True               # 标记已应用反馈
            result["feedback_count"] = len(relevant_feedback)  # 使用的反馈条目数

            # 记录调整详情
            print(f"文档{i+1}：将评分从{original_score:.4f}调整为{adjusted_score:.4f}，基于{len(relevant_feedback)}条反馈")

    # 根据调整后的评分重新排序结果，以确保高质量匹配排在前面
    results.sort(key=lambda x: x["similarity"], reverse=True)

    return results

# 使用反馈优化索引

def fine_tune_index(current_store, chunks, feedback_data):
    """
    通过高质量反馈增强向量存储，以提高检索质量随时间。
    该函数实现了一个持续学习过程：
    1. 识别高质量反馈（高度评分的问答对）
    2. 从成功交互中创建新的检索项
    3. 将这些项添加到向量存储中，带加权相关性评分

    参数：
        current_store (SimpleVectorStore)：包含原始文档块的当前向量存储
        chunks (List[str])：原始文档文本块
        feedback_data (List[Dict])：包含相关性和质量评分的历史用户反馈

    返回：
        SimpleVectorStore：增强后的向量存储，包含原始块和基于反馈的内容
    """
    print("使用高质量反馈优化索引...")

    # 过滤出高质量反馈（相关性和质量评分均为4或5）
    # 这样我们只从最成功的交互中学习
    good_feedback = [f for f in feedback_data if f['relevance'] >= 4 and f['quality'] >= 4]

    if not good_feedback:
        print("未找到可用于优化的高质量反馈。")
        return current_store  # 如果没有高质量反馈，返回原始存储不变

    # 初始化新的存储，将包含原始内容和增强内容
    new_store = SimpleVectorStore()

    # 首先将所有原始文档块及其现有元数据转移到新存储中
    for i in range(len(current_store.texts)):
        new_store.add_item(
            text=current_store.texts[i],
            embedding=current_store.vectors[i],
            metadata=current_store.metadata[i].copy()  # 使用副本以防止引用问题
        )

    # 从高质量反馈创建并添加增强内容
    for feedback in good_feedback:
        # 格式化一个新的文档，结合问题及其高质量答案
        # 这样创建的检索内容直接针对用户查询
        enhanced_text = f"问题：{feedback['query']}\n答案：{feedback['response']}"

        # 为这个新的合成文档生成嵌入向量
        embedding = create_embeddings(enhanced_text)

        # 将其添加到向量存储中，带特殊元数据以标识其来源和重要性
        new_store.add_item(
            text=enhanced_text,
            embedding=embedding,
            metadata={
                "type": "feedback_enhanced",  # 标记为基于反馈
                "query": feedback["query"],   # 存储原始查询以供参考
                "relevance_score": 1.2,       # 提高初始相关性以优先考虑这些项目
                "feedback_count": 1,          # 跟踪反馈整合
                "original_feedback": feedback # 保留完整的反馈记录
            }
        )

        print(f"添加了基于反馈的增强内容：{feedback['query'][:50]}...")

    # 记录增强的统计信息
    print(f"优化后的索引现在有{len(new_store.texts)}个项目（原始：{len(chunks)}）")
    return new_store

# 带反馈循环的完整RAG管道

def generate_response(query, context, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    根据查询和上下文生成响应。

    参数：
        query (str)：用户查询
        context (str)：来自检索文档的上下文文本
        model (str)：LLM模型
        temperature (float)：控制响应的随机性（默认为0）

    返回：
        str：生成的响应
    """
    # 定义系统提示，指导AI的行为
    system_prompt = """你是一个乐于助人的AI助手。仅根据提供的上下文回答用户的问题。如果无法在上下文中找到答案，请说明你没有足够的信息。"""

    # 创建用户提示，结合上下文和查询
    user_prompt = f"""上下文：
    {context}

    问题：{query}

    请根据上述上下文提供一个全面的回答。
    """

    # 调用OpenAI API生成基于系统和用户提示的响应
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0  # 使用温度=0以获得一致的、确定性的响应
    )

    # 返回生成的响应内容
    return response.choices[0].message.content

def rag_with_feedback_loop(query, vector_store, feedback_data, k=5, model="meta-llama/Llama-3.2-3B-Instruct"):
    """
    带反馈循环的完整RAG管道。

    参数：
        query (str)：用户查询
        vector_store (SimpleVectorStore)：包含文档块的向量存储
        feedback_data (List[Dict])：反馈历史
        k (int)：要检索的文档数
        model (str)：LLM模型

    返回：
        Dict：包含查询、检索文档和响应的结果
    """
    print(f"\n=== 使用反馈增强的RAG处理查询 ===")
    print(f"查询：{query}")

    # 步骤1：创建查询嵌入
    query_embedding = create_embeddings(query)

    # 步骤2：基于查询嵌入进行初始检索
    results = vector_store.similarity_search(query_embedding, k=k)

    # 步骤3：根据反馈调整检索到的文档的相关性评分
    adjusted_results = adjust_relevance_scores(query, results, feedback_data)

    # 步骤4：从调整后的结果中提取文本以构建上下文
    retrieved_texts = [result["text"] for result in adjusted_results]

    # 步骤5：通过连接检索到的文本构建上下文
    context = "\n\n---\n\n".join(retrieved_texts)

    # 步骤6：根据上下文和查询生成响应
    print("生成响应...")
    response = generate_response(query, context, model)

    # 步骤7：编译最终结果
    result = {
        "query": query,
        "retrieved_documents": adjusted_results,
        "response": response
    }

    print("\n=== 响应 ===")
    print(response)

    return result

# 完整工作流程：从初始设置到反馈收集

def full_rag_workflow(pdf_path, query, feedback_data=None, feedback_file="feedback_data.json", fine_tune=False):
    """
    带反馈整合的完整RAG工作流程，用于持续改进。
    该函数编排了完整的检索增强生成过程：
    1. 加载历史反馈数据
    2. 处理和分块文档
    3. 可选：使用过去反馈优化向量索引
    4. 带反馈感知的相关性检索和生成
    5. 收集新用户反馈以供未来改进
    6. 存储反馈以实现系统随时间学习

    参数：
        pdf_path (str)：PDF文档的路径
        query (str)：用户的自然语言查询
        feedback_data (List[Dict], optional)：预加载的反馈数据，如果为None，则从文件加载
        feedback_file (str)：存储反馈历史的JSON文件路径
        fine_tune (bool)：是否使用成功的过去问答对优化索引

    返回：
        Dict：包含响应和检索元数据的结果
    """
    # 步骤1：加载历史反馈用于相关性调整（如果未显式提供）
    if feedback_data is None:
        feedback_data = load_feedback_data(feedback_file)
        print(f"从{feedback_file}加载了{len(feedback_data)}条反馈")

    # 步骤2：通过提取、分块和嵌入管道处理文档
    chunks, vector_store = process_document(pdf_path)

    # 步骤3：使用成功过去的问答对优化向量索引
    # 这会创建从成功交互中增强的可检索内容
    if fine_tune and feedback_data:
        vector_store = fine_tune_index(vector_store, chunks, feedback_data)

    # 步骤4：执行带反馈感知的相关性检索
    # 注意：这取决于在别处定义的rag_with_feedback_loop函数
    result = rag_with_feedback_loop(query, vector_store, feedback_data)

    # 步骤5：收集用户反馈以提高未来性能
    print("\n=== 你愿意为这个响应提供反馈吗？ ===")
    print("相关性评分（1-5，5表示最相关）：")
    relevance = input()

    print("质量评分（1-5，5表示最高质量）：")
    quality = input()

    print("任何评论？（可选，按Enter跳过）")
    comments = input()

    # 步骤6：将反馈格式化为结构化数据
    feedback = get_user_feedback(
        query=query,
        response=result["response"],
        relevance=int(relevance),
        quality=int(quality),
        comments=comments
    )

    # 步骤7：持久化反馈以实现持续系统学习
    store_feedback(feedback, feedback_file)
    print("已记录反馈。感谢你的反馈！")

    return result



####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


