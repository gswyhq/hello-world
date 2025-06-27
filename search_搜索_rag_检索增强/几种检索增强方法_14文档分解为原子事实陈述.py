
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法14：
# 一种将文档分解为原子事实陈述的高级技术，以实现更准确的检索。与传统分块仅按字符数分割文本不同，原子分块保留了单个事实的语义完整性。
#
# 原子分块（Proposition Chunking）通过以下方式实现更精确的检索：
#
# 1. 将内容分解为原子、自包含的事实
# 2. 创建更小、更细粒度的检索单元
# 3. 使查询与相关内容之间的匹配更加精确
# 4. 过滤掉低质量或不完整的原子

# proposition 被定义为文本中的 原子表达（不能进一步分解的单个语义元素，可用于构成更大的语义单位） ，用于检索和表达文本中的独特事实或特定概念，能够以简明扼要的方式表达，使用自然语言完整地呈现一个独立的概念或事实，不需要额外的信息来解释。

# 文本分块
# 将它分成较小的、重叠的块以提高检索准确性。

def chunk_text(text, chunk_size=800, overlap=100):
    """
    将文本分割成重叠块。

    Args:
        text (str): 输入文本
        chunk_size (int): 每个块的大小（以字符为单位）
        overlap (int): 块之间的重叠部分（以字符为单位）

    Returns:
        List[Dict]: 包含文本和元数据的块列表
    """
    chunks = []  # 初始化一个空列表以存储块

    # 以指定的块大小和重叠部分遍历文本
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]  # 提取指定大小的块
        if chunk:  # 确保我们不添加空块
            chunks.append({
                "text": chunk,  # 块文本
                "chunk_id": len(chunks) + 1,  # 块的唯一ID
                "start_char": i,  # 块的起始字符索引
                "end_char": i + len(chunk)  # 块的结束字符索引
            })

    print(f"创建了 {len(chunks)} 个文本块")  # 打印创建的块数
    return chunks  # 返回块列表

# 原子块生成

def generate_propositions(chunk):
    """
    从文本块生成原子块、自包含的原子块。

    Args:
        chunk (Dict): 包含内容和元数据的文本块

    Returns:
        List[str]: 生成的原子块列表
    """
    # 系统提示，指示AI如何生成原子块
    system_prompt = """请将以下文本分解为简单的、自包含的原子块。
    确保每个原子块满足以下标准：

    1. 表达单一事实：每个原子块应陈述一个具体事实或声明。
    2. 无需上下文即可理解：原子块应自包含，即无需额外上下文即可理解。
    3. 使用全名，避免代词：避免使用代词或模糊引用；使用全名。
    4. 包含相关日期/限定词：如果适用，请包含必要的日期、时间和限定词以使事实精确。
    5. 包含单一主谓关系：专注于单一主语及其对应的动作或属性，避免使用连词或多个从句。

    仅输出原子块列表，不带任何额外文本或解释。"""

    # 用户提示，包含要转换为原子块的文本块
    user_prompt = f"要转换为原子块的文本：\n\n{chunk['text']}"

    # 生成模型响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",  # 使用更强的模型以确保准确的原子块生成
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0
    )

    # 从响应中提取原子块
    raw_propositions = response.choices[0].message.content.strip().split('\n')

    # 清理原子块（移除编号、项目符号等）
    clean_propositions = []
    for prop in raw_propositions:
        # 移除编号（1., 2., 等）和项目符号
        cleaned = re.sub(r'^\s*(\d+\.|-|\*)\s*', '', prop).strip()
        if cleaned and len(cleaned) > 10:  # 简单过滤空或非常短的原子块
            clean_propositions.append(cleaned)

    return clean_propositions

# 原子块质量检查
def evaluate_proposition(proposition, original_text):
    """
    根据准确性、清晰度、完整性和简洁性评估原子块的质量。

    Args:
        proposition (str): 要评估的原子块
        original_text (str): 原始文本以供比较

    Returns:
        Dict: 每个评估维度的分数
    """
    # 系统提示，指示AI如何评估原子块
    system_prompt = """你是一位擅长评估从文本中提取原子块质量的专家。
    在以下标准（1-10分）下对给定原子块进行评分：

    - 准确性：原子块反映原始文本信息的程度
    - 清晰度：无需额外上下文即可理解原子块的难易程度
    - 完整性：原子块是否包含必要的细节（日期、限定词等）
    - 简洁性：原子块是否简洁且未丢失重要信息

    响应必须是包含每个标准数值分数的有效JSON格式：
    {"accuracy": X, "clarity": X, "completeness": X, "conciseness": X}
    """

    # 用户提示，包含原子块和原始文本
    user_prompt = f"""原子块：{proposition}

    原始文本：{original_text}

    请提供您的评分分数，格式为JSON。"""

    # 生成模型响应
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"},
        temperature=0
    )

    # 解析JSON响应
    try:
        scores = json.loads(response.choices[0].message.content.strip())
        return scores
    except json.JSONDecodeError:
        # 如果JSON解析失败，返回默认分数
        return {
            "accuracy": 5,
            "clarity": 5,
            "completeness": 5,
            "conciseness": 5
        }

# 完整的原子块处理管道

def process_document_into_propositions(pdf_path, chunk_size=800, chunk_overlap=100, quality_thresholds=None):
    """
    将文档处理为经过质量检查的原子块。

    Args:
        pdf_path (str): PDF文件的路径
        chunk_size (int): 每个块的大小（以字符为单位）
        chunk_overlap (int): 块之间的重叠部分（以字符为单位）
        quality_thresholds (Dict): 原子块质量阈值

    Returns:
        Tuple[List[Dict], List[Dict]]: 原始块和原子块块
    """
    # 设置默认质量阈值（如果未提供）
    if quality_thresholds is None:
        quality_thresholds = {
            "accuracy": 7,
            "clarity": 7,
            "completeness": 7,
            "conciseness": 7
        }

    # 从PDF文件中提取文本
    text = extract_text_from_pdf(pdf_path)

    # 从提取的文本创建块
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # 初始化一个列表以存储所有原子块
    all_propositions = []

    print("从块生成原子块...")
    for i, chunk in enumerate(chunks):
        print(f"处理块 {i+1}/{len(chunks)}...")
        # 为当前块生成原子块
        chunk_propositions = generate_propositions(chunk)
        print(f"生成了 {len(chunk_propositions)} 个原子块")

        # 处理每个生成的原子块
        for prop in chunk_propositions:
            proposition_data = {
                "text": prop,
                "source_chunk_id": chunk["chunk_id"],
                "source_text": chunk["text"]
            }
            all_propositions.append(proposition_data)

    # 评估生成原子块的质量
    print("\n评估原子块质量...")
    quality_propositions = []

    for i, prop in enumerate(all_propositions):
        if i % 10 == 0:  # 每10个原子块更新一次状态
            print(f"评估原子块 {i+1}/{len(all_propositions)}...")
            # 评估当前原子块的质量
            scores = evaluate_proposition(prop["text"], prop["source_text"])
            prop["quality_scores"] = scores

            # 检查原子块是否通过质量阈值
            passes_quality = True
            for metric, threshold in quality_thresholds.items():
                if scores.get(metric, 0) < threshold:
                    passes_quality = False
                    break

            if passes_quality:
                quality_propositions.append(prop)
            else:
                print(f"原子块未通过质量检查：{prop['text'][:50]}...")

    print(f"\n保留了 {len(quality_propositions)}/{len(all_propositions)} 个原子块，经过质量过滤")
    return chunks, quality_propositions

# 构建两种方法的向量存储

def build_vector_stores(chunks, propositions):
    """
    为块和原子块方法构建向量存储。

    Args:
        chunks (List[Dict]): 原始文档块
        propositions (List[Dict]): 经过质量过滤的原子块

    Returns:
        Tuple[SimpleVectorStore, SimpleVectorStore]: 块和原子块向量存储
    """
    # 为块创建向量存储
    chunk_store = SimpleVectorStore()

    # 提取块文本并创建嵌入
    chunk_texts = [chunk["text"] for chunk in chunks]
    print(f"为 {len(chunk_texts)} 个块创建嵌入...")
    chunk_embeddings = create_embeddings(chunk_texts)

    # 将块添加到向量存储中，带元数据
    chunk_metadata = [{"chunk_id": chunk["chunk_id"], "type": "chunk"} for chunk in chunks]
    chunk_store.add_items(chunk_texts, chunk_embeddings, chunk_metadata)

    # 为原子块创建向量存储
    prop_store = SimpleVectorStore()

    # 提取原子块文本并创建嵌入
    prop_texts = [prop["text"] for prop in propositions]
    print(f"为 {len(prop_texts)} 个原子块创建嵌入...")
    prop_embeddings = create_embeddings(prop_texts)

    # 将原子块添加到向量存储中，带元数据
    prop_metadata = [
        {
            "type": "proposition",
            "source_chunk_id": prop["source_chunk_id"],
            "quality_scores": prop["quality_scores"]
        }
        for prop in propositions
    ]
    prop_store.add_items(prop_texts, prop_embeddings, prop_metadata)

    return chunk_store, prop_store

# 查询和检索函数

def retrieve_from_store(query, vector_store, k=5):
    """
    从向量存储中检索与查询相关的内容。

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 要搜索的向量存储
        k (int): 要检索的结果数

    Returns:
        List[Dict]: 检索到的项目，带分数和元数据
    """
    # 创建查询嵌入
    query_embedding = create_embeddings(query)

    # 在向量存储中搜索最相似的前k个项目
    results = vector_store.similarity_search(query_embedding, k=k)

    return results

# 原子分块评估

# PDF文件路径
pdf_path = "data/AI_Information.pdf"

# 定义测试查询以评估原子分块
test_queries = [
    "什么是人工智能开发中的主要伦理问题？",
    # "可解释的人工智能如何提高对AI系统的信任？",
    # "开发公平AI系统的关键挑战是什么？",
    # "人类监督在AI安全中扮演什么角色？"
]

# 参考答案以更全面地评估和比较结果
# 这些提供了事实依据，用于衡量生成响应的质量
reference_answers = [
    "人工智能开发中的主要伦理问题包括偏见和公平性、隐私、透明度、问责制、安全性和潜在的有害应用风险。",
    # "可解释的人工智能通过使AI决策过程透明和可理解，帮助用户验证公平性、识别潜在偏见，并更好地理解AI的局限性。",
    # "开发公平AI系统的关键挑战包括解决数据偏见、确保训练数据的多样性、创建透明的算法、在不同上下文中定义公平性以及平衡相互竞争的公平标准。",
    # "人类监督在AI安全中起着关键作用，通过监控系统行为、验证输出、必要时进行干预、设定伦理边界以及确保AI系统在整个运行过程中与人类价值观和意图保持一致。"
]

# 运行评估
evaluation_results = run_proposition_chunking_evaluation(
    pdf_path=pdf_path,
    test_queries=test_queries,
    reference_answers=reference_answers
)

# 打印总体分析
print("\n\n=== 总体分析 ===")
print(evaluation_results["overall_analysis"])



####################################################################################################################################




# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


