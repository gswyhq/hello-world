
# 检索增强生成 （RAG） 是一种将信息检索与生成模型相结合的混合方法。它通过整合外部知识来增强语言模型的性能，从而提高准确性和事实正确性。

####################################################################################################################################
# 方法22：将文档转换为结构化的知识图谱
# 方法概述：
# 1、数据获取与准备：获取并清理原始文本。
# 2、信息提取：识别关键实体（组织、人物、金钱、日期）及其关系（例如，“收购”，“投资”）。
# 3、知识图谱构建：将提取的信息结构化为 三元组，形成知识图谱的节点和边。
# 4、知识图谱精炼（概念性）：使用嵌入表示知识图谱组件，并概念性地探索链接预测。
# 5、持久化与使用：存储、查询（SPARQL）和可视化知识图谱。 我们将利用大型语言模型（LLMs）进行复杂的自然语言处理任务，如细致的实体和关系提取，同时使用传统库如 spaCy 进行初步探索和 rdflib 进行知识图谱管理。

# 目录
# 端到端流程：大数据与知识图谱（参考书籍）
# 初始设置：导入和配置
# 初始化 LLM 客户端和 spaCy 模型
# 定义 RDF 命名空间
# 第一阶段：数据获取与准备
# 步骤 1.1：数据获取
# 执行数据获取
# 步骤 1.2：数据清洗与预处理
# 执行数据清洗
# 第二阶段：信息抽取
# 步骤 2.1：实体抽取（命名实体识别 - NER）
# 2.1.1：使用 spaCy 进行实体探索 - 函数定义
# 2.1.1：使用 spaCy 进行实体探索 - 绘图函数定义
# 2.1.1：使用 spaCy 进行实体探索 - 执行
# 通用 LLM 调用函数定义
# 2.1.2：使用 LLM 进行实体类型选择 - 执行
# LLM JSON 输出解析函数定义
# 2.1.3：使用 LLM 进行目标实体抽取 - 执行
# 步骤 2.2：关系抽取
# 第三阶段：知识图谱构建
# 步骤 3.1：实体消歧与链接（简化）- 标准化函数
# 执行实体标准化和 URI 生成
# 步骤 3.2：模式/本体对齐 - RDF 类型映射函数
# 模式/本体对齐 - RDF 谓词映射函数
# 模式/本体对齐 - 示例
# 步骤 3.3：三元组生成
# 第四阶段：使用嵌入进行知识图谱精炼
# 步骤 4.1：生成知识图谱嵌入 - 嵌入函数定义
# 生成知识图谱嵌入 - 执行
# 步骤 4.2：链接预测（知识发现 - 概念性）- 余弦相似度函数
# 链接预测（概念性）- 相似度计算示例
# 步骤 4.3：添加预测链接（可选 & 概念性）- 函数定义
# 添加预测链接（概念性）- 执行示例
# 第五阶段：持久化与使用
# 步骤 5.1：知识图谱存储 - 存储函数定义
# 知识图谱存储 - 执行
# 步骤 5.2：查询与分析 - SPARQL 执行函数
# SPARQL 查询与分析 - 执行示例
# 步骤 5.3：可视化（可选）- 可视化函数定义
# 知识图谱可视化 - 执行
# 结论与未来工作

# 导入必要的库
import os
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pandas as pd
import time

# NLP 和知识图谱库
import spacy
from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, XSD, SKOS  # 添加 SKOS 用于 altLabel

# OpenAI 客户端用于 LLM
from openai import OpenAI

# 可视化
from pyvis.network import Network

# Hugging Face 数据集库
from datasets import load_dataset

# 用于嵌入相似度
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

print("库已导入。")

# 初始化 LLM 客户端和 spaCy 模型
client = None  # 初始化客户端为 None

if NEBIUS_API_KEY != "YOUR_NEBIUS_API_KEY" and NEBIUS_BASE_URL != "YOUR_NEBIUS_BASE_URL" and TEXT_GEN_MODEL_NAME != "YOUR_TEXT_GENERATION_MODEL_NAME":
    try:
        client = OpenAI(
            base_url=NEBIUS_BASE_URL,
            api_key=NEBIUS_API_KEY
        )
        print(f"OpenAI 客户端已初始化，base_url: {NEBIUS_BASE_URL}，使用模型: {TEXT_GEN_MODEL_NAME}")
    except Exception as e:
        print(f"初始化 OpenAI 客户端时出错: {e}")
        client = None  # 如果初始化失败，确保客户端为 None
else:
    print("警告：OpenAI 客户端未完全配置。LLM 功能将被禁用。请设置 NEBIUS_API_KEY、NEBIUS_BASE_URL 和 TEXT_GEN_MODEL_NAME。")

nlp_spacy = None  # 初始化 nlp_spacy 为 None

try:
    nlp_spacy = spacy.load("en_core_web_sm")
    print("spaCy 模型 'en_core_web_sm' 已加载。")
except OSError:
    print("spaCy 模型 'en_core_web_sm' 未找到。正在下载...（这可能需要一些时间）")
    try:
        spacy.cli.download("en_core_web_sm")
        nlp_spacy = spacy.load("en_core_web_sm")
        print("spaCy 模型 'en_core_web_sm' 下载并加载成功。")
    except Exception as e:
        print(f"下载或加载 spaCy 模型失败: {e}")
        print("请尝试在终端运行: python -m spacy download en_core_web_sm 并重启内核。")
        nlp_spacy = None  # 如果加载失败，确保 nlp_spacy 为 None

# 定义 RDF 命名空间
EX = Namespace("http://example.org/kg/")
SCHEMA = Namespace("http://schema.org/")

print(f"自定义命名空间 EX 定义为: {EX}")
print(f"Schema.org 命名空间 SCHEMA 定义为: {SCHEMA}")

# 步骤 1.1：数据获取
def acquire_articles(dataset_name="cnn_dailymail", version="3.0.0", split='train', sample_size=1000, keyword_filter=None):
    """从指定的 Hugging Face 数据集中加载文章，可选地进行过滤并取样。"""
    print(f"尝试加载数据集: {dataset_name}（版本: {version}, 分割: '{split}'）...")
    try:
        full_dataset = load_dataset(dataset_name, version, split=split, streaming=False)
        print(f"成功加载数据集。分割中的总记录数: {len(full_dataset)}")
    except Exception as e:
        print(f"加载数据集 {dataset_name} 时出错: {e}")
        print("请确保数据集可用或你有网络连接。")
        return []  # 失败时返回空列表

    raw_articles_list = []
    if keyword_filter:
        print(f"正在过滤包含以下关键词的文章: {keyword_filter}...")
        count = 0
        iteration_limit = min(len(full_dataset), sample_size * 20)  # 最多检查 20 倍样本大小的文章
        for i in tqdm(range(iteration_limit), desc="过滤文章"):
            record = full_dataset[i]
            if any(keyword.lower() in record['article'].lower() for keyword in keyword_filter):
                raw_articles_list.append(record)
                count += 1
            if count >= sample_size:
                print(f"在 {i+1} 条记录中找到 {sample_size} 条符合过滤条件的文章。")
                break
        if not raw_articles_list:
            print(f"警告：在前 {iteration_limit} 条记录中未找到包含关键词 {keyword_filter} 的文章。返回空列表。")
            return []
        raw_articles_list = raw_articles_list[:sample_size]
    else:
        print(f"未使用关键词过滤，正在取前 {sample_size} 条文章。")
        actual_sample_size = min(sample_size, len(full_dataset))
        raw_articles_list = list(full_dataset.select(range(actual_sample_size)))

    print(f"已获取 {len(raw_articles_list)} 条文章。")
    return raw_articles_list

print("函数 'acquire_articles' 已定义。")

# 执行数据获取
# 定义与科技公司收购相关的关键词
ACQUISITION_KEYWORDS = ["acquire", "acquisition", "merger", "buyout", "purchased by", "acquired by", "takeover"]
TECH_KEYWORDS = ["technology", "software", "startup", "app", "platform", "digital", "AI", "cloud"]

# 本演示中主要过滤收购相关的关键词
FILTER_KEYWORDS = ACQUISITION_KEYWORDS

SAMPLE_SIZE = 10  # 保持很小，以便快速进行 LLM 处理

# 初始化 raw_data_sample 为空列表
raw_data_sample = []
raw_data_sample = acquire_articles(sample_size=SAMPLE_SIZE, keyword_filter=FILTER_KEYWORDS)

if raw_data_sample:
    print(f"\n示例原始获取的文章（ID: {raw_data_sample[0]['id']}）:")
    print(raw_data_sample[0]['article'][:500] + "...")
    print(f"\n每条记录的字段数量: {len(raw_data_sample[0].keys())}")
    print(f"字段: {list(raw_data_sample[0].keys())}")
else:
    print("未获取到任何文章。涉及文章处理的后续步骤可能会被跳过或无输出。")

# 步骤 1.2：数据清洗与预处理
def clean_article_text(raw_text):
    """使用正则表达式清洗新闻文章的原始文本。"""
    text = raw_text

    # 移除 (CNN) 风格的前缀
    text = re.sub(r'^$CNN$\s*(--)?\s*', '', text)
    # 移除常见的署名和发布/更新行（模式可能需要根据具体数据集进行调整）
    text = re.sub(r'By .*? for Dailymail\.com.*?Published:.*?Updated:.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'PUBLISHED:.*?BST,.*?UPDATED:.*?BST,.*', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'Last updated at.*on.*', '', text, flags=re.IGNORECASE)
    # 移除 URL
    text = re.sub(r'https?://\S+|www\.\S+', '[URL]', text)
    # 移除 HTML 标签
    text = re.sub(r'<.*?>', '', text)
    # 移除电子邮件地址
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    # 标准化空白字符：将换行符、制表符替换为一个空格，再将多个空格替换为一个空格
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text).strip()
    # 可选：如果 LLM 有处理问题，可以转义引号，但通常高质量模型不需要
    # text = text.replace('"', '\\"').replace("'", "\'")
    return text

print("函数 'clean_article_text' 已定义。")

# 执行数据清洗
cleaned_articles = []  # 初始化为空列表

if raw_data_sample:  # 仅在 raw_data_sample 不为空时执行
    print(f"正在清洗 {len(raw_data_sample)} 篇获取的文章...")
    for record in tqdm(raw_data_sample, desc="清洗文章"):
        cleaned_text_content = clean_article_text(record['article'])
        cleaned_articles.append({
            "id": record['id'],
            "original_text": record['article'],  # 保留原始文本以供参考
            "cleaned_text": cleaned_text_content,
            "summary": record.get('highlights', '')  # CNN/DM 数据集中 'highlights' 是摘要
        })
    print(f"清洗完成。总共清洗了 {len(cleaned_articles)} 篇文章。")
    if cleaned_articles:  # 检查处理后列表是否为空
        print(f"\n清洗后的文章示例（ID: {cleaned_articles[0]['id']}）:")
        print(cleaned_articles[0]['cleaned_text'][:500] + "...")
else:
    print("上一步未获取到原始文章，因此跳过清洗步骤。")

# 确保 cleaned_articles 始终被定义，即使为空
if 'cleaned_articles' not in globals():
    cleaned_articles = []
    print("由于之前未创建，已将 'cleaned_articles' 初始化为空列表。")

# 第二阶段：信息抽取
# 步骤 2.1：实体抽取（命名实体识别 - NER）
# 1、spaCy 探索性 NER：使用 spaCy 的预训练模型快速获取文章中常见的实体类型。这有助于我们了解数据集中实体的大致分布。
# 2、LLM 驱动的实体类型选择：基于 spaCy 的输出和我们的具体目标（如技术收购），我们将提示一个 LLM 来建议一组更聚焦、更相关的实体类型。
# 3、LLM 面向目标的 NER：使用 LLM 和精炼的实体类型列表对文章进行 NER，以提高在特定领域中的准确性和相关性。LLM 在此过程中非常强大，尤其是当使用精心设计的提示时。

# 2.1.1：使用 spaCy 进行实体探索 - 函数定义
def get_spacy_entity_counts(articles_data, text_field='cleaned_text', sample_size_spacy=50):
    """使用 spaCy 处理文章样本并统计实体标签。"""
    if not nlp_spacy:
        print("spaCy 模型未加载。跳过 spaCy 实体计数。")
        return Counter()
    if not articles_data:
        print("未提供文章数据给 spaCy 进行实体计数。跳过。")
        return Counter()

    label_counter = Counter()
    # 为快速分析，处理较小的样本
    sample_to_process = articles_data[:min(len(articles_data), sample_size_spacy)]

    print(f"正在使用 spaCy 处理 {len(sample_to_process)} 篇文章以统计实体标签...")
    for article in tqdm(sample_to_process, desc="spaCy NER for counts"):
        doc = nlp_spacy(article[text_field])
        for ent in doc.ents:
            label_counter[ent.label_] += 1
    return label_counter


print("函数 'get_spacy_entity_counts' 已定义。")

# 2.1.1：使用 spaCy 进行实体探索 - 可视化函数定义
def plot_entity_distribution(label_counter_to_plot):
    """从 Counter 对象绘制实体标签的分布图。"""
    if not label_counter_to_plot:
        print("没有实体计数可供绘制。")
        return

    # 获取最常见的 15 个实体类型，或全部（如果少于 15 个）
    top_items = label_counter_to_plot.most_common(min(15, len(label_counter_to_plot)))
    if not top_items:  # 处理计数器非空但 most_common(0) 等边缘情况
        print("从实体计数中没有可绘制的项目。")
        return

    labels, counts = zip(*top_items)

    plt.figure(figsize=(12, 7))
    plt.bar(labels, counts, color='skyblue')
    plt.title("最常见的实体类型分布（通过 spaCy）")
    plt.ylabel("频率")
    plt.xlabel("实体标签")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()  # 调整布局以确保所有内容显示完整
    plt.show()


print("函数 'plot_entity_distribution' 已定义。")

# 2.1.1：使用 spaCy 进行实体探索 - 执行
spacy_entity_counts = Counter()  # 初始化为空计数器

if cleaned_articles and nlp_spacy:
    # 为保持快速，使用较小的固定样本量进行 spaCy 分析
    spacy_analysis_sample_size = min(len(cleaned_articles), 20)
    print(f"正在对 {spacy_analysis_sample_size} 篇清洗后的文章运行 spaCy NER...")
    spacy_entity_counts = get_spacy_entity_counts(cleaned_articles, sample_size_spacy=spacy_analysis_sample_size)

    if spacy_entity_counts:
        print("\nspaCy 实体计数（来自样本）:")
        for label, count in spacy_entity_counts.most_common():
            print(f"  {label}: {count}")
        plot_entity_distribution(spacy_entity_counts)
    else:
        print("spaCy NER 未从样本中返回任何实体计数。")
else:
    print("跳过 spaCy 实体分析：没有清洗后的文章可用或 spaCy 模型未加载。")

# 通用 LLM 调用函数定义
def call_llm_for_response(system_prompt, user_prompt, model_to_use=TEXT_GEN_MODEL_NAME, temperature=0.2):
    """
    接收系统提示（LLM 的指令）和用户提示（具体的输入/查询）。
    向配置好的 LLM 接口发起 API 请求。
    从 LLM 的响应中提取文本内容。
    如果 LLM 客户端未初始化或 API 调用失败，进行基本的错误处理。
    """
    if not client:
        print("LLM 客户端未初始化。跳过 LLM 调用。")
        return "LLM_CLIENT_NOT_INITIALIZED"

    try:
        print(f"\n调用 LLM（模型: {model_to_use}, 温度: {temperature}）...")
        # 用于调试，取消注释可查看提示（可能很长）
        # print(f"系统提示（前200字符）: {system_prompt[:200]}...")
        # print(f"用户提示（前200字符）: {user_prompt[:200]}...")

        response = client.chat.completions.create(
            model=model_to_use,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature  # 降低温度以获得更集中/确定的输出
        )
        content = response.choices[0].message.content.strip()
        print("LLM 响应已接收。")
        return content
    except Exception as e:
        print(f"调用 LLM 时出错: {e}")
        return f"LLM_ERROR: {str(e)}"


print("函数 'call_llm_for_response' 已定义。")

# 2.1.2：使用 LLM 进行实体类型选择 - 执行
ENTITY_TYPE_SELECTION_SYSTEM_PROMPT = (
    "你是一个专注于科技新闻分析的知识图谱构建专家。"
    "你将获得一个从新闻文章中提取的实体标签及其频率列表。"
    "你的任务是从中选择并返回一个**最相关**的实体标签列表，用于构建一个专注于**科技公司收购**的知识图谱。"
    "优先考虑收购方和被收购方（ORG）、交易金额（MONEY）、公告/完成日期（DATE）、关键人物（CEO、创始人等，PERSON），以及相关技术产品/服务或行业。"
    "对于你输出列表中的每一个实体标签，请提供一个简洁的括号说明或清晰的示例。"
    "示例：ORG（参与收购的公司，如 Google、Microsoft），MONEY（交易金额或投资，如 10 亿美元），DATE（收购公告或完成日期，如 2023 年 7 月 26 日）。"
    "输出必须**仅**为一个由标签和括号说明组成的逗号分隔列表。"
    "不要包含任何介绍性语句、问候语、总结或其他格式外的文本。"
)

llm_selected_entity_types_str = ""  # 初始化
DEFAULT_ENTITY_TYPES_STR = "ORG（收购方或被收购公司，如 TechCorp），PERSON（关键高管，如 CEO），MONEY（收购价格，如 5 亿美元），DATE（收购公告日期），PRODUCT（关键产品/服务），GPE（公司所在地，如硅谷）"

if spacy_entity_counts and client:  # 如果有 spaCy 的统计结果且 LLM 客户端可用
    # 从 spaCy 实体统计中生成提示字符串
    spacy_labels_for_prompt = ", ".join(
        [f"{label}（频率: {count})" for label, count in spacy_entity_counts.most_common()])
    user_prompt_for_types = f"从以下新闻文章中发现的实体标签及其频率中: [{spacy_labels_for_prompt}]。请根据指示选择并格式化最相关的实体类型，用于构建科技公司收购的知识图谱。"

    llm_selected_entity_types_str = call_llm_for_response(ENTITY_TYPE_SELECTION_SYSTEM_PROMPT, user_prompt_for_types)

    if "LLM_CLIENT_NOT_INITIALIZED" in llm_selected_entity_types_str or "LLM_ERROR" in llm_selected_entity_types_str or not llm_selected_entity_types_str.strip():
        print("\nLLM 实体类型选择失败或返回为空。使用默认实体类型。")
        llm_selected_entity_types_str = DEFAULT_ENTITY_TYPES_STR
    else:
        print("\nLLM 建议的科技收购知识图谱实体类型：")
        # 后处理：确保输出为干净的列表，即使 LLM 添加了额外内容
        # 这是一个简单启发式方法，对于不合规的 LLM 可能需要更健壮的解析
        if not re.match(r"^([A-Z_]+（.*?）)(, [A-Z_]+（.*?）)*$", llm_selected_entity_types_str.strip()):
            print(f"警告：LLM 的实体类型输出可能不符合预期的严格格式。原始输出: '{llm_selected_entity_types_str}'")
            # 尝试简单清理：提取看起来像实体列表的最长行
            lines = llm_selected_entity_types_str.strip().split('\n')
            best_line = ""
            for line in lines:
                if '(' in line and ')' in line and len(line) > len(best_line):
                    best_line = line
            if best_line:
                llm_selected_entity_types_str = best_line
                print(f"尝试清理后: '{llm_selected_entity_types_str}'")
            else:
                print("清理失败，回退到默认实体类型。")
                llm_selected_entity_types_str = DEFAULT_ENTITY_TYPES_STR
else:
    print("\n跳过 LLM 实体类型选择（spaCy 统计不可用或 LLM 客户端未初始化）。使用默认实体类型。")
    # 如果跳过 NER，确保 articles_with_entities 中有空的实体列表
    if cleaned_articles:  # 只有在有清理后的文章时才执行
        num_articles_to_fallback = min(len(cleaned_articles), MAX_ARTICLES_FOR_LLM_NER)
        for article_dict_fallback in cleaned_articles[:num_articles_to_fallback]:
            fallback_data = article_dict_fallback.copy()
            fallback_data['llm_entities'] = []
            articles_with_entities.append(fallback_data)
        print(f"已填充 'articles_with_entities'，共 {len(articles_with_entities)} 条目，'llm_entities' 列表为空。")

print(f"\n最终用于 NER 的实体类型列表: {llm_selected_entity_types_str}")

# LLM JSON 输出解析函数定义
def parse_llm_json_output(llm_output_str):
    """
    解析 LLM 的 JSON 输出，处理可能的 Markdown 代码块和常见问题。
    """
    if not llm_output_str or "LLM_CLIENT_NOT_INITIALIZED" in llm_output_str or "LLM_ERROR" in llm_output_str:
        print("无法解析 LLM 输出：LLM 未运行、出错或输出为空。")
        return []  # 返回空列表

    # 尝试从 Markdown 代码块中提取 JSON
    match = re.search(r'```json\s*([\s\S]*?)\s*```', llm_output_str, re.IGNORECASE)
    if match:
        json_str = match.group(1).strip()
    else:
        # 如果没有 Markdown 块，假设整个字符串是 JSON（或需要清理）
        # LLM 有时会在 JSON 列表前添加介绍性文本。尝试找到列表的开始。
        list_start_index = llm_output_str.find('[')
        list_end_index = llm_output_str.rfind(']')
        if list_start_index != -1 and list_end_index != -1 and list_start_index < list_end_index:
            json_str = llm_output_str[list_start_index:list_end_index+1].strip()
        else:
            json_str = llm_output_str.strip()  # 回退到整个字符串

    try:
        parsed_data = json.loads(json_str)
        if isinstance(parsed_data, list):
            return parsed_data
        else:
            print(f"警告：LLM 输出是有效的 JSON 但不是列表（类型: {type(parsed_data)}）。返回空列表。")
            print(f"问题 JSON 字符串（或部分）: {json_str[:200]}...")
            return []
    except json.JSONDecodeError as e:
        print(f"从 LLM 输出中解析 JSON 时出错: {e}")
        print(f"问题 JSON 字符串（或部分）: {json_str[:500]}...")
        # 可选：如果标准解析失败，使用更激进的正则表达式回退
        # 该方法有风险，可能会提取部分 JSON。请谨慎使用。
        # entities_found = []
        # for match_obj in re.finditer(r'\{\s*"text":\s*".*?",\s*"type":\s*".*?"\s*\}', json_str):
        #     try:
        #         entities_found.append(json.loads(match_obj.group(0)))
        #     except json.JSONDecodeError:
        #         continue  # 跳过格式错误的单个对象
        # if entities_found:
        #     print(f"警告：由于 JSON 错误，使用激进正则表达式恢复了 {len(entities_found)} 个实体。")
        #     return entities_found
        return []
    except Exception as e:
        print(f"LLM JSON 输出解析过程中发生意外错误: {e}")
        return []

print("函数 'parse_llm_json_output' 已定义。")

# 2.1.3：使用 LLM 进行目标实体抽取 - 执行
LLM_NER_SYSTEM_PROMPT_TEMPLATE = (
    "你是一个专注于识别**科技公司收购**信息的专家命名实体识别系统。"
    "从提供的新闻文章中识别并提取实体。"
    "需要关注的实体类型是：{entity_types_list_str}。"
    "确保提取的 'text' 是文章中的**精确片段**。"
    "输出**仅**为一个有效的 JSON 列表，每个对象包含 'text'（提取的实体字符串）和 'type'（实体类型，如 ORG、PERSON、MONEY）键。"
    "示例：[{'text': '联合国', 'type': 'ORG'}, {'text': '奥巴马', 'type': 'PERSON'}, {'text': 'iPhone 15', 'type': 'PRODUCT'}]"
)

articles_with_entities = []  # 初始化
MAX_ARTICLES_FOR_LLM_NER = 3  # 本演示中处理少量文章

if cleaned_articles and client and llm_selected_entity_types_str and "LLM_" not in llm_selected_entity_types_str:
    # 准备系统提示，使用动态选择的实体类型
    ner_system_prompt = LLM_NER_SYSTEM_PROMPT_TEMPLATE.format(entity_types_list_str=llm_selected_entity_types_str)

    # 确定要处理的文章数量
    num_articles_to_process_ner = min(len(cleaned_articles), MAX_ARTICLES_FOR_LLM_NER)
    print(f"开始 LLM NER，处理 {num_articles_to_process_ner} 篇文章...")

    for i, article_dict in enumerate(tqdm(cleaned_articles[:num_articles_to_process_ner], desc="LLM NER 处理")):
        print(f"\n处理文章 ID: {article_dict['id']} 的 NER（{i + 1}/{num_articles_to_process_ner}）...")

        # 如果文章文本过长，进行截断（例如 > 12000 字符）
        max_text_chars = 12000  # 约 3000 词，对多数模型是安全的
        article_text_for_llm = article_dict['cleaned_text'][:max_text_chars]
        if len(article_dict['cleaned_text']) > max_text_chars:
            print(
                f"  警告：文章文本从 {len(article_dict['cleaned_text'])} 字符截断为 {max_text_chars} 字符以适应 LLM NER。")

        llm_ner_raw_output = call_llm_for_response(ner_system_prompt, article_text_for_llm)
        extracted_entities_list = parse_llm_json_output(llm_ner_raw_output)

        # 存储结果
        current_article_data = article_dict.copy()  # 复制以避免修改原始列表项
        current_article_data['llm_entities'] = extracted_entities_list
        articles_with_entities.append(current_article_data)

        print(f"  提取了 {len(extracted_entities_list)} 个实体，文章 ID: {article_dict['id']}")
        if extracted_entities_list:
            # 打印部分实体示例，最多 3 个
            print(f"  示例实体: {json.dumps(extracted_entities_list[:min(3, len(extracted_entities_list))], indent=2)}")

        if i < num_articles_to_process_ner - 1:  # 最后一篇文章不等待
            time.sleep(1)  # 小延迟，礼貌对待 API

    if articles_with_entities:
        print(f"\n完成 LLM NER。已处理 {len(articles_with_entities)} 篇文章并存储实体。")
else:
    print("跳过 LLM NER：前提条件（清理后的文章、LLM 客户端或有效的实体类型字符串）未满足。")
    # 如果跳过 NER，确保 articles_with_entities 中有空的实体列表
    if cleaned_articles:  # 只有在有清理后的文章时才执行
        num_articles_to_fallback = min(len(cleaned_articles), MAX_ARTICLES_FOR_LLM_NER)
        for article_dict_fallback in cleaned_articles[:num_articles_to_fallback]:
            fallback_data = article_dict_fallback.copy()
            fallback_data['llm_entities'] = []
            articles_with_entities.append(fallback_data)
        print(f"已填充 'articles_with_entities'，共 {len(articles_with_entities)} 条目，'llm_entities' 列表为空。")

# 确保 articles_with_entities 已定义
if 'articles_with_entities' not in globals():
    articles_with_entities = []
    print("已将 'articles_with_entities' 初始化为空列表。")

# 第2.2步：关系抽取（Relationship Extraction）
# 示例系统提示模板
RELATIONSHIP_EXTRACTION_SYSTEM_PROMPT_TEMPLATE = (
    "你是一个从文本中抽取实体之间关系的专家系统，特别关注**科技公司收购**。"
    "示例：[{'subject_text': 'Innovatech Ltd.', 'subject_type': 'ORG', 'predicate': 'ACQUIRED', 'object_text': 'Global Solutions Inc.', 'object_type': 'ORG'}, {'subject_text': 'Global Solutions Inc.', 'subject_type': 'ORG', 'predicate': 'HAS_PRICE', 'object_text': '$250M', 'object_type': 'MONEY'}]"
    "如果在提供的实体之间没有找到指定类型的相关关系，请输出一个空的JSON列表[]。不要输出任何其他文本或解释。"
)

# 第3阶段：知识图谱构建（Knowledge Graph Construction）
# 第3.1步：实体消歧与链接（简化版） - 标准化函数
def normalize_entity_text(text_to_normalize, entity_type_str):
    """标准化实体文本以实现更好的链接（简化版本）
    去除首尾空格。
    对于 ORG 类型实体，尝试去除常见的公司后缀（如 "Inc."、"Ltd."、"Corp."），将类似 "Example Corp" 和 "Example Corporation" 的变体统一为 "Example"。
    （可选）考虑小写化，但有时会丢失重要区分（如 "IT" 作为代词 vs. 行业术语）。
    标准化后的文本将用于为每个唯一实体生成唯一URI。
    """
    normalized_t = text_to_normalize.strip()

    if entity_type_str == 'ORG':
        # 常见的公司后缀列表
        suffixes = [
            'Inc.', 'Incorporated', 'Ltd.', 'Limited', 'LLC', 'L.L.C.',
            'Corp.', 'Corporation', 'PLC', 'Public Limited Company',
            'GmbH', 'AG', 'S.A.', 'S.A.S.', 'B.V.', 'Pty Ltd', 'Co.', 'Company',
            'Solutions', 'Technologies', 'Systems', 'Group', 'Holdings'
        ]
        # 按长度降序排序，优先去除较长的后缀
        suffixes.sort(key=len, reverse=True)
        for suffix in suffixes:
            if normalized_t.lower().endswith(suffix.lower()):
                suffix_start_index = normalized_t.lower().rfind(suffix.lower())
                normalized_t = normalized_t[:suffix_start_index].strip()
                break

        # 去除后缀后可能残留的逗号或句号
        normalized_t = re.sub(r'[-,.]*$', '', normalized_t).strip()

    # 通用清理：去除NER可能误识别的's
    if normalized_t.endswith("'s") or normalized_t.endswith("s'"):
        normalized_t = normalized_t[:-2].strip()

    # 考虑是否小写化。对于ORG类型可能可以接受，对于PERSON类型则不太合适。
    # 在本演示中，我们保留后缀去除后的原始大小写。
    # normalized_t = normalized_t.lower()  # 如需激进标准化，可取消注释

    return normalized_t.strip() if normalized_t else text_to_normalize

# 执行实体标准化与URI生成
articles_with_normalized_entities = []  # 初始化
unique_entities_map = {}  # 映射 (normalized_text, type) -> URI，确保URI一致性

if articles_with_relations:  # 仅在有关系的文章上执行
    print("Normalizing entities and preparing for triple generation...")
    for article_data_rel in tqdm(articles_with_relations, desc="Normalizing Entities & URI Gen"):
        current_article_normalized_ents = []
        if 'llm_entities' in article_data_rel and isinstance(article_data_rel['llm_entities'], list):
            for entity_dict in article_data_rel['llm_entities']:
                if not (isinstance(entity_dict, dict) and 'text' in entity_dict and 'type' in entity_dict):
                    print(f"  跳过格式错误的实体对象: {str(entity_dict)[:100]} 在文章 {article_data_rel['id']}")
                    continue

                original_entity_text = entity_dict['text']
                entity_type_val = entity_dict['type']
                simple_entity_type = entity_type_val.split(' ')[0].upper()
                entity_dict['simple_type'] = simple_entity_type

                normalized_entity_text = normalize_entity_text(original_entity_text, simple_entity_type)
                if not normalized_entity_text:
                    normalized_entity_text = original_entity_text

                entity_map_key = (normalized_entity_text, simple_entity_type)
                if entity_map_key not in unique_entities_map:
                    safe_uri_text_part = re.sub(r'[^a-zA-Z0-9_\-]', '_', normalized_entity_text.replace(' ', '_'))
                    safe_uri_text_part = safe_uri_text_part[:80]
                    if not safe_uri_text_part:
                        import hashlib
                        safe_uri_text_part = f"entity_{hashlib.md5(normalized_entity_text.encode()).hexdigest()[:8]}"
                    unique_entities_map[entity_map_key] = EX[f"{safe_uri_text_part}_{simple_entity_type}"]

                entity_dict_copy = entity_dict.copy()
                entity_dict_copy['normalized_text'] = normalized_entity_text
                entity_dict_copy['uri'] = unique_entities_map[entity_map_key]
                current_article_normalized_ents.append(entity_dict_copy)

        article_data_output_norm = article_data_rel.copy()
        article_data_output_norm['normalized_entities'] = current_article_normalized_ents
        articles_with_normalized_entities.append(article_data_output_norm)

    if articles_with_normalized_entities and articles_with_normalized_entities[0].get('normalized_entities'):
        print("\n第一个文章的标准化实体示例（前3个）：")
        for ent_example in articles_with_normalized_entities[0]['normalized_entities'][:3]:
            print(f"  原始文本: '{ent_example['text']}'，类型: {ent_example['type']}（简化类型: {ent_example['simple_type']}），标准化文本: '{ent_example['normalized_text']}'，URI: <{ent_example['uri']}>")
    print(f"\n已处理 {len(articles_with_normalized_entities)} 篇文章的实体标准化和URI生成。")
    print(f"总共创建了 {len(unique_entities_map)} 个唯一实体URI。")
else:
    print("跳过实体标准化和URI生成：没有可用的关系文章。")
    if articles_with_entities:
        for article_data_fallback_re in articles_with_entities:
            fallback_data_re = article_data_fallback_re.copy()
            fallback_data_re['llm_relations'] = []
            articles_with_relations.append(fallback_data_re)
        print(f"已填充 'articles_with_relations'，共 {len(articles_with_relations)} 个条目，关系列表为空。")

# 步骤 3.2: 模式/本体对齐 - RDF 类型映射函数
def get_rdf_type_for_entity(simple_entity_type_str):
    """将简单实体类型字符串（如'ORG'）映射到RDF类。"""
    type_mapping = {
        'ORG': SCHEMA.Organization,
        'PERSON': SCHEMA.Person,
        'MONEY': SCHEMA.PriceSpecification,  # 或自定义 EX.MonetaryValue
        'DATE': SCHEMA.Date,  # 注意：schema.org/Date 是数据类型，事件日期可用 schema.org/Event
        'PRODUCT': SCHEMA.Product,
        'GPE': SCHEMA.Place,    # 地理政治实体
        'LOC': SCHEMA.Place,    # 通用位置
        'EVENT': SCHEMA.Event,
        'CARDINAL': RDF.Statement,  # 若上下文明确可更具体，通常作为字面量
        'FAC': SCHEMA.Place  # 设施
    }
    return type_mapping.get(simple_entity_type_str.upper(), EX[simple_entity_type_str.upper()])  # 回退到自定义类型

print("函数 'get_rdf_type_for_entity' 已定义。")

# 模式/本体对齐 - RDF 谓词映射函数
def get_rdf_predicate(predicate_str_from_llm):
    """将LLM抽取的谓词字符串映射到EX命名空间的RDF属性。"""
    sanitized_predicate = predicate_str_from_llm.strip().replace(" ", "_").upper()
    return EX[sanitized_predicate]

print("函数 'get_rdf_predicate' 已定义。")

# 通过示例展示实体类型和关系谓词如何映射到RDF术语，验证映射逻辑。
print("模式对齐函数已就绪。示例映射：")
example_entity_type = 'ORG'
example_predicate_str = 'ACQUIRED'
print(f"  实体类型 '{example_entity_type}' 映射到 RDF 类: <{get_rdf_type_for_entity(example_entity_type)}>")
print(f"  谓词 '{example_predicate_str}' 映射到 RDF 属性: <{get_rdf_predicate(example_predicate_str)}>")

example_entity_type_2 = 'MONEY'
example_predicate_str_2 = 'HAS_PRICE'
print(f"  实体类型 '{example_entity_type_2}' 映射到 RDF 类: <{get_rdf_type_for_entity(example_entity_type_2)}>")
print(f"  谓词 '{example_predicate_str_2}' 映射到 RDF 属性: <{get_rdf_predicate(example_predicate_str_2)}>")

# 步骤 3.3: 三元组生成
kg = Graph()  # 初始化空RDF图
kg.bind("ex", EX)
kg.bind("schema", SCHEMA)
kg.bind("rdf", RDF)
kg.bind("rdfs", RDFS)
kg.bind("xsd", XSD)
kg.bind("skos", SKOS)

triples_generated_count = 0

if articles_with_normalized_entities:
    print(f"正在为 {len(articles_with_normalized_entities)} 篇文章生成RDF三元组...")
    for article_data_final in tqdm(articles_with_normalized_entities, desc="生成三元组"):
        # 创建文章URI并添加元数据
        article_uri = EX[f"article_{article_data_final['id'].replace('-', '_')}"]
        kg.add((article_uri, RDF.type, SCHEMA.Article))
        kg.add((article_uri, SCHEMA.headline, Literal(article_data_final.get('summary', article_data_final['id']))))
        triples_generated_count += 2

        # 实体处理与关系解析
        entity_text_to_uri_map_current_article = {}

        # Add entity triples
        for entity_obj in article_data_final.get('normalized_entities', []):
            entity_uri_val = entity_obj['uri']  # This is the canonical URI from unique_entities_map
            rdf_entity_type_val = get_rdf_type_for_entity(entity_obj['simple_type'])
            normalized_label = entity_obj['normalized_text']
            original_label = entity_obj['text']

            kg.add((entity_uri_val, RDF.type, rdf_entity_type_val))
            kg.add((entity_uri_val, RDFS.label, Literal(normalized_label, lang='en')))
            triples_generated_count += 2
            if normalized_label != original_label:
                kg.add((entity_uri_val, SKOS.altLabel, Literal(original_label, lang='en')))
                triples_generated_count += 1

            # Link article to mentioned entities
            kg.add((article_uri, SCHEMA.mentions, entity_uri_val))
            triples_generated_count += 1

            # Populate the local map for resolving relations within this article
            entity_text_to_uri_map_current_article[original_label] = entity_uri_val

        # Add relation triples
        for relation_obj in article_data_final.get('llm_relations', []):
            subject_orig_text = relation_obj.get('subject_text')
            object_orig_text = relation_obj.get('object_text')
            predicate_str = relation_obj.get('predicate')

            # Resolve subject and object texts to their canonical URIs using the article-specific map
            subject_resolved_uri = entity_text_to_uri_map_current_article.get(subject_orig_text)
            object_resolved_uri = entity_text_to_uri_map_current_article.get(object_orig_text)

            if subject_resolved_uri and object_resolved_uri and predicate_str:
                predicate_rdf_prop = get_rdf_predicate(predicate_str)
                kg.add((subject_resolved_uri, predicate_rdf_prop, object_resolved_uri))
                triples_generated_count += 1
            else:
                if not subject_resolved_uri:
                    print(
                        f"  Warning: Could not find URI for subject '{subject_orig_text}' in article {article_data_final['id']}. Relation skipped: {relation_obj}")
                if not object_resolved_uri:
                    print(
                        f"  Warning: Could not find URI for object '{object_orig_text}' in article {article_data_final['id']}. Relation skipped: {relation_obj}")

    print(f"\n三元组生成完成。共添加约 {triples_generated_count} 个候选三元组。")
    print(f"图谱中实际三元组总数: {len(kg)}")
    if len(kg) > 0:
        print("\n前5条三元组示例（N3格式）:")
        for i, (s, p, o) in enumerate(kg):
            print(f"  {s.n3(kg.namespace_manager)} {p.n3(kg.namespace_manager)} {o.n3(kg.namespace_manager)}.")
            if i >= 4: break
else:
    print("跳过三元组生成：无可用数据。")

# 第四阶段：使用嵌入向量进行知识图谱优化
# 步骤 4.1：生成KG嵌入向量 – 嵌入函数定义
def get_embeddings_for_texts(texts_list, embedding_model_name=EMBEDDING_MODEL_NAME):
    """使用指定模型通过LLM客户端为文本列表获取嵌入向量。"""
    if not client:
        print("LLM客户端未初始化。跳过嵌入生成。")
        return {text: [] for text in texts_list}  # 返回空嵌入的字典

    if not texts_list:
        print("未提供用于生成嵌入的文本。")
        return {}

    embeddings_map_dict = {}
    print(f"正在为 {len(texts_list)} 个唯一文本获取嵌入向量，使用模型 '{embedding_model_name}'...")

    # 为提高效率并遵守API限制，按批次处理文本
    # 某些API可以直接处理列表输入，某些可能需要分批处理
    # 假设当前客户端可以处理列表输入，否则需要循环处理

    # 检查输入是否为字符串列表
    if not all(isinstance(text, str) for text in texts_list):
        print("错误：输入 'texts_list' 必须是字符串列表。")
        return {text: [] for text in texts_list if isinstance(text, str)}  # 尽量保留有效部分

    # 移除空字符串以避免API错误
    valid_texts_list = [text for text in texts_list if text.strip()]
    if not valid_texts_list:
        print("没有有效的（非空）文本用于嵌入。")
        return {}

    try:
        # 假设客户端的 embeddings.create 方法可以处理列表输入
        response = client.embeddings.create(
            model=embedding_model_name,
            input=valid_texts_list  # 传入有效文本列表
        )
        # 响应数据应为嵌入对象列表，与输入顺序一致
        for i, data_item in enumerate(response.data):
            embeddings_map_dict[valid_texts_list[i]] = data_item.embedding

        print(f"已为 {len(embeddings_map_dict)} 个文本获取嵌入向量。")
        # 对于空文本或失败的文本，添加空嵌入（如果调用者需要）
        for text in texts_list:
            if text not in embeddings_map_dict:
                embeddings_map_dict[text] = []
        return embeddings_map_dict

    except Exception as e:
        print(f"获取嵌入向量时出错（批量尝试）：{e}")
        print("批量失败，正在回退到逐个请求模式...")
        # 如果批量处理失败或不被支持，回退到逐个请求
        embeddings_map_dict_fallback = {}
        for text_input_item in tqdm(valid_texts_list, desc="生成嵌入向量（回退模式）"):
            try:
                response_item = client.embeddings.create(
                    model=embedding_model_name,
                    input=text_input_item
                )
                embeddings_map_dict_fallback[text_input_item] = response_item.data[0].embedding
                if len(valid_texts_list) > 10:  # 如果处理大量项目，添加小延迟
                    time.sleep(0.1)
            except Exception as e_item:
                print(f"  获取文本 '{text_input_item[:50]}...' 的嵌入向量时出错：{e_item}")
                embeddings_map_dict_fallback[text_input_item] = []  # 出错时存储空列表

        # 对于空文本或失败的文本，添加空嵌入（如果调用者需要）
        for text in texts_list:
            if text not in embeddings_map_dict_fallback:
                embeddings_map_dict_fallback[text] = []
        return embeddings_map_dict_fallback

print("函数 'get_embeddings_for_texts' 已定义。")

# 生成KG嵌入向量 – 执行
entity_embeddings = {}  # 初始化：映射实体URI -> 嵌入向量

if unique_entities_map and client:  # 如果有唯一实体和LLM客户端，则继续
    # 提取需要生成嵌入的唯一标准化实体文本
    entity_normalized_texts_to_embed = list(set([key[0] for key in unique_entities_map.keys() if key[0].strip()]))

    if entity_normalized_texts_to_embed:
        print(f"准备为 {len(entity_normalized_texts_to_embed)} 个唯一标准化实体文本获取嵌入向量。")

        # 获取这些唯一文本的嵌入
        text_to_embedding_map = get_embeddings_for_texts(entity_normalized_texts_to_embed)

        # 将这些嵌入向量映射回实体URI
        for (normalized_text_key, entity_type_key), entity_uri_val_emb in unique_entities_map.items():
            if normalized_text_key in text_to_embedding_map and text_to_embedding_map[normalized_text_key]:
                entity_embeddings[entity_uri_val_emb] = text_to_embedding_map[normalized_text_key]

        if entity_embeddings:
            print(f"\n成功为 {len(entity_embeddings)} 个实体URI生成并映射嵌入向量。")
            # 显示一个示例
            first_uri_with_embedding = next(iter(entity_embeddings.keys()), None)
            if first_uri_with_embedding:
                emb_example = entity_embeddings[first_uri_with_embedding]
                # 从KG中获取该URI的标签
                label_for_uri = kg.value(subject=first_uri_with_embedding, predicate=RDFS.label, default=str(first_uri_with_embedding))
                print(f"  URI <{first_uri_with_embedding}> 的示例嵌入向量（标签：'{label_for_uri}'）:")
                print(f"    向量（前5维）: {str(emb_example[:5])}...")
                print(f"    向量维度: {len(emb_example)}")
        else:
            print("没有成功将嵌入向量映射到实体URI。")
    else:
        print("没有找到需要生成嵌入向量的唯一实体文本。")
else:
    print("跳过嵌入生成：未识别到唯一实体，或LLM客户端不可用。")

# 确保 entity_embeddings 已定义
if 'entity_embeddings' not in globals():
    entity_embeddings = {}
    print("已将 'entity_embeddings' 初始化为空字典。")

# 步骤 4.2：链接预测（知识发现 - 概念性） – 余弦相似度函数
def get_cosine_similarity(embedding1, embedding2):
    """使用sklearn计算两个嵌入向量之间的余弦相似度。"""
    if not isinstance(embedding1, (list, np.ndarray)) or not isinstance(embedding2, (list, np.ndarray)):
        return 0.0
    if not embedding1 or not embedding2:
        return 0.0

    # 确保它们是numpy数组，并为余弦相似度函数转换为二维
    vec1 = np.array(embedding1).reshape(1, -1)
    vec2 = np.array(embedding2).reshape(1, -1)

    # 检查维度是否匹配
    if vec1.shape[1] != vec2.shape[1]:
        print(f"警告：余弦相似度计算的嵌入维度不匹配：{vec1.shape[1]} vs {vec2.shape[1]}")
        return 0.0  # 或者作为错误处理

    return cosine_similarity(vec1, vec2)[0][0]

print("函数 'get_cosine_similarity' 已定义。")

# 链接预测（概念性） – 相似度计算示例
if len(entity_embeddings) >= 2:
    print("\n概念性链接预测：使用名称嵌入计算实体之间的语义相似度。")

    # 获取所有有嵌入的URI
    uris_with_embeddings = [uri for uri, emb in entity_embeddings.items() if emb]

    # 尝试找到两个ORG实体进行更有意义的比较
    org_entity_uris_with_embeddings = []
    for uri_cand in uris_with_embeddings:
        # 从KG中检查类型
        rdf_types_for_uri = list(kg.objects(subject=uri_cand, predicate=RDF.type))
        if SCHEMA.Organization in rdf_types_for_uri or EX.ORG in rdf_types_for_uri:
            org_entity_uris_with_embeddings.append(uri_cand)

    entity1_uri_sim = None
    entity2_uri_sim = None

    if len(org_entity_uris_with_embeddings) >= 2:
        entity1_uri_sim = org_entity_uris_with_embeddings[0]
        entity2_uri_sim = org_entity_uris_with_embeddings[1]
        print(f"找到至少两个具有嵌入的ORG实体用于相似度比较。")
    elif len(uris_with_embeddings) >= 2:  # 如果ORG不够，回退到任意两个实体
        entity1_uri_sim = uris_with_embeddings[0]
        entity2_uri_sim = uris_with_embeddings[1]
        print(f"未能找到两个ORG实体。使用两个通用实体进行相似度比较。")
    else:
        print("没有足够的实体（少于2个）用于演示相似度。")

    if entity1_uri_sim and entity2_uri_sim:
        embedding1_val = entity_embeddings.get(entity1_uri_sim)
        embedding2_val = entity_embeddings.get(entity2_uri_sim)

        # 从图谱中获取这些URI的标签以提供上下文
        label1_val = kg.value(subject=entity1_uri_sim, predicate=RDFS.label, default=str(entity1_uri_sim))
        label2_val = kg.value(subject=entity2_uri_sim, predicate=RDFS.label, default=str(entity2_uri_sim))

        calculated_similarity = get_cosine_similarity(embedding1_val, embedding2_val)
        print(f"\n  '{label1_val}' (<{entity1_uri_sim}>) 与 '{label2_val}' (<{entity2_uri_sim}>): {calculated_similarity:.4f}")

        # 对相似度进行简单解释（示例阈值）
        if calculated_similarity > 0.8:
            print(f"  解释：这些实体基于名称嵌入具有高度相似性。")
        elif calculated_similarity > 0.6:
            print(f"  解释：这些实体基于名称嵌入具有中等相似性。")
        else:
            print(f"  解释：这些实体基于名称嵌入具有低相似性。")

    print("\n注意：这是一个语义相似性的概念性演示。真正的链接预测需要在现有图谱三元组上训练专用的KGE模型（如TransE、ComplEx），以预测缺失的（主体，谓词，客体）事实，而不仅仅是实体之间的泛化相似性。")
else:
    print("跳过概念性链接预测：没有足够的实体嵌入（至少需要2个）。")

# 第4.3步：添加预测链接（可选 & 概念性） - 函数定义
def add_inferred_triples_to_graph(target_graph, list_of_inferred_triples):
    """
    将一组推断出的 (subject_uri, predicate_uri, object_uri) 三元组添加到图中。
    """
    if not list_of_inferred_triples:
        print("没有提供要添加的推断三元组。")
        return target_graph, 0

    added_count = 0
    for s_uri, p_uri, o_uri in list_of_inferred_triples:
        # 基本验证：确保它们是 URIRefs 或 Literals
        if isinstance(s_uri, URIRef) and isinstance(p_uri, URIRef) and (isinstance(o_uri, URIRef) or isinstance(o_uri, Literal)):
            target_graph.add((s_uri, p_uri, o_uri))
            added_count += 1
        else:
            print(f"  警告：跳过格式错误的推断三元组: ({s_uri}, {p_uri}, {o_uri})")

    print(f"已添加 {added_count} 个概念性推断三元组到图中。")
    return target_graph, added_count

print("函数 'add_inferred_triples_to_graph' 已定义。")

# 添加预测链接（概念性） - 执行示例
# 概念性：假设我们有一个来自其他模型的高置信度预测链接
# 例如，如果 similarity 检查中的 entity1_uri_sim 和 entity2_uri_sim 显示出非常高的相似度，
# 并且我们有一个谓词 ex:isSemanticallySimilarTo，我们可能会添加它。
conceptual_inferred_triples_list = []

# 示例：如果我们在上一步中定义了 entity1_uri_sim、entity2_uri_sim 和 calculated_similarity
# 以下变量可能不存在，除非前面的单元格已运行并存在有效实体
SIMILARITY_THRESHOLD_FOR_INFERENCE = 0.85  # 示例阈值
if 'entity1_uri_sim' in locals() and 'entity2_uri_sim' in locals() and 'calculated_similarity' in locals():
    if entity1_uri_sim and entity2_uri_sim and calculated_similarity > SIMILARITY_THRESHOLD_FOR_INFERENCE:
        print(f"概念性推断：实体 '{kg.label(entity1_uri_sim)}' 和 '{kg.label(entity2_uri_sim)}' 非常相似（{calculated_similarity:.2f}）。")
        # 定义一个概念性谓词
        EX.isHighlySimilarTo = EX["isHighlySimilarTo"]  # 如果尚未定义则定义
        conceptual_inferred_triples_list.append((entity1_uri_sim, EX.isHighlySimilarTo, entity2_uri_sim))
        # 对称关系（可选，取决于谓词定义）
        # conceptual_inferred_triples_list.append((entity2_uri_sim, EX.isHighlySimilarTo, entity1_uri_sim))

if conceptual_inferred_triples_list:
    print(f"\n尝试添加 {len(conceptual_inferred_triples_list)} 个概念性推断三元组...")
    kg, num_added = add_inferred_triples_to_graph(kg, conceptual_inferred_triples_list)
    if num_added > 0:
        print(f"添加概念性推断后，图中总三元组数为：{len(kg)}")
else:
    print("\n没有生成要添加的概念性推断三元组。")

# 第5阶段：持久化与使用
# 第5.1步：知识图谱存储 - 保存函数定义
def save_graph_to_turtle(graph_to_save, output_filepath="knowledge_graph.ttl"):
    """
    将 RDF 图保存为 Turtle 文件。
    """
    if not len(graph_to_save):
        print("图为空。没有内容可保存。")
        return False
    try:
        # 确保格式字符串正确，例如 'turtle', 'xml', 'n3', 'nt'
        graph_to_save.serialize(destination=output_filepath, format='turtle')
        print(f"知识图谱（共 {len(graph_to_save)} 个三元组）已成功保存到：{output_filepath}")
        return True
    except Exception as e:
        print(f"保存图到 {output_filepath} 时出错：{e}")
        return False

print("函数 'save_graph_to_turtle' 已定义。")

# 知识图谱存储 - 执行
KG_OUTPUT_FILENAME = "tech_acquisitions_kg.ttl"
if len(kg) > 0:
    print(f"尝试保存包含 {len(kg)} 个三元组的图...")
    save_graph_to_turtle(kg, KG_OUTPUT_FILENAME)
else:
    print(f"知识图谱 ('kg') 为空。跳过保存到 '{KG_OUTPUT_FILENAME}'。")

# 第5.2步：查询与分析 - SPARQL 执行函数
def execute_sparql_query(graph_to_query, query_string_sparql):
    """
    在图上执行 SPARQL 查询并打印结果，返回一个字典列表。
    """
    if not len(graph_to_query):
        print("无法执行 SPARQL 查询：图为空。")
        return []

    print(f"\n执行 SPARQL 查询：\n{query_string_sparql}")
    try:
        query_results = graph_to_query.query(query_string_sparql)
    except Exception as e:
        print(f"执行 SPARQL 查询时出错：{e}")
        return []

    if not query_results:
        print("查询成功执行但未返回结果。")
        return []

    results_list_of_dicts = []
    print(f"查询结果（共 {len(query_results)} 条）：")
    for row_idx, row_data in enumerate(query_results):
        # 将行转换为字典以便访问和打印
        result_item_dict = {}
        if hasattr(row_data, 'labels'):  # rdflib 6.x+ 提供 .labels 和 .asdict()
            result_item_dict = {str(label): str(value) for label, value in row_data.asdict().items()}
        else:  # 旧版本 rdflib 或 .asdict() 不可用时的回退
            # 此部分可能需要根据旧版本中 row_data 的实际结构进行调整
            # 为了简化，如果 .labels 不存在，我们只生成一个字符串值列表
            result_item_dict = {f"col_{j}": str(item_val) for j, item_val in enumerate(row_data)}

        results_list_of_dicts.append(result_item_dict)

        # 打印部分结果
        if row_idx < 10:  # 最多打印 10 条结果
            print(f"  行 {row_idx+1}: {result_item_dict}")
        elif row_idx == 10:
            print(f"  ... (还有 {len(query_results) - 10} 条结果)")
            print(f"  ... (and {len(query_results) - 10} more results)")

    return results_list_of_dicts

print("函数 'execute_sparql_query' 已定义。")

# SPARQL 查询与分析 - 执行示例
[31]
if len(kg) > 0:
    print("\n--- 执行示例 SPARQL 查询 ---")

    # 查询1：查找知识图谱中提到的所有组织及其标签
    sparql_query_1 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX schema: <http://schema.org/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT DISTINCT ?org_uri ?org_label
    WHERE {
        ?org_uri a schema:Organization ;
                 rdfs:label ?org_label .
    }
    ORDER BY ?org_label
    LIMIT 10
    """
    query1_results = execute_sparql_query(kg, sparql_query_1)

    # 查询2：查找一家公司收购另一家公司的关系
    # 假设：?acquiredCompany ex:ACQUIRED ?acquiringCompany
    sparql_query_2 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?acquiredCompanyLabel ?acquiringCompanyLabel
    WHERE {
        ?acquiredCompany ex:ACQUIRED ?acquiringCompany .
        ?acquiredCompany rdfs:label ?acquiredCompanyLabel .
        ?acquiringCompany rdfs:label ?acquiringCompanyLabel .
        # 确保两者都是组织
        ?acquiredCompany a schema:Organization .
        ?acquiringCompany a schema:Organization .
    }
    LIMIT 10
    """
    query2_results = execute_sparql_query(kg, sparql_query_2)

    # 查询3：查找带有价格信息的收购（由公司表示）
    sparql_query_3 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?companyLabel ?priceLabel ?dateLabel
    WHERE {
        ?company ex:HAS_PRICE ?priceEntity .
        ?company rdfs:label ?companyLabel .
        ?priceEntity rdfs:label ?priceLabel .
        # 确保 ?company 是组织，?priceEntity 是价格规范（MONEY）
        ?company a schema:Organization .
        ?priceEntity a schema:PriceSpecification .

        # 可选：查找与该收购事件/公司相关的日期
        OPTIONAL { 
            ?company ex:ANNOUNCED_ON ?dateEntity .
            ?dateEntity rdfs:label ?dateLabelRaw .
            # 如果 dateEntity 是 schema:Date，其标签可能是日期字符串
            # 如果 dateEntity 是事件，它可能有 schema:startDate 或类似字段
            BIND(COALESCE(?dateLabelRaw, STR(?dateEntity)) As ?dateLabel)            
        }
    }
    LIMIT 10
    """
    query3_results = execute_sparql_query(kg, sparql_query_3)

    # 查询4：统计每家收购公司的收购次数
    sparql_query_4 = """
    PREFIX ex: <http://example.org/kg/>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX schema: <http://schema.org/>

    SELECT ?acquiringCompanyLabel (COUNT(?acquiredCompany) AS ?numberOfAcquisitions)
    WHERE {
        ?acquiredCompany ex:ACQUIRED ?acquiringCompany .
        ?acquiringCompany rdfs:label ?acquiringCompanyLabel .
        ?acquiringCompany a schema:Organization .
        ?acquiredCompany a schema:Organization .
    }
    GROUP BY ?acquiringCompanyLabel
    ORDER BY DESC(?numberOfAcquisitions)
    LIMIT 10
    """
    query4_results = execute_sparql_query(kg, sparql_query_4)

else:
    print("知识图谱 ('kg') 为空。跳过 SPARQL 查询执行。")

# 第5.3步：可视化（可选） - 可视化函数定义
def visualize_subgraph_pyvis(graph_to_viz, output_filename="kg_visualization.html", sample_size_triples=75):
    """使用 pyvis 可视化一个子图，并保存为 HTML 文件。"""
    if not len(graph_to_viz):
        print("图谱为空，没有内容可以可视化。")
        return None

    net = Network(notebook=True, height="800px", width="100%", cdn_resources='remote', directed=True)
    net.repulsion(node_distance=150, spring_length=200)
    # net.show_buttons(filter_=['physics', 'nodes', 'edges', 'interaction'])
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08,
          "damping": 0.4,
          "avoidOverlap": 0.5
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "forceAtlas2Based",
        "timestep": 0.5,
        "stabilization": {"iterations": 150}
      }
    }
    """)

    added_nodes_set = set()

    # 为了更清晰的可视化，我们关注主语和宾语都是资源（URI）的三元组
    # 并尝试选取包含结构的样本，而不仅仅是单个节点的属性赋值。
    # 为了简化，这里我们选取所有三元组的一个样本。
    triples_for_visualization = list(graph_to_viz)[:min(sample_size_triples, len(graph_to_viz))]

    if not triples_for_visualization:
        print("从样本中未选取任何三元组用于可视化。")
        return None

    print(f"正在准备 {len(triples_for_visualization)} 个样本三元组的可视化...")

    for s_uri, p_uri, o_val in tqdm(triples_for_visualization, desc="构建 Pyvis 可视化"):
        # 获取标签或使用 URI 的一部分
        s_label_str = str(
            graph_to_viz.value(subject=s_uri, predicate=RDFS.label, default=s_uri.split('/')[-1].split('#')[-1]))
        p_label_str = str(p_uri.split('/')[-1].split('#')[-1])

        s_node_id = str(s_uri)
        s_node_title = f"{s_label_str}\nURI: {s_uri}"
        s_node_group_uri = graph_to_viz.value(s_uri, RDF.type)
        s_node_group = str(s_node_group_uri.split('/')[-1].split('#')[-1]) if s_node_group_uri else "UnknownType"

        if s_uri not in added_nodes_set:
            net.add_node(s_node_id, label=s_label_str, title=s_node_title, group=s_node_group)
            added_nodes_set.add(s_uri)

        if isinstance(o_val, URIRef):  # 如果宾语是一个资源，添加为节点并绘制边
            o_label_str = str(
                graph_to_viz.value(subject=o_val, predicate=RDFS.label, default=o_val.split('/')[-1].split('#')[-1]))
            o_node_id = str(o_val)
            o_node_title = f"{o_label_str}\nURI: {o_val}"
            o_node_group_uri = graph_to_viz.value(o_val, RDF.type)
            o_node_group = str(o_node_group_uri.split('/')[-1].split('#')[-1]) if o_node_group_uri else "UnknownType"

            if o_val not in added_nodes_set:
                net.add_node(o_node_id, label=o_label_str, title=o_node_title, group=o_node_group)
                added_nodes_set.add(o_val)
            net.add_edge(s_node_id, o_node_id, title=p_label_str, label=p_label_str)
        else:  # 如果宾语是一个字面量，将其作为主语节点的属性添加到标题（提示信息）中
            # 这样可以避免图中出现大量字面量节点。
            # 如果该节点已添加，则更新其标题
            for node_obj in net.nodes:
                if node_obj['id'] == s_node_id:
                    node_obj['title'] += f"\n{p_label_str}: {str(o_val)}"
                    break

    try:
        net.save_graph(output_filename)
        print(f"交互式知识图谱可视化已保存为 HTML 文件: {output_filename}")
        # 如果在 Jupyter Lab/Notebook 中运行，并且 notebook=True 且环境支持，图应该会内联显示。
        # 有时需要显式显示，或手动打开 HTML 文件。
    except Exception as e:
        print(f"保存或尝试显示图谱可视化时出错: {e}")
    return net  # 返回网络对象


print("函数 'visualize_subgraph_pyvis' 已定义。")

# 知识图谱可视化 - 执行
VIZ_OUTPUT_FILENAME = "tech_acquisitions_kg_interactive_viz.html"
pyvis_network_object = None  # 初始化

if len(kg) > 0:
    print(f"正在尝试可视化一个包含 {len(kg)} 个三元组的图谱样本...")
    # 可视化最多 75 个三元组的样本
    pyvis_network_object = visualize_subgraph_pyvis(kg, output_filename=VIZ_OUTPUT_FILENAME, sample_size_triples=75)
else:
    print(f"知识图谱 ('kg') 为空。跳过可视化。")

# 尝试在 Jupyter 中内联显示（可能需要信任笔记本或特定的 Jupyter 配置）
if pyvis_network_object is not None:
    try:
        # 这应该会在经典笔记本或 Lab 中内联显示
        from IPython.display import HTML, display
        # display(HTML(VIZ_OUTPUT_FILENAME))  # 从文件加载，pyvis 也可能直接渲染
        # pyvis_network_object.show(VIZ_OUTPUT_FILENAME)  # 另一种方式：在新标签页中打开或尝试内联显示
        print(f"\n要查看可视化，请在浏览器中打开文件 '{VIZ_OUTPUT_FILENAME}'。")
        print("如果在 Jupyter Notebook/Lab 中，图可能也会显示在此消息上方。")
        # 如果 pyvis_network_object 是单元格的最后一个语句，并且 notebook=True，Jupyter 会尝试渲染它。
    except Exception as e_display:
        print(f"无法在内联中自动显示可视化 ({e_display})。请手动打开 '{VIZ_OUTPUT_FILENAME}'。")

# 如果在 Jupyter 中，这行代码可能会自动渲染图谱
if pyvis_network_object:
    pyvis_network_object  # 这一行在某些 Jupyter 环境中对自动显示至关重要


####################################################################################################################################


# 资料来源：https://github.com/FareedKhan-dev/all-rag-techniques.git


