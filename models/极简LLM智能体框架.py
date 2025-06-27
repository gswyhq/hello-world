
# 导入模块

import asyncio, warnings, copy, time
# asyncio: 用于异步编程，处理协程和事件循环。
# warnings: 发出警告信息，提示潜在问题。
# copy: 提供对象复制功能，用于创建对象的浅拷贝。
# time: 处理时间相关操作，如睡眠等待。

# BaseNode 基类
class BaseNode:
    def __init__(self):
        self.params = {}  # 存储节点参数
        self.successors = {}  # 存储后续节点，键为动作名，值为节点对象

    def set_params(self, params):
        self.params = params  # 设置参数

    def next(self, node, action="default"):
        if action in self.successors:
            warnings.warn(f"Overwriting successor for action '{action}'")  # 重复设置相同action时警告
        self.successors[action] = node  # 添加后续节点
        return node  # 支持链式调用

    def prep(self, shared): pass  # 准备阶段，子类实现
    def exec(self, prep_res): pass  # 执行阶段，子类实现
    def post(self, shared, prep_res, exec_res): pass  # 后处理阶段，子类实现

    def _exec(self, prep_res):
        return self.exec(prep_res)  # 内部执行方法，调用exec

    def _run(self, shared):
        p = self.prep(shared)  # 调用准备
        e = self._exec(p)  # 调用执行
        return self.post(shared, p, e)  # 调用后处理

    def run(self, shared):
        if self.successors:
            warnings.warn("Node won't run successors. Use Flow.")  # 单独运行节点时提示使用Flow
        return self._run(shared)  # 执行节点流程

    def __rshift__(self, other):
        return self.next(other)  # 重载>>运算符，设置默认action的后续节点

    def __sub__(self, action):
        if isinstance(action, str):
            return _ConditionalTransition(self, action)  # 重载-运算符，创建条件转移对象
        raise TypeError("Action must be a string")

# _ConditionalTransition 条件转移类
class _ConditionalTransition:
    def __init__(self, src, action):
        self.src = src  # 源节点
        self.action = action  # 转移动作名

    def __rshift__(self, tgt):
        return self.src.next(tgt, self.action)  # 重载>>，设置特定action的后续节点

# Node 节点类（带重试）
class Node(BaseNode):
    def __init__(self, max_retries=1, wait=0):
        super().__init__()
        self.max_retries = max_retries  # 最大重试次数
        self.wait = wait  # 重试等待时间

    def exec_fallback(self, prep_res, exc):
        raise exc  # 重试失败后抛出异常，子类可覆盖

    def _exec(self, prep_res):
        for self.cur_retry in range(self.max_retries):
            try:
                return self.exec(prep_res)  # 尝试执行
            except Exception as e:
                if self.cur_retry == self.max_retries - 1:
                    return self.exec_fallback(prep_res, e)  # 最后一次重试失败调用fallback
                if self.wait > 0:
                    time.sleep(self.wait)  # 等待后重试

# BatchNode 批量节点
class BatchNode(Node):
    def _exec(self, items):
        return [super(BatchNode, self)._exec(i) for i in (items or [])]  # 逐个处理批量项

# Flow 流程控制类
class Flow(BaseNode):
    def __init__(self, start=None):
        super().__init__()
        self.start_node = start  # 起始节点

    def start(self, start):
        self.start_node = start
        return start  # 设置起始节点并返回

    def get_next_node(self, curr, action):
        nxt = curr.successors.get(action or "default")  # 根据action获取下一节点
        if not nxt and curr.successors:
            warnings.warn(f"Flow ends: '{action}' not found in {list(curr.successors)}")  # 找不到节点时警告
        return nxt

    def _orch(self, shared, params=None):
        curr = copy.copy(self.start_node)  # 复制起始节点避免状态污染
        p = params or {**self.params}  # 合并参数
        last_action = None
        while curr:
            curr.set_params(p)
            last_action = curr._run(shared)  # 执行当前节点
            curr = copy.copy(self.get_next_node(curr, last_action))  # 获取下一节点
        return last_action

    def _run(self, shared):
        p = self.prep(shared)  # 流程准备
        o = self._orch(shared)  # 执行流程链
        return self.post(shared, p, o)  # 流程后处理

    def post(self, shared, prep_res, exec_res):
        return exec_res  # 默认返回执行结果

# BatchFlow 批量流程
class BatchFlow(Flow):
    def _run(self, shared):
        pr = self.prep(shared) or []  # 准备批量参数
        for bp in pr:
            self._orch(shared, {**self.params, **bp})  # 逐个参数执行流程
        return self.post(shared, pr, None)

# 异步相关类
# AsyncNode 异步节点
class AsyncNode(Node):
    async def prep_async(self, shared): pass  # 异步准备
    async def exec_async(self, prep_res): pass  # 异步执行
    async def exec_fallback_async(self, prep_res, exc): raise exc  # 异步异常处理
    async def post_async(self, shared, prep_res, exec_res): pass  # 异步后处理

    async def _exec(self, prep_res):
        for i in range(self.max_retries):
            try:
                return await self.exec_async(prep_res)  # 异步执行
            except Exception as e:
                if i == self.max_retries - 1:
                    return await self.exec_fallback_async(prep_res, e)  # 异步fallback
                if self.wait > 0:
                    await asyncio.sleep(self.wait)  # 异步等待

    async def run_async(self, shared):
        if self.successors:
            warnings.warn("Node won't run successors. Use AsyncFlow.")
        return await self._run_async(shared)  # 异步运行节点

    async def _run_async(self, shared):
        p = await self.prep_async(shared)  # 异步准备
        e = await self._exec(p)  # 异步执行
        return await self.post_async(shared, p, e)  # 异步后处理

    def _run(self, shared):
        raise RuntimeError("Use run_async.")  # 禁用同步运行

# AsyncBatchNode 异步批量节点
class AsyncBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        return [await super(AsyncBatchNode, self)._exec(i) for i in items]  # 异步逐个处理批量项

# AsyncParallelBatchNode 异步并行批量节点
class AsyncParallelBatchNode(AsyncNode, BatchNode):
    async def _exec(self, items):
        return await asyncio.gather(*(super()._exec(i) for i in items))  # 并行处理批量项

# AsyncFlow 异步流程
class AsyncFlow(Flow, AsyncNode):
    async def _orch_async(self, shared, params=None):
        curr = copy.copy(self.start_node)
        p = params or {**self.params}
        last_action = None
        while curr:
            curr.set_params(p)
            if isinstance(curr, AsyncNode):
                last_action = await curr._run_async(shared)  # 异步执行节点
            else:
                last_action = curr._run(shared)  # 同步执行节点
            curr = copy.copy(self.get_next_node(curr, last_action))
        return last_action

    async def _run_async(self, shared):
        p = await self.prep_async(shared)  # 异步准备
        o = await self._orch_async(shared)  # 异步流程执行
        return await self.post_async(shared, p, o)  # 异步后处理

    async def post_async(self, shared, prep_res, exec_res):
        return exec_res

# AsyncBatchFlow 异步批量流程
class AsyncBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        for bp in pr:
            await self._orch_async(shared, {**self.params, **bp})  # 异步逐个处理批量参数
        return await self.post_async(shared, pr, None)

# AsyncParallelBatchFlow 异步并行批量流程
class AsyncParallelBatchFlow(AsyncFlow, BatchFlow):
    async def _run_async(self, shared):
        pr = await self.prep_async(shared) or []
        await asyncio.gather(*(self._orch_async(shared, {**self.params, **bp}) for bp in pr))  # 并行处理批量参数
        return await self.post_async(shared, pr, None)

# 总结
# BaseNode：定义节点基础结构，包含参数、后继节点和生命周期方法。
# Node：实现带重试机制的节点。
# Flow：管理节点间的流程执行，支持条件分支。
# Batch类：处理批量任务，支持同步和异步。
# Async类：利用asyncio实现异步执行，提升并发性能。
# 运算符重载：简化节点间的连接和条件转移定义。

# 资料来源：https://github.com/The-Pocket/PocketFlow
# 应用示例
# | 名称                                                                                                | 难度          | 描述                           |
# | ------------------------------------------------------------------------------------------------- | ----------- | ---------------------------- |
# | [聊天](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat)                 | ☆☆☆<br>*入门* | 带有对话历史的基础聊天机器人               |
# | [结构化输出](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-structured-output) | ☆☆☆<br>*入门* | 通过提示从简历中提取结构化数据              |
# | [工作流](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-workflow)            | ☆☆☆<br>*入门* | 一个写作工作流，包括大纲编写、内容创作和样式应用     |
# | [智能体](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-agent)               | ☆☆☆<br>*入门* | 一个可以搜索网络并回答问题的研究智能体          |
# | [RAG](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-rag)                 | ☆☆☆<br>*入门* | 一个简单的检索增强生成过程                |
# | [批处理](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-batch)               | ☆☆☆<br>*入门* | 一个将markdown内容翻译成多种语言的批处理器    |
# | [流式处理](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-llm-streaming)      | ☆☆☆<br>*入门* | 具有用户中断功能的实时LLM流式演示           |
# | [聊天护栏](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat-guardrail)     | ☆☆☆<br>*入门* | 一个仅处理旅行相关查询的旅行顾问聊天机器人        |
# | [Map-Reduce](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-map-reduce)   | ★☆☆<br>*初级* | 使用map-reduce模式进行批量评估的简历资格处理器 |
# | [多智能体](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-multi-agent)        | ★☆☆<br>*初级* | 两个智能体之间异步通信的禁忌词游戏            |
# | [监督者](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-supervisor)          | ★☆☆<br>*初级* | 研究智能体变得不可靠...让我们构建一个监督流程     |
# | [并行](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel-batch)       | ★☆☆<br>*初级* | 展示3倍加速的并行执行演示                |
# | [并行流](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel-batch-flow) | ★☆☆<br>*初级* | 展示使用多个过滤器实现8倍加速的并行图像处理演示     |
# | [多数投票](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-majority-vote)      | ★☆☆<br>*初级* | 通过聚合多次解决方案尝试来提高推理准确性         |
# | [思考](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-thinking)             | ★☆☆<br>*初级* | 通过思维链解决复杂推理问题                |
# | [记忆](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat-memory)          | ★☆☆<br>*初级* | 具有短期和长期记忆的聊天机器人              |
# | [Text2SQL](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-text2sql)       | ★☆☆<br>*初级* | 使用自动调试循环将自然语言转换为SQL查询        |
# | [MCP](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-mcp)                 | ★☆☆<br>*初级* | 使用模型上下文协议进行数值运算的智能体          |
# | [A2A](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-a2a)                 | ★☆☆<br>*初级* | 使用智能体到智能体协议包装的智能体，用于智能体间通信   |
# | [Web HITL](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-web-hitl)       | ★☆☆<br>*初级* | 具有SSE更新的人工审核循环的最小Web服务       |
#
# 以下是更复杂LLM应用的示例：
#
# | 应用名称                                                                                              | 难度          | 主题                                                                                                                                                                                                                                                   | 人类设计                                                                                                 | 智能体代码                                                                                           |
# | ------------------------------------------------------------------------------------------------- | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
# | [用Cursor构建Cursor](https://github.com/The-Pocket/Tutorial-Cursor)<br>我们很快将达到奇点...                  | ★★★<br>*高级* | [智能体](https://the-pocket.github.io/PocketFlow/design_pattern/agent.html)                                                                                                                                                                             | [设计文档](https://github.com/The-Pocket/Tutorial-Cursor/blob/main/docs/design.md)                       | [Flow代码](https://github.com/The-Pocket/Tutorial-Cursor/blob/main/flow.py)                       |
# | [代码库知识构建器](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge)<br>生命太短暂，不应该困惑地盯着他人的代码      | ★★☆<br>*中级* | [工作流](https://the-pocket.github.io/PocketFlow/design_pattern/workflow.html)                                                                                                                                                                          | [设计文档](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge/blob/main/docs/design.md)           | [Flow代码](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge/blob/main/flow.py)           |
# | [询问AI Paul Graham](https://github.com/The-Pocket/Tutorial-YC-Partner)<br>询问AI Paul Graham，以防你没被录取 | ★★☆<br>*中级* | [RAG](https://the-pocket.github.io/PocketFlow/design_pattern/rag.html)<br>[Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html)<br>[TTS](https://the-pocket.github.io/PocketFlow/utility_function/text_to_speech.html) | [设计文档](https://github.com/The-Pocket/Tutorial-AI-Paul-Graham/blob/main/docs/design.md)               | [Flow代码](https://github.com/The-Pocket/Tutorial-AI-Paul-Graham/blob/main/flow.py)               |
# | [Youtube摘要器](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple)<br>像你5岁一样向你解释YouTube视频   | ★☆☆<br>*初级* | [Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html)                                                                                                                                                                  | [设计文档](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple/blob/main/docs/design.md)          | [Flow代码](https://github.com/The-Pocket/Tutorial-Youtube-Made-Simple/blob/main/flow.py)          |
# | [冷启动生成器](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization)<br>将冷门线索转变为热门的即时破冰工具  | ★☆☆<br>*初级* | [Map Reduce](https://the-pocket.github.io/PocketFlow/design_pattern/mapreduce.html)<br>[Web搜索](https://the-pocket.github.io/PocketFlow/utility_function/websearch.html)                                                                              | [设计文档](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization/blob/master/docs/design.md) | [Flow代码](https://github.com/The-Pocket/Tutorial-Cold-Email-Personalization/blob/master/flow.py) |



