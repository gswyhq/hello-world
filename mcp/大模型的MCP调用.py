import asyncio
import json
import logging
import os
import re
import time
import shutil
import traceback
import requests
from contextlib import AsyncExitStack
from typing import Any
from logging.handlers import TimedRotatingFileHandler
import httpx
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
import gradio as gr

# 创建日志记录器
logger = logging.getLogger('my_logger')
logger.setLevel(logging.DEBUG)

# 创建处理器，每天生成一个新文件，保留7天的日志
file_handler = TimedRotatingFileHandler(
    'logs/app.log',  # 日志文件名
    when='D',    # 按天分割
    interval=1,  # 每隔1天分割
    backupCount=365,  # 保留7天的日志
    encoding='utf-8' # 设置文件编码为UTF-8
)

# 设置日志格式
formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 创建控制台处理器（新增部分）
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 将处理器添加到记录器
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# 记录日志
logger.info('This is a log message.')


############################################################################################################################


############################################################################################################################
class Server:
    """Manages MCP server connections and tool execution."""

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        self.name: str = name
        self.config: dict[str, Any] = config
        self.session: ClientSession | None = None
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()
        self.exit_stack: AsyncExitStack = AsyncExitStack()

    async def initialize(self) -> None:
        """Initialize the server connection."""
        try:
            logger.info(f"config{self.config}")
            sse_transport = await self.exit_stack.enter_async_context(
                sse_client(self.config['url'])
            )
            read, write = sse_transport

            session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )
            await session.initialize()
            self.session = session
        except Exception as e:
            logger.error(f"Error initializing server {self.name}: {e}")
            logger.error(f"错误详情：{traceback.format_exc()}")
            await self.cleanup()
            raise

    async def list_tools(self) -> list[Any]:
        """List available tools from the server.

        Returns:
            A list of available tools.

        Raises:
            RuntimeError: If the server is not initialized.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        tools_response = await self.session.list_tools()
        tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == "tools":
                tools.extend(
                    Tool(tool.name, tool.description, tool.inputSchema)
                    for tool in item[1]
                )

        return tools

    async def execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        retries: int = 2,
        delay: float = 1.0,
    ) -> Any:
        """Execute a tool with retry mechanism.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Tool arguments.
            retries: Number of retry attempts.
            delay: Delay between retries in seconds.

        Returns:
            Tool execution result.

        Raises:
            RuntimeError: If server is not initialized.
            Exception: If tool execution fails after all retries.
        """
        if not self.session:
            raise RuntimeError(f"Server {self.name} not initialized")

        attempt = 0
        while attempt < retries:
            try:
                logger.info(f"Executing {tool_name}...")
                result = await self.session.call_tool(tool_name, arguments)

                return result

            except Exception as e:
                attempt += 1
                logger.info(
                    f"Error executing tool: {e}. Attempt {attempt} of {retries}."
                )
                if attempt < retries:
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logger.error("Max retries reached. Failing.")
                    raise

    async def cleanup(self) -> None:
        """Clean up server resources."""
        async with self._cleanup_lock:
            try:
                await self.exit_stack.aclose()
                self.session = None
            except Exception as e:
                logger.error(f"Error during cleanup of server {self.name}: {e}")


class Tool:
    """Represents a tool with its properties and formatting."""

    def __init__(
        self, name: str, description: str, input_schema: dict[str, Any]
    ) -> None:
        self.name: str = name
        self.description: str = description
        self.input_schema: dict[str, Any] = input_schema

    def format_for_llm(self) -> str:
        """Format tool information for LLM.

        Returns:
            A formatted string describing the tool.
        """
        args_desc = []
        if "properties" in self.input_schema:
            for param_name, param_info in self.input_schema["properties"].items():
                arg_desc = (
                    f"- {param_name}: {param_info.get('description', 'No description')}"
                )
                if param_name in self.input_schema.get("required", []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return f"""
Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class LLMClient:
    """Manages communication with the LLM provider."""

    def __init__(self, api_key: str) -> None:
        self.api_key: str = api_key

    def get_response(self, messages: list[dict[str, str]]) -> str:
        """Get a response from the LLM.

        Args:
            messages: A list of message dictionaries.

        Returns:
            The LLM's response as a string.

        Raises:
            httpx.RequestError: If the request to the LLM fails.
        """

        try:
            r = requests.post("http://localhost:8002/v1/chat/completions",
                              headers={"Content-Type": "application/json"},
                              json={
                                    "model": "Qwen2.5-32B-Instruct",
                                    "messages": messages
                                  }
                              )
            ret = r.json()
            logger.info(f"模型请求结果：{ret}")
            return ret.get("data", {}).get('answer', '')
        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logger.error(error_message)

            return (
                f"I encountered an error: {error_message}. "
                "Please try again or rephrase your request."
            )


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: list[Server], llm_client: LLMClient) -> None:
        self.servers: list[Server] = servers
        self.llm_client: LLMClient = llm_client
        self.system_message = ''


    async def cleanup_servers(self) -> None:
        """Clean up all servers properly."""
        cleanup_tasks = [
            asyncio.create_task(server.cleanup()) for server in self.servers
        ]
        if cleanup_tasks:
            try:
                await asyncio.gather(*cleanup_tasks, return_exceptions=True)
            except Exception as e:
                logger.info(f"Warning during final cleanup: {e}")

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed.

        Args:
            llm_response: The response from the LLM.

        Returns:
            The result of tool execution or the original response.
        """
        import json

        try:
            pattern = re.compile(r'<think>\s*(.*?)\s*</think>', re.DOTALL)
            llm_response = re.sub(pattern, '', llm_response)
            r = re.search('```json(.*?)```', llm_response, re.DOTALL)
            if r:
                llm_response = r.groups()[0]
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logger.info(f"Executing tool: {tool_call['tool']}")
                logger.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = await server.list_tools()
                    if any(tool.name == tool_call["tool"] for tool in tools):
                        try:
                            url = server.config['url']
                            logger.info(f"mcp请求：url: {url}, {tool_call}")
                            result = await server.execute_tool(
                                tool_call["tool"], tool_call["arguments"]
                            )
                            logger.info(f"mcp请求结果：{result}")
                            if isinstance(result, dict) and "progress" in result:
                                progress = result["progress"]
                                total = result["total"]
                                percentage = (progress / total) * 100
                                logger.info(
                                    f"Progress: {progress}/{total} ({percentage:.1f}%)"
                                )
                            tool = [tool for tool in tools if tool.name == tool_call["tool"]][0]
                            system_message = (
                                "你是一个将工具执行结果转换为口语化响应的助理：\n\n"
                                f"工具描述：{tool.format_for_llm()}\n"
                                f"工具执行结果: {result}\n"
                                "要求：\n"
                                "1. 将工具执行结果转换为自然的口语化响应\n"
                                "2. 保持回复简洁但信息完整\n"
                                "3. 聚焦最相关的信息\n"
                                "4. 结合用户问题的上下文\n"
                                "5. 避免直接重复原始数据\n\n"

                            )
                            return system_message

                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logger.error(error_msg)
                            logger.error(f"错误详情：{traceback.format_exc()}")
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError as e:
            logger.error(f"解析调用工具json出错: {e}")
            logger.error(f"错误详情：{traceback.format_exc()}")
            return llm_response

    async def bot(self, history, request: gr.Request):
        await chat_session.start()
        user_id = request.session_hash
        start_time = time.time()
        logger.info(f"用户ID：{user_id}，请求的问题：{history[-1][0]}")
        print("输入：", history)
        question = history[-1][0]
        history[-1][1] = ""

        try:

            messages = [{"role": "system", "content": self.system_message}]

            user_input = question
            messages.append({"role": "user", "content": question})

            llm_response = self.llm_client.get_response(messages)
            history[-1][1] = llm_response.replace("<think>", "")
            logger.info("\nAssistant: %s", llm_response)

            result = await self.process_llm_response(llm_response)

            if result != llm_response:
                # messages.append({"role": "assistant", "content": llm_response})
                messages = []
                messages.append({"role": "system", "content": result})
                messages.append({"role": "user", "content": user_input})

                final_response = self.llm_client.get_response(messages)
                logger.info("\nFinal response: %s", final_response)
                messages.append(
                    {"role": "assistant", "content": final_response}
                )
                history[-1][1] += '\n'+final_response.replace("<think>", "")
            else:
                messages.append({"role": "assistant", "content": llm_response})

        except Exception as e:
            logger.error(f"请求出错：{e}")
            logger.error(f"错误详情：{traceback.format_exc()}")
            history[-1][1] += f"请求出错：{e}"

        yield history

    async def start(self) -> None:
        """Main chat session handler."""
        if self.system_message:
            return
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                logger.error(f"Failed to initialize server: {e}")
                await self.cleanup_servers()
                return

        all_tools = []
        for server in self.servers:
            tools = await server.list_tools()
            all_tools.extend(tools)

        tools_description = "\n".join([tool.format_for_llm() for tool in all_tools])

        self.system_message = (
            "## 角色：你是一个可以通过以下工具提供帮助的助理：\n\n"
            "## 任务：根据用户问题选择合适的工具。如果不需要工具，请直接回复。\n\n"
            "## 要求：当需要使用工具时，你必须且只能使用以下精确的JSON对象格式响应：\n"
            "{\n"
            '    "tool": "tool-name",\n'
            '    "arguments": {\n'
            '        "argument-name": "value"\n'
            "    }\n"
            "}\n\n"

            f"## 工具描述：{tools_description}\n\n"
        )
        logger.info(f"system_message: {self.system_message}")



############################################################################################################################

def user(user_message, history):
    return "", history + [[user_message, None]]

############################################################################################################################

server_config = {
    "mcpServers": {
        'tag': {'url': 'http://192.168.3.34:8003/sse', 'transport': 'sse'},
        'add': {'url': 'http://192.168.3.34:8042/sse', 'transport': 'sse'}
    }
}

servers = [
    Server(name, srv_config)
    for name, srv_config in server_config["mcpServers"].items()
]

llm_client = LLMClient("llm_api_key")
chat_session = ChatSession(servers, llm_client)

with gr.Blocks() as demo:
    with gr.Tabs():
        with gr.TabItem("模型问答"):
            chatbot = gr.Chatbot(label='模型问答对话框', height=700, render_markdown=True, sanitize_html=False)
            msg = gr.Textbox(label='请在这里输入你的问题', value='7854+547812=')

            msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
                chat_session.bot, [chatbot], chatbot
            )

async def main() -> None:
    demo.queue(status_update_rate="auto", default_concurrency_limit=120, max_size=120)
    demo.launch(server_name='0.0.0.0', server_port=7860, share=True)

if __name__ == "__main__":
    asyncio.run(main())

# 参考资料：https://github.com/modelcontextprotocol/python-sdk/blob/main/examples/clients/simple-chatbot/mcp_simple_chatbot/main.py


