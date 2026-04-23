import re
import logging
from typing import Literal
from openai import OpenAI

from src.utils.config_loader import load_config


class RouterNode:
    """路由节点：判断用户问题是否需要检索（RAG）还是直接聊天（Chat）"""

    def __init__(self, config: dict = None):
        """初始化路由节点

        Args:
            config: 配置字典，如果为 None 则自动加载
        """
        if config is None:
            config = load_config()

        # 获取 generator 配置（用于连接 Llama.cpp）
        generator_config = config.get("generator", {})
        agent_config = config.get("agent", {})

        self.model_name = generator_config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = generator_config.get("base_url", "http://localhost:8080/v1")
        self.api_key = generator_config.get("api_key", "llama-cpp")
        self.temperature = generator_config.get("temperature", 0.1)
        self.max_tokens = generator_config.get("max_tokens", 50)

        # 路由配置
        self.default_path = agent_config.get("router", {}).get("default_path", "RAG")

        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Few-shot 示例
        self.few_shot_examples = """
        示例1:
        用户问题: "求解方程 $x^2 + 2x + 1 = 0$"
        思考: 这是一个具体的数学问题，需要用到代数知识，应该检索教材中的相关内容。
        输出: RAG

        示例2:
        用户问题: "你好，今天天气怎么样？"
        思考: 这是一个日常聊天问题，与数学教材无关。
        输出: Chat

        示例3:
        用户问题: "什么是线性代数？"
        思考: 这是一个数学概念的定义问题，需要检索教材中的定义部分。
        输出: RAG

        示例4:
        用户问题: "帮我写一首诗"
        思考: 这是一个创作请求，与数学无关。
        输出: Chat
        """

        self.system_prompt = f"""你是一个路由判断助手。你的任务是根据用户问题判断应该使用检索增强生成（RAG）还是直接聊天（Chat）。

{self.few_shot_examples}

判断规则：
1. 如果问题是关于数学概念、定理、证明、计算、公式等数学相关内容，输出 "RAG"
2. 如果问题是日常聊天、问候、与数学无关的内容，输出 "Chat"
3. 只输出 "RAG" 或 "Chat"，不要输出其他任何内容

现在请判断以下问题："""

        logging.info(f"RouterNode 初始化完成，使用模型: {self.model_name}")

    def _call_llm(self, question: str) -> str:
        """调用 LLM 进行路由判断"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": question}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"路由节点 LLM 调用失败: {e}")
            return self.default_path

    def _parse_output(self, llm_output: str) -> Literal["RAG", "Chat"]:
        """解析 LLM 输出，提取路由决策"""
        # 清理输出，提取 RAG 或 Chat
        cleaned = llm_output.strip().upper()

        # 使用正则匹配 RAG 或 Chat
        if re.search(r'\bRAG\b', cleaned, re.IGNORECASE):
            return "RAG"
        elif re.search(r'\bCHAT\b', cleaned, re.IGNORECASE):
            return "Chat"
        else:
            # 输出模糊，使用默认路径
            logging.warning(f"路由节点输出模糊: '{llm_output}'，使用默认路径: {self.default_path}")
            return self.default_path

    def route(self, question: str) -> Literal["RAG", "Chat"]:
        """执行路由判断

        Args:
            question: 用户问题

        Returns:
            "RAG" 或 "Chat"
        """
        logging.info(f"路由节点处理问题: {question}")

        # 调用 LLM 获取原始输出
        llm_output = self._call_llm(question)
        logging.info(f"路由节点 LLM 输出: {llm_output}")

        # 解析输出
        route_decision = self._parse_output(llm_output)
        logging.info(f"路由决策: {route_decision}")

        return route_decision

    def __call__(self, state: dict) -> dict:
        """LangGraph 节点接口

        Args:
            state: 当前状态字典

        Returns:
            更新后的状态字典
        """
        question = state.get("question", "")

        if not question:
            logging.error("路由节点: 状态中缺少 question 字段")
            return {"route": self.default_path, "error": "缺少问题输入"}

        route_decision = self.route(question)

        return {
            "route": route_decision
        }