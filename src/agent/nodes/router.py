import re
import logging
from typing import Literal
from openai import OpenAI

from src.utils.config_loader import load_config


class RouterNode:
    """路由节点 (v2.2)：二元分流 Chat / Math

    - Chat: 纯社交闲聊 → 直接 Chat_Node
    - Math: 数学问题（含简单计算、复杂定理证明）→ 统一走三段式自适应流程
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        generator_config = config.get("generator", {})
        agent_config = config.get("agent", {})

        self.model_name = generator_config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = generator_config.get("base_url", "http://localhost:8080/v1")
        self.api_key = generator_config.get("api_key", "llama-cpp")
        self.temperature = generator_config.get("temperature", 0.1)
        self.max_tokens = generator_config.get("max_tokens", 50)

        self.default_path = agent_config.get("router", {}).get("default_path", "Math")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.few_shot_examples = """
        示例1:
        用户问题: "求解方程 $x^2 + 2x + 1 = 0$"
        输出: Math

        示例2:
        用户问题: "你好，今天天气怎么样？"
        输出: Chat

        示例3:
        用户问题: "什么是线性无关？"
        输出: Math

        示例4:
        用户问题: "证明史密斯标准型的唯一性"
        输出: Math

        示例5:
        用户问题: "最近老师脾气好差，作业又这么多"
        输出: Chat

        示例6:
        用户问题: "计算行列式 $\\begin{vmatrix}1&2\\\\3&4\\end{vmatrix}$"
        输出: Math

        示例7:
        用户问题: "帮我写一首诗"
        输出: Chat

        示例8:
        用户问题: "$\\sin^2 x + \\cos^2 x = 1$ 怎么证明？"
        输出: Math

        示例9:
        用户问题: "证明 $\\lambda$-矩阵的初等因子唯一性"
        输出: Math

        示例10:
        用户问题: "请说明线性映射、线性变换、正交变换和酉变换的区别"
        输出: Math
        """

        self.system_prompt = f"""你是一个路由判断助手。将用户问题分为两类：Chat、Math。

{self.few_shot_examples}

分类规则：
1. **Chat**：日常聊天、问候、情绪表达、非数学内容（诗歌、天气、美食等）
2. **Math**：任何与数学相关的问题，包括简单计算、概念定义、定理证明等

注意：所有数学问题都归为 Math，哪怕是非常复杂的定理证明。系统内部会自动判断是否需要查教材。
只输出 "Chat" 或 "Math"。

现在请判断以下问题："""

        logging.info(f"RouterNode (v2.2) 初始化完成，模型: {self.model_name}，默认路径: {self.default_path}")

    def _call_llm(self, question: str) -> str:
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

    def _parse_output(self, llm_output: str) -> Literal["Chat", "Math"]:
        cleaned = llm_output.strip().upper()

        if re.search(r'\bCHAT\b', cleaned):
            return "Chat"
        else:
            # 任何包含 Math 或默认情况都走 Math
            return "Math"

    def route(self, question: str) -> Literal["Chat", "Math"]:
        logging.info(f"路由节点处理问题: {question[:60]}...")
        llm_output = self._call_llm(question)
        logging.info(f"路由节点 LLM 输出: {llm_output}")
        route_decision = self._parse_output(llm_output)
        logging.info(f"路由决策: {route_decision}")
        return route_decision

    def __call__(self, state: dict) -> dict:
        question = state.get("question", "")

        if not question:
            return {"route": self.default_path, "error": "缺少问题输入"}

        route_decision = self.route(question)

        return {
            "route": route_decision,
            "logic_path": f"{state.get('logic_path', 'start')} > Router({route_decision})"
        }
