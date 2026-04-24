import re
import logging
from typing import Literal
from openai import OpenAI

from src.utils.config_loader import load_config


class RouterNode:
    """路由节点 (v2.0)：将入口流量分为 Chat / Math / Math_RAG 三类

    - Chat: 纯社交闲聊 → 直接 Chat_Node
    - Math: 简单计算/基础定义 → Math_Solver 直接生成
    - Math_RAG: 复杂定理证明/深奥概念 → 完整 RAG 流程
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

        self.default_path = agent_config.get("router", {}).get("default_path", "Math_RAG")

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.few_shot_examples = """
        示例1:
        用户问题: "求解方程 $x^2 + 2x + 1 = 0$"
        思考: 这是一元二次方程求根，属于标准计算，可以直接求解。
        输出: Math

        示例2:
        用户问题: "你好，今天天气怎么样？"
        思考: 日常问候，与数学无关。
        输出: Chat

        示例3:
        用户问题: "什么是线性无关？"
        思考: 这是一个基本数学概念的定义，可以直接解答。
        输出: Math

        示例4:
        用户问题: "证明史密斯标准型的唯一性"
        思考: 这是一个复杂定理的证明，需要查阅教材以确保正确性。
        输出: Math_RAG

        示例5:
        用户问题: "最近老师脾气好差，作业又这么多"
        思考: 纯情绪表达/抱怨，与数学无关。
        输出: Chat

        示例6:
        用户问题: "计算行列式 $\\begin{vmatrix}1&2\\\\3&4\\end{vmatrix}$"
        思考: 行列式计算，有固定公式，可以直接算。
        输出: Math

        示例7:
        用户问题: "定理10说矩阵相似于若尔当标准型，为什么？"
        思考: 询问定理背后的原因，需要教材中的详细论证。
        输出: Math_RAG

        示例8:
        用户问题: "$\\sin^2 x + \\cos^2 x = 1$ 怎么证明？"
        思考: 三角恒等式，有标准推导。
        输出: Math

        示例9:
        用户问题: "帮我写一首诗"
        思考: 创作请求，与数学无关。
        输出: Chat

        示例10:
        用户问题: "证明 $\\lambda$-矩阵的初等因子唯一性"
        思考: 高阶抽象代数定理，需检索教材中的严格证明。
        输出: Math_RAG
        """

        self.system_prompt = f"""你是一个数学路由判断助手。将用户问题分为三类：Chat、Math、Math_RAG。

{self.few_shot_examples}

分类规则：
1. **Chat**：日常聊天、问候、情绪表达、非数学内容
2. **Math**：简单数学计算、标准公式求解、基础概念定义（可直接推理得出）
3. **Math_RAG**：复杂定理证明、深奥概念辨析、需要教材参考文献的进阶内容

重要：过滤掉问题中的情绪化语言（如"太难了"、"老师好凶"），只根据数学核心内容判断。

只输出 "Chat" 或 "Math" 或 "Math_RAG"，不要输出其他任何内容。

现在请判断以下问题："""

        logging.info(f"RouterNode (v2.0) 初始化完成，模型: {self.model_name}")

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

    def _parse_output(self, llm_output: str) -> Literal["Chat", "Math", "Math_RAG"]:
        """解析 LLM 输出，提取三类路由决策"""
        cleaned = llm_output.strip().upper()

        if re.search(r'\bCHAT\b', cleaned):
            return "Chat"
        elif re.search(r'\bMATH_RAG\b', cleaned):
            return "Math_RAG"
        elif re.search(r'\bMATH\b', cleaned):
            return "Math"
        else:
            logging.warning(f"路由节点输出模糊: '{llm_output}'，使用默认: {self.default_path}")
            return self.default_path

    def route(self, question: str) -> Literal["Chat", "Math", "Math_RAG"]:
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
