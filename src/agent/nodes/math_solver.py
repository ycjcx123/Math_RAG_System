import re
import logging
from openai import OpenAI

from src.utils.config_loader import load_config


class MathSolverNode:
    """数学直接求解节点：不依赖 RAG，让 1.7B 直接尝试解答"""

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        generator_config = config.get("generator", {})

        self.model_name = generator_config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = generator_config.get("base_url", "http://localhost:8080/v1")
        self.api_key = generator_config.get("api_key", "llama-cpp")
        self.temperature = generator_config.get("temperature", 0.1)
        self.max_tokens = generator_config.get("max_tokens", 2048)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt = (
            "你是一个数学解题助手。请直接解答以下数学问题。\n\n"
            "要求：\n"
            "1. 分步推理，每步标注依据（公式、定理名称或运算规则）\n"
            "2. 使用标准 LaTeX 数学符号\n"
            "3. 最终答案清晰突出（如 \\boxed{答案}）\n"
            "4. 如果问题超出你的知识范围，请明确说'需要查阅教材'\n"
            "示例：\n"
            "问题：$x^2 - 4 = 0$\n"
            "解：由平方差公式得\n"
            "$$x^2 - 4 = (x+2)(x-2) = 0$$\n"
            "因此 $x = 2$ 或 $x = -2$。\n"
            "$$\\boxed{x = \\pm 2}$$"
        )

        logging.info(f"MathSolverNode 初始化完成，模型: {self.model_name}")

    def solve(self, question: str) -> str:
        """直接生成数学解答"""
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

            draft = response.choices[0].message.content.strip()
            # 清理思维链标签
            draft = re.sub(r'<think>.*?</think>', '', draft, flags=re.DOTALL).strip()
            return draft

        except Exception as e:
            logging.error(f"MathSolver LLM 调用失败: {e}")
            return ""

    def __call__(self, state: dict) -> dict:
        """LangGraph 节点接口"""
        question = state.get("question", "")
        loop_count = state.get("loop_count", 0)

        if not question:
            return {"internal_draft": "", "error": "缺少问题输入"}

        logging.info(f"MathSolver 求解问题: {question}")
        draft = self.solve(question)

        return {
            "internal_draft": draft,
            "logic_path": f"{state.get('logic_path', '')} > Math_Solver"
        }
