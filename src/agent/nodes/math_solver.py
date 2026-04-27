import re
import logging
from openai import OpenAI

from src.utils.config_loader import load_config


class MathSolverNode:
    """数学直接求解节点 (v2.2)：支持首次求解 + Self-Refine 修正"""

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

        self.self_refine_prompt = (
            "你是一个数学修正助手。你之前已经做出了一个解答，现在根据反馈意见修正你的答案。\n\n"
            "要求：\n"
            "1. 仔细阅读反馈意见，找出解答中的不足\n"
            "2. 保留正确的部分，只修正有问题的步骤\n"
            "3. 分步推理，每步标注依据\n"
            "4. 使用标准 LaTeX 数学符号\n"
            "5. 最终答案清晰突出（如 \\boxed{答案}）\n"
            "6. 如果仍然不确定，请明确说'需要查阅教材'"
        )

        logging.info(f"MathSolverNode (v2.2) 初始化完成，模型: {self.model_name}")

    def solve(self, question: str) -> str:
        """首次直接求解"""
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
            draft = re.sub(r'<think>.*?</think>', '', draft, flags=re.DOTALL).strip()
            return draft

        except Exception as e:
            logging.error(f"MathSolver LLM 调用失败: {e}")
            return ""

    def solve_with_critique(self, question: str, draft: str, critique: str) -> str:
        """基于 critique 进行 Self-Refine 修正"""
        try:
            user_content = (
                f"原问题: {question}\n\n"
                f"你之前的回答:\n{draft}\n\n"
                f"反馈意见:\n{critique}\n\n"
                f"请根据上述反馈修正你的回答："
            )

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.self_refine_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            revised = response.choices[0].message.content.strip()
            revised = re.sub(r'<think>.*?</think>', '', revised, flags=re.DOTALL).strip()
            return revised

        except Exception as e:
            logging.error(f"MathSolver Self-Refine 调用失败: {e}")
            return draft  # 保底：返回原有草稿

    def __call__(self, state: dict) -> dict:
        """LangGraph 节点接口"""
        question = state.get("question", "")
        self_refine_count = state.get("self_refine_count", 0)
        critique = state.get("critique", "")

        if not question:
            return {"internal_draft": "", "error": "缺少问题输入"}

        if self_refine_count > 0 and critique:
            # Self-Refine 模式
            draft = state.get("internal_draft", "")
            logging.info(f"MathSolver Self-Refine (第{self_refine_count}轮): {question[:50]}...")
            revised = self.solve_with_critique(question, draft, critique)
            return {
                "internal_draft": revised,
                "self_refine_count": self_refine_count,
                "logic_path": f"{state.get('logic_path', '')} > Math_Solver(Self-Refine#{self_refine_count})"
            }
        else:
            # 首次求解
            logging.info(f"MathSolver 首次求解: {question[:60]}...")
            draft = self.solve(question)
            return {
                "internal_draft": draft,
                "logic_path": f"{state.get('logic_path', '')} > Math_Solver"
            }
