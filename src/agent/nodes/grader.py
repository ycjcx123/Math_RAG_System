import re
import json
import logging
from typing import Tuple, List
from openai import OpenAI

from src.utils.config_loader import load_config


class ReflectiveGraderNode:
    """反思评分节点 (v2.2)：三段式评分的核心

    评分分段:
      - pass_threshold(85) ~ 100: Fast-Track（直接输出）
      - rag_threshold(60) ~ pass_threshold(85): Self-Refine（自我修正）
      - 0 ~ rag_threshold(60): RAG（检索教材补充）
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
        self.max_tokens = generator_config.get("max_tokens", 256)

        grader_config = agent_config.get("reflective_grader", {})
        self.pass_threshold = grader_config.get("pass_threshold", 85)
        self.rag_threshold = grader_config.get("rag_threshold", 60)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.few_shot_examples = """
        示例1:
        问题: "求解方程 $x^2 - 4 = 0$"
        回答: "解：由平方差公式，$x^2-4=(x+2)(x-2)=0$，因此 $x=2$ 或 $x=-2$。"
        评分: {"score": 95, "critique": "", "reasoning": "步骤完整，推导正确，依据清晰。"}

        示例2:
        问题: "证明柯西-施瓦茨不等式"
        回答: "由内积定义直接可得。"
        评分: {"score": 30, "critique": "缺少具体推导步骤；未说明内积空间假设；需引用定理的完整表述", "reasoning": "回答过于简略，没有展示证明过程。"}

        示例3:
        问题: "求矩阵 $\\begin{pmatrix}1&2\\\\3&4\\end{pmatrix}$ 的特征值"
        回答: "特征多项式为 $\\det(A-\\lambda I) = \\lambda^2 - 5\\lambda - 2$，解得 $\\lambda = \\frac{5 \\pm \\sqrt{33}}{2}$"
        评分: {"score": 70, "critique": "特征多项式计算结果正确，但应检查行列式计算是否正确（应为 $\\lambda^2 - 5\\lambda - 2$ 或 $\\lambda^2 - 5\\lambda - 2$？）；建议查阅教材中特征值的定义章节确认", "reasoning": "结果正确但缺少中间展开步骤，不便于验证。"}

        示例4:
        问题: "什么是向量空间？"
        回答: "向量空间是满足八条公理的集合，包含加法与数乘运算。"
        评分: {"score": 60, "critique": "回答方向正确但过于简略；需要列出八条公理的具体内容（加法交换律、结合律、零元、负元、数乘分配律等）。", "reasoning": "回答只有一句话，未展开公理细节。"}

        示例5:
        问题: "用拉格朗日中值定理证明不等式 $\\ln(1+x) < x$"
        回答: "设 $f(x)=\\ln(1+x)$，取区间 $[0,x]$，由拉格朗日中值定理存在 $\\xi\\in(0,x)$ 使 $\\frac{\\ln(1+x)-\\ln(1+0)}{x-0}=f'(\\xi)=\\frac{1}{1+\\xi}<1$，所以 $\\ln(1+x)<x$"
        评分: {"score": 85, "critique": "", "reasoning": "定理引用正确，推导完整，不等式方向正确。"}
        """

        self.system_prompt = f"""你是一个严格的数学评分助手。你的任务是对数学解答进行评估和纠错。

{self.few_shot_examples}

评分规则（0-100分）：
- **0-39**：严重错误或完全缺失关键步骤
- **40-59**：方向正确但重要步骤缺失
- **60-79**：基本正确但有可改进之处（缺少引用、步骤不够清晰）
- **80-84**：基本正确但需小幅修正
- **85-100**：正确且完整，可直接作为最终答案

如果得分 >= {self.pass_threshold}，critique 设为空字符串。
如果得分在 {self.rag_threshold}~{self.pass_threshold - 1} 之间，critique 应具体说明修正方向。
如果得分 < {self.rag_threshold}，critique 应建议查阅教材的具体章节。

输出要求：严格输出 JSON 对象，包含三个字段：
- "score": 整数 0-100
- "critique": 字符串（通过时为空）
- "reasoning": 评分理由

不要输出 Markdown 代码块或额外解释。"""

        logging.info(f"ReflectiveGraderNode 初始化完成，通过阈值: {self.pass_threshold}")

    def _call_llm(self, question: str, draft: str) -> str:
        try:
            user_content = f"问题: {question}\n\n回答:\n{draft}"

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"ReflectiveGrader LLM 调用失败: {e}")
            return '{"score": 0, "critique": "评分节点异常，转入RAG流程", "reasoning": "LLM调用失败"}'

    def _parse_json(self, text: str) -> dict:
        """鲁棒的 JSON 解析"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_patterns = [
            r'```json\s*(.*?)\s*```',
            r'```\s*(.*?)\s*```',
            r'\{(.*)\}',
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    if pattern.startswith(r'```'):
                        return json.loads(match.strip())
                    else:
                        return json.loads("{" + match + "}")
                except (json.JSONDecodeError, AttributeError):
                    continue

        logging.warning(f"无法解析评分 JSON: {text[:100]}...")
        return {"score": 0, "critique": "JSON解析失败", "reasoning": ""}

    def grade(self, question: str, draft: str) -> Tuple[int, str, bool]:
        """
        评分并返回 (score, critique, passed)
        """
        logging.info(f"ReflectiveGrader 评估: {question[:40]}...")

        if not draft:
            return 0, "草稿为空，需要查阅教材", False

        llm_output = self._call_llm(question, draft)
        logging.info(f"评分 LLM 输出: {llm_output[:120]}...")

        result = self._parse_json(llm_output)

        score = result.get("score", 0)
        critique = result.get("critique", "")

        # 确保 score 在 0-100
        score = max(0, min(100, int(score)))

        passed = score >= self.pass_threshold

        logging.info(f"评分结果: {score}/100 {'✅通过' if passed else '❌不通过'}")
        if critique:
            logging.info(f"纠错意见: {critique[:120]}")

        return score, critique, passed

    def __call__(self, state: dict) -> dict:
        question = state.get("question", "")
        draft = state.get("internal_draft", "")
        self_refine_count = state.get("self_refine_count", 0)

        if not question:
            return {"score": 0, "error": "缺少问题输入"}

        score, critique, passed = self.grade(question, draft)

        result = {
            "score": score,
            "critique": critique if not passed else "",
            "logic_path": f"{state.get('logic_path', '')} > ReflectiveGrader({score}/100)"
        }

        # 三段式来源标记
        if score >= self.pass_threshold:
            result["generation_source"] = "fast_track"
        elif score >= self.rag_threshold:
            # Self-Refine 计数器递增（由 graph 的 _reflective_decision_v22 决策）
            result["self_refine_count"] = self_refine_count + 1
            result["generation_source"] = "self_refine"
        else:
            result["generation_source"] = "rag"

        return result
