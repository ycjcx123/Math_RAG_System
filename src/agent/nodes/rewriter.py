import re
import json
import logging
from typing import Dict, Any
from openai import OpenAI

from src.utils.config_loader import load_config


class QueryRewriterNode:
    """查询重写节点 (v2.0)：阶梯式重写策略

    - loop_count=0 (Precise): 精确提取定理名/编号/术语
    - loop_count=1 (Concept Expansion): 扩展为上位/相关概念
    - 如果提供 critique: 基于纠错意见提取检索关键词
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
        self.max_tokens = generator_config.get("max_tokens", 100)

        rewriter_config = agent_config.get("rewriter", {})
        self.fallback_to_original = rewriter_config.get("fallback_to_original", True)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # ---- 第一轮：精确提取 ----
        self.system_prompt_precise = """你是一个数学检索关键词提取助手。从用户问题中提取适合检索教材的关键词。

输出格式：严格 JSON --- {"tool": "RAG", "context": "关键词"}

规则（第一轮-精确提取）：
1. 提取数学定理的准确名称、编号（如"定理10"、"柯西-施瓦茨不等式"）
2. 提取标准数学术语（如"初等因子"、"史密斯标准型"）
3. 如果提供[纠错意见]，优先从中提取缺失的定理/概念名
4. 关键词简洁准确，3-6个词为宜

示例：
问题: "证明定理10中λ-矩阵的相抵标准型唯一"
输出: {"tool": "RAG", "context": "定理10 λ-矩阵 相抵 标准型 唯一性"}

问题: "什么是线性空间？"
输出: {"tool": "RAG", "context": "线性空间 定义 向量空间"}

[纠错意见]: "缺少对拉格朗日中值定理的引用"
问题: "证明ln(1+x) < x"
输出: {"tool": "RAG", "context": "拉格朗日中值定理 不等式 证明 ln(1+x)"}"""

        # ---- 第二轮：概念扩展 ----
        self.system_prompt_expand = """你是一个数学检索关键词扩展助手。第一轮检索未找到相关内容，请将关键词扩展。

输出格式：严格 JSON --- {"tool": "RAG", "context": "扩展后的关键词"}

规则（第二轮-概念扩展）：
1. 将原关键词扩展为上位概念或相关概念
2. 添加同义词或相近术语
3. 覆盖更广的范围，增加命中概率
4. 5-8个关键词/短语，用空格分隔

示例：
原词: "初等因子"
扩展: "初等因子 不变因子 行列式因子 λ-矩阵 史密斯标准型 矩阵对角化 多项式分解"

原词: "定理10"
扩展: "定理10 λ-矩阵 相抵 初等变换 标准型"

原词: "特征值"
扩展: "特征值 特征向量 特征多项式 相似对角化 谱分解"

原词: "线性无关"
扩展: "线性无关 线性相关 向量组的秩 极大线性无关组 线性组合"""

        logging.info(f"QueryRewriterNode (v2.0) 初始化完成，模型: {self.model_name}")

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"查询重写 LLM 调用失败: {e}")
            return ""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """鲁棒的 JSON 解析"""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        json_patterns = [
            r'\{[^{}]*"tool"[^{}]*"RAG"[^{}]*"context"[^{}]*[^{}]*\}',
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

        logging.warning(f"无法从文本中提取 JSON: {text[:100]}...")
        return {}

    def rewrite(self, question: str, loop_count: int = 0, critique: str = "") -> Dict[str, Any]:
        """
        阶梯式重写

        Args:
            question: 原始问题
            loop_count: 当前轮次 (0=精确, 1=扩展)
            critique: 反思评分节点的纠错意见

        Returns:
            {"tool": "RAG", "context": "关键词", "success": bool, "tier": int}
        """
        tier = loop_count  # 0=精确, 1=扩展, 2+=fallback

        logging.info(f"查询重写 (第{tier+1}轮): {question[:50]}...")

        if critique:
            logging.info(f"基于纠错意见重写: {critique[:80]}...")

        # 构造 prompt
        if tier == 0:
            # 第一轮：精确提取
            user_content = f"问题: {question}"
            if critique:
                user_content += f"\n[纠错意见]: {critique}"
            llm_output = self._call_llm(self.system_prompt_precise, user_content)
        elif tier == 1:
            # 第二轮：概念扩展，将原问题作为输入
            user_content = f"原词: {question}"
            if critique:
                user_content = f"原词: {question}\n上一轮不足: {critique}"
            llm_output = self._call_llm(self.system_prompt_expand, user_content)
        else:
            # Fallback: 直接使用原始问题
            logging.info(f"达到最大重写轮次 ({tier})，使用原始问题作为关键词")
            return {
                "tool": "RAG",
                "context": question,
                "success": True,
                "tier": tier,
                "note": "fallback_to_original"
            }

        logging.info(f"重写 LLM 输出: {llm_output[:100]}...")

        if not llm_output:
            if self.fallback_to_original:
                return {
                    "tool": "RAG",
                    "context": question,
                    "success": False,
                    "tier": tier,
                    "error": "LLM调用失败，使用原始问题"
                }
            return {
                "tool": "RAG",
                "context": "",
                "success": False,
                "tier": tier,
                "error": "LLM调用失败"
            }

        json_result = self._extract_json(llm_output)

        if json_result and "tool" in json_result and "context" in json_result:
            json_result["success"] = True
            json_result["tier"] = tier
            logging.info(f"重写成功 (第{tier+1}轮): {json_result['context']}")
            return json_result

        if self.fallback_to_original:
            return {
                "tool": "RAG",
                "context": question,
                "success": False,
                "tier": tier,
                "error": "JSON解析失败，使用原始问题"
            }

        return {
            "tool": "RAG",
            "context": "",
            "success": False,
            "tier": tier,
            "error": "JSON解析失败"
        }

    def __call__(self, state: dict) -> dict:
        question = state.get("question", "")
        route = state.get("route", "Math_RAG")
        loop_count = state.get("loop_count", 0)
        critique = state.get("critique", "")

        if not question:
            return {"error": "缺少问题输入"}

        if route not in ("Math_RAG", "Math"):
            logging.info(f"路由为 {route}，跳过查询重写")
            return {"rewritten_query": question, "extracted_keywords": question}

        rewrite_result = self.rewrite(question, loop_count, critique)

        return {
            "rewritten_query": rewrite_result.get("context", question),
            "extracted_keywords": rewrite_result.get("context", question)
        }
