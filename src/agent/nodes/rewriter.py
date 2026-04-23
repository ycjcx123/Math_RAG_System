import re
import json
import logging
from typing import Dict, Any
from openai import OpenAI

from src.utils.config_loader import load_config


class QueryRewriterNode:
    """查询重写节点：提取关键词，优化查询用于检索"""

    def __init__(self, config: dict = None):
        """初始化查询重写节点

        Args:
            config: 配置字典，如果为 None 则自动加载
        """
        if config is None:
            config = load_config()

        # 获取 generator 配置
        generator_config = config.get("generator", {})
        agent_config = config.get("agent", {})

        self.model_name = generator_config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = generator_config.get("base_url", "http://localhost:8080/v1")
        self.api_key = generator_config.get("api_key", "llama-cpp")
        self.temperature = generator_config.get("temperature", 0.1)
        self.max_tokens = generator_config.get("max_tokens", 100)

        # 重写配置
        rewriter_config = agent_config.get("rewriter", {})
        self.fallback_to_original = rewriter_config.get("fallback_to_original", True)

        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt = """你是一个数学查询优化助手。你的任务是将用户的问题重写为更适合检索的关键词或查询语句。

输出要求：
1. 必须输出严格的 JSON 格式：{"tool": "RAG", "context": "关键词"}
2. "tool" 字段固定为 "RAG"
3. "context" 字段包含提取的关键词或优化后的查询语句
4. 关键词应该简洁、准确，包含数学概念、定理名称、公式等核心信息

示例：
用户问题: "求解方程 $x^2 + 2x + 1 = 0$"
输出: {"tool": "RAG", "context": "一元二次方程 求解 公式法"}

用户问题: "什么是线性无关？"
输出: {"tool": "RAG", "context": "线性无关 定义 向量组"}

用户问题: "证明勾股定理"
输出: {"tool": "RAG", "context": "勾股定理 证明 直角三角形"}

现在请处理以下问题："""

        logging.info(f"QueryRewriterNode 初始化完成，使用模型: {self.model_name}")

    def _call_llm(self, question: str) -> str:
        """调用 LLM 进行查询重写"""
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
            logging.error(f"查询重写节点 LLM 调用失败: {e}")
            return ""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """从文本中提取 JSON 对象

        Args:
            text: 可能包含 JSON 的文本

        Returns:
            解析后的 JSON 字典，如果解析失败返回空字典
        """
        # 尝试直接解析
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 尝试提取 JSON 块（处理可能包含其他文本的情况）
        json_patterns = [
            r'\{[^{}]*"tool"[^{}]*"RAG"[^{}]*"context"[^{}]*[^{}]*\}',  # 简单匹配
            r'```json\s*(.*?)\s*```',  # 匹配 ```json ``` 块
            r'```\s*(.*?)\s*```',  # 匹配 ``` ``` 块
            r'\{(.*)\}',  # 匹配最外层大括号
        ]

        for pattern in json_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    if pattern.startswith(r'```'):
                        # 如果是代码块，直接解析匹配的内容
                        return json.loads(match.strip())
                    else:
                        # 如果是大括号匹配，需要添加大括号
                        json_str = "{" + match + "}"
                        return json.loads(json_str)
                except (json.JSONDecodeError, AttributeError):
                    continue

        # 所有尝试都失败
        logging.warning(f"无法从文本中提取 JSON: {text[:100]}...")
        return {}

    def rewrite(self, question: str) -> Dict[str, Any]:
        """执行查询重写

        Args:
            question: 原始用户问题

        Returns:
            包含重写结果的字典，格式: {"tool": "RAG", "context": "关键词", "success": bool}
        """
        logging.info(f"查询重写节点处理问题: {question}")

        # 调用 LLM 获取原始输出
        llm_output = self._call_llm(question)
        logging.info(f"查询重写节点 LLM 输出: {llm_output}")

        if not llm_output:
            # LLM 调用失败，使用回退策略
            if self.fallback_to_original:
                logging.info("LLM 调用失败，回退到原始问题")
                return {
                    "tool": "RAG",
                    "context": question,
                    "success": False,
                    "error": "LLM调用失败，使用原始问题"
                }
            else:
                return {
                    "tool": "RAG",
                    "context": "",
                    "success": False,
                    "error": "LLM调用失败且未启用回退"
                }

        # 尝试提取 JSON
        json_result = self._extract_json(llm_output)

        if json_result and "tool" in json_result and "context" in json_result:
            # JSON 解析成功
            json_result["success"] = True
            logging.info(f"查询重写成功: {json_result}")
            return json_result
        else:
            # JSON 解析失败，使用回退策略
            if self.fallback_to_original:
                logging.warning(f"JSON 解析失败，回退到原始问题。LLM输出: {llm_output}")
                return {
                    "tool": "RAG",
                    "context": question,
                    "success": False,
                    "error": "JSON解析失败，使用原始问题"
                }
            else:
                logging.error(f"JSON 解析失败且未启用回退。LLM输出: {llm_output}")
                return {
                    "tool": "RAG",
                    "context": "",
                    "success": False,
                    "error": "JSON解析失败且未启用回退"
                }

    def __call__(self, state: dict) -> dict:
        """LangGraph 节点接口

        Args:
            state: 当前状态字典

        Returns:
            更新后的状态字典
        """
        question = state.get("question", "")
        route = state.get("route", "RAG")

        if not question:
            logging.error("查询重写节点: 状态中缺少 question 字段")
            return {"error": "缺少问题输入"}

        if route != "RAG":
            # 如果不是 RAG 路径，跳过重写
            logging.info(f"路由为 {route}，跳过查询重写")
            return {
                "rewritten_query": question,
                "extracted_keywords": question
            }

        # 执行查询重写
        rewrite_result = self.rewrite(question)

        return {
            "rewritten_query": rewrite_result.get("context", question),
            "extracted_keywords": rewrite_result.get("context", question)
        }