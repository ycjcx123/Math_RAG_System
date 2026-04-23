import re
import logging
from typing import List
from openai import OpenAI

from src.utils.config_loader import load_config


class GraderNode:
    """评分节点：判断检索到的文档是否相关"""

    def __init__(self, config: dict = None):
        """初始化评分节点

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
        self.max_tokens = generator_config.get("max_tokens", 50)

        # 评分配置
        grader_config = agent_config.get("grader", {})
        self.relevance_threshold = grader_config.get("relevance_threshold", 0.5)

        # 初始化客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        self.system_prompt = """你是一个数学文档相关性评估助手。你的任务是判断提供的文档是否包含回答用户问题所需的数学定理、定义或关键信息。

输出要求：
1. 只输出 "Yes" 或 "No"，不要输出其他任何内容
2. "Yes" 表示文档包含回答问题所需的关键数学信息
3. "No" 表示文档不包含回答问题所需的关键数学信息

判断标准：
- 如果文档包含问题中提到的数学概念、定理、公式的定义或解释，输出 "Yes"
- 如果文档包含解决问题所需的步骤、方法或示例，输出 "Yes"
- 如果文档与问题完全无关，或只包含边缘信息，输出 "No"
- 如果文档为空或没有实质内容，输出 "No"

现在请评估以下文档是否相关："""

        logging.info(f"GraderNode 初始化完成，使用模型: {self.model_name}")

    def _prepare_context(self, documents: List[str]) -> str:
        """准备文档上下文"""
        if not documents:
            return "【文档为空】"

        context_str = ""
        for i, doc in enumerate(documents[:3]):  # 只取前3个文档进行评估
            context_str += f"文档 {i+1}:\n{doc[:500]}\n\n"  # 限制长度

        return context_str.strip()

    def _call_llm(self, question: str, context: str) -> str:
        """调用 LLM 进行相关性评估"""
        try:
            user_content = f"问题: {question}\n\n文档:\n{context}"

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
            logging.error(f"评分节点 LLM 调用失败: {e}")
            return "No"  # 默认不相关

    def _parse_output(self, llm_output: str) -> bool:
        """解析 LLM 输出，判断是否相关"""
        cleaned = llm_output.strip().upper()

        # 使用正则匹配 Yes/No
        if re.search(r'\bYES\b', cleaned):
            return True
        elif re.search(r'\bNO\b', cleaned):
            return False
        else:
            # 输出模糊，默认不相关
            logging.warning(f"评分节点输出模糊: '{llm_output}'，默认不相关")
            return False

    def grade(self, question: str, documents: List[str]) -> bool:
        """评估文档相关性

        Args:
            question: 用户问题
            documents: 检索到的文档列表

        Returns:
            True 表示相关，False 表示不相关
        """
        logging.info(f"评分节点评估问题: {question}")
        logging.info(f"文档数量: {len(documents)}")

        if not documents:
            logging.info("文档为空，直接判定为不相关")
            return False

        # 准备上下文
        context = self._prepare_context(documents)

        # 调用 LLM 获取原始输出
        llm_output = self._call_llm(question, context)
        logging.info(f"评分节点 LLM 输出: {llm_output}")

        # 解析输出
        is_relevant = self._parse_output(llm_output)
        logging.info(f"相关性判断: {'相关' if is_relevant else '不相关'}")

        return is_relevant

    def __call__(self, state: dict) -> dict:
        """LangGraph 节点接口

        Args:
            state: 当前状态字典

        Returns:
            更新后的状态字典
        """
        question = state.get("question", "")
        documents = state.get("documents", [])
        loop_count = state.get("loop_count", 0)

        if not question:
            logging.error("评分节点: 状态中缺少 question 字段")
            return {"is_relevant": False, "error": "缺少问题输入"}

        if not documents:
            logging.warning("评分节点: 文档为空")
            return {
                "is_relevant": False,
                "loop_count": loop_count + 1
            }

        # 执行评分
        is_relevant = self.grade(question, documents)

        # 仅当不相关时才递增 loop_count（表示一次检索尝试失败）
        if not is_relevant:
            return {
                "is_relevant": is_relevant,
                "loop_count": loop_count + 1
            }

        return {
            "is_relevant": is_relevant
        }