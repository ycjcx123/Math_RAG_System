import re
import logging
from typing import List, Optional
from openai import OpenAI

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Generator:
    def __init__(self, config: dict):
        """
        初始化生成器
        :param config: 通过 load_config.get("generator") 获得的配置字典
        """
        self.model_name = config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "llama-cpp")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 2048)
        self.n_ctx = config.get("n_ctx", 4096)        # 服务端上下文窗口大小
        self.max_context_chars = config.get("max_context_chars", 2000)  # 参考资料安全截断长度

        # 初始化 OpenAI 兼容客户端（设置超时避免卡死）
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=600
        )

        logging.info(f"Generator 初始化完成，使用模型: {self.model_name}, "
                     f"n_ctx={self.n_ctx}, max_tokens={self.max_tokens}")

    def _build_system_prompt(self, contexts: Optional[List[str]], custom_sys_prompt: Optional[str]) -> str:
        """
        私有方法：构建系统提示词
        """
        # 如果提供了自定义系统提示词，则优先使用
        if custom_sys_prompt:
            if contexts:
                context_str = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(contexts)])
                return f"{custom_sys_prompt}\n\n### 参考资料：\n{context_str}"
            return custom_sys_prompt

        # 默认的数学助手逻辑
        if contexts:
            context_str = "\n".join([f"[{i+1}] {text}" for i, text in enumerate(contexts)])
            return (
                "你是一个专业的数学助手。请根据以下提供的参考资料回答用户的问题。\n"
                "如果资料中没有相关信息，请诚实回答你不知道，不要胡乱推导。\n"
                "请直接输出最终的证明过程或答案。\n\n"
                f"### 参考资料：\n{context_str}"
            )
        
        return "你是一个专业的数学助手。请根据你自己的知识回答用户的问题。请直接输出最终的证明过程或答案。"

    def _estimate_overhead(self, query: str, contexts: Optional[List[str]]) -> int:
        """估算 prompt 总长度（字符数），用于上下文截断判断"""
        total = len(query)
        if contexts:
            total += sum(len(c) for c in contexts)
        return total

    def _truncate_contexts(self, contexts: List[str], query: str) -> List[str]:
        """截断参考资料使其不超出 n_ctx（保守估算：1 char ≈ 1 token）"""
        # 保守预留：system prompt ~300 chars + max_tokens 输出 + 300 chars 安全垫
        overhead = 600 + self.max_tokens
        budget = self.n_ctx - overhead

        result = []
        used = len(query)
        for i, ctx in enumerate(contexts):
            if used + len(ctx) <= budget:
                result.append(ctx)
                used += len(ctx)
            elif used + self.max_context_chars <= budget:
                # 截断过长片段
                keep = self.max_context_chars
                truncated = ctx[:keep] + "\n...[截断]"
                result.append(truncated)
                used += keep + 20
                logging.warning(f"截断第 {i+1} 个参考资料 ({len(ctx)}→{keep} chars)")
            else:
                logging.warning(f"丢弃第 {i+1} 个参考资料（预算不足，已用 {used}/{budget}）")
                break
        return result

    def generate(self, query: str, contexts: Optional[List[str]] = None, custom_sys_prompt: Optional[str] = None) -> str:
        """
        执行推理生成
        :param query: 用户提出的数学问题
        :param contexts: 检索到的参考片段列表
        :param custom_sys_prompt: 可选的自定义系统提示词
        :return: 清理后的回答文本
        """
        # 上下文安全截断
        if contexts:
            raw_chars = sum(len(c) for c in contexts)
            contexts = self._truncate_contexts(contexts, query)
            truncated_chars = sum(len(c) for c in contexts)
            if truncated_chars < raw_chars:
                logging.info(f"参考资料截断: {raw_chars} → {truncated_chars} chars")

        system_content = self._build_system_prompt(contexts, custom_sys_prompt)

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            response_content = response.choices[0].message.content
            if not response_content:
                logging.error(f"模型返回空内容 ({self.model_name})")
                return ""

            # 清理思维链标签 (针对某些模型自带 <think> 标签的情况)
            clean_response = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
            if not clean_response:
                logging.warning(f"清理后回答为空（原始内容仅包含 think 标签）")
            return clean_response

        except Exception as e:
            logging.error(f"本地推理调用失败 ({self.model_name}): {e}")
            return ""