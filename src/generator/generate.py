import re
import logging
from typing import List, Optional
from openai import OpenAI

# 配置日志，方便调试
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaGenerator:
    def __init__(self, config: dict):
        """
        初始化生成器
        :param config: 通过 load_config.get("generator") 获得的配置字典
        """
        self.model_name = config.get("model_name", "my-qwen3")
        self.base_url = config.get("base_url", "http://localhost:11434/v1")
        self.api_key = config.get("api_key", "ollama")
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 4096)
        
        # 初始化 OpenAI 兼容客户端
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logging.info(f"Generator 初始化完成，使用模型: {self.model_name}")

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
                "请直接输出最终的证明过程或答案，不要进行内部思考或输出思维链内容。\n\n"
                f"### 参考资料：\n{context_str}"
            )
        
        return "你是一个专业的数学助手。请根据你自己的知识回答用户的问题。请直接输出最终的证明过程或答案，不要进行内部思考或输出思维链内容。"

    def generate(self, query: str, contexts: Optional[List[str]] = None, custom_sys_prompt: Optional[str] = None) -> str:
        """
        执行推理生成
        :param query: 用户提出的数学问题
        :param contexts: 检索到的参考片段列表
        :param custom_sys_prompt: 可选的自定义系统提示词
        :return: 清理后的回答文本
        """
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
            
            # 清理思维链标签 (针对某些模型自带 <think> 标签的情况)
            clean_response = re.sub(r'<think>.*?</think>', '', response_content, flags=re.DOTALL).strip()
            
            return clean_response

        except Exception as e:
            logging.error(f"本地推理调用失败 ({self.model_name}): {e}")
            return ""