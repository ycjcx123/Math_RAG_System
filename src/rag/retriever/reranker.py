import requests
from typing import List, Tuple, Optional
import logging

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
class Reranker:
    def __init__(self, config: dict):
        """
        config应为load_config.get("retriever")相关配置:
        """
        config = config.get("rerank_model")
        self.url = config["url"]
        self.api_key = config["SILICONFLOW_API_KEY"]
        self.top_n = config.get("top_n", 3)
        self.return_documents = config.get("return_documents", False)
        self.model_name=config.get("model_name")

    def rerank(self, query: str, nodes: List) -> Tuple[List[str], List[str]]:
        """
        对检索出的节点进行重排
        :param nodes: llama_index 的 NodeWithScore 列表
        :return: (rerank_ids, rerank_texts) 按重排分数降序排列，长度 <= self.top_n
        """
        if not nodes:
            return [], []

        doc_texts = [node.get_content() for node in nodes]
        doc_ids = [node.node.node_id for node in nodes]

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": doc_texts,
            "top_n": self.top_n,
            "return_documents": self.return_documents
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            top_indices = [res["index"] for res in results]
            final_ids = [doc_ids[idx] for idx in top_indices]
            final_texts = [doc_texts[idx] for idx in top_indices]
            return final_ids, final_texts
        except Exception as e:
            logging.error(f"Rerank API 错误: {e}，降级使用前 {self.top_n} 个原始结果")
            return doc_ids[:self.top_n], doc_texts[:self.top_n]

    def rerank_texts(self, query: str, texts: List[str], top_n: int = 3) -> List[str]:
        """
        对纯文本列表进行重排（不依赖 NodeWithScore 对象）
        :param query: 查询文本
        :param texts: 待重排的文本列表
        :param top_n: 返回 top-n 条
        :return: 按分数降序排列的文本列表
        """
        if not texts:
            return []

        payload = {
            "model": self.model_name,
            "query": query,
            "documents": texts,
            "top_n": top_n,
            "return_documents": False
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(self.url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            results = response.json().get("results", [])
            top_indices = [res["index"] for res in results]
            return [texts[idx] for idx in top_indices]
        except Exception as e:
            logging.error(f"RerankTexts API 错误: {e}，降级使用前 {top_n} 条")
            return texts[:top_n]