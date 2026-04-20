import torch
from typing import List, Optional
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client
import logging

from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



class QdrantRetriever:
    def __init__(self, config: dict):
        """
        config 应包含 retriever相关配置
        """
        self.collection_name=config.get("collection_name","math_rag_hybrid")
        self.top_k=config.get("top_k", 20)

        Data_cfg=config.get("Database",{})
        self.host=Data_cfg.get("host","localhost")
        self.port=Data_cfg.get("port",6333)
        

        embed_cfg = config["embedding_model"]
        self.device=embed_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 初始化全局 embedding 模型
        Settings.embed_model = HuggingFaceEmbedding(
            model_name=embed_cfg["model_path"],
            device=self.device,
            embed_batch_size=embed_cfg.get("embed_batch_size", 1),
            query_instruction="为数学查询检索相关文档："  # 可根据需要调整
        )
        Settings.llm = None

    def _get_retriever(self, collection_name=None, hybrid: bool = True, top_k: int = 20):
        """
        构建并返回一个 retriever 对象
        :param collection_name: Qdrant 中的集合名
        :param hybrid: 是否启用混合检索（稠密+稀疏）
        :param top_k: 召回数量
        """
        if collection_name is None:
            collection_name = self.collection_name
        client = qdrant_client.QdrantClient(
            host=self.host,
            port=self.port
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=hybrid
        )
        index = VectorStoreIndex.from_vector_store(vector_store)
        retriever = index.as_retriever(
            similarity_top_k=self.top_k,
            vector_store_query_mode="hybrid" if hybrid else "default",
            alpha=0.5
        )
        return retriever

    def retrieve(self, query: str, collection_name=None, hybrid: bool = True, top_k: int = 20) -> List:
        """
        执行检索，返回节点列表 (llama_index 的 NodeWithScore 对象)
        """
        retriever = self._get_retriever(collection_name, hybrid, self.top_k)
        nodes = retriever.retrieve(query)
        return nodes

if __name__ == '__main__':
    config=load_config()
    retriever=config.get("retriever")
    a=QdrantRetriever(config=retriever)
    a.retrieve(query="例1 解方程 $x^4 + 1 = 0$", collection_name=retriever.get("collection_name"))
    print(a)