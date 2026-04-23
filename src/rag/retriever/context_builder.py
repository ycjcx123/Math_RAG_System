import logging
import qdrant_client
from qdrant_client import models
from typing import List, Tuple, Dict

# 导入原始模块
from src.utils.config_loader import load_config
from .reranker import Reranker
from .searcher import QdrantRetriever

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class BlockAggregator:
    def __init__(self, config: dict):
        """
        :param config: 传入的是 load_config.get("retriever") 的内容
        """
        self.config = config
        self.block_aggregate=config.get("block_aggregate")

        self.collection_name = config.get("collection_name", "math_rag_hybrid")
        
        # 1. 初始化基础检索器 (来自 searcher.py)
        # 注意：QdrantRetriever 内部会初始化 Settings.embed_model
        self.base_retriever = QdrantRetriever(config=config)
        
        # 2. 初始化重排器 (来自 reranker.py)
        self.reranker = Reranker(config=config)
        
        # 3. 初始化底层的 Qdrant Client 用于 Block 滚动查询
        db_cfg = config.get("Database", {})
        self.q_client = qdrant_client.QdrantClient(
            host=db_cfg.get("host", "localhost"),
            port=db_cfg.get("port", 6333)
        )

        # 配置参数
        self.top_n = config.get("rerank_model", {}).get("top_n", 3)
        self.rerank_pool_size = config.get("rerank_pool_size", 10)  # 为了聚合预留的重排池大小

    def _get_full_block_text(self, block_id: int) -> str:
        """
        根据 block_id 从 Qdrant 中拉取所有切片并合并
        """
        try:
            # 使用 scroll 获取所有具有相同 block_id 的点
            hits, _ = self.q_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[models.FieldCondition(key="block_id", match=models.MatchValue(value=block_id))]
                ),
                limit=100,  # 假设一个 block 不会超过 100 个切片
                with_payload=True
            )
            
            if not hits:
                return ""

            # 排序逻辑：通常切片的 ID 是连续递增的，或者是 metadata 里有 index 字段
            # 这里仿照你提供的脚本，尝试按 ID 排序
            hits_sorted = sorted(hits, key=lambda x: int(x.id) if str(x.id).isdigit() else 0)
            
            merged_text = "\n".join([h.payload.get("text", "") for h in hits_sorted])
            return f"--- 逻辑块 ID: {block_id} ---\n{merged_text}"
        
        except Exception as e:
            logging.error(f"拉取 Block {block_id} 失败: {e}")
            return ""

    def retrieve_and_aggregate(self, query: str) -> List[str]:
        """
        执行完整的：检索 -> 重排 -> 块聚合 流程
        :return: 聚合后的文本列表，长度不超过 top_n
        """
        if not self.block_aggregate:
            logging.info("Block聚合在config中被禁止")
            return
        
        # 1. 基础召回 (top_k=20)
        nodes = self.base_retriever.retrieve(query, self.collection_name)
        if not nodes:
            return []

        # 2. 调用重排
        # 实际上，reranker.rerank 返回的是重排后的 IDs 和 Texts，但丢失了 Node 对象
        # 我们需要找到这些文本对应的原始 Node 以获取 metadata
        final_ids, final_texts = self.reranker.rerank(query, nodes)
        
        # 建立映射方便找回 Node (metadata)
        node_map = {node.node.node_id: node for node in nodes}

        # 3. 策略性筛选：保证 block_id 唯一性
        aggregated_contexts = []
        seen_block_ids = set()

        for node_id in final_ids:
            node = node_map.get(node_id)
            if not node:
                continue
                
            block_id = node.metadata.get("block_id")

            if block_id is not None:
                if block_id in seen_block_ids:
                    continue
                seen_block_ids.add(block_id)
                
                # 执行全块拉取合并
                block_text = self._get_full_block_text(block_id)
                if block_text:
                    aggregated_contexts.append(block_text)
            else:
                # 如果没有 block_id，视为独立切片直接添加
                aggregated_contexts.append(node.get_content())

            # 达到 top_n 则停止
            if len(aggregated_contexts) >= self.top_n:
                break

        return aggregated_contexts
    
if __name__ == "__main__":
    # 模拟外部传入的 config (load_config.get("retriever"))
    config =(load_config.get("retriever"))

    aggregator = BlockAggregator(config=config)
    results = aggregator.retrieve_and_aggregate("例1 解方程 $x^4 + 1 = 0$")
    
    for i, res in enumerate(results):
        print(f"\n[Result {i+1}]\n{res[:200]}...")