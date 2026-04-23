import logging
from src.rag.retriever import QdrantRetriever, Reranker, BlockAggregator

from src.utils.config_loader import load_config

class ChatPipeline:
    def __init__(self,config,query):
        """config=load_config()"""
        self.query=query
        self.block_aggregate=config["retriever"].get("block_aggregate", False)
        self.retriever=QdrantRetriever(config["retriever"])
        self.reranker=Reranker(config["retriever"])
        self.block_aggregator=BlockAggregator(config["retriever"])
    
    def run(self):
        """根据retriever的配置，决定是否块聚合
            返回：contexts
        """
        logging.info("正在执行：召回--重排--生成流程")
        if self.block_aggregate:
            logging.info("正在执行：块聚合")
            block_contexts=self.block_aggregator.retrieve_and_aggregate(query=self.query)
            logging.info("块召回已完成")
            return block_contexts
        else:
            logging.info("块聚合被禁用，执行常规召回")
            node=self.retriever.retrieve(query=self.query)
            contexts_id, contexts=self.reranker.rerank(query=self.query,nodes=node)
            logging.info("常规召回已完成")
            return contexts
    
if __name__=="__main__":
    query="求解方程 $x^4 + 1 = 0$"
    config=load_config()
    pipeline=ChatPipeline(config=config,query=query)
    normal_answer,block_answer, _, _=pipeline.run()
    print(normal_answer)
    print(block_answer)