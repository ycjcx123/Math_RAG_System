import logging
from src.parser import FormulaFixer
from src.chunked import ChunkProcessor
from src.utils import InsertQdrant

from src.utils.config_loader import load_config

# 简单配置一下log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IngestPipeline:
    def __init__(self, config):
        self.config = config
        # 在初始化时就准备好各个组件
        self.fixer = FormulaFixer(self.config.get("fix", {}))
        self.chunk_processor = ChunkProcessor(self.config.get("chunked", {}))
        self.inserter = InsertQdrant(self.config.get("insert", {}).get("math_rag_system", {}))

    def run(self):
        logging.info("开始执行：公式修复--分块--插入qdrant的全流程")
        self.fixer.run()
        self.chunk_processor.run()
        self.inserter.run()
        logging.info("公式修复--分块--插入qdrant的：完成")

if __name__ == "__main__":
    config = load_config()
    pipeline = IngestPipeline(config)
    pipeline.run()