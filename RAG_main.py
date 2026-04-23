from src.pipeline import chat_pipeline, ingest_pipeline
from src.utils.config_loader import load_config

if __name__ =="__main__":
    config = load_config()

    # 步骤一：将解析后的markdown文件，进行：修复-分块-插入
    ingest = ingest_pipeline.IngestPipeline(config)
    ingest.run()

    # 步骤二：针对query，去数据库中召回，重排，生成答案，返回两种答案格式，分别是常规答案和块答案
    query="求解方程 $x^4 + 1 = 0$"
    chat=chat_pipeline.ChatPipeline(config=config,query=query)
    normal_answer,block_answer=chat.run()
    print(normal_answer)
    print(block_answer)