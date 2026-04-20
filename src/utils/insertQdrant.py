import json
import torch
import qdrant_client
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel
import logging
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
import os

from .config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class InsertQdrant:
    def __init__(self, insert_config):
        """insert_config=load_config.get("insert")"""

        self.json_path=insert_config.get("math_chunk_path")
        self.collection_name=insert_config.get("collection_name")

        database=insert_config.get("Database")
        self.host=database.get("host")
        self.port=database.get("port")

        model_config=insert_config.get("embedding_model")
        self.model_path=model_config.get("model_path")
        self.fp16=model_config.get("use_fp16", True)
        self.device=model_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        self.model = BGEM3FlagModel(self.model_path, use_fp16=self.fp16, device=self.device) 
        self.client=qdrant_client.QdrantClient(host=self.host,port=self.port)

    
    def _process_and_upload(self):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        points = []
        total = len(data)
        
        logging.info(f"开始生成混合向量并入库，共 {total} 条数据...")

        for i, item in enumerate(data):
            idx = int(item["id"])
            text_content = item["text"]
            metadata = item["metadata"]
            
            # 构建 Payload 拓扑指针
            payload = metadata.copy()
            payload["text"] = text_content
            payload["prev_id"] = idx - 1 if idx > 0 else None
            payload["next_id"] = idx + 1 if idx < total - 1 else None
            payload["block_id"] = metadata.get("block_id", None)

            # 核心：同时提取 Dense 和 Sparse 向量
            # return_dense=True, return_sparse=True 是关键
            output = self.model.encode(text_content, return_dense=True, return_sparse=True)
            
            # 1. 获取 Dense 向量
            dense_vec = output['dense_vecs'].tolist()
            
            # 2. 获取 Sparse 向量 (词汇权重字典格式转换)
            lexical_weights = output['lexical_weights']
            # BGE-M3 的 lexical_weights 格式为 {'token_id_str': weight_float, ...}
            # Qdrant 需要 indices (整数列表) 和 values (浮点数列表)
            sparse_indices = [int(k) for k in lexical_weights.keys()]
            sparse_values = list(lexical_weights.values())

            # 组装 Qdrant 要求的命名向量结构
            point_vector = {
                "text-dense": dense_vec,
                "text-sparse-new": models.SparseVector(
                    indices=sparse_indices, 
                    values=sparse_values
                )
            }

            points.append(
                models.PointStruct(
                    id=idx,
                    vector=point_vector,
                    payload=payload
                )
            )

            # 分批入库
            if len(points) >= 100:
                self.client.upsert(collection_name=self.collection_name, points=points)
                points = []
                logging.info(f"进度: {i+1}/{total}")

        # 上传剩余部分
        if points:
            self.client.upsert(collection_name=self.collection_name, points=points)
        
        logging.info("混合向量入库完成！")

    def run(self):
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "text-sparse-new": models.SparseVectorParams() # 必须叫这个，响应 LlamaIndex 的召唤
                }
            )
            # 为 Payload 建立索引，保证精准查找 block 和 path 时极速返回
            self.client.create_payload_index(self.collection_name, "block_id", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "path", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "chapter", models.PayloadSchemaType.KEYWORD)

        if self.client.collection_exists(self.collection_name):
            logging.info(f"集合 {self.collection_name} 已存在，跳过创建...")
            return
        else:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(size=1024, distance=models.Distance.COSINE),
                },
                sparse_vectors_config={
                    "text-sparse-new": models.SparseVectorParams() # 必须叫这个，响应 LlamaIndex 的召唤
                }
            )
            # 为 Payload 建立索引，保证精准查找 block 和 path 时极速返回
            self.client.create_payload_index(self.collection_name, "block_id", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "path", models.PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection_name, "chapter", models.PayloadSchemaType.KEYWORD)
            
        self._process_and_upload()

class NormalInsert:
    def __init__(self, config):
        # 1. 基础配置
        self.markdown_path = config.get("markdown_path")
        self.output_json_path = config.get("output_json_path", "output.json")
        self.chunk_size = config.get("chunk_size", 512)
        self.overlap = config.get("overlap", 50)
        
        # 2. 向量检索配置
        self.hybrid = config.get("hybrid", True)
        base_name = config.get("collection_name", "default_collection")
        self.collection_name = f"{base_name}_{'hybrid' if self.hybrid else 'dense'}"
        
        # 3. 数据库与模型
        db_config = config.get("Database", {})
        self.client = qdrant_client.QdrantClient(host=db_config.get("host", "localhost"), port=db_config.get("port", 6333))
        
        model_config = config.get("embedding_model", {})
        self.model = BGEM3FlagModel(
            model_config.get("model_path"), 
            use_fp16=model_config.get("use_fp16", True),
            device=config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self.batch_size = 200
        self.DENSE_NAME = "text-dense"
        self.SPARSE_NAME = "text-sparse-new"

    def _get_nodes(self):
        """核心切分逻辑：利用 Llama-Index 快速分块"""
        logging.info(f"正在读取并切分文档: {self.markdown_path}")
        documents = SimpleDirectoryReader(input_files=[self.markdown_path]).load_data()
        
        # 清除元数据防止干扰 Token 计算
        for doc in documents:
            doc.metadata = {} 
            
        node_parser = TokenTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.overlap)
        return node_parser.get_nodes_from_documents(documents)

    def generate_json(self):
        """函数 1：直接生成符合格式的 JSON 文件"""
        nodes = self._get_nodes()
        json_data = [{"id": str(i + 1), "text": node.get_content()} for i, node in enumerate(nodes)]
        
        with open(self.output_json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        
        logging.info(f"JSON 文件已生成: {self.output_json_path} (共 {len(json_data)} 条)")
        return json_data

    def insert_to_db(self, data_list):
        """函数 2：执行入库逻辑"""
        # 初始化集合
        self._prepare_collection()
        
        total = len(data_list)
        for i in range(0, total, self.batch_size):
            batch = data_list[i : i + self.batch_size]
            batch_texts = [item["text"] for item in batch]
            
            # 模型推理
            output = self.model.encode(batch_texts, return_dense=True, return_sparse=True)
            dense_vecs = output['dense_vecs']
            lexical_weights = output['lexical_weights']
            
            points = []
            for j, item in enumerate(batch):
                idx = i + j + 1
                vectors = {self.DENSE_NAME: dense_vecs[j].tolist()}
                
                if self.hybrid:
                    vectors[self.SPARSE_NAME] = models.SparseVector(
                        indices=[int(k) for k in lexical_weights[j].keys()],
                        values=list(lexical_weights[j].values())
                    )

                points.append(models.PointStruct(
                    id=idx, 
                    vector=vectors, 
                    payload={"id": item["id"], "text": item["text"]}
                ))

            self.client.upsert(collection_name=self.collection_name, points=points)
            logging.info(f"入库进度: {min(i + self.batch_size, total)} / {total}")

    def _prepare_collection(self):
        """初始化 Qdrant 集合"""
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={self.DENSE_NAME: models.VectorParams(size=1024, distance=models.Distance.COSINE)},
            sparse_vectors_config={self.SPARSE_NAME: models.SparseVectorParams()} if self.hybrid else None
        )

    def run(self):
        """函数 3：顺序执行生成与入库"""
        if os.path.exists(self.output_json_path):
            with open(self.output_json_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
            logging.info(f"从已有文件加载数据：{self.output_json_path}")
        else:
            data_list = self.generate_json()
        self.insert_to_db(data_list)
        logging.info("所有流程执行完毕！")

def math_rag():
    config=load_config()
    insert_config=config.get("insert")["math_rag_system"]
    insert_qdrant=InsertQdrant(insert_config)
    insert_qdrant.run()
def normal():
    config=load_config()
    insert_config=config.get("insert")["baseline"]
    qdrant=NormalInsert(insert_config)
    qdrant.run()
if __name__ == "__main__":
    # 执行入库
    math_rag()
    normal()