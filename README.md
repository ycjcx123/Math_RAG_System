# MathRAG-Algebra: 高等代数语义感知的 RAG 系统

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.10%2B-green)
![Framework](https://img.shields.io/badge/framework-LlamaIndex-orange)

## 📌 项目背景
传统的 RAG 系统在处理数学教材时存在两大痛点：
1. **OCR 质量受限**：公式符号错乱、括号失衡、跨页公式断裂。
2. **语义块破碎**：传统的固定长度切分会切断长证明（Proof）和解题步骤（Solution），导致生成结果逻辑断裂。

本项目针对**高等代数**教材进行了彻底重构，通过**逻辑断点切分**与**公式局部修复**技术，显著提升了数学场景下的检索准确率与生成质量。

---

## ✨ 核心特性

### 1. 公式自动化修复 (Formula Fixer)
- **发现机制**：基于严格正则规则与 PyLaTeX 渲染校验，自动识别异常公式块。
- **局部重写策略**：利用 LLM 仅对 `[TARGET]` 区域进行修复，在纠正 LaTeX 语法的同时最大限度保留非数学原文，避免无效 Token 干扰。

### 2. 语义感知切分 (Logic-aware Chunking)
- **逻辑断点识别**：按 `定义|定理|证明|命题` 等数学关键词边界切分。
- **结构路径注入**：每个 Chunk 头部自动补全章节路径（如 `[第7章 > 7.1 > 定义]`），强化检索时的环境特征。
- **Block 聚合逻辑**：通过 `block_id` 关联长证明的不同片段，确保在生成阶段能够召回完整的逻辑链条。

### 3. 高性能检索 Pipeline
- **混合检索**：结合 BGE-M3 的稠密 (Dense) 与稀疏 (Sparse) 向量，通过 RRF 算法融合。
- **精细重排**：使用 BGE-Reranker-V2-M3 进行 Top-K 过滤，严格控制上下文窗口。

graph TD
    subgraph "Stage 1: 文档预处理 (Parser & Fixer)"
        A[原始 PDF/MD] --> B[MinerU OCR 解析]
        B --> C{公式检测}
        C -- 报错/语法错 --> D[LLM 局部重写 + PyLaTeX 校验]
        C -- 正常 --> E[结构化文本]
        D --> E
    end

    subgraph "Stage 2: 语义感知切分 (Splitter)"
        E --> F[标题路径注入]
        F --> G[逻辑边界切分: 定义/定理/证明]
        G --> H[Token 动态调整: 过短合并/过长切分]
        H --> I[元数据分配: Block ID & Part ID]
    end

    subgraph "Stage 3: 混合检索与增强 (Retriever)"
        I --> J[(Qdrant 向量库)]
        K[用户查询] --> L[Hybrid Search: Dense + Sparse]
        L --> M[RRF 排序融合]
        M --> N[BGE-Reranker 重排序]
        N --> O[Block 聚合: 根据 ID 还原完整逻辑块]
    end

    subgraph "Stage 4: 答案生成 (Generator)"
        O --> P[Qwen LLM 推理]
        P --> Q[输出最终答案]
    end

---

## 📊 实验表现

> **说明**：以下指标基于测试集 A（单点/对比问答）与测试集 B（长证明压力测试）测得。

### 检索性能对比
| 系统版本 | MAP@20 | MRR@20 | Recall@20 |
| :--- | :---: | :---: | :---: |
| Baseline (Dense) | 0.4358 | 0.5365 | 0.7178 |
| Baseline (Hybrid) | 0.3677 | 0.4916 | 0.6428 |
| **MathRAG (Ours)** | **0.6125** | **0.7083** | **0.7194** |

### 生成质量 (LLM-Judgement)
使用 GPT-4o 级模型（GLM-4 / DeepSeek）对正确性、忠诚度等四个维度进行 0-2 分盲测：
- **MathRAG 方案在“忠诚度 (Faithfulness)”维度提升显著（+15%）**，主要归功于公式修复与完整 Block 召回。

---

## 🛠️ 快速开始

### 1. 环境准备
```bash
# 克隆仓库
git clone https://github.com/ycjcx123/Math_RAG_System
cd MathRAG

# 部署向量数据库
docker run -d -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage:z" --name MathRAG qdrant/qdrant
```




--------------------------
关于部署的问题

暂时仅支持本地Qdrant数据库，建议使用docker部署
关于Qdrant数据库，建议执行：docker run -d -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage:z" --name MathRAG qdrant/qdrant

关于config内的参数:
    请仔细阅读configs/config.yaml中的注释
    尤其是api，如果使用.env文件的话，请自行对齐
    如果不打算使用.env文件的话，请直接在config中更换api_key;

关于测试：
    目前仅支持使用llm-judgement，仅支持glm和deepseek（OpenAi格式）的api；虽然deepseek使用的是OpenAi格式，但是字段可能需要自行配置
    注：下述指标基于重构前的核心逻辑版本（v0.1）测得。目前 v1.0 重构版已跑通全流程 Pipeline，由于 LLM-Judgement 运行成本及时间限制，全量测试集的新版本指标正在更新中，但核心性能趋势具有一致性。

关于模型：
    注意：Qwen3-1.7B 性能可能受限，推荐使用 Qwen2.5-7B/14B 或 DeepSeek-V3 以获得最佳生成体验。

```
Math_RAG_System
├─ .env                         # 存放 API_KEY（Qwen, GLM4, DeepSeek）
├─ download.py                  # 用来下载模型
├─ LICENSE
├─ main.py                      # 全局入口
├─ README.md
├─ model/                       # 用来存放模型
├─ test                         # 测试位置
│  ├─ longTest.json
│  ├─ Test.json
│  ├─ test.py                   # 测试代码
│  └─ result                    # 测试结果存放位置
│     ├─ Long_Test.json
│     └─ Test_Result2.json
├─ src                          # 源码
│  ├─ __init__.py
│  ├─ utils
│  │  ├─ config_loader.py       # 统一读取 yaml 配置文件
│  │  ├─ insertQdrant.py        # 用来插入数据库 
│  │  └─ __init__.py
│  ├─ retriever
│  │  ├─ context_builder.py     # 用来执行block聚合：召回-重排-block聚合
│  │  ├─ reranker.py            # 用来执行重排
│  │  ├─ searcher.py            # 用来召回
│  │  └─ __init__.py
│  ├─ pipeline
│  │  ├─ chat_pipeline.py       # 对话流水线：Searcher -> Reranker -> ContextBuilder -> LLM
│  │  ├─ ingest_pipeline.py     # 入库流水线：Parser -> Fixer -> Splitter -> Qdrant
│  │  └─ __init__.py
│  ├─ parser
│  │  ├─ formula_fixer.py       # 核心：正则扫描 + LLM 局部重写 + PyLaTeX 校验
│  │  └─ __init__.py
│  ├─ generator
│  │  ├─ generate.py            # Qwen3 调用逻辑
│  │  └─ __init__.py
│  ├─ evaluation
│  │  ├─ evaluator.py           # 用来端对端的测试 
│  │  ├─ score.py               # 用来评价召回、重排的质量
│  │  └─ __init__.py
│  └─ chunked
│     ├─ chunk.py               # 针对文件进行分块，逻辑断点切分、Token 动态调整、ID 分配逻辑、超长分块
│     └─ __init__.py
├─ Data
│  ├─ raw                       # 原始文件,这里为了简便，不提供原始教材
│  │  └─ full.md
│  └─ processed                 # 处理后的文件
│     ├─ baseline_chunks.json
│     ├─ fixed.md               
│     └─ math_chunks.json
└─ configs                      # 模板在这
   ├─ config.yaml
   ├─ config_example.yaml
   └─ fewShot.yaml              # LLM局部重写所使用的
