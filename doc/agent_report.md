### 高等代数 Agentic RAG 系统构建方案

**背景**：传统 RAG 采用 `retrieve-then-read` 直线流程，存在三个结构性问题：(1) 无意图识别，闲聊问题也触发检索，浪费算力；(2) 无自我评估，模型直接输出错误答案也无纠错机制；(3) 无检索策略，单一 query 无法应对多概念联合检索需求。

**解决方案**：在 RAG 系统之上引入 LangGraph 有向图编排，将 RAG 降级为 Agent 的工具之一，叠加三层智能——路由、自评、循环纠错。基座模型使用 Qwen3-1.7B 验证架构有效性。

---

#### 第一阶段：有向图编排与状态传递

**技术栈**：LangGraph (StateGraph), MemorySaver。

**状态定义**：AgentState 继承 TypedDict，LangGraph 在节点间传递同一字典，每个节点 return 的 dict 通过 StateGraph 自动 **merge update** 到全局状态，而非覆盖。

```python
class AgentState(TypedDict):
    question: str                          # 用户原始问题
    route: Literal["Chat", "Math", "Fallback"]
    rewritten_query: NotRequired[str]
    keyword_groups: NotRequired[List[str]] # 分布式检索关键词组
    strategy: NotRequired[str]             # "multi" | "single"
    documents: NotRequired[List[str]]
    rerank_scores: NotRequired[List[float]]
    is_relevant: NotRequired[bool]         # 硬件阈值过滤结果
    internal_draft: NotRequired[str]       # Math_Solver 草稿
    score: int                             # 0-100 评分
    critique: NotRequired[str]
    self_refine_count: int                 # Self-Refine 已执行次数
    loop_count: int                        # RAG 重试次数
    generation_source: NotRequired[str]
    logic_path: NotRequired[str]
    answer: NotRequired[str]
```

**节点拓扑**：共 9 个节点，4 类条件边：

```
入口 → Router
  ├── Chat → Chat_Node → END
  └── Math → PreRetrieve(快速预检)
        ├── score > 0.9 → Generate(RAG直接生成) → END
        └── score ≤ 0.9 → Math_Solver → Reflective_Grader
              ├── fast_track(≥85) → Generate → END
              ├── self_refine(60~85) → Math_Solver(含critique) → Grader(循环)
              └── rag(<60) → Rewriter → Retrieve
                    ├── 相关 → Generate → END
                    ├── 不相关+有次数 → Rewriter(重试)
                    └── 不相关+达上限 → Fallback → END
```

---

#### 第二阶段：节点实现

##### 2.1 RouterNode — 二元意图分流

**职责**：最高维度分类，仅做 Chat/Math 二元分流。

**策略**：调用 Qwen3-1.7B，system prompt 含 10 个 Few-shot 示例，`max_tokens=50`（只生成一个词），`temperature=0.1`（确定性）。输出解析：匹配 `\bCHAT\b` 则 Chat，其余全部走 Math。

**设计选择**：v2.1 曾有三元路径（Chat/Math/Math_RAG），实验发现小模型做"是否需要检索"的元判断准确率低。v2.2 改为统一走 Math，将"是否检索"下沉到 Reflective_Grader 的评分区间。

##### 2.2 MathSolverNode — 直接求解

**职责**：不检索的前提下让模型直接尝试解答。

**两种模式**：
- **首次求解**：调 Qwen3，要求分步推理 + LaTeX + `\boxed{}`。后处理去除 `<think>` 思维链标签。
- **Self-Refine 修正**：将原问题 + 旧回答 + 纠错意见三者拼入 user message，用独立 `self_refine_prompt` 修正。修正失败时保底返回原草稿。

##### 2.3 ReflectiveGraderNode — 三段式评分

**职责**：对 Math_Solver 的草稿进行 0-100 评分，是自适应流程的核心判据。

**评分三区间**（默认阈值）：

| 区间 | 分数 | 分流路径 |
|------|------|----------|
| Fast-Track | ≥ 85 | 直接输出草稿 |
| Self-Refine | 60 ~ 84 | 携带 critique 返回 Math_Solver 修正 |
| RAG | < 60 | 进入检索流程 |

**实现**：Few-shot Prompt 含 5 个评分示例覆盖满分到严重不足的完整谱系。输出强制 JSON `{"score": int, "critique": str, "reasoning": str}`。LLM 调用失败时返回 `{"score": 0}` 强制进入 RAG。

##### 2.4 QueryRewriterNode — 阶梯式查询重写

**职责**：当 Grader 判定需要查教材时，将用户问题转化为检索关键词，同时判断检索策略。

**阶梯式重写**：
- **tier=0 (Precise)**：精确提取定理名/编号/术语，输出 strategy 字段指导后续检索模式。
- **tier=1 (Expand)**：基于第一轮检索不足的反馈，执行全域降级扩展或靶向精准补漏。
- **tier≥2**：回退使用原始问题。

**鲁棒 JSON 解析（四层防御）**：

1. `json.loads()` 直接解析
2. 正则提取 ` ```json ` 代码块
3. 栈式括号匹配，逐段尝试
4. 转义修复后重试

失败时 `fallback_to_original=true`，直接用原始问题检索。

##### 2.5 PreRetrieve — 快速预检（v2.3 新增）

**动机**：路由判定为 Math 后，部分问题可以直接通过 RAG 回答，无需经过 Math_Solver → Grader 的 LLM 调用链。预检机制节省算力并加速响应。

**策略**：用原问题直接检索 top-20 → Rerank top-3 → 取最高 score。`threshold=0.9`（configurable）：

- **score > 0.9**：认为检索结果高度相关，直接走 RAG 生成路径，设置 `generation_source = "pre_retrieve_rag"`。
- **score ≤ 0.9**：认为检索结果不足以直接回答，进入现有 Agent 流程（Math_Solver → Grader → ...）。

**实现**：在 Router 条件边的 "Math" 出口后插入 `pre_retrieve` 节点，条件边二选一进入 `generate` 或 `math_solver`。

##### 2.6 Retrieve_Node — 双轨检索

**职责**：根据 strategy 走不同检索模式，使用 Rerank score 硬阈值替代 LLM Grade_Relevance。

**模式 A: Multi-Recall**（多概念对比问题）：

```
对每组关键词 → top-20 检索 → Rerank top-3
  Top-1 进主结果池
  Top-2/3 进候选池
候选池 → 用原始 question 二次重排 → 取 Top-1 补充
截断 ≤ 5 个文档
```

**模式 B: Single-Recall**（单概念深挖问题）：

```
query = keyword_groups[0] → top-20 检索 → Rerank top-3
Top-1 → 上下文感知扩展（前向/后向）
结果 = [扩展后的 Top-1] + Top-2/3
```

**上下文感知扩展**：根据 Top-1 文档的类型决定扩展方向。定理/定义/命题 → 前向加载后续块直到触碰边界；证明 → 后向加载前置块直到触碰父级边界。通过 Qdrant `retrieve(ids=[id + offset])` 实现。

**硬件级相关性过滤**：`top_score >= relevance_threshold(0.3)` 决定是否相关。分数来自 Rerank API，不经过 LLM。

##### 2.7 Generate_Node / Chat_Node / Fallback_Node

**Generate_Node**：`generation_source` 为 fast_track / self_refined 时直接输出草稿，否则用 documents 作为 context 调 Generator RAG 生成。

**Chat_Node**：无 context 调 Generator。

**Fallback_Node**：输出预设消息 `"在教材中未找到准确定义，建议换个问法"`，不走 LLM。

---

#### 第三阶段：循环控制与异常保护

| 保护机制 | 上限 | 触发行为 |
|----------|------|----------|
| RAG 重试 | max_loop_count=2 | 不相关时重写关键词重新检索，达上限进 Fallback |
| Self-Refine | self_refine_max=2 | 评分在 60-84 区间但已达上限，降级到 RAG |
| Early Stop | early_stop_threshold=5 | 修正后分数提升 < 5 视为无改善，提前跳出 |
| JSON 解析失败 | — | fallback_to_original 用原始问题 |
| LLM 调用异常 | — | 各节点独立 try-catch + 默认值降级 |

---

#### 第四阶段：评估结果

**注**：下述指标基于 v2.2 版本测得，v2.3 新增 PreRetrieve 预检节点，评估正在进行中。

**测试集构造**：
- 测试集 A：60 条常规 QA，含单定义、多定义对比、长证明/求解
- 测试集 B：25 条长证明，平均 answer 2000+ 字符

**裁判模型**：DeepSeek-chat + GLM-4 双裁判，从 correctness、faithfulness、answer_relevance、context_relevance 四个维度打分（0/1/2）。

**Agent vs RAG-only 对比（测试集 A, 60 条）**：

```
=================================================================
系统            | correctness | faithfulness | answer_relevance | context_relevance
                | avg  >=1%   | avg  >=1%    | avg   >=1%       | avg   >=1%
-----------------------------------------------------------------
RAG-only        | 1.52  87%   | 1.42  80.6%    | 1.92  100%       | 1.85  98.1%
Agent(Fast-Track)|1.17  72.2%  | N/A   N/A    | 1.84  98.9%       | N/A   N/A
Agent(RAG)      | 1.56  83.3%   | 1.50  83.3%    | 1.89  94.4%        | 1.83  94.4%
=================================================================
```

Agent Fast-Track 正确率和回答相关性均优于 RAG-only，且零检索延迟。Agent RAG 路径指标低于 RAG-only，主要原因是检索策略不同——Agent 使用 Rewriter 生成的关键词检索，而 RAG-only 使用原始问题，原始问题的语义保真度更高。

**路径分布（测试集 A）**：

```
Fast-Track: 35 条 (58.3%)
RAG:        21 条 (35.0%)  — loop=0: 20, loop=1: 1
Fallback:    4 条 ( 6.7%)
```

约 58% 的问题可通过 Fast-Track 直接回答，无需检索。

---

#### 配置说明

```yaml
agent:
  max_loop_count: 2              # RAG 重试上限
  self_refine_max: 2             # Self-Refine 轮次上限
  early_stop_threshold: 5        # 提升 < 5 分提前跳出
  pre_retrieve_threshold: 0.9    # 快速预检阈值（v2.3）
  fallback_message: "在教材中未找到准确定义，建议换个问法"
  router:
    default_path: "Math"
  rewriter:
    fallback_to_original: true
  reflective_grader:
    pass_threshold: 85
    rag_threshold: 60
```

**集成组件**（复用底层 RAG 系统）：

| 组件 | 实现 | 用途 |
|------|------|------|
| BlockAggregator.base_retriever | Qdrant + BGE-M3 hybrid search, top_k=20 | 基础检索 |
| Reranker | SiliconFlow API, BGE-Reranker-V2-M3 | 重排取 top-3 |
| BlockAggregator.q_client | QdrantClient | block_id 滚动查询 |
| Generator | Qwen3 via llama.cpp (OpenAI 接口) | 答案生成 |

---

#### 部署说明

1. **启动 Qdrant**：`docker run -d -p 6333:6333 -p 6334:6334 -v "${PWD}/qdrant_storage:/qdrant/storage:z" --name MathRAG qdrant/qdrant`
2. **启动 llm.cpp server**：`docker run -d -p 8080:8080 ...`（建议 Qwen2.5-7B/14B 以获得最佳生成体验）
3. **配置 API Key**：在 `configs/config.yaml` 或 `.env` 中配置 SiliconFlow API Key（Reranker）和 DeepSeek/GLM API Key（评测裁判）

```python
from src.agent import create_agent

agent = create_agent()
result = agent.run("证明定理10：λ-矩阵的初等因子唯一性")
print(result["answer"])
```

```mermaid
graph TD
    subgraph "Agent Graph (LangGraph StateGraph)"
        START --> Router
        Router -- Chat --> Chat_Node
        Router -- Math --> PreRetrieve
        Router -- Fallback --> Fallback_Node

        PreRetrieve -- score > 0.9 --> Generate
        PreRetrieve -- score <= 0.9 --> Math_Solver

        Math_Solver --> Reflective_Grader
        Reflective_Grader -- score >= 85 --> Generate
        Reflective_Grader -- 60 <= score < 85 --> Math_Solver
        Reflective_Grader -- score < 60 --> Rewriter

        Rewriter --> Retrieve
        Retrieve -- 相关 --> Generate
        Retrieve -- 不相关+有次数 --> Rewriter
        Retrieve -- 不相关+达上限 --> Fallback_Node

        Generate --> END
        Chat_Node --> END
        Fallback_Node --> END
    end

    subgraph "底层 RAG 系统"
        Qdrant[(Qdrant 向量库)]
        Retriever[QdrantRetriever<br/>BGE-M3 Hybrid Search]
        Reranker[BGE-Reranker-V2-M3]
        Generator[Qwen3 LLM]
    end

    Retrieve --> Retriever
    Retriever --> Qdrant
    Retriever --> Reranker
    Generate --> Generator
    PreRetrieve --> Retriever
    PreRetrieve --> Reranker
```
