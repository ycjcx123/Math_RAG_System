from typing import TypedDict, List, Optional, Literal
from typing_extensions import NotRequired


class AgentState(TypedDict):
    """Agent 状态定义 (v2.2)"""

    # 用户输入
    question: str

    # 路由决策: Chat / Math（二元分类）
    route: Literal["Chat", "Math", "Fallback"]

    # 查询重写结果
    rewritten_query: NotRequired[str]
    extracted_keywords: NotRequired[str]
    keyword_groups: NotRequired[List[str]]    # 分布式检索：最多 4 组关键词
    strategy: NotRequired[str]                # "multi" | "single"（Rewriter 输出）

    # 检索结果 + 相关性（硬件级过滤, 替代原 LLM Grade_Relevance）
    documents: NotRequired[List[str]]
    document_ids: NotRequired[List[str]]
    rerank_scores: NotRequired[List[float]]   # Rerank API 返回的分数
    is_relevant: NotRequired[bool]

    # Math_Solver 直接生成的草稿
    internal_draft: NotRequired[str]

    # Reflective_Grader 输出
    score: int              # 0-100 数值评分
    critique: NotRequired[str]  # <85 分时的纠错意见

    # Self-Refine 循环控制
    self_refine_count: int  # Self-Refine 已执行次数

    # 循环控制（RAG 重试）
    loop_count: int

    # 生成来源追踪
    generation_source: NotRequired[str]  # "fast_track" | "self_refined" | "rag"

    # 决策追踪
    logic_path: NotRequired[str]

    # 最终答案
    answer: NotRequired[str]

    # 错误信息
    error: NotRequired[str]


def create_initial_state(question: str) -> AgentState:
    """创建初始 Agent 状态"""
    return {
        "question": question,
        "route": "Math",              # 默认走 Math
        "loop_count": 0,
        "self_refine_count": 0,
        "score": 0,
        "documents": [],
        "document_ids": [],
        "rerank_scores": [],
        "is_relevant": False,
        "answer": "",
        "internal_draft": "",
        "critique": "",
        "logic_path": "start",
        "error": ""
    }
