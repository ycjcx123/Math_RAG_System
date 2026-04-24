from typing import TypedDict, List, Optional, Literal
from typing_extensions import NotRequired


class AgentState(TypedDict):
    """Agent 状态定义，用于 LangGraph 状态管理 (v2.0)"""

    # 用户输入
    question: str

    # 路由决策: Chat / Math / Math_RAG
    route: Literal["Chat", "Math", "Math_RAG", "Fallback"]

    # 查询重写结果
    rewritten_query: NotRequired[str]
    extracted_keywords: NotRequired[str]

    # 检索结果
    documents: NotRequired[List[str]]
    document_ids: NotRequired[List[str]]

    # Math_Solver 直接生成的草稿
    internal_draft: NotRequired[str]

    # Reflective_Grader 输出
    score: int              # 0-100 数值评分
    critique: NotRequired[str]  # <80 分时的纠错意见

    # 评分结果（传统 Grader 保留）
    is_relevant: NotRequired[bool]

    # 循环控制
    loop_count: int

    # 决策追踪
    logic_path: NotRequired[str]  # 记录完整决策路径

    # 最终答案
    answer: NotRequired[str]

    # 错误信息
    error: NotRequired[str]


def create_initial_state(question: str) -> AgentState:
    """创建初始 Agent 状态"""
    return {
        "question": question,
        "route": "Math_RAG",          # 默认走 Math_RAG
        "loop_count": 0,
        "score": 0,                   # 初始评分 0
        "documents": [],
        "document_ids": [],
        "is_relevant": False,
        "answer": "",
        "internal_draft": "",
        "critique": "",
        "logic_path": "start",
        "error": ""
    }
