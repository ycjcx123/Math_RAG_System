from typing import TypedDict, List, Optional, Literal
from typing_extensions import NotRequired


class AgentState(TypedDict):
    """Agent 状态定义，用于 LangGraph 状态管理"""

    # 用户输入
    question: str

    # 路由决策
    route: Literal["RAG", "Chat", "Fallback"]

    # 查询重写结果
    rewritten_query: NotRequired[str]
    extracted_keywords: NotRequired[str]

    # 检索结果
    documents: NotRequired[List[str]]
    document_ids: NotRequired[List[str]]

    # 评分结果
    is_relevant: NotRequired[bool]

    # 循环控制
    loop_count: int

    # 最终答案
    answer: NotRequired[str]

    # 错误信息
    error: NotRequired[str]


def create_initial_state(question: str) -> AgentState:
    """创建初始 Agent 状态"""
    return {
        "question": question,
        "route": "RAG",  # 默认走 RAG 路径
        "loop_count": 0,
        "documents": [],
        "document_ids": [],
        "is_relevant": False,
        "answer": "",
        "error": ""
    }