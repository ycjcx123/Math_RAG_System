import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state
from .nodes.router import RouterNode
from .nodes.rewriter import QueryRewriterNode
from .nodes.grader import GraderNode
from src.utils.config_loader import load_config


class AgentGraph:
    """Agent 图编排"""

    def __init__(self, config: dict = None):
        """初始化 Agent 图

        Args:
            config: 配置字典，如果为 None 则自动加载
        """
        if config is None:
            config = load_config()

        self.config = config
        agent_config = config.get("agent", {})

        # 获取配置参数
        self.max_loop_count = agent_config.get("max_loop_count", 2)
        self.fallback_message = agent_config.get("fallback_message",
                                                "在教材中未找到准确定义，建议换个问法")

        # 初始化节点
        self.router_node = RouterNode(config)
        self.rewriter_node = QueryRewriterNode(config)
        self.grader_node = GraderNode(config)

        # 初始化 RAG 工具（使用现有的 retriever）
        from src.pipeline.retriever import ChatPipeline
        self.retriever = ChatPipeline(config, query="")  # query 会在运行时设置

        # 初始化生成器
        from src.rag.generator import Generator
        self.generator = Generator(config.get("generator", {}))

        # 构建图
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)

        logging.info(f"AgentGraph 初始化完成，最大循环次数: {self.max_loop_count}")

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 图结构"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("router", self.router_node)
        workflow.add_node("rewriter", self.rewriter_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grader", self.grader_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("fallback", self._fallback_node)
        workflow.add_node("chat", self._chat_node)

        # 设置入口点
        workflow.set_entry_point("router")

        # 添加边（路由决策）
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "RAG": "rewriter",
                "Chat": "chat",
                "Fallback": "fallback"
            }
        )

        # RAG 路径
        workflow.add_edge("rewriter", "retrieve")
        workflow.add_edge("retrieve", "grader")

        # 评分后决策
        workflow.add_conditional_edges(
            "grader",
            self._grade_decision,
            {
                "relevant": "generate",
                "not_relevant": "router",  # 重新路由（循环）
                "max_loop": "fallback"     # 达到最大循环次数
            }
        )

        # 最终节点
        workflow.add_edge("generate", END)
        workflow.add_edge("chat", END)
        workflow.add_edge("fallback", END)

        return workflow

    def _route_decision(self, state: AgentState) -> Literal["RAG", "Chat", "Fallback"]:
        """路由决策函数"""
        loop_count = state.get("loop_count", 0)

        # 检查是否达到最大循环次数
        if loop_count >= self.max_loop_count:
            logging.warning(f"达到最大循环次数 ({loop_count})，跳转到 Fallback")
            return "Fallback"

        # 使用路由节点的决策
        route = state.get("route", "RAG")
        return route

    def _grade_decision(self, state: AgentState) -> Literal["relevant", "not_relevant", "max_loop"]:
        """评分决策函数"""
        loop_count = state.get("loop_count", 0)
        is_relevant = state.get("is_relevant", False)

        # 检查是否达到最大循环次数
        if loop_count >= self.max_loop_count:
            logging.warning(f"达到最大循环次数 ({loop_count})，跳转到 Fallback")
            return "max_loop"

        # 根据相关性决定下一步
        if is_relevant:
            return "relevant"
        else:
            logging.info(f"文档不相关，循环计数: {loop_count}")
            return "not_relevant"

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """检索节点：调用 RAG 工具获取文档"""
        question = state.get("question", "")
        rewritten_query = state.get("rewritten_query", question)

        logging.info(f"检索节点: 使用查询 '{rewritten_query}' 进行检索")

        try:
            # 更新 retriever 的查询
            self.retriever.query = rewritten_query

            # 执行检索
            documents = self.retriever.run()

            # 提取文档文本
            document_texts = []
            if isinstance(documents, tuple):
                # 处理可能返回的多种格式
                if len(documents) >= 2:
                    document_texts = documents[0] if isinstance(documents[0], list) else []
                else:
                    document_texts = documents if isinstance(documents, list) else []
            elif isinstance(documents, list):
                document_texts = documents
            else:
                document_texts = [str(documents)]

            logging.info(f"检索到 {len(document_texts)} 个文档片段")

            return {
                "documents": document_texts
            }

        except Exception as e:
            logging.error(f"检索失败: {e}")
            return {
                "documents": [],
                "error": f"检索失败: {str(e)}"
            }

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """生成节点：使用检索到的文档生成答案"""
        question = state.get("question", "")
        documents = state.get("documents", [])

        logging.info(f"生成节点: 基于 {len(documents)} 个文档生成答案")

        try:
            answer = self.generator.generate(
                query=question,
                contexts=documents
            )

            return {
                "answer": answer
            }

        except Exception as e:
            logging.error(f"生成失败: {e}")
            return {
                "answer": f"生成答案时出错: {str(e)}",
                "error": str(e)
            }

    def _chat_node(self, state: AgentState) -> Dict[str, Any]:
        """聊天节点：直接生成回答（不依赖检索）"""
        question = state.get("question", "")

        logging.info(f"聊天节点: 直接回答 '{question}'")

        try:
            answer = self.generator.generate(
                query=question,
                contexts=None  # 不提供上下文，直接生成
            )

            return {
                "answer": answer
            }

        except Exception as e:
            logging.error(f"聊天生成失败: {e}")
            return {
                "answer": f"生成回答时出错: {str(e)}",
                "error": str(e)
            }

    def _fallback_node(self, state: AgentState) -> Dict[str, Any]:
        """Fallback 节点：处理失败情况"""
        logging.info("进入 Fallback 节点")

        return {
            "answer": self.fallback_message,
            "route": "Fallback"
        }

    def run(self, question: str, config: dict = None) -> Dict[str, Any]:
        """运行 Agent

        Args:
            question: 用户问题
            config: 可选配置覆盖

        Returns:
            包含最终答案和状态的字典
        """
        logging.info(f"Agent 开始处理问题: {question}")

        # 创建初始状态
        initial_state = create_initial_state(question)

        # 运行图
        try:
            final_state = self.compiled_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": "math_agent"}}
            )

            # 提取结果
            result = {
                "question": question,
                "answer": final_state.get("answer", ""),
                "route": final_state.get("route", "Unknown"),
                "loop_count": final_state.get("loop_count", 0),
                "documents_retrieved": len(final_state.get("documents", [])),
                "is_relevant": final_state.get("is_relevant", False),
                "error": final_state.get("error", "")
            }

            logging.info(f"Agent 处理完成，循环次数: {result['loop_count']}")
            return result

        except Exception as e:
            logging.error(f"Agent 运行失败: {e}")
            return {
                "question": question,
                "answer": f"Agent 系统错误: {str(e)}",
                "route": "Error",
                "loop_count": 0,
                "documents_retrieved": 0,
                "is_relevant": False,
                "error": str(e)
            }


# 简化接口
def create_agent(config: dict = None) -> AgentGraph:
    """创建 Agent 实例"""
    return AgentGraph(config)