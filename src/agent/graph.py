import logging
from typing import Literal, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state
from .nodes.router import RouterNode
from .nodes.rewriter import QueryRewriterNode
from .nodes.grader import ReflectiveGraderNode
from .nodes.math_solver import MathSolverNode
from src.utils.config_loader import load_config


class AgentGraph:
    """Agent 图编排 (v2.0)

    新增拓扑：
    入口 → Router
      ├── Chat → Chat_Node → END
      ├── Math → Math_Solver → Reflective_Grader
      │         ├── score>=80 → Generate → END  (草稿直接输出)
      │         └── score<80  → Rewriter → Retrieve → Grader →
      │                                              ├── relevant → Generate → END
      │                                              └── not_relevant → Router/Fallback
      └── Math_RAG → Rewriter → Retrieve → Grader →
                                           ├── relevant → Generate → END
                                           └── not_relevant → Router/Fallback
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config
        agent_config = config.get("agent", {})

        self.max_loop_count = agent_config.get("max_loop_count", 2)
        self.fallback_message = agent_config.get("fallback_message",
                                                  "在教材中未找到准确定义，建议换个问法")

        # 初始化节点
        self.router_node = RouterNode(config)
        self.rewriter_node = QueryRewriterNode(config)
        self.math_solver_node = MathSolverNode(config)
        self.reflective_grader_node = ReflectiveGraderNode(config)

        # 注意：grader.py 现在被 ReflectiveGraderNode 占用
        # 文档相关性评分用内联节点 _grade_relevance_node 实现
        self.grader_node = None  # 不再使用原有的 GraderNode

        # 初始化 RAG 工具
        from src.pipeline.retriever import ChatPipeline
        self.retriever = ChatPipeline(config, query="")

        # 初始化生成器
        from src.rag.generator import Generator
        self.generator = Generator(config.get("generator", {}))

        # 构建图
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)

        logging.info(f"AgentGraph v2.0 初始化完成，最大循环次数: {self.max_loop_count}")

    # ==================== 图构建 ====================

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # 节点
        workflow.add_node("router", self.router_node)
        workflow.add_node("math_solver", self.math_solver_node)
        workflow.add_node("reflective_grader", self.reflective_grader_node)
        workflow.add_node("rewriter", self.rewriter_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade_relevance", self._grade_relevance_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("fallback", self._fallback_node)

        workflow.set_entry_point("router")

        # === Router → 三元分流 ===
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "Chat": "chat",
                "Math": "math_solver",
                "Math_RAG": "rewriter",
                "Fallback": "fallback"
            }
        )

        # === Math 路径：解 → 自检 ===
        workflow.add_edge("math_solver", "reflective_grader")

        # === Reflective_Grader 条件分流 ===
        workflow.add_conditional_edges(
            "reflective_grader",
            self._reflective_decision,
            {
                "generate": "generate",     # score>=80，直接输出草稿
                "rewriter": "rewriter"       # score<80，进入 RAG 流程补全
            }
        )

        # === RAG 路径 (Math_RAG 入口 + Reflective 失败降级) ===
        workflow.add_edge("rewriter", "retrieve")
        workflow.add_edge("retrieve", "grade_relevance")

        # === 文档相关性评分 → 条件分流 ===
        workflow.add_conditional_edges(
            "grade_relevance",
            self._grade_decision,
            {
                "relevant": "generate",
                "not_relevant": "router",   # 重新路由（循环）
                "max_loop": "fallback"
            }
        )

        # === 终止节点 ===
        workflow.add_edge("generate", END)
        workflow.add_edge("chat", END)
        workflow.add_edge("fallback", END)

        return workflow

    # ==================== 决策函数 ====================

    def _route_decision(self, state: AgentState) -> Literal["Chat", "Math", "Math_RAG", "Fallback"]:
        """路由决策：Chat / Math / Math_RAG / Fallback"""
        loop_count = state.get("loop_count", 0)

        if loop_count >= self.max_loop_count:
            logging.warning(f"达到最大循环次数 ({loop_count})，跳转到 Fallback")
            return "Fallback"

        route = state.get("route", "Math_RAG")
        return route

    def _reflective_decision(self, state: AgentState) -> Literal["generate", "rewriter"]:
        """反思评分决策：>=80 直接输出，<80 进入 RAG 补全"""
        score = state.get("score", 0)
        loop_count = state.get("loop_count", 0)

        logging.info(f"反思评分决策: score={score}, loop_count={loop_count}")

        if score >= 80:
            logging.info("反思评分通过，直接输出草稿")
            return "generate"

        # score < 80，检查是否还有循环余量
        if loop_count >= self.max_loop_count:
            logging.warning(f"反思评分不通过且达到循环上限，进入 Fallback")
            return "rewriter"  # 但后续 grade_decision 会拦住

        logging.info(f"反思评分不通过，携带 critique 进入 RAG 检索补全")
        return "rewriter"

    def _grade_decision(self, state: AgentState) -> Literal["relevant", "not_relevant", "max_loop"]:
        """文档相关性评分决策"""
        loop_count = state.get("loop_count", 0)
        is_relevant = state.get("is_relevant", False)

        if loop_count >= self.max_loop_count:
            logging.warning(f"达到最大循环次数 ({loop_count})，跳转到 Fallback")
            return "max_loop"

        if is_relevant:
            return "relevant"

        logging.info(f"文档不相关，循环计数: {loop_count}")
        return "not_relevant"

    # ==================== 节点实现 ====================

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")
        rewritten_query = state.get("rewritten_query", question)

        logging.info(f"检索节点: 使用查询 '{rewritten_query}'")

        try:
            self.retriever.query = rewritten_query
            documents = self.retriever.run()

            document_texts = []
            if isinstance(documents, tuple):
                if len(documents) >= 2:
                    document_texts = documents[0] if isinstance(documents[0], list) else []
                else:
                    document_texts = documents if isinstance(documents, list) else []
            elif isinstance(documents, list):
                document_texts = documents
            else:
                document_texts = [str(documents)]

            logging.info(f"检索到 {len(document_texts)} 个文档片段")
            return {"documents": document_texts}

        except Exception as e:
            logging.error(f"检索失败: {e}")
            return {"documents": [], "error": f"检索失败: {str(e)}"}

    def _grade_relevance_node(self, state: AgentState) -> Dict[str, Any]:
        """
        文档相关性评分节点（内联实现，替代原有的 GraderNode）
        注意：这与 ReflectiveGraderNode 不同，后者评估 Math_Solver 的草稿
        本节点评估检索到的文档与问题的相关性
        """
        question = state.get("question", "")
        documents = state.get("documents", [])
        loop_count = state.get("loop_count", 0)

        if not documents:
            logging.warning("文档为空，判定为不相关")
            return {"is_relevant": False, "loop_count": loop_count + 1}

        # 用 LLM 简单判断文档是否相关（复用 Generator）
        if documents:
            # 取前 2 个文档的前 400 字做快速评估
            sample = "\n".join([d[:400] for d in documents[:2]])
            prompt = (
                f"问题: {question}\n\n"
                f"参考资料:\n{sample}\n\n"
                f"这些资料是否包含回答该问题所需的关键数学信息？只回答 Yes 或 No。"
            )

            try:
                response = self.generator.client.chat.completions.create(
                    model=self.generator.model_name,
                    messages=[
                        {"role": "system", "content": "你是一个文档相关性判断助手。只回答 Yes 或 No。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                answer = response.choices[0].message.content.strip().upper()
                is_relevant = "YES" in answer
            except Exception:
                is_relevant = False

            logging.info(f"文档相关性判断: {'相关' if is_relevant else '不相关'}")

            if not is_relevant:
                return {"is_relevant": is_relevant, "loop_count": loop_count + 1}

            return {"is_relevant": is_relevant}

        return {"is_relevant": False, "loop_count": loop_count + 1}

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """生成节点：优先使用 internal_draft（反思通过的草稿），否则走 RAG 生成"""
        question = state.get("question", "")
        documents = state.get("documents", [])
        internal_draft = state.get("internal_draft", "")

        # 如果来自反思评分通过，直接使用草稿
        if internal_draft:
            score = state.get("score", 0)
            if score >= 80:
                logging.info("生成节点: 使用反思通过的草稿")
                return {"answer": internal_draft}

        # 否则使用 RAG 生成
        logging.info(f"生成节点: 基于 {len(documents)} 个文档生成答案")

        try:
            answer = self.generator.generate(query=question, contexts=documents if documents else None)
            return {"answer": answer}
        except Exception as e:
            logging.error(f"生成失败: {e}")
            return {"answer": f"生成答案时出错: {str(e)}", "error": str(e)}

    def _chat_node(self, state: AgentState) -> Dict[str, Any]:
        question = state.get("question", "")

        try:
            answer = self.generator.generate(query=question, contexts=None)
            return {"answer": answer}
        except Exception as e:
            logging.error(f"聊天生成失败: {e}")
            return {"answer": f"生成回答时出错: {str(e)}", "error": str(e)}

    def _fallback_node(self, state: AgentState) -> Dict[str, Any]:
        return {"answer": self.fallback_message, "route": "Fallback"}

    # ==================== 运行入口 ====================

    def run(self, question: str, config: dict = None) -> Dict[str, Any]:
        logging.info(f"Agent v2.0 开始处理问题: {question}")

        initial_state = create_initial_state(question)

        try:
            final_state = self.compiled_graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": "math_agent"}}
            )

            result = {
                "question": question,
                "answer": final_state.get("answer", ""),
                "route": final_state.get("route", "Unknown"),
                "loop_count": final_state.get("loop_count", 0),
                "score": final_state.get("score", 0),
                "critique": final_state.get("critique", ""),
                "documents_retrieved": len(final_state.get("documents", [])),
                "is_relevant": final_state.get("is_relevant", False),
                "logic_path": final_state.get("logic_path", ""),
                "error": final_state.get("error", "")
            }

            logging.info(f"Agent 处理完成，路径: {result['logic_path']}, 循环: {result['loop_count']}")
            return result

        except Exception as e:
            logging.error(f"Agent 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "question": question,
                "answer": f"Agent 系统错误: {str(e)}",
                "route": "Error",
                "loop_count": 0,
                "score": 0,
                "critique": "",
                "documents_retrieved": 0,
                "is_relevant": False,
                "error": str(e)
            }


def create_agent(config: dict = None) -> AgentGraph:
    return AgentGraph(config)
