import logging
from typing import Literal, Dict, Any, List, Tuple
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import AgentState, create_initial_state
from .nodes.router import RouterNode
from .nodes.rewriter import QueryRewriterNode
from .nodes.grader import ReflectiveGraderNode
from .nodes.math_solver import MathSolverNode
from src.utils.config_loader import load_config
from src.rag.retriever import BlockAggregator, Reranker


class AgentGraph:
    """Agent 图编排 (v2.2)

    拓扑（简化版）：
    入口 → Router
      ├── Chat → Chat_Node → END
      └── Math → PreRetrieve(快速预检)
            ├── score > 0.9 → Generate(直接RAG) → END
            └── score ≤ 0.9 → Math_Solver → Reflective_Grader
                  ├── fast_track(≥85) → Generate → END
                  ├── self_refine(60~85) → Math_Solver(含critique) → Grader(循环)
                  └── rag(<60) → Rewriter → Retrieve(内嵌阈值过滤)
                                            ├── 相关 → Generate → END
                                            ├── 不相关 → Rewriter(重试) → ...
                                            └── 达上限 → Fallback → END
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        self.config = config
        agent_config = config.get("agent", {})

        self.max_loop_count = agent_config.get("max_loop_count", 2)
        self.self_refine_max = agent_config.get("self_refine_max", 2)
        self.early_stop_threshold = agent_config.get("early_stop_threshold", 5)
        self.fallback_message = agent_config.get("fallback_message",
                                                  "在教材中未找到准确定义，建议换个问法")
        self.pre_retrieve_threshold = agent_config.get("pre_retrieve_threshold", 0.9)

        # 初始化节点
        self.router_node = RouterNode(config)
        self.rewriter_node = QueryRewriterNode(config)
        self.math_solver_node = MathSolverNode(config)
        self.reflective_grader_node = ReflectiveGraderNode(config)

        # 初始化检索工具
        retriever_config = config.get("retriever", {})
        self.block_aggregator = BlockAggregator(retriever_config)
        self.secondary_reranker = Reranker(retriever_config)
        self.relevance_threshold = retriever_config.get("relevance_threshold", 0.3)

        # 初始化生成器
        from src.rag.generator import Generator
        self.generator = Generator(config.get("generator", {}))

        # 构建图
        self.graph = self._build_graph()
        self.checkpointer = MemorySaver()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)

        logging.info(f"AgentGraph v2.2 初始化完成，循环上限: {self.max_loop_count}, "
                     f"Self-Refine上限: {self.self_refine_max}, "
                     f"相关性阈值: {self.relevance_threshold}")

    # ==================== 图构建 ====================

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # === 节点注册 ===
        workflow.add_node("router", self.router_node)
        workflow.add_node("math_solver", self._math_solver_wrapper)
        workflow.add_node("reflective_grader", self.reflective_grader_node)
        workflow.add_node("rewriter", self.rewriter_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("chat", self._chat_node)
        workflow.add_node("fallback", self._fallback_node)

        workflow.set_entry_point("router")

        # === Router → 二元分流 (Chat / Math) ===
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "Chat": "chat",
                "Math": "pre_retrieve",
                "Fallback": "fallback"
            }
        )

        # === PreRetrieve: 快速预检 → score > threshold → 直接 RAG 生成 ===
        workflow.add_node("pre_retrieve", self._pre_retrieve_node)
        workflow.add_conditional_edges(
            "pre_retrieve",
            self._pre_retrieve_decision,
            {
                "generate": "generate",
                "math_solver": "math_solver"
            }
        )

        # === Math_Solver → Reflective_Grader (唯一出口) ===
        workflow.add_edge("math_solver", "reflective_grader")

        # === Reflective_Grader → 三段式分流 ===
        workflow.add_conditional_edges(
            "reflective_grader",
            self._reflective_decision_v22,
            {
                "fast_track": "generate",      # score >= pass_threshold(85): 直接输出
                "self_refine": "math_solver",   # rag_threshold(60) <= score < 85: 自修正
                "rag": "rewriter"               # score < rag_threshold(60): 直接进 RAG
            }
        )

        # === RAG 路径：Rewriter → Retrieve → (条件) Generate / Rewriter 重试 ===
        workflow.add_edge("rewriter", "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self._retrieve_decision,
            {
                "generate": "generate",
                "rewriter": "rewriter",
                "fallback": "fallback"
            }
        )

        # === 终止节点 ===
        workflow.add_edge("generate", END)
        workflow.add_edge("chat", END)
        workflow.add_edge("fallback", END)

        return workflow

    # ==================== 决策函数 ====================

    def _route_decision(self, state: AgentState) -> Literal["Chat", "Math", "Fallback"]:
        """路由决策：二元分流 Chat / Math"""
        loop_count = state.get("loop_count", 0)

        if loop_count >= self.max_loop_count:
            logging.warning(f"达到最大循环次数 ({loop_count})，跳转到 Fallback")
            return "Fallback"

        return state.get("route", "Math")

    def _reflective_decision_v22(self, state: AgentState) -> Literal["fast_track", "self_refine", "rag"]:
        """三段式自适应分流（在 Reflective_Grader 执行后调用）"""
        score = state.get("score", 0)
        self_refine_count = state.get("self_refine_count", 0)

        logging.info(f"三段式分流: score={score}, self_refine_count={self_refine_count}")

        if score >= self.reflective_grader_node.pass_threshold:
            logging.info("Fast-Track: 草稿可信度高，直接输出")
            return "fast_track"

        if score >= self.reflective_grader_node.rag_threshold:
            # Self-Refine 循环：检查是否超限
            if self_refine_count >= self.self_refine_max:
                logging.info(f"Self-Refine 已达上限 ({self_refine_count})，降级到 RAG")
                return "rag"
            logging.info(f"Self-Refine Track (第{self_refine_count + 1}轮)，继续修正")
            return "self_refine"

        logging.info(f"RAG Track: score={score} < 阈值，进入 RAG 检索")
        return "rag"

    def _retrieve_decision(self, state: AgentState) -> Literal["generate", "rewriter", "fallback"]:
        """检索结果决策：基于 Rerank score 阈值过滤（替代原 LLM Grade_Relevance）"""
        is_relevant = state.get("is_relevant", False)
        loop_count = state.get("loop_count", 0)

        if is_relevant:
            return "generate"

        if loop_count >= self.max_loop_count:
            logging.warning(f"检索不相关且达到循环上限 ({loop_count})，跳转到 Fallback")
            return "fallback"

        logging.info(f"检索结果不相关 (loop={loop_count})，返回 Rewriter 重试")
        return "rewriter"

    def _pre_retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """快速预检节点：路由判定为 Math 后，先用原问题检索并检查 Rerank score

        - score > threshold: 直接走 RAG 生成
        - score <= threshold: 走现有 Agent 流程
        """
        question = state.get("question", "")
        logging.info(f"PreRetrieve 快速预检: {question[:50]}...")

        try:
            nodes = self.block_aggregator.base_retriever.retrieve(question)
            if not nodes:
                logging.warning("PreRetrieve 检索无结果")
                return {"pre_retrieve_decision": "agent", "pre_retrieve_score": 0.0,
                        "logic_path": f"{state.get('logic_path', '')} > PreRetrieve(no_results -> agent)"}

            texts = [n.get_content() for n in nodes]
            top_texts, top_scores = self.secondary_reranker.rerank_texts_with_scores(
                query=question, texts=texts, top_n=3
            )

            if not top_scores:
                return {"pre_retrieve_decision": "agent", "pre_retrieve_score": 0.0,
                        "logic_path": f"{state.get('logic_path', '')} > PreRetrieve(no_scores -> agent)"}

            top_score = max(top_scores)
            logging.info(f"PreRetrieve 最高 score: {top_score:.4f}, 阈值: {self.pre_retrieve_threshold}")

            if top_score > self.pre_retrieve_threshold:
                logging.info(f"PreRetrieve 高分 ({top_score:.4f})，直接 RAG 生成")
                return {
                    "documents": top_texts,
                    "rerank_scores": top_scores,
                    "is_relevant": True,
                    "pre_retrieve_decision": "direct_rag",
                    "pre_retrieve_score": top_score,
                    "generation_source": "pre_retrieve_rag",
                    "loop_count": 0,
                    "logic_path": f"{state.get('logic_path', '')} > PreRetrieve({top_score:.4f} -> direct_RAG)"
                }

            logging.info(f"PreRetrieve 低分 ({top_score:.4f})，进入 Agent 流程")
            return {
                "pre_retrieve_decision": "agent",
                "pre_retrieve_score": top_score,
                "logic_path": f"{state.get('logic_path', '')} > PreRetrieve({top_score:.4f} -> agent)"
            }
        except Exception as e:
            logging.error(f"PreRetrieve 异常: {e}")
            return {"pre_retrieve_decision": "agent", "pre_retrieve_score": 0.0,
                    "logic_path": f"{state.get('logic_path', '')} > PreRetrieve(error -> agent)"}

    def _pre_retrieve_decision(self, state: AgentState) -> Literal["generate", "math_solver"]:
        """PreRetrieve 决策：direct_rag → Generate, 否则 → Math_Solver"""
        return "generate" if state.get("pre_retrieve_decision", "agent") == "direct_rag" else "math_solver"

    # ==================== 节点包装器 ====================

    def _math_solver_wrapper(self, state: AgentState) -> Dict[str, Any]:
        """
        Math_Solver 包装器：
        - 首次进入：正常求解
        - Self-Refine 进入：携带 critique 修正
        """
        self_refine_count = state.get("self_refine_count", 0)
        critique = state.get("critique", "")

        if self_refine_count > 0 and critique:
            previous_score = state.get("score", 0)
            revised = self.math_solver_node.solve_with_critique(
                question=state.get("question", ""),
                draft=state.get("internal_draft", ""),
                critique=critique
            )
            return {
                "internal_draft": revised,
                "previous_score": previous_score,
                "logic_path": f"{state.get('logic_path', '')} > Math_Solver(Self-Refine#{self_refine_count})"
            }

        return self.math_solver_node(state)

    # ==================== 检索节点 ====================

    def _retrieve_node(self, state: AgentState) -> Dict[str, Any]:
        """
        检索节点 (v2.2)：
        - 根据 strategy(multi/single) 走不同检索模式
        - 使用 Rerank score + threshold 作相关性判断（替代原 LLM Grade_Relevance）
        """
        question = state.get("question", "")
        strategy = state.get("strategy", "multi")
        keyword_groups: List[str] = state.get("keyword_groups", [question])
        loop_count = state.get("loop_count", 0)

        logging.info(f"检索节点: strategy={strategy}, {len(keyword_groups)} 组关键词")

        if strategy == "single":
            documents, all_scores = self._single_recall(question, keyword_groups)
        else:
            documents, all_scores = self._multi_recall(question, keyword_groups)

        # Rerank score 阈值过滤（核心替代 LLM Grade_Relevance）
        top_score = max(all_scores) if all_scores else 0.0
        is_relevant = top_score >= self.relevance_threshold

        logging.info(f"检索完成: {len(documents)} 个文档, "
                     f"最高 Rerank score: {top_score:.4f}, "
                     f"阈值: {self.relevance_threshold}, "
                     f"{'相关' if is_relevant else '不相关'}")

        return {
            "documents": documents,
            "rerank_scores": all_scores,
            "is_relevant": is_relevant,
            "loop_count": loop_count + 1 if not is_relevant else loop_count
        }

    def _multi_recall(self, question: str, keyword_groups: List[str]) -> Tuple[List[str], List[float]]:
        """Multi-Recall: 分布式检索 + 二次重排（不使用 block 聚合）"""
        primary_list = []
        secondary_pool = []
        all_scores = []

        for i, kw in enumerate(keyword_groups):
            try:
                # 1. 基础检索（top_k=20）
                nodes = self.block_aggregator.base_retriever.retrieve(kw)
                if not nodes:
                    continue

                # 2. Rerank（不聚合 block，直接取 top-3）
                top_ids, top_texts = self.block_aggregator.reranker.rerank(kw, nodes)
                if not top_texts:
                    continue

                primary_list.append(top_texts[0])
                if len(top_texts) > 1:
                    secondary_pool.extend(top_texts[1:])
            except Exception as e:
                logging.error(f"第{i+1}组关键词检索失败: {e}")
                continue

        if secondary_pool:
            try:
                second_results, second_scores = self.secondary_reranker.rerank_texts_with_scores(
                    query=question, texts=secondary_pool, top_n=1
                )
                if second_results:
                    primary_list.append(second_results[0])
                    all_scores.extend(second_scores)
            except Exception as e:
                logging.error(f"二次重排失败: {e}")

        # 截断 ≤ 5
        result = primary_list[:5]

        # 没有获取到 rerank scores 时，用阈值作保底判断
        return result, all_scores if all_scores else [self.relevance_threshold]

    def _single_recall(self, question: str, keyword_groups: List[str]) -> Tuple[List[str], List[float]]:
        """Single-Recall: 全局寻优 + 上下文感知扩展"""
        query = keyword_groups[0] if keyword_groups else question

        try:
            nodes = self.block_aggregator.base_retriever.retrieve(query)
            if not nodes:
                return [query], [0.0]

            rerank_ids, rerank_texts = self.block_aggregator.reranker.rerank(query, nodes)
            _, rerank_scores = self.secondary_reranker.rerank_texts_with_scores(
                query=query, texts=rerank_texts, top_n=3
            )

            if not rerank_texts:
                return [query], [0.0]

            # 上下文感知扩展（仅对 top-1）
            expanded = self._context_aware_expand(rerank_ids, rerank_texts, query)

            # top-2, top-3 静默辅助
            result = expanded + rerank_texts[1:3]
            return result[:5], rerank_scores

        except Exception as e:
            logging.error(f"Single-Recall 失败: {e}")
            return [query], [0.0]

    def _context_aware_expand(self, rerank_ids: List[str], rerank_texts: List[str], query: str) -> List[str]:
        """上下文感知扩展（仅对 top-1）"""
        if not rerank_ids or not rerank_texts:
            return []

        top_text = rerank_texts[0]
        chunk_type = self._detect_chunk_type(top_text)

        try:
            nodes = self.block_aggregator.base_retriever.retrieve(query)
            node_map = {node.node.node_id: node for node in nodes}
            node = node_map.get(rerank_ids[0])
            if node:
                chunk_id = node.metadata.get("id")
                if chunk_id is not None:
                    expanded = self._get_adjacent_chunks(int(chunk_id), chunk_type)
                    if expanded:
                        return [top_text] + expanded
        except Exception as e:
            logging.warning(f"上下文扩展失败: {e}")

        return [top_text]

    def _detect_chunk_type(self, text: str) -> str:
        """从文本中检测 chunk 类型"""
        for keyword, type_name in [
            ("证明", "proof"), ("定理", "theorem"), ("定义", "definition"),
            ("例题", "example"), ("命题", "proposition"), ("推论", "corollary")
        ]:
            if keyword in text[:200]:
                return type_name
        return "other"

    def _get_adjacent_chunks(self, chunk_id: int, chunk_type: str) -> List[str]:
        """获取相邻 chunk（基于类型感知）"""
        try:
            q_client = self.block_aggregator.q_client
            collection = self.block_aggregator.collection_name

            if chunk_type in ("theorem", "definition", "proposition", "corollary"):
                # 前向：加载后续区块直到触碰新定理/定义/例题
                texts = []
                for offset in range(1, 10):
                    hit = q_client.retrieve(
                        collection_name=collection, ids=[chunk_id + offset],
                        with_payload=True
                    )
                    if not hit:
                        break
                    text = hit[0].payload.get("text", "")
                    if any(b in text[:100] for b in ("**定理", "**定义", "**例题", "**推论", "**命题")):
                        break
                    texts.append(text)
                return texts

            elif chunk_type == "proof":
                # 后向：加载前置区块直到触碰父级边界
                texts = []
                for offset in range(1, 10):
                    hit = q_client.retrieve(
                        collection_name=collection, ids=[chunk_id - offset],
                        with_payload=True
                    )
                    if not hit:
                        break
                    text = hit[0].payload.get("text", "")
                    if any(b in text[:100] for b in ("**证明", "**解", "**例题", "**定理", "**定义")):
                        break
                    texts.insert(0, text)
                return texts

            return []
        except Exception as e:
            logging.warning(f"获取相邻 chunk 失败: {e}")
            return []

    # ==================== 生成 / 聊天 / 兜底 ====================

    def _generate_node(self, state: AgentState) -> Dict[str, Any]:
        """生成节点：Fast-Track / Self-Refined 草稿优先，否则 RAG"""
        question = state.get("question", "")
        documents = state.get("documents", [])
        internal_draft = state.get("internal_draft", "")
        generation_source = state.get("generation_source", "rag")

        if generation_source in ("fast_track", "self_refined") and internal_draft:
            logging.info(f"生成节点: 使用 {generation_source} 草稿")
            return {"answer": internal_draft}

        logging.info(f"生成节点: 基于 {len(documents)} 个文档 RAG 生成")
        try:
            answer = self.generator.generate(query=question, contexts=documents or None)
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
            return {"answer": f"生成回答时出错: {str(e)}", "error": str(e)}

    def _fallback_node(self, state: AgentState) -> Dict[str, Any]:
        return {"answer": self.fallback_message, "route": "Fallback"}

    # ==================== 运行入口 ====================

    def run(self, question: str, config: dict = None) -> Dict[str, Any]:
        logging.info(f"Agent v2.2 开始处理问题: {question}")

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
                "self_refine_count": final_state.get("self_refine_count", 0),
                "score": final_state.get("score", 0),
                "critique": final_state.get("critique", ""),
                "generation_source": final_state.get("generation_source", "unknown"),
                "documents": final_state.get("documents", []),
                "documents_retrieved": len(final_state.get("documents", [])),
                "is_relevant": final_state.get("is_relevant", False),
                "logic_path": final_state.get("logic_path", ""),
                "error": final_state.get("error", "")
            }

            logging.info(f"Agent 处理完成，路径: {result['logic_path']}, "
                         f"来源: {result['generation_source']}")
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
                "self_refine_count": 0,
                "score": 0,
                "critique": "",
                "generation_source": "error",
                "documents_retrieved": 0,
                "is_relevant": False,
                "error": str(e)
            }


def create_agent(config: dict = None) -> AgentGraph:
    return AgentGraph(config)
