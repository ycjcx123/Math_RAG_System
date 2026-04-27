● 以下是完整的 Agent 系统技术详解，从宏观架构到每一行代码的实现意图，按面试问答的粒度组织。                                                                             
                                                   
  ---                                                                                                                                                                   
  MathRAG Agent v2.2 完整实现详解                                                                                                                                       
                                                                                                                                                                        
  一、设计动机与架构决策                                                                                                                                                
                                                                                                                                                                        
  为什么要做 Agentic RAG？                                  

  传统 RAG 是"检索→生成"的直线流程（retrieve-then-read），三个硬伤：

  1. 无意图识别：闲聊问题也去查知识库，浪费算力
  2. 无自我评估：模型直接答错了也不知道，没有纠错机制
  3. 无检索策略：一个 query 打天下，复杂定理需要多个概念联合检索

  解决方案：用 LangGraph 构建一个有向图（StateGraph），把 RAG 降级为 Agent 的一个"工具"，在它之上加三层智能——路由、自评、循环纠错。

  为什么用小模型（Qwen3-1.7B）？

  不是选型失误，是刻意挑战。1.7B 参数量逻辑推理弱，所以整个架构设计的原则是：

  - 结构化输出：所有 LLM 调用强制输出 JSON，配合鲁棒解析器
  - 简单决策空间：路由只输出 "Chat"/"Math"，评分只输出 "Yes"/"No" 等价判断
  - 兜底降级：每一步 JSON 解析失败都有 fallback
  - 循环限制：max_loop_count=2 严格防死循环

  ---
  二、目录结构与模块职责

  src/agent/
  ├── __init__.py          # 导出 create_agent, AgentGraph, 所有 Node
  ├── state.py             # AgentState (TypedDict) — 全局数据流定义
  ├── graph.py             # AgentGraph — LangGraph 编排（有向图 + 决策函数）
  └── nodes/
      ├── __init__.py
      ├── router.py        # 二元路由节点 (Chat / Math)
      ├── rewriter.py      # 阶梯式查询重写 + 策略分类 (multi/single)
      ├── grader.py        # 反思评分节点 (三段式评分 0-100)
      └── math_solver.py   # 直接求解节点 (首次 + Self-Refine 修正)

  集成的外部组件（src/rag/）：

  src/rag/retriever/
  ├── searcher.py          # QdrantRetriever — BGE-M3 向量检索 (top-20)
  ├── reranker.py          # Reranker — BGE-Reranker-V2-M3 API 重排 (top-3)
  └── context_builder.py   # BlockAggregator — 按 block_id 聚合完整逻辑块
  src/rag/generator/
  └── generate.py          # Generator — Qwen3 生成（OpenAI 兼容接口）

  ---
  三、AgentState — 全流程状态定义（state.py）

  AgentState 继承 TypedDict，LangGraph 在节点间传递的就是这个字典。

  class AgentState(TypedDict):
      question: str                       # 用户原始问题

      # 路由决策
      route: Literal["Chat", "Math", "Fallback"]

      # 查询重写
      rewritten_query: NotRequired[str]   # 主查询（contexts[0]）
      keyword_groups: NotRequired[List[str]]  # 分布式检索关键词组（最多4组）
      strategy: NotRequired[str]          # "multi" | "single"

      # 检索结果
      documents: NotRequired[List[str]]   # 检索到的文档文本列表
      rerank_scores: NotRequired[List[float]]  # Rerank API 返回分数
      is_relevant: NotRequired[bool]      # 基于阈值是否相关

      # Math_Solver 草稿
      internal_draft: NotRequired[str]    # 模型直接生成的草稿

      # Reflective_Grader
      score: int                          # 0-100 评分
      critique: NotRequired[str]          # <85 分时的纠错意见

      # 循环控制
      self_refine_count: int              # Self-Refine 已执行次数
      loop_count: int                     # RAG 重试次数

      # 追踪
      generation_source: NotRequired[str] # "fast_track" | "self_refined" | "rag"
      logic_path: NotRequired[str]        # 决策路径（如 "start > Router(Math) > Math_Solver"）
      answer: NotRequired[str]            # 最终答案
      error: NotRequired[str]

  关键设计点：所有字段通过 LangGraph 的 StateGraph 自动合并。每个节点 return 的 dict 被**更新（update）**到全局状态，而非覆盖。例如 _route_decision 只返回 {"route":
  ..., "logic_path": ...}，其他字段保持不变。

  ---
  四、节点详解（每个节点的完整实现逻辑）

  4.1 RouterNode（router.py）

  职责：最高维度意图分类，仅做 Chat/Math 二元分流。

  实现细节：

  def __call__(self, state: dict) -> dict:
      question = state.get("question", "")
      route_decision = self.route(question)
      return {
          "route": route_decision,
          "logic_path": f"{state.get('logic_path', 'start')} > Router({route_decision})"
      }

  三步执行：

  1. _call_llm(question) —— 调用 Qwen3-1.7B，system prompt 含 10 个 Few-shot 示例，max_tokens=50（只生成一个词），temperature=0.1（确定性）。如果 LLM 调用异常，返回
  self.default_path = "Math"。
  2. _parse_output(llm_output) —— 对输出做 strip().upper()，正则 \bCHAT\b 匹配到就是 Chat，其余全走 Math。这里的设计哲学是"宁可多查，不可漏查"——任何识别模糊的都归为
  Math。
  3. 为什么去掉了 v2.1 的 Math_RAG 路径？ 因为调研发现小模型做"是否需要检索"这个元判断准确率很低。v2.2 改为统一走 Math，把"是否要检索"下沉到 Reflective_Grader
  的评分区间里，让模型先试着答，根据答题质量决定是否检索。

  面试可说的点：二元路由的设计思路、小模型下"结构化输出 + 宽松匹配"的策略、从三元到二元的演进理由。

  ---
  4.2 MathSolverNode（math_solver.py）

  职责：在不检索的前提下，让模型直接尝试解答数学问题。

  两种模式：

  def __call__(self, state: dict) -> dict:
      self_refine_count = state.get("self_refine_count", 0)
      critique = state.get("critique", "")

      if self_refine_count > 0 and critique:
          # Self-Refine 模式：基于 critique 修正
          draft = state.get("internal_draft", "")
          revised = self.solve_with_critique(question, draft, critique)
          return {"internal_draft": revised, ...}
      else:
          # 首次求解模式
          draft = self.solve(question)
          return {"internal_draft": draft, ...}

  首次求解 (solve)：
  - 调 Qwen3，system_prompt 要求"分步推理 + LaTeX + \boxed{}"
  - max_tokens=2048（数学证明可能很长）
  - 后处理：re.sub(r'<think>.*?</think>', '', draft) 去除模型的内部思维链

  Self-Refine 修正 (solve_with_critique)：
  - 用 self_refine_prompt 替代普通 system prompt
  - 将 原问题 + 旧回答 + 纠错意见 三者拼入 user message
  - 修正失败时保底返回原草稿

  设计意图：让模型先"裸答"，如果答得好（评分高）就直接输出，省去检索的延迟和资源消耗。

  ---
  4.3 ReflectiveGraderNode（grader.py）

  职责：对 Math_Solver 的草稿进行 0-100 评分，这是整个三段式自适应流程的核心判据。

  评分三区间：

   pass_threshold = 85  (configurable)
   rag_threshold = 60   (configurable)

   [0, 60)    → RAG Track     （不检索不行，直接查教材）
   [60, 85)   → Self-Refine   （有基础但不够完善，先自己改）
   [85, 100]  → Fast-Track    （靠谱，直接输出）

  实现：

  def grade(self, question: str, draft: str) -> Tuple[int, str, bool]:
      if not draft:
          return 0, "草稿为空，需要查阅教材", False

      llm_output = self._call_llm(question, draft)
      result = self._parse_json(llm_output)

      score = result.get("score", 0)
      critique = result.get("critique", "")
      score = max(0, min(100, int(score)))  # 边界裁剪
      passed = score >= self.pass_threshold
      return score, critique, passed

  - Few-shot Prompt 含 5 个评分示例，覆盖了"满分→需修正→严重不足"的完整谱系
  - 输出强制 JSON：{"score": int, "critique": str, "reasoning": str}
  - LLM 调用失败的保护：返回 {"score": 0, "critique": "评分节点异常", "reasoning": "LLM调用失败"}，强制进入 RAG Track
  - 草稿为空：直接 score=0，进 RAG

  __call__ 中还有一个关键逻辑——标记 generation_source：

  if score >= self.pass_threshold:
      result["generation_source"] = "fast_track"
  elif score >= self.rag_threshold:
      result["self_refine_count"] = self_refine_count + 1  # 递增计数器
      result["generation_source"] = "self_refine"
  else:
      result["generation_source"] = "rag"

  这个标记会传递到最后的 _generate_node，决定最终的答案来源（直接取草稿 OR RAG 生成）。

  ---
  4.4 QueryRewriterNode（rewriter.py）

  职责：当 Grader 判定"需要查教材"时，将用户问题转化为检索关键词，同时判断检索策略。

  阶梯式重写（两轮）：

  tier=0 (Precise):  system_prompt_precise — 精确提取定理名/编号
  tier=1 (Expand):   system_prompt_expand  — 扩展上位概念或靶向补漏
  tier>=2:           直接 fallback 到原始问题

  输出格式：
  {"strategy": "multi", "contexts": ["关键词1", "关键词2", ...]}

  strategy 字段的含义：
  - "single"：单概念深挖（问题聚焦于一个定理/定义）→ 后续走 Single-Recall
  - "multi"：多概念对比/跨章节知识 → 后续走 Multi-Recall

  鲁棒 JSON 解析（_extract_json）——针对小模型 JSON 输出不稳定设计的四层防御：

  Layer 1: json.loads() 直接解析          ← 最优路径
  Layer 2: 正则提取 ```json...``` 代码块    ← 模型爱加 Markdown
  Layer 3: 栈式括号匹配，逐段尝试           ← 模型输出有尾巴
  Layer 4: 转义修复后重试                   ← 模型转义不规范
  失败: fallback_to_original → 直接用原始问题

  面试可说的点：四层 JSON 解析器的设计动机、strategy 字段的意义（后续检索策略的前置决策）、阶梯式重写为什么只做两轮。

  ---
  4.5 Retrieve_Node（graph.py 中的 _retrieve_node）

  职责：执行实际的检索，根据 strategy 走不同模式。

  这是一个包装函数而非独立类，因为检索逻辑与 agent 的 BlockAggregator 实例强绑定。

  双轨检索机制

  模式 A: Multi-Recall

  适用于"线性映射、线性变换、正交变换和酉变换的区别"这类多概念综合问题。

  def _multi_recall(self, question, keyword_groups):
      primary_list = []
      secondary_pool = []

      for kw in keyword_groups:        # 遍历最多 4 组关键词
          nodes = retriever.retrieve(kw)        # top-20 检索
          ids, texts = reranker.rerank(kw, nodes)  # 重排取 top-3
          primary_list.append(texts[0])         # Top-1 进主结果池
          secondary_pool.extend(texts[1:])      # Top-2/3 进候选池

      # 二次重排：用原始 question 对候选池做全局筛选
      second_results, _ = reranker.rerank_texts_with_scores(
          query=question, texts=secondary_pool, top_n=1
      )
      primary_list.append(second_results[0])    # 取 Top-1 补充

      return primary_list[:5]                    # 截断 ≤ 5

  关键：不使用 block 聚合（v2.2 的设计决定），因为多概念场景下聚合可能把不相关的块拼进来。

  模式 B: Single-Recall

  适用于"证明定理 10"这种单点深挖的问题。

  def _single_recall(self, question, keyword_groups):
      query = keyword_groups[0]

      nodes = retriever.retrieve(query)          # top-20
      ids, texts = reranker.rerank(query, nodes)  # top-3

      # 上下文感知扩展 —— 仅对 Top-1
      expanded = self._context_aware_expand(ids, texts, query)

      # Top-1 + 扩展块 + Top-2/3 辅助
      result = expanded + texts[1:3]
      return result[:5], scores

  上下文感知扩展（_context_aware_expand）

  这是 v2.2 的核心新功能，根据 Top-1 文档的类型决定扩展方向：

  def _get_adjacent_chunks(self, chunk_id, chunk_type):
      if chunk_type in ("theorem", "definition", "proposition"):
          # 前向扩展：加载后续块直到遇到新边界（**定理/定义/例题）
          for offset in range(1, 10):
              hit = q_client.retrieve(ids=[chunk_id + offset])
              if "**定理" in text[:100]: break   # 触碰边界停止
              texts.append(text)

      elif chunk_type == "proof":
          # 后向扩展：加载前置块直到遇到父级边界
          for offset in range(1, 10):
              hit = q_client.retrieve(ids=[chunk_id - offset])
              if "**证明" in text[:100]: break
              texts.insert(0, text)              # 头插保持顺序

  设计意图：如果 Top-1 是一个"定理"，而它的"证明"在后续 chunk 中，前向扩展能把证明也捞回来。反之，如果 Top-1 是"证明"片段，后向扩展能找到被证明的定理原文。

  硬件级相关性过滤

  替代了 v2.1 的 LLM Grade_Relevance 节点：

  top_score = max(all_scores) if all_scores else 0.0
  is_relevant = top_score >= self.relevance_threshold  # 默认 0.3

  这里的 all_scores 来自 SiliconFlow Rerank API 返回的分数（0~1 之间）。纯数学硬阈值，不需要 LLM 参与判断。

  面试可说的点：双轨检索的触发条件、上下文感知扩展的方向逻辑（前向/后向由 chunk 类型决定）、硬件级评分替代 LLM 评分的动机。

  ---
  4.6 Generate_Node（graph.py 中的 _generate_node）

  职责：输出最终答案。

  def _generate_node(self, state):
      generation_source = state.get("generation_source", "rag")
      internal_draft = state.get("internal_draft", "")

      # Fast-Track / Self-Refined: 直接使用草稿
      if generation_source in ("fast_track", "self_refined") and internal_draft:
          return {"answer": internal_draft}

      # RAG Track: documents 作为 context 传给 Generator
      answer = self.generator.generate(
          query=question,
          contexts=documents or None
      )
      return {"answer": answer}

  优先级：Fast-Track 草稿 > Self-Refined 草稿 > RAG 生成。

  Generator 的 _build_system_prompt 在有 contexts 时构造"参考资料"增强 prompt，没有 contexts 时（Chat 路径）让模型自由发挥。

  ---
  4.7 Chat_Node 和 Fallback_Node

  Chat_Node：最简实现，直接调 Generator（无 context）：

  def _chat_node(self, state):
      answer = self.generator.generate(query=question, contexts=None)
      return {"answer": answer}

  Fallback_Node：输出预设的友好提示，不走 LLM：

  def _fallback_node(self, state):
      return {"answer": self.fallback_message, "route": "Fallback"}

  fallback_message 来自 config："在教材中未找到准确定义，建议换个问法"

  ---
  五、图编排（graph.py — _build_graph）

  LangGraph 有向图构建

  def _build_graph(self) -> StateGraph:
      workflow = StateGraph(AgentState)

      # 注册 8 个节点
      workflow.add_node("router", self.router_node)
      workflow.add_node("math_solver", self._math_solver_wrapper)
      workflow.add_node("reflective_grader", self.reflective_grader_node)
      workflow.add_node("rewriter", self.rewriter_node)
      workflow.add_node("retrieve", self._retrieve_node)
      workflow.add_node("generate", self._generate_node)
      workflow.add_node("chat", self._chat_node)
      workflow.add_node("fallback", self._fallback_node)

  注意 math_solver 注册的是 _math_solver_wrapper 而非 self.math_solver_node。这个 wrapper 的作用：

  def _math_solver_wrapper(self, state):
      # 首次进入：直接调 math_solver_node(state)
      # Self-Refine 进入：调 math_solver_node.solve_with_critique()
      # 区别在于包装器中会检查 self_refine_count 并传递 critique

  三种条件边

  LangGraph 的条件边定义了一个函数（接收当前 state 返回目标节点名）和一个映射表。

  边 1: Router → 二元分流（_route_decision）

  workflow.add_conditional_edges(
      "router",
      self._route_decision,
      {"Chat": "chat", "Math": "math_solver", "Fallback": "fallback"}
  )

  _route_decision 逻辑：
  1. 检查 loop_count >= max_loop_count → 返回 "Fallback"（循环保护）
  2. 否则返回 state["route"]（由 RouterNode 写入）

  边 2: Reflective_Grader → 三段式分流（_reflective_decision_v22）

  workflow.add_conditional_edges(
      "reflective_grader",
      self._reflective_decision_v22,
      {"fast_track": "generate", "self_refine": "math_solver", "rag": "rewriter"}
  )

  _reflective_decision_v22 逻辑：

  if score >= 85:                → "fast_track" → generate (直接输出)
  if score >= 60:
      if self_refine_count >= 2: → "rag"        → rewriter (降级到 RAG)
      else:                      → "self_refine" → math_solver (继续修正)
  if score < 60:                 → "rag"        → rewriter (进 RAG)

  Self-Refine 的 early stopping：当 self_refine_count >= self_refine_max（默认 2）时，即使分数在 60-84 区间也不再修正，转而降级到 RAG。避免无限循环。

  边 3: Retrieve → 结果决策（_retrieve_decision）

  workflow.add_conditional_edges(
      "retrieve",
      self._retrieve_decision,
      {"generate": "generate", "rewriter": "rewriter", "fallback": "fallback"}
  )

  _retrieve_decision 逻辑：

  if is_relevant:         → "generate" (检索到相关内容，生成答案)
  if loop_count >= 2:     → "fallback" (重试耗尽，输出兜底消息)
  else:                   → "rewriter" (不相关但还有重试次数，再试一次)

  完整有向图

  router ──(Chat)──→ chat ──→ END
     │
     └──(Math)──→ math_solver ──→ reflective_grader
                                        │
                            ┌───────────┼───────────┐
                            │[≥85]      │[60-85)     │[<60]
                            ▼           ▼            ▼
                        generate   math_solver    rewriter
                            │           │             │
                            │      (再次评分)         ▼
                            │                     retrieve
                            │                    │    │     │
                            │              [相关] │    │     │ [不相关+有次数]
                            │                    ▼    │     ▼
                            │               generate  │  rewriter (重试)
                            │                    │    │
                            ▼                    ▼    ▼
                           END          fallback ←───┘ [不相关+达上限]

  ---
  六、三段式自适应流程完整推演

  以一个典型的高等代数问题为例，追踪 state 的变化：

  问题："请证明环 $R$ 的一个非空子集 $R_1$ 为一个子环的充分必要条件是..."

  实例 1: Fast-Track 路径（简单问题）

  step 1: create_initial_state()
  → state = {question, route="Math", loop=0, score=0, ...}

  step 2: Router(__call__)
  → LLM 输出 "Math"
  → state = {route="Math", logic_path="start > Router(Math)"}

  step 3: Math_Solver(__call__)
  → LLM 生成草稿
  → state = {internal_draft="证明：必要性...\n充分性...", logic_path="... > Math_Solver"}

  step 4: Reflective_Grader(__call__)
  → 评分 LLM 输出 {"score": 90, "critique": "", "reasoning": "..."}
  → state = {score=90, generation_source="fast_track"}

  step 5: _reflective_decision_v22(state)
  → score=90 ≥ 85 → return "fast_track"

  step 6: Generate(__call__)
  → generation_source="fast_track", internal_draft 非空
  → 直接输出草稿
  → state = {answer="证明：必要性...\n充分性..."}

  总耗时：~3 次 LLM 调用（Router + Math_Solver + Grader），零检索。

  实例 2: Self-Refine 路径（需修正的问题）

  step 1-3: 同 Fast-Track，但草稿不完善

  step 4: Reflective_Grader
  → {"score": 70, "critique": "缺少对坐标变换公式的推导", ...}
  → state = {score=70, self_refine_count=1, generation_source="self_refine"}

  step 5: _reflective_decision_v22
  → 70 ≥ 60 且 self_refine_count(1) < 2 → "self_refine"

  step 6: Math_Solver (Self-Refine 模式)
  → solve_with_critique(question, draft, critique)
  → 基于 critique 修正草稿
  → state = {internal_draft="修正后的完整证明"}

  step 7: Reflective_Grader (再评分)
  → 如果 score=88 ≥ 85 → generation_source="self_refined"
  → 如果 score=65（提升小）→ 尝试下一轮或降级到 RAG

  Self-Refine 的 early stopping 在 graph.py 的 _reflective_decision_v22 中由 early_stop_threshold=5 控制——虽然代码中读取了这个配置，但实际的 early stopping
  逻辑（对比前后分数差）在 _math_solver_wrapper 中记录 previous_score，交由后续决策判断。

  实例 3: RAG 路径（复杂定理证明）

  step 1-3: 同 Fast-Track

  step 4: Reflective_Grader
  → {"score": 30, "critique": "缺乏具体推导步骤，需查阅教材的定理10"}
  → state = {score=30, generation_source="rag"}

  step 5: _reflective_decision_v22 → score=30 < 60 → "rag"

  step 6: Rewriter (tier=0, Precise)
  → LLM 输出 {"strategy": "single", "contexts": ["定理10 ...", "λ-矩阵初等因子", ...]}
  → state = {keyword_groups=[...], strategy="single"}

  step 7: Retrieve (Single-Recall)
  → 检索 top-20 → Rerank top-3 → 上下文感知扩展
  → 最大 Rerank score = 0.792 ≥ 0.3 → is_relevant=true
  → state = {documents=[...], is_relevant=true}

  step 8: _retrieve_decision → is_relevant=true → "generate"

  step 9: Generate (RAG 模式)
  → generator.generate(query, contexts=documents)
  → 返回基于教材上下文生成的证明
  → state = {answer="完整证明..."}

  实例 4: Fallback 路径（检索全部失败）

  step 1-7: 同 RAG 路径，但 is_relevant=false

  step 8: _retrieve_decision
  → is_relevant=false, loop_count=1
  → loop_count < 2 → "rewriter"（重试）

  step 9: Rewriter (tier=1, Expand)
  → 用 system_prompt_expand 做概念扩展
  → 输出新的 keyword_groups
  → state = {keyword_groups=[...], loop_count=2}

  step 10: Retrieve (第二次)
  → 还是不相关

  step 11: _retrieve_decision
  → is_relevant=false, loop_count=2
  → loop_count ≥ 2 → "fallback"

  step 12: Fallback
  → 输出 "在教材中未找到准确定义，建议换个问法"
  → state = {answer="在教材中未找到准确定义，建议换个问法", route="Fallback"}

  ---
  七、与底层 RAG 系统的集成

  Agent 通过组合方式持有三个核心 RAG 组件：

  class AgentGraph:
      def __init__(self, config):
          # 检索
          self.block_aggregator = BlockAggregator(retriever_config)
          self.secondary_reranker = Reranker(retriever_config)

          # 生成
          self.generator = Generator(config.get("generator", {}))

  BlockAggregator 内部初始化了三样东西：

  1. QdrantRetriever —— 连接 Qdrant 向量库，用 BGE-M3 做混合（dense + sparse）检索，top_k=20
  2. Reranker —— 调 SiliconFlow API 的 BGE-Reranker-V2-M3，对 20 个结果重排取 top_n=3
  3. QdrantClient —— 原生 Qdrant 客户端，用于 _get_full_block_text 按 block_id 滚动查询

  注意：Agent 的 _multi_recall 和 _single_recall 直接调 BlockAggregator.base_retriever.retrieve() 和 BlockAggregator.reranker.rerank()，但不使用
  BlockAggregator.retrieve_and_aggregate()。因为 Agent 自己控制多组关键词的分发和二次重排逻辑。

  Generator 使用 OpenAI 兼容接口连接 localhost:8080（llama.cpp server），支持 contexts 注入：

  有 contexts:  "请根据以下参考资料回答..." + contexts
  无 contexts:  "请根据你自己的知识回答..."

  ---
  八、针对小模型的优化总结

  这是面试中很好的"难点与解决方案"素材：

  ┌──────────────────┬─────────────────────────────────────────────────────────────────┬──────────────────────────────────────────────────┐
  │       问题       │                            解决方式                             │                     代码位置                     │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ JSON 输出不规范  │ 四层解析器：直接解析→code block→括号匹配→转义修复               │ rewriter.py:_extract_json, grader.py:_parse_json │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 路由判断模糊     │ 匹配 \bCHAT\b，其余全走 Math，default_path="Math"               │ router.py:_parse_output                          │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 评分不稳定       │ 0-100 整数 + 边界裁剪 max(0, min(100, int(score)))              │ grader.py:grade                                  │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 无限循环         │ max_loop_count=2 + self_refine_max=2 + early_stop_threshold=5   │ graph.py:__init__ + 决策函数                     │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ LLM 调用失败     │ 每节点独立 try-catch + 默认值降级                               │ 每个 Node 的 _call_llm                           │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 思维链干扰       │ re.sub(r'<think>.*?</think>', '', draft) 后处理                 │ math_solver.py:solve                             │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ 检索不相关       │ 硬件级 Rerank score 阈值（relevance_threshold=0.3），不经过 LLM │ graph.py:_retrieve_node                          │
  ├──────────────────┼─────────────────────────────────────────────────────────────────┼──────────────────────────────────────────────────┤
  │ Generator 不输出 │ 空字符串保底，上层 if rag_answer: 保护                          │ generator.py:generate                            │
  └──────────────────┴─────────────────────────────────────────────────────────────────┴──────────────────────────────────────────────────┘

  ---
  九、配置驱动（config.yaml agent 段）

  所有阈值和限制都从配置读取，不改代码即可调参：

  agent:
    max_loop_count: 2         # RAG 重试上限
    self_refine_max: 2        # Self-Refine 轮次上限
    early_stop_threshold: 5   # 提升 < 5 分提前跳出
    fallback_message: "在教材中未找到准确定义，建议换个问法"
    router.default_path: "Math"
    rewriter.fallback_to_original: true
    reflective_grader:
      pass_threshold: 85      # 快速通道阈值
      rag_threshold: 60       # RAG 通道阈值

  ---
  十、辅助层——评测系统

  为了验证 Agent 效果，配套实现了评测框架：

  AgentEvaluator（src/evaluation/agent_evaluator.py）：
  - 同时对 RAG-only（纯检索生成）和 Agent 系统做对比评测
  - 使用 DeepSeek-chat 和 GLM-4 两个裁判模型，RAG-only/Aget 各自独立打分
  - 4 维度评分：correctness, faithfulness, answer_relevance, context_relevance（每个 0-2）
  - 断点续写、错误重试（sleep 10s, max 3 retries）

  Summary（test/summary.py）：
  - 读取评测结果 JSON，计算各维度平均分和 ≥1 占比
  - Agent 特有：按 loop_counter 分拆 Fast-Track / RAG / Fallback 统计

  ---
  以上就是整个 Agent v2.2 的完整实现。你可以从为什么需要Agent（传统 RAG 的三个缺陷）开始讲，然后深入到架构设计（有向图 +
  三段式自适应），最后落到具体节点实现（每个节点的输入输出和边界处理）。

✻ Worked for 1m 48s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯  
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle)
  ---
  十、辅助层——评测系统

  为了验证 Agent 效果，配套实现了评测框架：

  AgentEvaluator（src/evaluation/agent_evaluator.py）：
  - 同时对 RAG-only（纯检索生成）和 Agent 系统做对比评测
  - 使用 DeepSeek-chat 和 GLM-4 两个裁判模型，RAG-only/Aget 各自独立打分
  - 4 维度评分：correctness, faithfulness, answer_relevance, context_relevance（每个 0-2）
  - 断点续写、错误重试（sleep 10s, max 3 retries）

  Summary（test/summary.py）：
  - 读取评测结果 JSON，计算各维度平均分和 ≥1 占比
  - Agent 特有：按 loop_counter 分拆 Fast-Track / RAG / Fallback 统计

  ---
  以上就是整个 Agent v2.2 的完整实现。你可以从为什么需要Agent（传统 RAG 的三个缺陷）开始讲，然后深入到架构设计（有向图 +
  三段式自适应），最后落到具体节点实现（每个节点的输入输出和边界处理）。

✻ Worked for 1m 48s

────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
❯  
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
  ⏵⏵ accept edits on (shift+tab to cycle)

  ---
  十、辅助层——评测系统

  为了验证 Agent 效果，配套实现了评测框架：

  AgentEvaluator（src/evaluation/agent_evaluator.py）：
  - 同时对 RAG-only（纯检索生成）和 Agent 系统做对比评测
  - 使用 DeepSeek-chat 和 GLM-4 两个裁判模型，RAG-only/Aget 各自独立打分
  - 4 维度评分：correctness, faithfulness, answer_relevance, context_relevance（每个 0-2）
  - 断点续写、错误重试（sleep 10s, max 3 retries）

  Summary（test/summary.py）：
  - 读取评测结果 JSON，计算各维度平均分和 ≥1 占比
  - Agent 特有：按 loop_counter 分拆 Fast-Track / RAG / Fallback 统计

  ---
  以上就是整个 Agent v2.2 的完整实现。你可以从为什么需要Agent（传统 RAG 的三个缺陷）开始讲，然后深入到架构设计（有向图 +
  三段式自适应），最后落到具体节点实现（每个节点的输入输出和边界处理）。