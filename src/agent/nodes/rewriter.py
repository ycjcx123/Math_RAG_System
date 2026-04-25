import re
import json
import logging
from typing import Dict, Any, List
from openai import OpenAI

from src.utils.config_loader import load_config


class QueryRewriterNode:
    """查询重写节点 (v2.0)：阶梯式重写策略

    - loop_count=0 (Precise): 精确提取定理名/编号/术语
    - loop_count=1 (Concept Expansion): 扩展为上位/相关概念
    - 如果提供 critique: 基于纠错意见提取检索关键词
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()

        generator_config = config.get("generator", {})
        agent_config = config.get("agent", {})

        self.model_name = generator_config.get("model_name", "Qwen3-1.7B-Q8_0.gguf")
        self.base_url = generator_config.get("base_url", "http://localhost:8080/v1")
        self.api_key = generator_config.get("api_key", "llama-cpp")
        self.temperature = generator_config.get("temperature", 0.1)
        self.max_tokens = generator_config.get("max_tokens", 100)

        rewriter_config = agent_config.get("rewriter", {})
        self.fallback_to_original = rewriter_config.get("fallback_to_original", True)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # ---- 第一轮：精确提取（分布式检索） ----
        self.system_prompt_precise = """""你是一个数学检索专家。你的任务是提取检索关键词。
对于定理证明题，你必须遵循“锚点优先”原则：

输出格式：严格 JSON --- {"tool": "RAG", "context": ["完整命题锚点", "核心术语1", "核心术语2", "补充术语"]}

规则：
1. 第一组（context[0]）：必须是定理的编号加上最完整的命题原文（保留 LaTeX 公式）。这是最高优先级的检索锚点。
2. 后续组：将问题拆解为核心术语（如“初等因子”、“$\lambda$-矩阵”），用于弥补原文匹配的不足。
3. 剥离噪音：去除“证明下述定理”、“老师脾气不好”等无关口语。
4. 数量：1-4 组，按重要性从前到后填充。如果问题很短，只需 1 组。

示例：
问题："证明下述定理：定理10 设 $A(\lambda)$ 是满秩矩阵...则其因式方幂是初等因子"
输出：{"tool":"RAG", "context":["定理10 设 $A(\lambda)$ 是满秩矩阵...其因式方幂是初等因子", "$\lambda$-矩阵初等因子", "满秩矩阵对角化", "初等变换"]}

问题："线性空间维数公式怎么证"
输出：{"tool":"RAG", "context":["线性空间维数公式 dim(V1+V2)", "维数定理证明", "线性空间基与维数"]}

问题："请帮我证明下述命题：设 $f(x), g(x) \\in K[x]$ ，则\n$$\n\\deg (f (x) \\pm g (x)) \\leqslant \\max  \\left\\{\\deg f (x), \\deg g (x) \\right\\}\n$$\n$$\n\\deg (f (x) g (x)) = \\deg f (x) + \\deg g (x)\n$$"
输出：{"tool":"RAG", "context":["多项式次数性质", "deg(f±g)≤max(deg f,deg g)", "deg(fg)=deg f+deg g", "多项式运算次数公式"]}

问题："唯一分解定理怎么证明"
输出：{"tool":"RAG", "context":["唯一分解定理", "多项式唯一分解", "不可约多项式分解唯一性"]}

问题："证明定理10中\\lambda-矩阵的相抵标准型唯一"
输出：{"tool":"RAG", "context":["定理10 \\lambda-矩阵相抵标准型", "\\lambda-矩阵初等变换", "\\lambda-矩阵相抵"]}

问题："请说明线性映射、线性变换、正交变换和酉变换在定义上的核心区别"
输出：{"tool":"RAG", "context":["线性映射定义", "线性变换定义", "正交变换定义", "酉变换定义"]}"""

        # ---- 第二轮：概念扩展（分布式检索） ----
        self.system_prompt_expand = """你是一个数学检索扩展专家。第一轮检索未能满足需求，你需要根据反馈进行“全域降级扩展”或“靶向精准补漏”。

输出格式：严格 JSON --- {"tool": "RAG", "context": ["关键词组1", "关键词组2", "关键词组3", "关键词组4"]}

规则：
1. 数量约束：至少 1 组，最多 4 组，按重要性从前到后填充，且重点不同，多于 4 组将导致系统崩溃。
2. 独立性：不要重复第一轮已失败的关键词。
3. 双模态逻辑：
   - 情况 A [上一轮不足]为空：说明第一轮完全没搜到。请执行“全域降级”，搜索该定理所属的章节名、上位概念或数学领域的通用同义词（例如：由“定理10”扩展为“Smith标准型”或“矩阵相抵”）。
   - 情况 B [上一轮不足]不为空：说明搜到了但内容有缺失。请执行“靶向补漏”，将[上一轮不足]中的评价直接转化为搜索词。

示例 1 (情况 A - 检索空转)：
原问题："证明定理10：关于初等因子的分解"
第一轮关键词：["定理10 设 A(\\lambda) 是满秩矩阵...初等因子", "定理10证明"]
上一轮不足：""
输出：{"tool":"RAG", "context":["\\lambda-矩阵的初等因子定义", "矩阵化为相抵标准型", "多项式矩阵的Smith标准型", "不变因子与初等因子的关系"]}

示例 2 (情况 A - 检索空转)：
原问题："请解释什么是有限维线性空间的基"
第一轮关键词：["有限维线性空间的基定义", "线性空间基的性质"]
上一轮不足：""
输出：{"tool":"RAG", "context":["向量组的极大线性无关组", "线性空间的维数与基", "基变换与坐标变换", "线性代数基础概念-基"]}

示例 3 (情况 B - 补漏重搜)：
原问题："证明线性变换 A 在不同基下的矩阵是相似的"
第一轮关键词：["线性变换在不同基下的矩阵", "矩阵相似定义"]
上一轮不足："缺少对坐标变换公式 P^-1AP 的具体推导逻辑"
输出：{"tool":"RAG", "context":["坐标变换公式推导", "过渡矩阵与线性变换矩阵的关系", "矩阵相似的几何意义", "基变换下的矩阵表示"]}

示例 4 (情况 B - 补漏重搜)：
原问题："求解方程 x^2 - 4 = 0 的复数根"
第一轮关键词：["x^2 - 4 = 0 求解", "复数根定义"]
上一轮不足："未提及代数基本定理在多项式根个数中的应用"
输出：{"tool":"RAG", "context":["代数基本定理", "n次多项式的复根个数", "复数域上的多项式分解"]}

示例 5
原问题："证明定理10中\\lambda-矩阵的相抵标准型唯一"
第一轮关键词：["定理10 \\lambda-矩阵相抵标准型", "\\lambda-矩阵初等变换"]
上一轮不足："\\lambda-矩阵定义不准确"
输出：{"tool":"RAG", "context":["\\lambda-矩阵定义", "\\lambda-矩阵相抵标准型", "\\lambda-矩阵初等变换", "史密斯标准型"]}"""

        logging.info(f"QueryRewriterNode (v2.0) 初始化完成，模型: {self.model_name}")

    def _call_llm(self, system_prompt: str, user_content: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            logging.error(f"查询重写 LLM 调用失败: {e}")
            return ""

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """从文本中提取第一个合法 JSON（鲁棒版）"""

        # 先尝试直接解析（最优路径）
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # 提取 ```json ``` 块
        code_block_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(code_block_pattern, text, re.DOTALL)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue

        # 使用“括号匹配”提取 JSON
        stack = []
        start = None

        for i, ch in enumerate(text):
            if ch == '{':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch == '}':
                if stack:
                    stack.pop()
                    if not stack and start is not None:
                        candidate = text[start:i + 1]
                        # 先尝试标准解析
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        # LaTeX 反斜杠修复后重试（如 \lambda → \\lambda）
                        try:
                            # 将未被合法转义的 \X 变为 \\X
                            # JSON 合法转义: \" \\ \/ \b \f \n \r \t \uXXXX
                            fixed = re.sub(
                                r'\\([^"\\/bfnrtu]|$)',
                                r'\\\\\1',
                                candidate
                            )
                            return json.loads(fixed)
                        except json.JSONDecodeError:
                            continue

        logging.warning(f"无法从文本中提取 JSON: {text[:200]}...")
        return {}

    def rewrite(self, question: str, loop_count: int = 0, critique: str = "",
                first_round_keywords: List[str] = None) -> Dict[str, Any]:
        """
        阶梯式重写（分布式检索）

        Args:
            question: 原始问题
            loop_count: 当前轮次 (0=精确, 1=扩展)
            critique: 反思评分节点的纠错意见
            first_round_keywords: 第一轮输出的关键词数组

        Returns:
            {"tool": "RAG", "context": ["关键词组1", ...], "success": bool, "tier": int}
        """
        tier = loop_count  # 0=精确, 1=扩展, 2+=fallback

        logging.info(f"查询重写 (第{tier+1}轮): {question[:50]}...")

        if critique:
            logging.info(f"基于纠错意见重写: {critique[:80]}...")

        # 构造 prompt
        if tier == 0:
            # 第一轮：精确提取
            user_content = f"问题: {question}"
            if critique:
                user_content += f"\n[纠错意见]: {critique}"
            llm_output = self._call_llm(self.system_prompt_precise, user_content)
        elif tier == 1:
            # 第二轮：概念扩展，传入第一轮关键词
            user_content = f"原问题: {question}"
            if first_round_keywords:
                user_content += f"\n第一轮关键词: {json.dumps(first_round_keywords, ensure_ascii=False)}"
            if critique:
                user_content += f"\n上一轮不足: {critique}"
            llm_output = self._call_llm(self.system_prompt_expand, user_content)
        else:
            # Fallback: 直接使用原始问题（包装为数组）
            logging.info(f"达到最大重写轮次 ({tier})，使用原始问题作为关键词")
            return {
                "tool": "RAG",
                "context": [question],
                "success": True,
                "tier": tier,
                "note": "fallback_to_original"
            }

        logging.info(f"重写 LLM 输出: {llm_output[:100]}...")

        if not llm_output:
            if self.fallback_to_original:
                return {
                    "tool": "RAG",
                    "context": [question],
                    "success": False,
                    "tier": tier,
                    "error": "LLM调用失败，使用原始问题"
                }
            return {
                "tool": "RAG",
                "context": [],
                "success": False,
                "tier": tier,
                "error": "LLM调用失败"
            }

        json_result = self._extract_json(llm_output)

        if json_result and "tool" in json_result and "context" in json_result:
            json_result["success"] = True
            json_result["tier"] = tier
            logging.info(f"重写成功 (第{tier+1}轮): {json_result['context']}")
            return json_result

        if self.fallback_to_original:
            return {
                "tool": "RAG",
                "context": [question],
                "success": False,
                "tier": tier,
                "error": "JSON解析失败，使用原始问题"
            }

        return {
            "tool": "RAG",
            "context": [],
            "success": False,
            "tier": tier,
            "error": "JSON解析失败"
        }

    def __call__(self, state: dict) -> dict:
        question = state.get("question", "")
        route = state.get("route", "Math_RAG")
        loop_count = state.get("loop_count", 0)
        critique = state.get("critique", "")
        first_round_kws = state.get("keyword_groups", [])

        if not question:
            return {"error": "缺少问题输入"}

        if route not in ("Math_RAG", "Math"):
            logging.info(f"路由为 {route}，跳过查询重写")
            return {"rewritten_query": question, "extracted_keywords": question}

        rewrite_result = self.rewrite(question, loop_count, critique, first_round_kws)
        context = rewrite_result.get("context", [question])

        # 确保 context 为列表，最多取前 4 组
        if isinstance(context, str):
            context = [context]
        context = context[:4]

        return {
            "rewritten_query": context[0] if context else question,
            "extracted_keywords": " ".join(context) if context else question,
            "keyword_groups": context
        }
