import re
import json
import logging
from typing import Dict, Any, List
from openai import OpenAI

from src.utils.config_loader import load_config


class QueryRewriterNode:
    """查询重写节点 (v2.2)：阶梯式重写 + 策略分类

    - 输出含 strategy 字段 ("multi" / "single")，指导后续检索模式
    - loop_count=0 (Precise): 精确提取定理名/编号/术语
    - loop_count=1 (Concept Expansion): 扩展为上位/相关概念
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

        # ---- 第一轮：精确提取（分布式检索 + 策略分类） ----
        self.system_prompt_precise = (
            "你是一个数学检索专家。你的任务是提取检索关键词并判断检索策略。\n\n"
            "输出格式：严格 JSON --- {\"strategy\": \"multi/single\", \"contexts\": [\"关键词组1\", \"关键词组2\", ...]}\n\n"
            "策略规则：\n"
            '- "single": 问题聚焦于单个定理/定义/证明（如"证明定理10"、"什么是线性空间"）。contexts 第一组为完整锚点。\n'
            '- "multi": 问题涉及多个概念对比或需要跨章节知识（如"说明A、B、C的区别"）。contexts 每组对应一个独立概念。\n\n'
            "检索关键词规则：\n"
            "1. 第一组（contexts[0]）：必须是定理的编号加上最完整的命题原文（保留 LaTeX 公式）。这是最高优先级的检索锚点。\n"
            '2. 后续组：将问题拆解为核心术语（如"初等因子"、"$\\lambda$-矩阵"），用于弥补原文匹配的不足。\n'
            '3. 剥离噪音：去除"证明下述定理"、"老师脾气不好"等无关口语。\n'
            "4. 数量：1-4 组，按重要性从前到后填充。如果问题很短，只需 1 组。\n\n"
            "示例：\n"
            '问题："证明下述定理：定理10 设 $A(\\lambda)$ 是 $\\mathbf\{C\}[\\lambda]$ 上的 $n$ 级满秩矩阵，通过初等变换把 $A(\\lambda)$ 化成对角矩阵，然后把主对角线上每个次数大于0的多项式分解成互不相同的一次因式方幂的乘积，那么所有这些一次因式的方幂（相同的按出现的次数计算）就是 $A(\\lambda)$ 的初等因子。"\n'
            '输出：{"strategy": "single", "contexts": ["定理10 设 $A(\\lambda)$ 是 $\\mathbf\{C\}[\\lambda]$ 上的 $n$ 级满秩矩阵，通过初等变换把 $A(\\lambda)$ 化成对角矩阵，然后把主对角线上每个次数大于0的多项式分解成互不相同的一次因式方幂的乘积，那么所有这些一次因式的方幂（相同的按出现的次数计算）就是 $A(\\lambda)$ 的初等因子。", "$\\lambda$-矩阵初等因子", "满秩矩阵对角化", "初等变换"]}\n\n'
            '问题："线性空间维数公式怎么证"\n'
            '输出：{"strategy": "single", "contexts": ["线性空间维数公式 dim(V1+V2)", "维数定理证明", "线性空间基与维数"]}\n\n'
            '问题："请说明线性映射、线性变换、正交变换和酉变换在定义上的核心区别"\n'
            '输出：{"strategy": "multi", "contexts": ["线性映射定义", "线性变换定义", "正交变换定义", "酉变换定义"]}\n\n'
            '问题："比较矩阵相似与矩阵合同的区别"\n'
            '输出：{"strategy": "multi", "contexts": ["矩阵相似定义与性质", "矩阵合同定义与性质", "相似与合同的区别"]}\n\n'
            '问题："请帮我证明下述命题：环 $R$ 的一个非空子集 $R_\{1\}$ 为一个子环的充分必要条件是 $R_\{1\}$ 对于 $R$ 的减法与乘法都封闭，即\n$$\na, b \\in R _ \{1\} \\quad \\Longrightarrow \\quad a - b \\in R _ \{1\}, a b \\in R _ \{1\} 。\n$$"\n'
            '输出：{"strategy": "single", "contexts": ["环 $R$ 的一个非空子集 $R_\{1\}$ 为一个子环的充分必要条件是 $R_\{1\}$ 对于 $R$ 的减法与乘法都封闭，即\n$$\na, b \\in R _ \{1\} \\quad \\Longrightarrow \\quad a - b \\in R _ \{1\}, a b \\in R _ \{1\} 。\n$$", "子环的充分必要条件"]}'
        )

        # ---- 第二轮：概念扩展（分布式检索） ----
        self.system_prompt_expand = (
            "你是一个数学检索扩展专家。第一轮检索未能满足需求，你需要根据反馈进行"
            '"全域降级扩展"或"靶向精准补漏"。\n\n'
            "输出格式：严格 JSON --- {\"strategy\": \"single\", \"contexts\": [\"关键词组1\", \"关键词组2\", ...]}\n\n"
            "规则：\n"
            "1. 数量约束：至少 1 组，最多 4 组，按重要性从前到后填充。\n"
            "2. 独立性：不要重复第一轮已失败的关键词。\n"
            "3. 双模态逻辑：\n"
            '   - 情况 A [上一轮不足]为空：说明第一轮完全没搜到。请执行"全域降级"，搜索该定理所属的章节名、上位概念。\n'
            '   - 情况 B [上一轮不足]不为空：说明搜到了但内容有缺失。请执行"靶向补漏"，将评价直接转化为搜索词。\n\n'
            "示例 1 (情况 A - 检索空转)：\n"
            '原问题："证明定理10：关于初等因子的分解"\n'
            "第一轮关键词：[\"定理10 设 A(\\lambda) 是满秩矩阵...初等因子\", \"定理10证明\"]\n"
            '上一轮不足：""\n'
            '输出：{"strategy": "single", "contexts": ["\\lambda-矩阵的初等因子定义", "矩阵化为相抵标准型", "多项式矩阵的Smith标准型", "不变因子与初等因子的关系"]}\n\n'
            "示例 3 (情况 B - 补漏重搜)：\n"
            '原问题："证明线性变换 A 在不同基下的矩阵是相似的"\n'
            "第一轮关键词：[\"线性变换在不同基下的矩阵\", \"矩阵相似定义\"]\n"
            '上一轮不足："缺少对坐标变换公式 P^-1AP 的具体推导逻辑"\n'
            '输出：{"strategy": "single", "contexts": ["坐标变换公式推导", "过渡矩阵与线性变换矩阵的关系", "矩阵相似的几何意义", "基变换下的矩阵表示"]}\n\n'
            "示例 5\n"
            '原问题："证明定理10中\\lambda-矩阵的相抵标准型唯一"\n'
            "第一轮关键词：[\"定理10 \\lambda-矩阵相抵标准型\", \"\\lambda-矩阵初等变换\"]\n"
            '上一轮不足："\\lambda-矩阵定义不准确"\n'
            '输出：{"strategy": "single", "contexts": ["\\lambda-矩阵定义", "\\lambda-矩阵相抵标准型", "\\lambda-矩阵初等变换", "史密斯标准型"]}'
        )

        logging.info(f"QueryRewriterNode (v2.2) 初始化完成，模型: {self.model_name}")

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

        # 使用"括号匹配"提取 JSON
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
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            pass
                        try:
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
        阶梯式重写（分布式检索 + 策略分类）

        Returns:
            {"strategy": "multi"/"single", "contexts": ["关键词组1", ...], "success": bool, "tier": int}
        """
        tier = loop_count

        logging.info(f"查询重写 (第{tier+1}轮): {question[:50]}...")

        if critique and tier>=1:
            logging.info(f"基于纠错意见重写: {critique[:80]}...")

        if tier == 0:
            user_content = f"问题: {question}"
            # if critique:
            #     user_content += f"\n[纠错意见]: {critique}"
            llm_output = self._call_llm(self.system_prompt_precise, user_content)
        elif tier == 1:
            user_content = f"原问题: {question}"
            if first_round_keywords:
                user_content += f"\n第一轮关键词: {json.dumps(first_round_keywords, ensure_ascii=False)}"
            if critique:
                user_content += f"\n上一轮不足: {critique}"
            llm_output = self._call_llm(self.system_prompt_expand, user_content)
        else:
            logging.info(f"达到最大重写轮次 ({tier})，使用原始问题作为关键词")
            return {
                "strategy": "single",
                "contexts": [question],
                "success": True,
                "tier": tier,
                "note": "fallback_to_original"
            }

        logging.info(f"重写 LLM 输出: {llm_output[:100]}...")

        if not llm_output:
            if self.fallback_to_original:
                return {
                    "strategy": "single",
                    "contexts": [question],
                    "success": False,
                    "tier": tier,
                    "error": "LLM调用失败，使用原始问题"
                }
            return {
                "strategy": "single",
                "contexts": [],
                "success": False,
                "tier": tier,
                "error": "LLM调用失败"
            }

        json_result = self._extract_json(llm_output)

        if json_result and "strategy" in json_result and "contexts" in json_result:
            json_result["success"] = True
            json_result["tier"] = tier
            logging.info(f"重写成功 (第{tier+1}轮): strategy={json_result.get('strategy')}, "
                         f"{len(json_result['contexts'])} 组")
            return json_result

        # 兼容旧格式：有的模型可能仍输出 tool/context
        if json_result and "tool" in json_result and "context" in json_result:
            json_result["strategy"] = json_result.get("strategy", "multi")
            json_result["contexts"] = json_result.pop("context")
            json_result["success"] = True
            json_result["tier"] = tier
            return json_result

        if self.fallback_to_original:
            return {
                "strategy": "single",
                "contexts": [question],
                "success": False,
                "tier": tier,
                "error": "JSON解析失败，使用原始问题"
            }

        return {
            "strategy": "single",
            "contexts": [],
            "success": False,
            "tier": tier,
            "error": "JSON解析失败"
        }

    def __call__(self, state: dict) -> dict:
        question = state.get("question", "")
        route = state.get("route", "Math")
        loop_count = state.get("loop_count", 0)
        critique = state.get("critique", "")
        first_round_kws = state.get("keyword_groups", [])

        if not question:
            return {"error": "缺少问题输入"}

        if route != "Math":
            logging.info(f"路由为 {route}，跳过查询重写")
            return {"rewritten_query": question, "extracted_keywords": question}

        rewrite_result = self.rewrite(question, loop_count, critique, first_round_kws)
        strategy = rewrite_result.get("strategy", "multi")
        contexts = rewrite_result.get("contexts", rewrite_result.get("context", [question]))

        # 确保 contexts 为列表，最多取前 4 组
        if isinstance(contexts, str):
            contexts = [contexts]
        contexts = contexts[:4]

        return {
            "rewritten_query": contexts[0] if contexts else question,
            "extracted_keywords": " ".join(contexts) if contexts else question,
            "keyword_groups": contexts,
            "strategy": strategy
        }
