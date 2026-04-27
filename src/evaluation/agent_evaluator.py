"""Agent 评测器：对 RAG-only 和 Agent 系统进行端到端评估

用法：
    from src.evaluation.agent_evaluator import AgentEvaluator
    evaluator = AgentEvaluator()
    evaluator.evaluate("./test/Test.json", "./test/result/Test_agent_eval.json")
"""

import json
import logging
import os
import time
from typing import List, Dict, Any, Optional

import requests
from openai import OpenAI

from src.agent import create_agent
from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)


class AgentEvaluator:
    """Agent 评测器

    同时对 RAG-only 系统（ChatPipeline）和 Agentic RAG 系统（AgentGraph）
    进行端到端评测，使用 DeepSeek 和 GLM 作为裁判模型打分。
    """

    def __init__(self, config: dict = None):
        if config is None:
            config = load_config()
        self.full_config = config

        # 评测配置
        self.test_cfg = config.get("test", {})
        self.max_retries = self.test_cfg.get("max_retries", 3)

        # 裁判 system prompt
        self.JUDGE_SYSTEM_PROMPT = config.get("test", {}).get(
            "judgement_model", {}
        ).get("JUDGE_SYSTEM_PROMPT", "")

        # Fallback 消息
        self.fallback_message = config.get("agent", {}).get(
            "fallback_message", "在教材中未找到准确定义，建议换个问法"
        )

        # 初始化裁判客户端
        self._init_judges()

        # 初始化 Agent（全局复用，避免重复加载模型）
        logger.info("正在初始化 AgentGraph...")
        self.agent = create_agent(config)
        logger.info("AgentGraph 初始化完成")

    # ==================== Judge 初始化 ====================

    def _init_judges(self):
        judge_cfg = self.test_cfg.get("judgement_model", {})

        # DeepSeek 裁判
        ds_info = judge_cfg.get("DeepSeek", {})
        self.ds_client = OpenAI(
            api_key=ds_info.get("DEEPSEEK_API_KEY"),
            base_url=ds_info.get("url"),
        )

        # GLM 裁判
        self.glm_info = judge_cfg.get("GLM", {})
        self.glm_url = self.glm_info.get("url")
        self.glm_auth = self.glm_info.get("GLM_Authorization")
        self.glm_model = self.glm_info.get("model")

    # ==================== Judge 调用（带重试） ====================

    def _build_judge_prompt(
        self, query: str, ref_answer: str, contexts: List[str], model_answer: str
    ) -> str:
        """构建提交给裁判的 user prompt"""
        if contexts:
            context_str = "\n".join(
                f"[{i+1}] {text}" for i, text in enumerate(contexts)
            )
        else:
            context_str = "暂无参考资料"

        return (
            f"-【题目】: {query}\n"
            f"-【参考答案（仅供裁判参考，非模型输出）】: {ref_answer}\n"
            f"-【参考资料（仅供裁判参考，非模型输出）】: {context_str}\n"
            f"-【待评测的模型回答（必须以此为准）】: {model_answer}"
        )

    def _call_deepseek_judge(
        self, query: str, ref_answer: str, contexts: List[str], model_answer: str
    ) -> Dict:
        """调用 DeepSeek 裁判，失败时重试并 sleep(10)"""
        user_prompt = self._build_judge_prompt(query, ref_answer, contexts, model_answer)
        last_error = None

        for attempt in range(self.max_retries):
            try:
                response = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3,
                )
                content = response.choices[0].message.content
                return json.loads(content)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"DeepSeek Judge 失败 (第{attempt+1}/{self.max_retries}次): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(10)

        logger.error(f"DeepSeek Judge 重试耗尽: {last_error}")
        return {
            "reasoning": "",
            "scores": {
                "correctness": 0,
                "faithfulness": 0,
                "answer_relevance": 0,
                "context_relevance": 0,
            },
        }

    def _call_glm_judge(
        self, query: str, ref_answer: str, contexts: List[str], model_answer: str
    ) -> Dict:
        """调用 GLM 裁判（requests 直连），失败时重试并 sleep(10)"""
        user_prompt = self._build_judge_prompt(query, ref_answer, contexts, model_answer)
        last_error = None

        headers = {
            "Authorization": f"Bearer {self.glm_auth}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.glm_model,
            "messages": [
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
        }

        for attempt in range(self.max_retries):
            try:
                res = requests.post(self.glm_url, json=payload, headers=headers, timeout=180)
                res.raise_for_status()
                content = res.json()["choices"][0]["message"]["content"]
                if isinstance(content, str):
                    return json.loads(content)
                return content
            except Exception as e:
                last_error = e
                logger.warning(
                    f"GLM Judge 失败 (第{attempt+1}/{self.max_retries}次): {e}"
                )
                if attempt < self.max_retries - 1:
                    time.sleep(10)

        logger.error(f"GLM Judge 重试耗尽: {last_error}")
        return {
            "reasoning": "",
            "scores": {
                "correctness": 0,
                "faithfulness": 0,
                "answer_relevance": 0,
                "context_relevance": 0,
            },
        }

    # ==================== RAG-only 测试 ====================

    def _run_rag_only(self, query: str) -> tuple:
        """运行 RAG-only 系统，返回 (contexts, answer)

        复用 Agent 的 BlockAggregator 和 Generator 以避免重复加载
        BGE-M3 模型导致 OOM。
        """
        try:
            contexts = self.agent.block_aggregator.retrieve_and_aggregate(query=query)
            contexts = contexts or []
            answer = self.agent.generator.generate(query=query, contexts=contexts) or ""
            return contexts, answer
        except Exception as e:
            logger.error(f"RAG-only pipeline 运行失败: {e}")
            return [], f"错误: {str(e)}"

    # ==================== Agent 测试 ====================

    def _run_agent(self, query: str) -> Dict[str, Any]:
        """运行 Agent 系统，返回完整结果 dict"""
        try:
            return self.agent.run(query)
        except Exception as e:
            logger.error(f"Agent 运行失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "answer": f"Agent系统错误: {str(e)}",
                "route": "Error",
                "generation_source": "error",
                "loop_count": 0,
                "score": 0,
                "documents": [],
                "error": str(e),
            }

    def _compute_loop_counter(self, agent_result: Dict[str, Any]) -> int:
        """
        计算 loop_counter 语义值:
          -1 = 未调用 RAG（Fast-Track / Self-Refined）
          -2 = Fallback / 系统错误
           0 = RAG 一次命中
           1 = RAG 第二次命中
        """
        route = agent_result.get("route", "")
        generation_source = agent_result.get("generation_source", "")

        if route in ("Fallback", "Error") or generation_source == "error":
            return -2
        if generation_source in ("fast_track", "self_refined"):
            return -1
        # RAG 路径：loop_count 0=一次命中, 1=二次命中
        return agent_result.get("loop_count", 0)

    def _get_agent_contexts(
        self, agent_result: Dict[str, Any], loop_counter: int
    ) -> List[str]:
        """获取 Agent 生成答案时使用的参考资料"""
        if loop_counter < 0:
            return []  # 未调用 RAG 或 Fallback
        return agent_result.get("documents", [])

    def _make_fallback_entry(self, agent_result: Dict[str, Any]) -> Dict:
        """Fallback 固定输出"""
        forced = {
            "correctness": 0,
            "faithfulness": 2,
            "answer_relevance": 0,
            "context_relevance": 0,
        }
        answer = agent_result.get("answer", self.fallback_message)
        loop_counter = self._compute_loop_counter(agent_result)
        return {
            "response": answer,
            "loop_counter": loop_counter,
            "deepseek": {"reasoning": "", "scores": dict(forced)},
            "GLM": {"reasoning": "", "scores": dict(forced)},
        }

    # ==================== 主评测流程 ====================

    def evaluate(self, test_file_path: str, output_path: str) -> List[Dict]:
        """执行完整评测

        Args:
            test_file_path: 测试集 JSON 文件路径
            output_path: 结果输出 JSON 文件路径
        """
        # 加载测试集
        with open(test_file_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        logger.info(f"加载测试集: {test_file_path} ({len(dataset)} 条)")

        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 断点续写：加载已有结果
        results = []
        last_id = 0
        if os.path.exists(output_path):
            with open(output_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                results = existing
                completed_ids = {item.get("id") for item in existing if item.get("id")}
                if completed_ids:
                    last_id = max(completed_ids)
                    logger.info(
                        f"断点续写: 已有 {len(results)} 条，从 id={last_id + 1} 继续"
                    )

        # 遍历测试集
        for i, item in enumerate(dataset, start=1):
            if i <= last_id:
                continue

            query = item.get("query", "")
            ref_answer = item.get("answer_text", "")

            logger.info(f"\n{'='*50}")
            logger.info(f"[{i}/{len(dataset)}] {query[:60]}...")

            # ====== RAG-only ======
            logger.info("--- RAG-only ---")
            rag_contexts, rag_answer = self._run_rag_only(query)

            if rag_answer:
                rag_ds = self._call_deepseek_judge(
                    query, ref_answer, rag_contexts, rag_answer
                )
                rag_glm = self._call_glm_judge(
                    query, ref_answer, rag_contexts, rag_answer
                )
            else:
                rag_ds = {
                    "reasoning": "",
                    "scores": {
                        "correctness": 0,
                        "faithfulness": 0,
                        "answer_relevance": 0,
                        "context_relevance": 0,
                    },
                }
                rag_glm = dict(rag_ds)

            rag_entry = {
                "response": rag_answer,
                "deepseek": rag_ds,
                "GLM": rag_glm,
            }

            # ====== Agent ======
            logger.info("--- Agent ---")
            agent_result = self._run_agent(query)
            loop_counter = self._compute_loop_counter(agent_result)

            if loop_counter == -2:
                # Fallback: 固定分数，不调 Judge
                agent_entry = self._make_fallback_entry(agent_result)
            else:
                agent_answer = agent_result.get("answer", "")
                agent_contexts = self._get_agent_contexts(agent_result, loop_counter)

                if agent_answer:
                    agent_ds = self._call_deepseek_judge(
                        query, ref_answer, agent_contexts, agent_answer
                    )
                    agent_glm = self._call_glm_judge(
                        query, ref_answer, agent_contexts, agent_answer
                    )
                else:
                    agent_ds = {
                        "reasoning": "",
                        "scores": {
                            "correctness": 0,
                            "faithfulness": 0,
                            "answer_relevance": 0,
                            "context_relevance": 0,
                        },
                    }
                    agent_glm = dict(agent_ds)

                agent_entry = {
                    "response": agent_answer,
                    "loop_counter": loop_counter,
                    "deepseek": agent_ds,
                    "GLM": agent_glm,
                }

            # 组装输出条目
            entry = {
                "id": i,
                "query": query,
                "answer": ref_answer,
                "RAG": rag_entry,
                "Agent": agent_entry,
            }
            results.append(entry)

            # 实时保存
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)

            # 打印摘要
            logger.info(
                f"  RAG response: {rag_answer[:60] if rag_answer else '(空)'}..."
            )
            logger.info(f"  Agent loop_counter={loop_counter}, "
                        f"response: {agent_entry['response'][:60] if agent_entry['response'] else '(空)'}...")

        logger.info(f"\n评测完成! 共 {len(results)} 条，结果保存至: {output_path}")
        return results
