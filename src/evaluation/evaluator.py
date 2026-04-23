# src/evaluation/evaluator.py
import json
import os
import time
import logging
import requests
from typing import List, Dict, Any, Optional
from openai import OpenAI

# 导入组件
from src.pipeline.chat_pipeline import ChatPipeline
from src.utils.config_loader import load_config

class MathEvaluator:
    def __init__(self, config):
        """config 由 load_config() 返回"""
        self.full_config = config
        self.test_cfg = self.full_config.get("test", {})
        self.judge_cfg = self.test_cfg.get("judgement_model", {})
        self.max_retries = self.test_cfg.get("max_retries", 3)
        self.JUDGE_SYSTEM_PROMPT = self.test_cfg.get("JUDGE_SYSTEM_PROMPT", "")
        self.test_file_path = self.test_cfg.get("test_file_path")
        
        # 初始化裁判客户端
        self._init_judges()
        
        # 基础输出目录（从 output_path 提取目录）
        self.output_path = self.test_cfg.get("output_path",
                                             f"./test/result/{self.test_file_path.split('/')[-1]}")
        self.output_dir = os.path.dirname(self.output_path)
        os.makedirs(self.output_dir, exist_ok=True)
        self.output_filename_base = os.path.splitext(os.path.basename(self.output_path))[0]

    def _init_judges(self):
        """初始化不同裁判模型的客户端"""
        # DeepSeek 裁判
        ds_info = self.judge_cfg.get("DeepSeek", {})
        self.ds_client = OpenAI(
            api_key=ds_info.get("DEEPSEEK_API_KEY"),
            base_url=ds_info.get("url")
        )
        
        # GLM 裁判
        self.glm_info = self.judge_cfg.get("GLM", {})
        self.glm_url = self.glm_info.get("url")
        self.glm_auth = self.glm_info.get("GLM_Authorization")
        self.glm_model=self.glm_info.get("model")

    def _call_deepseek_judge(self, query, contexts, model_answer, ref_answer) -> Dict:
        """调用 DeepSeek 作为裁判，返回 JSON 字典"""
        user_prompt = f"""
        -【题目】: {query}
        -【参考答案（仅供裁判参考，非模型输出）】: {ref_answer}
        -【参考资料（仅供裁判参考，非模型输出）】: {contexts}
        -【待评测的模型回答（必须以此为准）】: {model_answer}"""
        
        for attempt in range(self.max_retries):
            try:
                response = self.ds_client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.3
                )
                return json.loads(response.choices[0].message.content)
            except Exception as e:
                logging.error(f"DeepSeek Judge 失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
                else:
                    return {}

    def _call_glm_judge(self, query, contexts, model_answer, ref_answer) -> Dict:
        """调用 GLM 作为裁判，返回 JSON 字典"""
        user_prompt = f"""
        -【题目】: {query}
        -【参考答案（仅供裁判参考，非模型输出）】: {ref_answer}
        -【参考资料（仅供裁判参考，非模型输出）】: {contexts}
        -【待评测的模型回答（必须以此为准）】: {model_answer}"""

        headers = {"Authorization": f"Bearer {self.glm_auth}", "Content-Type": "application/json"}
        payload = {
            "model": self.glm_model,
            "messages": [
                {"role": "system", "content": self.JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            "response_format": {"type": "json_object"}
        }
        
        for attempt in range(self.max_retries):
            try:
                res = requests.post(self.glm_url, json=payload, headers=headers, timeout=60)
                res.raise_for_status()
                content = res.json()['choices'][0]['message']['content']
                # 确保返回的是可解析的 JSON 字符串
                return json.loads(content) if isinstance(content, str) else content
            except Exception as e:
                logging.error(f"GLM Judge 失败 (尝试 {attempt+1}/{self.max_retries}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(10)
                else:
                    return {}

    def run_task(self, task_name: str, task_config: Dict[str, Any], last_index: int = 1):
        """
        运行单个任务
        :param task_name: 任务名称，如 'no_fix'
        :param task_config: 任务配置，包含 collection_name, block_aggregate
        :param last_index: 从第几条开始（用于断点续写）
        """
        logging.info(f"开始评测任务: {task_name}")
        
        # 加载测试数据集
        with open(self.test_file_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # 确定输出文件路径
        output_file = os.path.join(self.output_dir, f"{self.output_filename_base}_{task_name}.json")
        
        # 加载已有结果（断点续写）
        results = []
        if os.path.exists(output_file) and last_index > 1:
            with open(output_file, 'r', encoding='utf-8') as f:
                existing = json.load(f)
                if isinstance(existing, list):
                    results = [item for item in existing if item.get('id', 0) < last_index]
                else:
                    results = []
        
        # 遍历数据集
        for i, item in enumerate(dataset[last_index-1:], start=last_index):
            query = item['query']
            ref_answer = item.get('answer_text', "")
            
            logging.info(f"[{task_name}] 评测 [{i}/{len(dataset)}]: {query[:30]}...")
            
            # 动态修改 retriever 配置
            run_config = self.full_config.copy()
            run_config['retriever'] = self.full_config.get('retriever', {}).copy()
            run_config['retriever']['collection_name'] = task_config.get('collection_name')
            run_config['retriever']['block_aggregate'] = task_config.get('block_aggregate', False)
            
            # 初始化 Pipeline
            pipeline = ChatPipeline(config=run_config, query=query)
            
            try:
                # 获取回答和上下文（contexts 是文本列表，不包含 ID）
                contexts, answer = pipeline.run()
            except Exception as e:
                logging.error(f"Pipeline 运行失败: {e}")
                contexts, answer = [], f"错误: {str(e)}"
            
            # 调用裁判打分
            ds_score = self._call_deepseek_judge(query, contexts, answer, ref_answer)
            glm_score = self._call_glm_judge(query, contexts, answer, ref_answer)
            
            # 构建条目（不包含 rerank_id）
            entry = {
                "id": i,
                "query": query,
                "answer": answer,   # 可选字段，保留原始回答
                task_name: {
                    "response": answer,
                    "deepseek": ds_score,
                    "GLM": glm_score
                }
            }
            results.append(entry)
            
            # 实时保存
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
        
        logging.info(f"任务 {task_name} 完成，结果保存至: {output_file}")
    
    def run_all_tasks(self, last_test_index_per_task: Optional[Dict[str, int]] = None):
        """
        运行所有配置在 test.task 下的任务
        :param last_test_index_per_task: 可选，每个任务的断点索引，例如 {"no_fix": 5, "baseline_dense": 1}
        """
        tasks = self.test_cfg.get("task", {})
        if not tasks:
            logging.warning("未找到任何任务配置 (test.task 为空)")
            return
        
        for task_name, task_cfg in tasks.items():
            last_index = 1
            if last_test_index_per_task and task_name in last_test_index_per_task:
                last_index = last_test_index_per_task[task_name]
            self.run_task(task_name, task_cfg, last_index)