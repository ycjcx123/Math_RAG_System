import json
import logging
from typing import List, Tuple, Union

from src.retriever import QdrantRetriever, Reranker, BlockAggregator

class Score:
    def __init__(self,config):
        """config=load_config.get()"""
        self.test_config=config.get("test")
        self.retriever_config=config.get("retriever")

        self.retriever=QdrantRetriever(self.retriever_config)
        self.reranker=Reranker(self.retriever_config)

    def _get_chunk_id(self, query: str, task_config: dict):
        """
        执行检索和重排，返回原始节点列表和重排后的 ID 列表
        :param query: 查询文本
        :param task_config: 任务配置，需包含 collection_name 和 hybrid 字段
        :return: (nodes, rerank_ids)
            nodes: llama_index 的 NodeWithScore 列表（原始召回结果）
            rerank_ids: 重排后的 chunk ID 列表
        """
        collection_name = task_config.get("collection_name")
        hybrid = task_config.get("hybrid", False)

        # 检索
        nodes = self.retriever.retrieve(
            query=query,
            collection_name=collection_name,
            hybrid=hybrid
        )

        # 重排
        rerank_ids, _ = self.reranker.rerank(query=query, nodes=nodes)

        return nodes, rerank_ids

    def _evaluate_std(self, retrieved_ids: List[str], expected: Union[List[str], List[List[str]]]) -> Tuple[float, float, float]:
        """
        标准评测模式（展平 expected，计算 recall / MRR / MAP）
        :param retrieved_ids: 检索出的 chunk ID 列表
        :param expected: 正确答案，可为单层列表或嵌套列表（嵌套时会自动展平）
        :return: (recall, mrr, map)
        """
        retrieved = [str(x) for x in retrieved_ids]

        # 展平 expected
        if expected and isinstance(expected[0], list):
            expected_flat = list(set(str(item) for sublist in expected for item in sublist))
        else:
            expected_flat = list(set(str(item) for item in expected))

        if not expected_flat:
            return 0.0, 0.0, 0.0

        hit_ranks = {}
        for pos, cid in enumerate(retrieved, start=1):
            if cid in expected_flat and cid not in hit_ranks:
                hit_ranks[cid] = pos

        num_hits = len(hit_ranks)
        total_expected = len(expected_flat)

        recall = num_hits / total_expected
        mrr = 1.0 / min(hit_ranks.values()) if num_hits > 0 else 0.0

        if num_hits == 0:
            map_score = 0.0
        else:
            sorted_ranks = sorted(hit_ranks.values())
            ap_sum = sum((i + 1) / rank for i, rank in enumerate(sorted_ranks))
            map_score = ap_sum / total_expected

        return recall, mrr, map_score

    def _evaluate_new(self, retrieved_ids: List[str], expected: List[List[str]]) -> Tuple[float, float, float]:
        """
        严苛组命中评测模式（每个 group 必须完全命中才算一次命中）
        :param retrieved_ids: 检索出的 chunk ID 列表
        :param expected: 正确答案，必须为嵌套列表，每个子列表是一个 group
        :return: (recall, mrr, map)
        """
        retrieved = [str(x) for x in retrieved_ids]
        # 确保每个 group 转为 set
        groups = [[str(x) for x in g] if isinstance(g, list) else [str(g)] for g in expected]
        total_groups = len(groups)
        if total_groups == 0:
            return 0.0, 0.0, 0.0

        group_needed = [set(g) for g in groups]
        group_seen = [set() for _ in groups]
        group_first_rank = [None] * total_groups

        for pos, cid in enumerate(retrieved, start=1):
            for i, needed in enumerate(group_needed):
                if cid in needed:
                    group_seen[i].add(cid)
                    if group_first_rank[i] is None and group_seen[i] == needed:
                        group_first_rank[i] = pos

        hit_indices = [i for i, r in enumerate(group_first_rank) if r is not None]
        hit_count = len(hit_indices)
        recall = hit_count / total_groups
        mrr = 1.0 / min(group_first_rank[i] for i in hit_indices) if hit_count > 0 else 0.0

        if hit_count == 0:
            map_score = 0.0
        else:
            sorted_hits = sorted([group_first_rank[i] for i in hit_indices])
            ap_sum = sum((i + 1) / rank for i, rank in enumerate(sorted_hits))
            map_score = ap_sum / hit_count

        return recall, mrr, map_score

    def _cumulate_score(self, raw_nodes: List, rerank_ids: List[str], expected: Union[List[str], List[List[str]]]) -> dict:
        """
        计算单个查询的原始召回和重排结果的所有指标（std 和 new 模式）
        :param raw_nodes: 原始检索返回的节点列表（含 NodeWithScore）
        :param rerank_ids: 重排后的 ID 列表
        :param expected: 正确答案
        :return: 字典，键为 "raw_std", "raw_new", "rerank_std", "rerank_new"，值为 (recall, mrr, map)
        """
        raw_ids = [node.node.node_id for node in raw_nodes] if raw_nodes else []

        raw_std = self._evaluate_std(raw_ids, expected)
        raw_new = self._evaluate_new(raw_ids, expected)
        rerank_std = self._evaluate_std(rerank_ids, expected)
        rerank_new = self._evaluate_new(rerank_ids, expected)

        return {
            "raw_std": raw_std,
            "raw_new": raw_new,
            "rerank_std": rerank_std,
            "rerank_new": rerank_new
        }

    def _print_results(self, all_results: List[dict]):
        """格式化打印评测结果表格"""
        header = (f"{'系统 (Top-K)':<25} | {'MAP':<8} | {'MRR':<8} | {'Recall':<8} | "
                  f"{'MAP_new':<8} | {'MRR_new':<8} | {'Recall_new':<8}")
        print("\n" + "=" * 95)
        print(header)
        print("-" * 95)
        for res in all_results:
            s = res["System"]
            r = res["results"]   # 顺序: recall_std, mrr_std, map_std, recall_new, mrr_new, map_new
            print(f"{s:<25} | {r[2]:<8.4f} | {r[1]:<8.4f} | {r[0]:<8.4f} | "
                  f"{r[5]:<8.4f} | {r[4]:<8.4f} | {r[3]:<8.4f}")
        print("=" * 95)

    # ---------------------------- 公共接口 ----------------------------

    def run_evaluation(self):
        """
        执行完整的评测流程：
        1. 读取数据集
        2. 遍历 test_config 中配置的所有任务
        3. 对每个任务计算原始召回和重排后的指标（std 和 new 模式）
        4. 打印对比表格
        """
        tasks = self.test_config.get("tasks", [])
        if not tasks:
            print("没有配置评测任务，请在 test.tasks 中添加任务定义。")
            return

        if not self.dataset_path:
            print("未配置数据集路径 (test.dataset_path)")
            return

        with open(self.dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        all_results = []

        for task in tasks:
            print(f"\n>>> 正在处理: {task['name']} (召回 Top-{self.initial_top_k} & 重排 Top-{self.final_top_n})...")
            metrics = {
                "raw_std": [0.0, 0.0, 0.0],
                "raw_new": [0.0, 0.0, 0.0],
                "rerank_std": [0.0, 0.0, 0.0],
                "rerank_new": [0.0, 0.0, 0.0]
            }
            count = 0

            for item in dataset:
                query = item.get("query")
                expected = item.get(task["expected_key"])
                if not query or not expected:
                    continue

                count += 1
                nodes, rerank_ids = self._get_chunk_id(query, task)
                scores = self._cumulate_score(nodes, rerank_ids, expected)

                # 累加各项指标
                for key in metrics:
                    for i in range(3):
                        metrics[key][i] += scores[key][i]

            if count == 0:
                print(f"警告：任务 {task['name']} 没有有效样本，跳过。")
                continue

            # 计算平均值
            def avg(lst):
                return [round(v / count, 4) for v in lst]

            all_results.append({
                "System": f"{task['name']} (Raw@{self.initial_top_k})",
                "results": avg(metrics["raw_std"]) + avg(metrics["raw_new"])
            })
            all_results.append({
                "System": f"{task['name']} (Rerank@{self.final_top_n})",
                "results": avg(metrics["rerank_std"]) + avg(metrics["rerank_new"])
            })

        self._print_results(all_results)