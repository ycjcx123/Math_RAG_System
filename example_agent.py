#!/usr/bin/env python3
"""
MathRAG Agent 示例脚本 (v2.2)

演示升级后的二元路由 (Chat/Math) + 三段式自适应流程 (Fast-Track / Self-Refine / RAG)。
"""

import sys
import logging
from src.utils.config_loader import load_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def check_dependencies():
    missing_deps = []
    try:
        import langgraph
    except ImportError:
        missing_deps.append("langgraph")
    try:
        import langchain_core
    except ImportError:
        missing_deps.append("langchain-core")
    if missing_deps:
        print(f"缺少必要的依赖: {', '.join(missing_deps)}")
        print("请安装: pip install langgraph langchain-core")
        return False
    return True


def run_agent_example():
    print("=" * 60)
    print("MathRAG Agent v2.2 示例")
    print("特性: 二元路由 + 三段式自适应 (Fast-Track/Self-Refine/RAG)")
    print("=" * 60)

    if not check_dependencies():
        sys.exit(1)

    print("\n加载配置...")
    try:
        config = load_config()
    except Exception as e:
        print(f"配置加载失败: {e}")
        sys.exit(1)

    print("创建 Agent...")
    try:
        from src.agent import create_agent
        agent = create_agent(config)
    except Exception as e:
        print(f"Agent 创建失败: {e}")
        sys.exit(1)

    # 测试用例覆盖两种路由 + 三段式流程
    test_queries = [
        # Chat 路径
        "你好，今天天气怎么样？",
        # Math 路径 - Fast-Track
        "求解方程 $x^2 - 4 = 0$",
        # Math 路径 - Self-Refine (需修正)
        "什么是线性无关？",
        # Math 路径 - RAG (复杂定理)
        "证明下述定理：定理10 设 $A(\\lambda)$ 是 $\\mathbf{C}[\\lambda]$ 上的 $n$ 级满秩矩阵，通过初等变换把 $A(\\lambda)$ 化成对角矩阵，然后把主对角线上每个次数大于0的多项式分解成互不相同的一次因式方幂的乘积，那么所有这些一次因式的方幂（相同的按出现的次数计算）就是 $A(\\lambda)$ 的初等因子。"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"测试 {i}: {query[:60]}...")
        print(f"{'='*40}")

        try:
            result = agent.run(query)

            print(f"  路由路径: {result.get('route', 'Unknown')}")
            print(f"  决策路径: {result.get('logic_path', 'N/A')}")
            print(f"  生成来源: {result.get('generation_source', 'N/A')}")
            print(f"  Self-Refine轮数: {result.get('self_refine_count', 0)}")
            print(f"  RAG循环次数: {result.get('loop_count', 0)}")
            print(f"  反思评分: {result.get('score', 'N/A')}/100")
            if result.get('critique'):
                print(f"  纠错意见: {result.get('critique', '')[:80]}...")
            print(f"  检索文档数: {result.get('documents_retrieved', 0)}")
            if result.get('error'):
                print(f"  错误信息: {result.get('error')}")

            print(f"\n答案:")
            print(f"{'-'*30}")
            print(result.get('answer', '无答案')[:300])
            print(f"{'-'*30}")

        except Exception as e:
            print(f"查询处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


def main():
    print("MathRAG Agentic RAG 系统 v2.2")
    print("特性: 二元路由 + 三段式自适应 + 双轨检索 + 硬件级阈值过滤")
    print()
    response = input("是否运行示例？(y/n): ").strip().lower()
    if response != 'y':
        print("退出。")
        return
    run_agent_example()


if __name__ == "__main__":
    main()
