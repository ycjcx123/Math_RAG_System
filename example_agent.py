#!/usr/bin/env python3
"""
MathRAG Agent 示例脚本

演示如何使用新构建的 Agentic RAG 系统。
注意：需要先安装 langgraph: pip install langgraph langchain-core
"""

import sys
import logging
from src.utils.config_loader import load_config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def check_dependencies():
    """检查必要的依赖是否已安装"""
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
        print(f"❌ 缺少必要的依赖: {', '.join(missing_deps)}")
        print("请使用以下命令安装:")
        print("  pip install langgraph langchain-core")
        return False

    return True


def run_agent_example():
    """运行 Agent 示例"""
    print("=" * 60)
    print("MathRAG Agent 示例")
    print("=" * 60)

    # 检查依赖
    if not check_dependencies():
        sys.exit(1)

    # 加载配置
    print("\n📋 加载配置...")
    try:
        config = load_config()
        print("✅ 配置加载成功")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        sys.exit(1)

    # 创建 Agent
    print("\n🤖 创建 Agent...")
    try:
        from src.agent import create_agent
        agent = create_agent(config)
        print("✅ Agent 创建成功")
    except Exception as e:
        print(f"❌ Agent 创建失败: {e}")
        print("可能的原因:")
        print("  1. LangGraph 未正确安装")
        print("  2. Llama.cpp Server 未运行")
        print("  3. Qdrant 数据库未连接")
        sys.exit(1)

    # 测试查询
    test_queries = [
        "证明下述定理：定理10 设 $A(\\lambda)$ 是 $\\mathbf{C}[\\lambda]$ 上的 $n$ 级满秩矩阵，通过初等变换把 $A(\\lambda)$ 化成对角矩阵，然后把主对角线上每个次数大于0的多项式分解成互不相同的一次因式方幂的乘积，那么所有这些一次因式的方幂（相同的按出现的次数计算）就是 $A(\\lambda)$ 的初等因子。"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*40}")
        print(f"测试 {i}: {query}")
        print(f"{'='*40}")

        try:
            result = agent.run(query)

            print(f"📊 路由决策: {result.get('route', 'Unknown')}")
            print(f"🔄 循环次数: {result.get('loop_count', 0)}")
            print(f"📄 检索文档数: {result.get('documents_retrieved', 0)}")
            print(f"✅ 文档相关性: {'是' if result.get('is_relevant', False) else '否'}")

            if result.get('error'):
                print(f"⚠️  错误信息: {result.get('error')}")

            print(f"\n💡 答案:")
            print(f"{'-'*30}")
            print(result.get('answer', '无答案'))
            print(f"{'-'*30}")

        except Exception as e:
            print(f"❌ 查询处理失败: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


def main():
    """主函数"""
    print("MathRAG Agentic RAG 系统示例")
    print("版本: 1.0")
    print("作者: MathRAG 团队")
    print()

    # 显示系统状态
    print("系统状态检查:")
    print("  ✅ 目录结构已重构")
    print("  ✅ 配置文件已更新")
    print("  ✅ Agent 框架已构建")
    print("  ⚠️  需要安装 langgraph 和 langchain-core")
    print()

    # 询问是否继续
    response = input("是否运行示例？(y/n): ").strip().lower()
    if response != 'y':
        print("退出示例。")
        return

    # 运行示例
    run_agent_example()


if __name__ == "__main__":
    main()