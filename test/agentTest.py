#!/usr/bin/env python3
"""
Agent 评测入口脚本

用法：
    python test/agentTest.py                            # 交互式输入路径
    python test/agentTest.py ./test/Test.json           # 直接指定路径
    python test/agentTest.py ./test/longTest.json       # 长测试集

输出路径：自动在 test/result/ 下生成 {文件名}_agent_eval.json
    如 Test.json → test/result/Test_agent_eval.json
"""

import sys
import os
import time
import logging

# 将项目根目录加入 Python 路径，确保 import src 可用
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.utils.config_loader import load_config
from src.evaluation.agent_evaluator import AgentEvaluator


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_test_path(test_file: str) -> str:
    """解析测试文件路径（支持相对路径）"""
    if not os.path.isabs(test_file):
        test_file = os.path.join(_project_root, test_file)
    return os.path.normpath(test_file)


def resolve_output_path(test_file: str) -> str:
    """自动推导输出路径: test/result/{filename}_agent_eval.json"""
    test_dir = os.path.dirname(test_file)
    result_dir = os.path.join(test_dir, "result")
    test_name = os.path.splitext(os.path.basename(test_file))[0]
    return os.path.join(result_dir, f"{test_name}_agent_eval.json")


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    print("=" * 60)
    print("MathRAG Agent 评测工具 (v2.2)")
    print("评测系统: RAG-only vs Agentic RAG")
    print("评测裁判: DeepSeek + GLM")
    print("=" * 60)

    # # 获取测试文件路径
    # if len(sys.argv) > 1:
    #     test_file = sys.argv[1]
    # else:
    #     test_file = input("\n请输入测试集 JSON 文件路径: ").strip()

    # test_file_list = ["./test/Test.json","./test/longTest.json"]
    test_file_list = ["./test/longTest.json"]
    for idx, test_file in enumerate(test_file_list):
        test_file = resolve_test_path(test_file)

        if not os.path.exists(test_file):
            print(f"错误: 文件不存在 - {test_file}")
            sys.exit(1)

        output_path = resolve_output_path(test_file)
        print(f"\n测试文件: {test_file}")
        print(f"输出路径: {output_path}")

        # 每个测试集之间重启 llama.cpp 容器，避免连续调用后服务崩溃
        if idx > 0:
            print("\n重启 llama.cpp 容器以清空服务状态...")
            import subprocess
            try:
                subprocess.run(["docker", "restart", "llama-server"],
                               capture_output=True, timeout=30, check=True)
                print("llama.cpp 容器重启完成")
                time.sleep(3)  # 等待服务完全启动
            except Exception as e:
                print(f"警告: 容器重启失败 ({e})，继续执行...")

        # 加载配置
        print("\n加载配置...")
        try:
            config = load_config()
        except Exception as e:
            print(f"配置加载失败: {e}")
            sys.exit(1)

        # 初始化评测器
        print("初始化 Agent 评测器 (加载模型可能需要几分钟)...")
        try:
            evaluator = AgentEvaluator(config)
        except Exception as e:
            print(f"评测器初始化失败: {e}")
            sys.exit(1)

        # 执行评测
        print(f"\n开始评测...")
        try:
            evaluator.evaluate(test_file, output_path)
        except KeyboardInterrupt:
            print("\n用户中断评测，已保存的结果不会丢失")
            sys.exit(0)
        except Exception as e:
            print(f"评测过程出错: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)


if __name__ == "__main__":
    main()
