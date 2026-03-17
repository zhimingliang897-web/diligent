"""多智能体系统测试脚本

用于验证 Supervisor + Workers 架构是否正常工作
"""

import os
import sys

# 确保能找到 agent 模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent.multi import build_multi_agent_graph


def test_simple_code_task():
    """测试简单的代码任务（单 Agent）"""
    print("\n" + "=" * 60)
    print("测试 1: 简单代码任务")
    print("=" * 60)

    graph = build_multi_agent_graph()

    initial_state = {
        "messages": [],
        "iteration_count": 0,
        "task_plan": [],
        "current_worker": "supervisor",
        "worker_results": {},
        "handoff_context": "",
        "original_query": "写一个计算斐波那契数列第n项的Python函数",
    }

    result = graph.invoke(initial_state)
    print("\n最终回答:")
    print("-" * 40)
    if result.get("messages"):
        print(result["messages"][-1].content)
    print("-" * 40)


def test_data_query():
    """测试数据查询任务（单 Agent）"""
    print("\n" + "=" * 60)
    print("测试 2: 数据查询任务")
    print("=" * 60)

    graph = build_multi_agent_graph()

    initial_state = {
        "messages": [],
        "iteration_count": 0,
        "task_plan": [],
        "current_worker": "supervisor",
        "worker_results": {},
        "handoff_context": "",
        "original_query": "帮我计算 1+2+3+...+100 的结果",
    }

    result = graph.invoke(initial_state)
    print("\n最终回答:")
    print("-" * 40)
    if result.get("messages"):
        print(result["messages"][-1].content)
    print("-" * 40)


def test_multi_agent_collaboration():
    """测试多 Agent 协作任务"""
    print("\n" + "=" * 60)
    print("测试 3: 多 Agent 协作任务")
    print("=" * 60)

    graph = build_multi_agent_graph()

    initial_state = {
        "messages": [],
        "iteration_count": 0,
        "task_plan": [],
        "current_worker": "supervisor",
        "worker_results": {},
        "handoff_context": "",
        "original_query": "分析一下 10, 15, 20, 25, 30, 28, 35, 40 这组数据的趋势，然后用简短的文字总结分析结果",
    }

    result = graph.invoke(initial_state)
    print("\n最终回答:")
    print("-" * 40)
    if result.get("messages"):
        print(result["messages"][-1].content)
    print("-" * 40)


if __name__ == "__main__":
    print("多智能体系统测试")
    print("=" * 60)

    # 可以选择运行单个测试或全部测试
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=int, choices=[1, 2, 3], help="运行指定测试 (1=代码, 2=数据, 3=协作)")
    args = parser.parse_args()

    if args.test == 1:
        test_simple_code_task()
    elif args.test == 2:
        test_data_query()
    elif args.test == 3:
        test_multi_agent_collaboration()
    else:
        # 运行所有测试
        test_simple_code_task()
        test_data_query()
        test_multi_agent_collaboration()

    print("\n测试完成!")
