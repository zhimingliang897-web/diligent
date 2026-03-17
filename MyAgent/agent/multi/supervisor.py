"""Supervisor Agent — 多智能体系统的调度中心

职责:
1. 意图识别: 分析用户请求的类型
2. 任务分解: 将复杂任务拆解为子任务
3. 执行编排: 决定 Worker 的执行顺序
4. Re-plan: 任务失败时重新规划
5. 结果汇总: 整合各 Worker 的输出
"""

import json
import logging
import re
from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from agent.llm import get_llm
from agent.multi.state import MultiAgentState, TaskItem, WorkerType
from agent.multi import event_bus

logger = logging.getLogger(__name__)

MAX_REPLAN_COUNT = 2
_VALID_WORKERS = {"code", "data", "writer"}


def _extract_plan_json(text: str) -> dict:
    """从 LLM 回复中健壮地提取 JSON 任务计划。"""
    json_str = None

    if "```json" in text:
        json_str = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        json_str = text.split("```", 1)[1].split("```", 1)[0].strip()

    if not json_str:
        match = re.search(r'\{[\s\S]*"tasks"[\s\S]*\}', text)
        if match:
            json_str = match.group(0)
        else:
            json_str = text.strip()

    plan = json.loads(json_str)

    tasks = plan.get("tasks", [])
    for t in tasks:
        worker = t.get("assigned_to", "data")
        if worker not in _VALID_WORKERS:
            logger.warning("无效的 worker '%s'，回退到 'data'", worker)
            t["assigned_to"] = "data"

    return plan


# Supervisor 的系统提示词
SUPERVISOR_SYSTEM_PROMPT = """你是一个客观强大、负责多智能体系统的 Supervisor（调度员）。你的职责是分析用户请求，将其分解为子任务，并分配给合适的专业 Agent 执行。
【注意】不要主动强调或辩论你的大模型基座身份与知识截止时期，专注于调度任务。对于涉及今天、实时等时效性信息的请求，优先交由 data 专家处理，因为其具备时间工具。

## 可用的 Worker Agents:

1. **code** (代码专家)
   - 擅长: Python 代码生成、算法实现、代码调试、数据可视化代码
   - 适用场景: 用户需要代码、程序、脚本时

2. **data** (数据分析专家)
   - 擅长: 数据分析、统计计算、知识库检索(RAG)、趋势分析、网络搜索、实时天气查询
   - 适用场景: 用户需要数据分析、事实查询、天气查询时

3. **writer** (写作专家)
   - 擅长: 文章撰写、内容润色、报告生成、摘要总结
   - 适用场景: 用户需要写文章、报告、文档时

## 任务分解规则:

1. **简单任务** (只需要一个 Agent):
   - "写一段排序代码" → [code]
   - "帮我查一下今天的天气" → [data] (注意任务描述写"使用气象工具获取天气"，绝不能写"网络搜索")
   - "帮我写一篇关于 AI 的文章" → [writer]

2. **复杂任务** (需要多个 Agent 协作):
   - "分析数据并写报告" → [data → writer]
   - "根据文档写代码" → [data → code]
   - "分析数据、写报告、生成可视化代码" → [data → writer → code]

## 你的输出格式:

你必须输出一个 JSON 对象，包含以下字段:
```json
{
  "analysis": "对用户请求的简短分析",
  "tasks": [
    {"id": 1, "description": "子任务描述", "assigned_to": "worker名称"},
    {"id": 2, "description": "子任务描述", "assigned_to": "worker名称"}
  ]
}
```

## 重要规则:
- assigned_to 只能是: "code", "data", "writer"
- tasks 数组按执行顺序排列
- 每个子任务的 description 要具体、可执行。**尤其是天气查询任务，请直接写“调用气象工具查询天气”，严禁出现“网络搜索”、“互联网搜”等任何暗示使用浏览器的词汇！**
- 如果任务很简单，只需要一个子任务即可
- 不要过度拆分，保持子任务数量合理（通常 1-3 个）"""



REPLAN_SYSTEM_PROMPT = """你是一个多智能体系统的 Supervisor（调度员）。之前的任务计划执行过程中出现了问题，你需要重新规划。

## 可用的 Worker Agents:

1. **code** (代码专家) - Python 代码生成、算法实现、代码调试
2. **data** (数据分析专家) - 数据分析、知识库检索(RAG)、网络搜索
3. **writer** (写作专家) - 文章撰写、内容润色、报告生成

## 你的任务:

根据失败原因，重新规划任务。你可以:
1. 调整任务顺序
2. 拆分任务为更小的步骤
3. 更换执行任务的 Agent
4. 添加前置任务（如先获取需要的信息）

## 输出格式:

```json
{
  "analysis": "问题分析和新计划说明",
  "tasks": [
    {"id": 1, "description": "子任务描述", "assigned_to": "worker名称"},
    ...
  ]
}
```

## 规则:
- 只输出剩余需要执行的任务
- 不要重复已经成功完成的任务
- 新计划要能解决之前的问题"""


AGGREGATOR_SYSTEM_PROMPT = """你是一个结果汇总专家。你的任务是将多个专业 Agent 的工作成果整合成一个连贯、完整的最终回答。

## 汇总规则:
1. 保留每个 Agent 的核心贡献
2. 消除重复内容
3. 确保逻辑连贯
4. 使用清晰的格式（Markdown）
5. 如果有代码，确保代码块格式正确

## 输出格式:
- 直接输出最终整合后的内容
- 不需要提及 Agent 的名称
- 内容要自然流畅，像是一个人写的"""


def create_supervisor_node(llm=None):
    """创建 Supervisor 节点函数"""
    if llm is None:
        llm = get_llm()

    def supervisor_node(state: MultiAgentState) -> dict:
        """Supervisor: 分析用户请求，分解任务，决定执行计划，处理 Re-plan"""
        original_query = state.get("original_query", "")
        messages = state.get("messages", [])
        needs_replan = state.get("needs_replan", False)
        replan_reason = state.get("replan_reason", "")
        replan_count = state.get("replan_count", 0)

        # 如果已经有任务计划
        task_plan = state.get("task_plan", [])

        # ================== Re-plan 逻辑 ==================
        if needs_replan and task_plan:
            # 检查是否超过最大 replan 次数
            if replan_count >= MAX_REPLAN_COUNT:
                print(f"  [Supervisor] Re-plan 次数已达上限 ({MAX_REPLAN_COUNT})，强制继续", flush=True)
                # 重置 replan 标记，尝试继续执行
                return {
                    "needs_replan": False,
                    "replan_reason": "",
                }

            print(f"  [Supervisor] 检测到任务失败，启动 Re-plan (第 {replan_count + 1} 次)", flush=True)
            print(f"  [Supervisor] 失败原因: {replan_reason}", flush=True)
            event_bus.emit(f"🔄 [Supervisor] 启动 Re-plan （第 {replan_count + 1} 次），失败原因: {replan_reason[:60]}")

            # 收集已完成任务的结果作为上下文
            completed_context = ""
            for task in task_plan:
                if task.get("status") == "completed" and task.get("result"):
                    completed_context += f"\n- [{task['assigned_to']}] {task['description']}: 已完成"

            # 收集失败任务信息
            failed_tasks = ""
            for task in task_plan:
                if task.get("status") == "failed":
                    failed_tasks += f"\n- [{task['assigned_to']}] {task['description']}: 失败 - {task.get('error', replan_reason)}"

            # 调用 LLM 重新规划
            replan_messages = [
                SystemMessage(content=REPLAN_SYSTEM_PROMPT),
                HumanMessage(content=f"""用户原始请求: {original_query}

已完成的任务:
{completed_context if completed_context else "（无）"}

失败的任务:
{failed_tasks}

失败原因: {replan_reason}

请重新规划剩余任务:"""),
            ]

            response = llm.invoke(replan_messages)
            response_text = response.content

            try:
                plan_data = _extract_plan_json(response_text)
                analysis = plan_data.get("analysis", "")
                tasks_data = plan_data.get("tasks", [])

                print(f"  [Supervisor] Re-plan 分析: {analysis}", flush=True)
                event_bus.emit(f"🧠 [Supervisor] Re-plan 分析: {analysis[:60]}")

                # 保留已完成的任务，添加新任务
                new_task_plan = [t for t in task_plan if t.get("status") == "completed"]
                start_id = len(new_task_plan) + 1

                for t in tasks_data:
                    task: TaskItem = {
                        "id": start_id,
                        "description": t.get("description", ""),
                        "assigned_to": t.get("assigned_to", "data"),
                        "status": "pending",
                        "result": "",
                        "error": "",
                    }
                    new_task_plan.append(task)
                    print(f"  [Supervisor] 新任务 {task['id']}: [{task['assigned_to']}] {task['description']}", flush=True)
                    start_id += 1

                # 设置第一个 pending 任务为进行中
                for task in new_task_plan:
                    if task.get("status") == "pending":
                        task["status"] = "in_progress"
                        first_worker = task["assigned_to"]
                        return {
                            "task_plan": new_task_plan,
                            "current_worker": first_worker,
                            "needs_replan": False,
                            "replan_reason": "",
                            "replan_count": replan_count + 1,
                            "messages": [AIMessage(content=f"[Supervisor] Re-plan 完成: {analysis}")],
                        }

                # 如果没有新任务，直接结束
                return {
                    "current_worker": "FINISH",
                    "needs_replan": False,
                    "replan_reason": "",
                    "replan_count": replan_count + 1,
                }

            except (json.JSONDecodeError, KeyError, IndexError) as e:
                print(f"  [Supervisor] Re-plan 解析失败: {e}", flush=True)
                # 解析失败，跳过失败任务继续执行
                return {
                    "needs_replan": False,
                    "replan_reason": "",
                    "replan_count": replan_count + 1,
                }

        # ================== 正常流程 ==================
        if task_plan:
            # 找到下一个待执行的任务
            for task in task_plan:
                if task.get("status") == "pending":
                    task["status"] = "in_progress"
                    next_worker = task.get("assigned_to")
                    print(f"  [Supervisor] 分配任务给 {next_worker}: {task.get('description')}", flush=True)
                    event_bus.emit(f"📤 [Supervisor] 分配任务给 [{next_worker.upper()}]: {task.get('description', '')[:50]}")
                    return {
                        "task_plan": task_plan,
                        "current_worker": next_worker,
                        "needs_replan": False,
                    }

            # 所有任务都完成或失败了，转到汇总
            completed_count = sum(1 for t in task_plan if t.get("status") == "completed")
            failed_count = sum(1 for t in task_plan if t.get("status") == "failed")
            print(f"  [Supervisor] 所有子任务处理完毕 (完成: {completed_count}, 失败: {failed_count})，准备汇总", flush=True)
            event_bus.emit(f"🏁 [Supervisor] 全部任务处理完毕，准备汇总输出（完成: {completed_count}，失败: {failed_count}）")
            return {"current_worker": "FINISH", "needs_replan": False}

        # ================== 首次调用：创建任务计划 ==================
        print("  [Supervisor] 分析用户请求...", flush=True)
        event_bus.emit("🧠 [Supervisor] 开始分析用户请求，规划任务...")

        planning_messages = [
            SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT),
            HumanMessage(content=f"用户请求: {original_query}"),
        ]

        response = llm.invoke(planning_messages)
        response_text = response.content

        # 解析 JSON 输出
        try:
            plan_data = _extract_plan_json(response_text)
            analysis = plan_data.get("analysis", "")
            tasks_data = plan_data.get("tasks", [])

            logger.info("[Supervisor] 任务分析: %s", analysis)

            # 构建任务计划
            task_plan = []
            for t in tasks_data:
                task: TaskItem = {
                    "id": t.get("id", len(task_plan) + 1),
                    "description": t.get("description", ""),
                    "assigned_to": t.get("assigned_to", "data"),
                    "status": "pending",
                    "result": "",
                    "error": "",
                }
                task_plan.append(task)
                print(f"  [Supervisor] 子任务 {task['id']}: [{task['assigned_to']}] {task['description']}", flush=True)

            # 设置第一个任务为进行中
            if task_plan:
                task_plan[0]["status"] = "in_progress"
                first_worker = task_plan[0]["assigned_to"]
            else:
                first_worker = "FINISH"

            return {
                "task_plan": task_plan,
                "current_worker": first_worker,
                "needs_replan": False,
                "replan_count": 0,
                "messages": [AIMessage(content=f"[Supervisor] 任务规划完成: {analysis}")],
            }

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"  [Supervisor] 任务解析失败: {e}，使用默认 data agent", flush=True)
            # 解析失败，默认使用 data agent
            default_task: TaskItem = {
                "id": 1,
                "description": original_query,
                "assigned_to": "data",
                "status": "in_progress",
                "result": "",
                "error": "",
            }
            return {
                "task_plan": [default_task],
                "current_worker": "data",
                "needs_replan": False,
                "replan_count": 0,
            }

    return supervisor_node


def create_aggregator_node(llm=None):
    """创建结果汇总节点函数"""
    if llm is None:
        llm = get_llm()

    def aggregator_node(state: MultiAgentState) -> dict:
        """汇总所有 Worker 的结果，生成最终回答"""
        original_query = state.get("original_query", "")
        worker_results = state.get("worker_results", {})
        task_plan = state.get("task_plan", [])

        print("  [Aggregator] 汇总各 Agent 结果...", flush=True)

        # 构建汇总上下文
        results_text = ""
        for task in task_plan:
            worker = task.get("assigned_to", "")
            status = task.get("status", "")
            result = worker_results.get(worker, task.get("result", ""))

            if status == "completed" and result:
                results_text += f"\n\n### {worker.upper()} Agent 的输出:\n{result}"
            elif status == "failed":
                error = task.get("error", "未知错误")
                results_text += f"\n\n### {worker.upper()} Agent (失败):\n错误: {error}"

        if not results_text.strip():
            # 如果没有结果，检查 worker_results 是否有内容
            if worker_results:
                # 尝试直接使用 worker_results 中的内容
                for worker_name, result in worker_results.items():
                    if result and result.strip():
                        results_text = f"\n\n### {worker_name.upper()} Agent 的输出:\n{result}"
                        break

            if not results_text.strip():
                return {
                    "messages": [AIMessage(content="抱歉，处理过程中没有产生有效结果。")],
                }

        # 如果只有一个成功的 Worker 的结果，直接使用
        successful_results = {k: v for k, v in worker_results.items() if v and v.strip()}
        if len(successful_results) == 1:
            final_result = list(successful_results.values())[0]
            # 去掉可能的 [worker]: 前缀
            if final_result.startswith("[") and "]: " in final_result:
                final_result = final_result.split("]: ", 1)[1]
            return {
                "messages": [AIMessage(content=final_result)],
            }

        # 多个 Worker 的结果需要汇总
        aggregation_messages = [
            SystemMessage(content=AGGREGATOR_SYSTEM_PROMPT),
            HumanMessage(content=f"""用户原始请求: {original_query}

各 Agent 的工作成果:
{results_text}

请将以上内容整合成一个完整、连贯的最终回答:"""),
        ]

        response = llm.invoke(aggregation_messages)
        final_result = response.content

        print("  [Aggregator] 汇总完成", flush=True)

        return {
            "messages": [AIMessage(content=final_result)],
        }

    return aggregator_node
