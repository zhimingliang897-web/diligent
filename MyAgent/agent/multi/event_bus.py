"""event_bus.py — 多智能体执行过程事件总线

简单的线程安全队列，供 Worker / Supervisor 写入执行日志，
供 webui.py 读取并实时展示给用户。
"""

import queue
from datetime import datetime

_event_queue: queue.Queue = queue.Queue()


def emit(event: str) -> None:
    """写入一条执行事件（带时间戳）"""
    ts = datetime.now().strftime("%H:%M:%S")
    _event_queue.put(f"[{ts}] {event}")


def drain() -> str:
    """取出队列中所有待显示的事件，返回拼接字符串"""
    lines = []
    while True:
        try:
            lines.append(_event_queue.get_nowait())
        except queue.Empty:
            break
    return "\n".join(lines)


def clear() -> None:
    """清空队列（新对话开始时调用）"""
    while not _event_queue.empty():
        try:
            _event_queue.get_nowait()
        except queue.Empty:
            break
