# callbacks.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class TokenCounter:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    calls: int = 0

    def add(self, usage: Dict[str, Any]) -> None:
        pt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        ct = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct) or 0)
        self.prompt_tokens += pt
        self.completion_tokens += ct
        self.total_tokens += tt
        self.calls += 1

    def reset(self) -> None:
        """重置本次会话的统计（每次新对话调用）"""
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        self.calls = 0

    def summary(self) -> str:
        """格式化显示字符串"""
        if self.calls == 0:
            return "暂无统计数据"
        return (
            f"📥 Prompt: {self.prompt_tokens:,} tokens\n"
            f"📤 补全: {self.completion_tokens:,} tokens\n"
            f"📊 合计: {self.total_tokens:,} tokens\n"
            f"🔁 LLM 调用: {self.calls} 次"
        )


class UsageCallback(BaseCallbackHandler):
    """把每次 LLM 调用的 token usage 累加到 counter 上。"""

    def __init__(self, counter: TokenCounter):
        self.counter = counter

    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage: Optional[Dict[str, Any]] = None

        if getattr(response, "llm_output", None):
            lo = response.llm_output or {}
            usage = lo.get("token_usage") or lo.get("usage")

        if not usage and getattr(response, "generations", None):
            try:
                gi = response.generations[0][0].generation_info or {}
                usage = gi.get("token_usage") or gi.get("usage")
            except Exception:
                pass

        if usage:
            self.counter.add(usage)


# ─── 全局单例 ───────────────────────────────────────────
_global_counter = TokenCounter()
_global_callback = UsageCallback(_global_counter)


def get_token_counter() -> TokenCounter:
    """获取全局 Token 计数器（供 webui.py 读取显示）"""
    return _global_counter


def get_usage_callback() -> UsageCallback:
    """获取全局回调（传给 LLM 的 callbacks 参数）"""
    return _global_callback


def reset_token_counter() -> None:
    """重置全局计数器（新对话时调用）"""
    _global_counter.reset()
