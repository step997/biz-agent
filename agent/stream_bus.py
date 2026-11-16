"""流式事件总线：让图内部（plan/tools/compose）在跑的过程中就把进展推给 SSE。

为什么不直接用 langgraph 的 stream_mode="messages"：
    langgraph 的同步 stream 只在"每个 super-step 结束"时 yield，节点内部（compose 的
    一次 LLM 生成）的 token 全被缓冲到节点收尾才吐一次 —— 实测首字 8.8s，比非流式还慢。
    所以这里用一张 thread_id → queue 的登记表，节点在自己的执行线程里直接 put，
    SSE 端只管取。没登记（走普通 /chat）时 emit 是空操作，不需要任何分支代码。
"""

from __future__ import annotations

import queue
from typing import Any

_BUS: dict[str, "queue.Queue[tuple[str, dict[str, Any]]]"] = {}
DONE = "__done__"
ERROR = "__error__"


def open_bus(thread_id: str) -> "queue.Queue[tuple[str, dict[str, Any]]]":
    q: "queue.Queue[tuple[str, dict[str, Any]]]" = queue.Queue()
    _BUS[thread_id] = q
    return q


def close_bus(thread_id: str) -> None:
    _BUS.pop(thread_id, None)


def emit(thread_id: str | None, kind: str, **data: Any) -> None:
    """从图内部发一个事件。没有监听者时静默返回。"""
    if not thread_id:
        return
    q = _BUS.get(thread_id)
    if q is None:
        return
    q.put((kind, data))
