"""Agent 状态定义（LangGraph 共享字典）。"""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    thread_id: str
    question: str            # 用户原始问题
    clean_question: str      # 过 Guard 后的问题
    guard_flags: list[str]
    blocked: bool            # Guard 拦截
    plan: list[dict[str, Any]]   # 待执行/已执行的工具调用
    tool_trace: list[dict[str, Any]]
    loop: int                # 规划轮次
    answer: str
    citations: list[str]
    confidence: str
    pending_approvals: list[str]
    error: str | None
    history: list[dict[str, Any]]   # 多轮历史（由 API 层写入，供指代消解与答复生成）
