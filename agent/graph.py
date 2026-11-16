"""LangGraph 编排：guard → plan → tools → (plan 循环) → compose。

- 多轮记忆：SqliteSaver 按 thread_id 持久化 state（含历史问答与上一单号）
- 写操作：create_approval 只落"待审批"，不做资金动作；人工在 /approvals 里决定
- 审计：工具层 on_call → audit.log_tool，图外也能查到
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from typing import Any

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agent import audit, guards, llm, planner, stream_bus
from agent.state import AgentState
from tools.biz_tools import build_registry
from tools.base import ToolRegistry

MAX_LOOP = 3
ORDER_NO = re.compile(r"SO\d{5,}")


def _registry(thread_id: str) -> ToolRegistry:
    reg = build_registry(on_call=lambda rec: audit.log_tool(thread_id, rec))
    return reg


# ───────────────────────────── 节点 ─────────────────────────────


def node_guard(state: AgentState) -> dict:
    history = state.get("history") or []
    res = guards.check_input(state.get("question", ""))
    # 先把 guard 结果推给前端（纯代码判断，毫秒级），用户点发送后立刻能看到轨迹在动
    stream_bus.emit(state.get("thread_id"), "guard", ok=res.allow, flags=res.flags)
    if not res.allow:
        return {"blocked": True, "guard_flags": res.flags, "answer": res.reason,
                "clean_question": "", "confidence": "blocked", "tool_trace": [], "plan": [],
                "pending_approvals": [], "citations": []}
    return {"blocked": False, "guard_flags": res.flags, "clean_question": res.text, "loop": 0,
            "tool_trace": [], "plan": [], "pending_approvals": [], "answer": "",
            "citations": [], "confidence": ""}


def node_plan(state: AgentState) -> dict:
    question = state.get("clean_question") or state.get("question", "")
    # 多轮指代消解："那这个订单呢" → 补上上一轮的单号
    if any(k in question for k in ("这个单", "该单", "那单", "这个订单", "它")) and not ORDER_NO.search(question):
        for turn in reversed(state.get("history") or []):
            if m := ORDER_NO.search(turn.get("question", "") + turn.get("answer", "")):
                question = f"{question}（订单号 {m.group(0)}）"
                break
    reg = _registry(state.get("thread_id", "default"))
    trace = state.get("tool_trace") or []
    plan, direct = planner.make_plan(question, state.get("loop", 0), trace, reg,
                                     history=state.get("history") or [])
    # 代码级护栏：只问规则时不动物写操作 —— 多轮记忆会把上一轮的"提交退款审批"带过来
    plan, dropped = planner.guard_plan(plan, question, reg)
    flags = list(state.get("guard_flags") or [])
    if dropped:
        flags.append("write_blocked:" + ",".join(dropped))
    # 同一轮里同一种工具只跑一次：模型重复规划的直接丢弃，避免"为了确认再查一遍"
    used = {t.get("tool") for t in trace}
    plan = [c for c in plan if (c or {}).get("tool") not in used]
    out: dict[str, Any] = {"plan": plan, "clean_question": question, "guard_flags": flags}
    if dropped and not plan and not trace and not direct:
        # 写操作被拦下后没别的事可做：这是制度问题，直接兜到制度检索，别让用户空手而归
        out["plan"] = [{"tool": "query_policy", "args": {"question": question, "top_k": 3}}]
    if direct and not plan:
        out["answer"] = direct
    if plan or flags:
        stream_bus.emit(state.get("thread_id"), "plan",
                        tools=[c.get("tool") for c in (out.get("plan") or plan)],
                        flags=out.get("guard_flags") or [])
    return out


def node_tools(state: AgentState) -> dict:
    reg = _registry(state.get("thread_id", "default"))
    trace = list(state.get("tool_trace") or [])
    approvals = list(state.get("pending_approvals") or [])
    seen: set[tuple] = {(t.get("tool"), json.dumps(t.get("args") or {}, sort_keys=True, ensure_ascii=False))
                        for t in trace}
    for call in state.get("plan") or []:
        call = dict(call or {})
        # 同一轮里重复的相同调用（同名 + 同参）只执行一次，避免模型重复规划浪费
        sig = (call.get("tool"), json.dumps(call.get("args") or {}, sort_keys=True, ensure_ascii=False))
        if sig in seen:
            continue
        seen.add(sig)
        res = reg.call(call["tool"], call.get("args") or {})
        trace.append({"tool": res.tool, "args": call.get("args") or {},
                      "ok": res.ok, "data": res.data, "error": res.error,
                      "latency_ms": res.latency_ms})
        stream_bus.emit(state.get("thread_id"), "tool", tool=res.tool,
                        args=call.get("args") or {}, ok=res.ok, error=res.error,
                        latency_ms=res.latency_ms)
        if res.ok and res.tool == "create_approval" and isinstance(res.data, dict):
            if no := res.data.get("approval_no"):
                approvals.append(no)
    return {"tool_trace": trace, "loop": (state.get("loop") or 0) + 1,
            "pending_approvals": approvals, "plan": []}


def node_compose(state: AgentState) -> dict:
    trace = state.get("tool_trace") or []
    question = state.get("clean_question") or state.get("question", "")
    if state.get("blocked"):
        stream_bus.emit(state.get("thread_id"), "delta", text=state.get("answer", ""))
        return {"answer": guards.mask_output(state.get("answer", "")), "confidence": "blocked"}
    if state.get("answer") and not trace:
        stream_bus.emit(state.get("thread_id"), "delta", text=state["answer"])
        return {"answer": guards.mask_output(state["answer"]), "confidence": "llm-direct"}

    citations = [h["citation"] for t in trace if t["tool"] == "query_policy"
                 and t.get("ok") and isinstance(t.get("data"), dict)
                 for h in t["data"].get("hits", [])]
    confidences = [t["data"].get("confidence") for t in trace if t["tool"] == "query_policy"
                   and isinstance(t.get("data"), dict)]

    if llm.mode() == "llm" and llm.available() and trace:
        try:
            answer = _compose_llm(question, trace, state.get("history") or [],
                                  thread_id=state.get("thread_id"))
        except Exception:  # noqa: BLE001
            answer = _compose_rules(question, trace)
    else:
        answer = _compose_rules(question, trace)
        stream_bus.emit(state.get("thread_id"), "delta", text=answer)
    return {"answer": guards.mask_output(answer), "citations": list(dict.fromkeys(citations)),
            "confidence": confidences[0] if confidences else "ok"}


# ───────────────────────────── 回答生成 ─────────────────────────────

COMPOSE_PROMPT = """你是 BizAgent 的答复生成器。根据工具结果用中文简洁回答（150 字内），要求：
1. 只用工具结果里出现的数据，数字/单号/时间照写，不要推测。
2. 制度类问题必须带 citations（形如 [return_policy#02]）。
3. 若工具结果为为空或 ok=false，直接说没查到并给下一步建议。
4. 涉及退款/补发/调价，说明"已提交审批，需人工确认"。
"""


def _compose_llm(question: str, trace: list[dict], history: list[dict],
                 thread_id: str | None = None) -> str:
    model = llm.get_llm()
    payload = json.dumps(trace, ensure_ascii=False, default=str)[:6000]
    hist = "\n".join(f"Q: {h.get('question')}\nA: {h.get('answer')}" for h in history[-3:])
    msg = f"历史对话：\n{hist or '（无）'}\n\n用户问题：{question}\n\n工具结果：\n{payload}"
    msgs = [{"role": "system", "content": COMPOSE_PROMPT}, {"role": "user", "content": msg}]
    # 用 stream 收集：token 边生成边通过 stream_bus 推给 SSE（无监听者时 emit 是空操作）
    try:
        buf: list[str] = []
        for chunk in model.stream(msgs):
            if text := getattr(chunk, "content", ""):
                buf.append(text)
                stream_bus.emit(thread_id, "delta", text=text)
        out = "".join(buf).strip()
        if out:
            return out
    except Exception:  # noqa: BLE001 —— 流式不支持就退回一次性调用
        pass
    return (model.invoke(msgs).content or "").strip()


def _fmt_money(x: Any) -> str:
    try:
        return f"¥{float(x):.2f}"
    except Exception:  # noqa: BLE001
        return str(x)


def _compose_rules(question: str, trace: list[dict]) -> str:
    if not trace:
        return "没查到可用的业务数据，请补充订单号或客户名后再试。"
    parts: list[str] = []
    for t in trace:
        tool, data = t["tool"], t.get("data")
        t_args = t.get("args") or {}
        if not t.get("ok"):
            parts.append(f"调用 {tool} 失败：{t.get('error')}")
            continue
        if tool == "query_orders":
            orders = (data or {}).get("orders") or []
            if not orders:
                parts.append("没有符合条件的订单（可换客户名或放宽时间范围）。")
                continue
            lines = [f"- {o['order_no']}　{o['customer']}　{o['status']}　{_fmt_money(o['amount'])}　{o['created_at'][:10]}"
                     for o in orders[:6]]
            parts.append(f"命中 {data.get('total', len(orders))} 单，最近 {len(lines)} 条：\n" + "\n".join(lines))
        elif tool == "get_order_detail":
            o = (data or {}).get("order") or {}
            items = (data or {}).get("items") or []
            lg = (data or {}).get("logistics") or {}
            item_txt = "、".join(f"{i['name']}×{i['qty']}" for i in items[:5]) or "无明细"
            parts.append(
                f"订单 {o.get('order_no')}（{o.get('customer')}）状态 {o.get('status')}，"
                f"金额 {_fmt_money(o.get('amount'))}，下单 {o.get('created_at')}；"
                f"商品：{item_txt}；物流：{lg.get('latest') or lg.get('note') or '暂无'}。")
        elif tool == "get_logistics":
            if not data.get("shipped"):
                parts.append(f"订单 {data.get('order_no')} 还没发货（当前状态 {data.get('status')}）。")
            else:
                parts.append(f"订单 {data['order_no']} 由 {data.get('carrier')} 承运，运单 {data.get('logistics_no')}，"
                             f"已发出 {data.get('elapsed_days')} 天，最新：{data.get('latest')}。")
        elif tool == "check_inventory":
            wh = "、".join(f"{w['warehouse']} {w['qty']} 件" for w in data.get("warehouses") or [])
            flag = "已低于安全库存" if data.get("below_safety") else "库存正常"
            parts.append(f"{data.get('name')}（{data.get('sku')}）合计 {data.get('total_qty')} 件，{flag}；{wh}。")
        elif tool == "low_stock_list":
            items = data.get("items") or []
            if not items:
                parts.append("当前没有低于安全库存的商品。")
            else:
                lines = [f"- {i['name']}（{i['sku']}）{i['warehouse']} 剩 {i['qty']}，缺 {i['shortage']}"
                         for i in items[:6]]
                parts.append(f"{len(items)} 个 SKU 低于安全库存：\n" + "\n".join(lines))
        elif tool == "query_tickets":
            items = data.get("items") or []
            if not items:
                parts.append("没有符合条件的工单。")
            else:
                overdue = [i for i in items if i.get("overdue")]
                lines = [f"- {i['ticket_no']}　{i['type']}　{i['status']}　截止 {i['sla_deadline']}"
                         + ("（已超时）" if i.get("overdue") else "")
                         for i in items[:6]]
                scope = "超时工单" if t_args.get("overdue_only") else "工单"
                parts.append(f"命中 {len(items)} 张{scope}，其中超时 {len(overdue)} 张：\n" + "\n".join(lines))
        elif tool == "create_ticket":
            parts.append(f"已创建工单 {data.get('ticket_no')}（{data.get('type')}），SLA {data.get('sla_hours')} 小时，"
                         f"截止 {data.get('sla_deadline')}。")
        elif tool == "create_approval":
            parts.append(f"已提交{data.get('kind')}审批单 {data.get('approval_no')}，需人工确认后执行（智能体不直接动钱）。")
        elif tool == "query_policy":
            hits = (data or {}).get("hits") or []
            conf = (data or {}).get("confidence")
            if not (data or {}).get("covered"):
                parts.append("制度库里没有找到可信依据（检索置信度低），这一条建议转人工确认，我不做推测。")
            else:
                head = "依据充分" if conf == "high" else "依据偏弱，建议人工复核"
                lines = [f"- {h['citation']} {h['content'][:110]}" for h in hits[:3]]
                parts.append(f"（{head}）\n" + "\n".join(lines))
    return "\n\n".join(parts)


# ───────────────────────────── 图构建 ─────────────────────────────


def _route_after_guard(state: AgentState) -> str:
    """被 Guard 拦下的请求直接去 compose（只做脱敏输出），不再进入规划，避免多余的模型调用。"""
    return "compose" if state.get("blocked") else "plan"


def _route_after_plan(state: AgentState) -> str:
    if state.get("blocked"):
        return "compose"
    if state.get("plan"):
        return "tools"
    return "compose"


def _route_after_tools(state: AgentState) -> str:
    if (state.get("loop") or 0) >= MAX_LOOP:
        return "compose"
    return "plan"


def build_graph(checkpointer: SqliteSaver | None = None):
    g = StateGraph(AgentState)
    g.add_node("guard", node_guard)
    g.add_node("plan", node_plan)
    g.add_node("tools", node_tools)
    g.add_node("compose", node_compose)
    g.set_entry_point("guard")
    g.add_conditional_edges("guard", _route_after_guard, {"plan": "plan", "compose": "compose"})
    g.add_conditional_edges("plan", _route_after_plan, {"tools": "tools", "compose": "compose"})
    g.add_conditional_edges("tools", _route_after_tools, {"plan": "plan", "compose": "compose"})
    g.add_edge("compose", END)
    return g.compile(checkpointer=checkpointer)


@lru_cache(maxsize=1)
def get_graph():
    import sqlite3
    audit.DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(audit.DB.parent / "checkpoints.db", check_same_thread=False)
    return build_graph(SqliteSaver(conn))
