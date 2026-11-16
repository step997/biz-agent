"""Agent 服务（对外统一入口，默认 8000）。前端、MCP、定时巡检都打这里或直接复用图。"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent import audit, llm, stream_bus
from agent.graph import get_graph
from tools.biz_tools import BIZ_API_KEY, BIZ_BASE_URL, build_registry

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "web"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("bizagent")


def _warm_rag() -> None:
    """后台预热检索：加载 bge 模型 + 建索引一次（约 9s），
    否则第一个问制度问题的用户要等这段时间。失败也不影响启动（自动降级词法）。"""
    try:
        from rag.index import get_index

        t = time.perf_counter()
        idx = get_index()
        idx.search("退款规则", 1)
        log.info("rag warmup ok backend=%s chunks=%d %.0fms",
                 idx.backend, len(idx.chunks), 1000 * (time.perf_counter() - t))
    except Exception:  # noqa: BLE001
        log.exception("rag warmup failed（将按需降级）")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_warm_rag, name="rag-warmup", daemon=True).start()
    yield


app = FastAPI(title="BizAgent 企业业务智能体", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    thread_id: str | None = None


def _history(thread_id: str, limit: int = 6) -> list[dict]:
    turns = [c for c in audit.recent_conversations(limit=80) if c["thread_id"] == thread_id]
    turns.reverse()
    return [{"question": t["question"], "answer": t["answer"]} for t in turns[-limit:]]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "mode": llm.mode(), "llm_available": llm.available(),
            "model": llm.MODEL if llm.available() else None}


@app.post("/chat")
def chat(body: ChatIn) -> dict:
    thread_id = body.thread_id or f"t-{uuid.uuid4().hex[:8]}"
    graph = get_graph()
    init = {"question": body.message, "thread_id": thread_id, "history": _history(thread_id)}
    t0 = time.perf_counter()
    try:
        out = graph.invoke(init, config={"configurable": {"thread_id": thread_id}})
    except Exception as exc:  # noqa: BLE001
        log.exception("chat failed thread=%s", thread_id)
        raise HTTPException(status_code=500, detail=f"agent 执行失败：{exc}") from exc
    cost_ms = (time.perf_counter() - t0) * 1000
    log.info("chat thread=%s mode=%s tools=%d conf=%s blocked=%s %.0fms q=%r",
             thread_id, llm.mode(), len(out.get("tool_trace") or []),
             out.get("confidence", ""), bool(out.get("blocked")), cost_ms, body.message[:40])

    audit.log_conversation(thread_id, body.message, out.get("answer", ""),
                           out.get("tool_trace") or [], out.get("confidence", ""),
                           bool(out.get("blocked")))
    return {
        "thread_id": thread_id,
        "answer": out.get("answer", ""),
        # tool_trace = 本轮真正执行的调用；recent_calls = 该会话历史审计（前端轨迹面板用前者）
        "tool_trace": [{k: t.get(k) for k in ("tool", "args", "ok", "error", "latency_ms")}
                       for t in (out.get("tool_trace") or [])],
        "recent_calls": audit.recent_calls(limit=12, thread_id=thread_id),
        "trace": [{k: t.get(k) for k in ("tool", "args", "ok", "error", "latency_ms")}
                  for t in (out.get("tool_trace") or [])],
        "guard_flags": out.get("guard_flags") or [],
        "citations": out.get("citations") or [],
        "confidence": out.get("confidence", ""),
        "blocked": bool(out.get("blocked")),
        "pending_approvals": out.get("pending_approvals") or [],
        "mode": llm.mode(),
    }


@app.get("/threads")
def threads(limit: int = 20) -> dict:
    """会话列表：按 thread_id 聚合最近对话（前端左侧栏用）。"""
    turns = audit.recent_conversations(limit=200)
    seen: dict[str, dict] = {}
    for t in turns:  # 已按时间倒序，第一次看到的就是该会话最新一条
        tid = t["thread_id"]
        row = seen.get(tid)
        if row is None:
            seen[tid] = {"thread_id": tid, "title": (t["question"] or "")[:40],
                         "last_ts": t["ts"], "turns": 1,
                         "confidence": t.get("confidence") or ""}
        else:
            row["turns"] += 1
    items = list(seen.values())[:limit]
    return {"count": len(items), "items": items}


@app.post("/chat/stream")
async def chat_stream(body: ChatIn):
    """SSE 流式对话：plan/tool 事件推执行轨迹，delta 事件推回答增量，done 事件收尾。

    图里用的是同步 SqliteSaver（不支持 async 接口），所以后台线程跑 graph.invoke，
    节点事件通过 stream_bus 塞进请求级 queue，这里只负责转发 —— 这样 /chat 与
    /chat/stream 走的是同一条图、同一份逻辑，没有第二条变慢的路径。
    """
    thread_id = body.thread_id or f"t-{uuid.uuid4().hex[:8]}"
    graph = get_graph()
    init = {"question": body.message, "thread_id": thread_id, "history": _history(thread_id)}
    config = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()

    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload, ensure_ascii=False, default=str)}\n\n"

    async def gen():
        q = stream_bus.open_bus(thread_id)
        yield sse({"event": "start", "thread_id": thread_id, "mode": llm.mode()})

        def run() -> None:
            try:
                state = graph.invoke(init, config=config)
                q.put(("__state__", state))
            except Exception as exc:  # noqa: BLE001
                q.put(("__error__", exc))
            finally:
                q.put((stream_bus.DONE, {}))

        threading.Thread(target=run, name="graph-run", daemon=True).start()
        buf: list[str] = []
        state: dict = {}
        try:
            while True:
                kind, data = await asyncio.to_thread(q.get)
                if kind == stream_bus.DONE:
                    break
                if kind == "__error__":
                    log.error("stream failed thread=%s: %s", thread_id, data)
                    yield sse({"event": "error", "message": f"执行失败：{data}"})
                    return
                if kind == "__state__":
                    state = data or {}
                    continue
                if kind == "delta":
                    text = data.get("text") or ""
                    if text:
                        buf.append(text)
                        yield sse({"event": "delta", "text": text})
                    continue
                yield sse({"event": "node", "node": kind, **data})
        finally:
            stream_bus.close_bus(thread_id)

        answer = state.get("answer", "") or "".join(buf).strip()
        cost_ms = (time.perf_counter() - t0) * 1000
        log.info("stream thread=%s mode=%s tools=%d conf=%s blocked=%s %.0fms q=%r",
                 thread_id, llm.mode(), len(state.get("tool_trace") or []),
                 state.get("confidence", ""), bool(state.get("blocked")), cost_ms,
                 body.message[:40])
        audit.log_conversation(thread_id, body.message, answer,
                               state.get("tool_trace") or [], state.get("confidence", ""),
                               bool(state.get("blocked")))
        yield sse({"event": "done",
                   "answer": answer,
                   "thread_id": thread_id,
                   "tool_trace": [{k: t.get(k) for k in ("tool", "args", "ok", "error", "latency_ms")}
                                  for t in (state.get("tool_trace") or [])],
                   "citations": state.get("citations") or [],
                   "confidence": state.get("confidence", ""),
                   "blocked": bool(state.get("blocked")),
                   "guard_flags": state.get("guard_flags") or [],
                   "pending_approvals": state.get("pending_approvals") or [],
                   "mode": llm.mode(),
                   "cost_ms": round(cost_ms)})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/threads/{thread_id}")
def thread(thread_id: str) -> dict:
    return {"thread_id": thread_id, "history": _history(thread_id, limit=20)}


@app.get("/audit/recent")
def audit_recent(limit: int = 50, thread_id: str | None = None) -> dict:
    return {"calls": audit.recent_calls(limit=limit, thread_id=thread_id),
            "conversations": audit.recent_conversations(limit=20)}


@app.get("/audit/stats")
def audit_stats() -> dict:
    return audit.stats()


@app.get("/audit/conversations")
def audit_conversations(limit: int = 20) -> dict:
    items = audit.recent_conversations(limit=limit)
    return {"count": len(items), "items": items}


@app.get("/tools")
def tools() -> dict:
    reg = build_registry()
    return {"count": len(reg.tools), "describe": reg.describe(), "schemas": reg.schemas()}


@app.get("/dashboard")
def dashboard() -> dict:
    """给前端看板：业务概览 + 智能体自身指标（拿不到业务系统时降级为 None）。"""
    overview = None
    try:
        r = httpx.get(f"{BIZ_BASE_URL}/stats/overview", headers={"X-Api-Key": BIZ_API_KEY}, timeout=5)
        r.raise_for_status()
        overview = r.json()
    except Exception as exc:  # noqa: BLE001
        overview = {"error": f"业务系统不可用：{exc}"}
    return {"biz": overview, "agent": audit.stats(), "mode": llm.mode()}


@app.get("/metrics")
def metrics() -> PlainTextResponse:
    """Prometheus 文本格式的极简指标：够 Grafana 拉一条曲线看工具失败率和耗时。"""
    s = audit.stats()
    calls = int(s.get("tool_calls") or 0)
    fails = int(s.get("tool_failures") or 0)
    avg = float(s.get("avg_latency_ms") or 0)
    lines = [
        "# HELP bizagent_tool_calls_total 工具调用次数",
        "# TYPE bizagent_tool_calls_total counter",
    ]
    for tool, n in sorted((s.get("by_tool") or {}).items()):
        lines.append(f'bizagent_tool_calls_total{{tool="{tool}"}} {n}')
    lines += [
        "# HELP bizagent_tool_failures_total 工具失败次数",
        "# TYPE bizagent_tool_failures_total counter",
        f"bizagent_tool_failures_total {fails}",
        "# HELP bizagent_tool_latency_ms 工具耗时（sum/count 由平均耗时 × 次数还原）",
        "# TYPE bizagent_tool_latency_ms summary",
        f"bizagent_tool_latency_ms_sum {avg * calls:.1f}",
        f"bizagent_tool_latency_ms_count {calls}",
        "# HELP bizagent_conversations_total 对话轮次",
        "# TYPE bizagent_conversations_total counter",
        f"bizagent_conversations_total {int(s.get('conversations') or 0)}",
        "# HELP bizagent_blocked_total 被 Guard 拦截的轮次",
        "# TYPE bizagent_blocked_total counter",
        f"bizagent_blocked_total {int(s.get('blocked_conversations') or 0)}",
        "# HELP bizagent_up 服务存活",
        "# TYPE bizagent_up gauge",
        "bizagent_up 1",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain; version=0.0.4")


@app.get("/approvals")
def approvals(status: str | None = None) -> dict:
    try:
        r = httpx.get(f"{BIZ_BASE_URL}/approvals", params={"status": status} if status else None,
                      headers={"X-Api-Key": BIZ_API_KEY}, timeout=8)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"业务系统不可用：{exc}") from exc


class Decision(BaseModel):
    decision: str = Field(pattern="^(approved|rejected)$")
    decided_by: str = "human"


@app.post("/approvals/{approval_no}/decide")
def decide(approval_no: str, body: Decision) -> dict:
    try:
        r = httpx.patch(f"{BIZ_BASE_URL}/approvals/{approval_no}", json=body.model_dump(),
                        headers={"X-Api-Key": BIZ_API_KEY}, timeout=8)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=exc.response.status_code, detail=exc.response.text) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"业务系统不可用：{exc}") from exc


# ---- 前端 ----
if WEB.exists():
    app.mount("/static", StaticFiles(directory=WEB), name="static")


@app.get("/")
def index():
    f = WEB / "index.html"
    if f.exists():
        return FileResponse(f)
    return {"service": "BizAgent", "docs": "/docs"}
