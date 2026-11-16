"""轻量基准：跑一遍关键路径，给出 p50/p95，用来看优化有没有效果。

前置：biz 服务已在 8001 起（AGENT_MODE=offline 只需要业务系统）。
用法：.venv\\Scripts\\python.exe scripts\\bench.py
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("AGENT_MODE", "offline")
os.environ.setdefault("BIZ_BASE_URL", "http://127.0.0.1:8001")
os.environ.setdefault("BIZ_API_KEY", "demo-key")

import httpx  # noqa: E402

from rag import index as rag_index  # noqa: E402
from tools.biz_tools import BIZ_BASE_URL, BIZ_API_KEY, build_registry  # noqa: E402


def stats(name: str, samples: list[float]) -> None:
    s = sorted(samples)
    p50 = statistics.median(s)
    p95 = s[int(len(s) * 0.95) - 1]
    print(f"{name:<28} n={len(s):<4} p50={p50:7.2f}ms  p95={p95:7.2f}ms  mean={statistics.mean(s):7.2f}ms")


def bench(fn, n: int = 20, warmup: int = 3) -> list[float]:
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t) * 1000)
    return out


def main() -> None:
    hdr = {"X-Api-Key": BIZ_API_KEY}

    # ① 业务系统 REST：单查询
    def biz_orders():
        httpx.get(f"{BIZ_BASE_URL}/orders", params={"customer_name": "王伟", "since_days": 7},
                  headers=hdr, timeout=10).raise_for_status()
    stats("biz GET /orders", bench(biz_orders, 20))

    # ② 走工具层（Pydantic 校验 + HTTP + 审计回调）
    reg = build_registry()
    def tool_orders():
        r = reg.call("query_orders", {"customer_name": "王伟", "since_days": 7})
        assert r.ok, r.error
    stats("tool query_orders", bench(tool_orders, 20))

    # ③ RAG：冷启动一次 + 缓存命中
    q = "退款几天内可以无理由退"
    t = time.perf_counter(); rag_index.answer_context(q, 3); cold = (time.perf_counter() - t) * 1000
    stats("rag answer_context (缓存后)", bench(lambda: rag_index.answer_context(q, 3), 50))
    print(f"{'rag 首次(含建索引)':<26} {cold:7.2f}ms")

    # ④ 整条图：offline 规划 + 工具 + 规则生成
    from agent.graph import get_graph
    g = get_graph()
    def turn():
        out = g.invoke({"question": "查一下王伟最近一周的订单", "thread_id": f"bench-{time.time_ns()}"},
                       config={"configurable": {"thread_id": f"bench-{time.time_ns()}"}})
        assert out.get("tool_trace")
    stats("graph 单轮（offline）", bench(turn, 8, 1))


if __name__ == "__main__":
    main()
