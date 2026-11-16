"""审计日志：每一次工具调用都落库，事后可查"谁、什么时候、查了什么、成不成功"。"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "agent.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  thread_id TEXT,
  tool TEXT NOT NULL,
  kind TEXT,
  args_json TEXT,
  ok INTEGER NOT NULL,
  error TEXT,
  latency_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC);
CREATE INDEX IF NOT EXISTS idx_audit_thread ON audit_log(thread_id, id DESC);
CREATE TABLE IF NOT EXISTS conversations (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL,
  thread_id TEXT,
  question TEXT,
  answer TEXT,
  tools_json TEXT,
  confidence TEXT,
  blocked INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_conv_ts ON conversations(ts DESC);
CREATE INDEX IF NOT EXISTS idx_conv_thread ON conversations(thread_id, id DESC);
"""


def _conn() -> sqlite3.Connection:
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB, check_same_thread=False, timeout=5)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.executescript(SCHEMA)
    return conn


# FastAPI 的同步端点跑在线程池里，审计库共用一条连接 → 写操作串行化
_LOCK = threading.Lock()


_CONN: sqlite3.Connection | None = None


def conn() -> sqlite3.Connection:
    global _CONN
    if _CONN is None:
        _CONN = _conn()
    return _CONN


def log_tool(thread_id: str | None, record: dict) -> None:
    with _LOCK:
        c = conn()
        c.execute(
            "INSERT INTO audit_log (ts, thread_id, tool, kind, args_json, ok, error, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (record.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S"), thread_id, record["tool"],
             record.get("kind"), json.dumps(record.get("args", {}), ensure_ascii=False),
             1 if record.get("ok") else 0, record.get("error"), record.get("latency_ms")),
        )
        c.commit()


def log_conversation(thread_id: str, question: str, answer: str, tools: list[dict],
                     confidence: str = "", blocked: bool = False) -> None:
    with _LOCK:
        c = conn()
        c.execute(
            "INSERT INTO conversations (ts, thread_id, question, answer, tools_json, confidence, blocked)"
            " VALUES (?,?,?,?,?,?,?)",
            (time.strftime("%Y-%m-%d %H:%M:%S"), thread_id, question, answer,
             json.dumps(tools, ensure_ascii=False), confidence, 1 if blocked else 0),
        )
        c.commit()


def recent_calls(limit: int = 50, thread_id: str | None = None) -> list[dict]:
    c = conn()
    if thread_id:
        rows = c.execute("SELECT * FROM audit_log WHERE thread_id = ? ORDER BY id DESC LIMIT ?",
                         (thread_id, limit)).fetchall()
    else:
        rows = c.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["args"] = json.loads(d.pop("args_json") or "{}")
        d["ok"] = bool(d["ok"])
        out.append(d)
    return out


def recent_conversations(limit: int = 20) -> list[dict]:
    rows = conn().execute(
        "SELECT * FROM conversations ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["tools"] = json.loads(d.pop("tools_json") or "[]")
        d["blocked"] = bool(d["blocked"])
        out.append(d)
    return out


def stats() -> dict:
    c = conn()
    total = c.execute("SELECT COUNT(*) n FROM audit_log").fetchone()["n"]
    failed = c.execute("SELECT COUNT(*) n FROM audit_log WHERE ok = 0").fetchone()["n"]
    avg = c.execute("SELECT COALESCE(AVG(latency_ms),0) a FROM audit_log").fetchone()["a"]
    by_tool = {r["tool"]: r["n"] for r in
               c.execute("SELECT tool, COUNT(*) n FROM audit_log GROUP BY tool ORDER BY n DESC")}
    convs = c.execute("SELECT COUNT(*) n FROM conversations").fetchone()["n"]
    blocked = c.execute("SELECT COUNT(*) n FROM conversations WHERE blocked = 1").fetchone()["n"]
    return {"tool_calls": total, "tool_failures": failed, "avg_latency_ms": round(avg, 1),
            "by_tool": by_tool, "conversations": convs, "blocked_conversations": blocked}
