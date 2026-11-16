"""制度语料加载：优先走业务系统 API（证明"知识也来自系统"），失败时退回本地缓存/数据库。"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "policies_cache.json"
BIZ_BASE_URL = os.getenv("BIZ_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
BIZ_API_KEY = os.getenv("BIZ_API_KEY", "demo-key")


def _from_api() -> list[dict]:
    r = httpx.get(f"{BIZ_BASE_URL}/policies/chunks", params={"limit": 2000},
                  headers={"X-Api-Key": BIZ_API_KEY}, timeout=10)
    r.raise_for_status()
    return r.json()["items"]


def _from_db() -> list[dict]:
    db = ROOT / "data" / "biz.db"
    if not db.exists():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, title, seq, content FROM policies ORDER BY doc_id, seq").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def load_chunks(use_cache: bool = True) -> list[dict]:
    if use_cache and CACHE.exists():
        try:
            data = json.loads(CACHE.read_text(encoding="utf-8"))
            if data:
                return data
        except Exception:  # noqa: BLE001
            pass
    chunks: list[dict] = []
    try:
        chunks = _from_api()
    except Exception:  # noqa: BLE001
        chunks = _from_db()
    if chunks and use_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(chunks, ensure_ascii=False, indent=1), encoding="utf-8")
    return chunks
