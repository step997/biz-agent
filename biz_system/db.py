"""Mock 企业业务系统 —— 数据层（SQLite）。

设计说明：
- 这是一个"假的 ERP/商城后台"，但接口、字段、状态机都按真实电商业务设计，
  目的是让上层的智能体有真实的业务系统可对接（JD 第 2 条：对接企业内部业务系统 API、数据库）。
- 业务系统独立进程运行（默认 :8001），智能体**只通过 HTTP 工具层**访问它，不直连数据库。
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("BIZ_DATA_DIR") or (ROOT / "data"))
DB_PATH = DATA_DIR / "biz.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    phone       TEXT NOT NULL,
    level       TEXT NOT NULL,              -- 普通 / VIP / 批发
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id          INTEGER PRIMARY KEY,
    sku         TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    category    TEXT NOT NULL,
    price       REAL NOT NULL,
    cost        REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    product_id   INTEGER NOT NULL,
    warehouse    TEXT NOT NULL,
    qty          INTEGER NOT NULL,
    safety_stock INTEGER NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (product_id, warehouse)
);

CREATE TABLE IF NOT EXISTS orders (
    id           INTEGER PRIMARY KEY,
    order_no     TEXT NOT NULL UNIQUE,
    customer_id  INTEGER NOT NULL,
    status       TEXT NOT NULL,             -- 待付款/待发货/已发货/已完成/退款中/已退款
    amount       REAL NOT NULL,
    created_at   TEXT NOT NULL,
    paid_at      TEXT,
    shipped_at   TEXT,
    logistics_no TEXT,
    carrier      TEXT
);

CREATE TABLE IF NOT EXISTS order_items (
    id         INTEGER PRIMARY KEY,
    order_id   INTEGER NOT NULL,
    product_id INTEGER NOT NULL,
    qty        INTEGER NOT NULL,
    unit_price REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    id           INTEGER PRIMARY KEY,
    ticket_no    TEXT NOT NULL UNIQUE,
    order_id     INTEGER,
    type         TEXT NOT NULL,             -- 物流/质量/退款/咨询
    content      TEXT NOT NULL,
    status       TEXT NOT NULL,             -- 待处理/处理中/已解决
    sla_deadline TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    resolved_at  TEXT
);

CREATE TABLE IF NOT EXISTS approvals (
    id           INTEGER PRIMARY KEY,
    approval_no  TEXT NOT NULL UNIQUE,
    kind         TEXT NOT NULL,             -- 退款/补发/调价
    payload_json TEXT NOT NULL,
    status       TEXT NOT NULL,             -- pending/approved/rejected
    requested_by TEXT NOT NULL,
    decided_by   TEXT,
    decided_at   TEXT,
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    id       INTEGER PRIMARY KEY,
    doc_id   TEXT NOT NULL,
    title    TEXT NOT NULL,
    chunk_id TEXT NOT NULL UNIQUE,
    seq      INTEGER NOT NULL,
    content  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_orders_status  ON orders(status);
CREATE INDEX IF NOT EXISTS idx_orders_created ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_items_order    ON order_items(order_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_order  ON tickets(order_id);
CREATE INDEX IF NOT EXISTS idx_customers_name ON customers(name);
CREATE INDEX IF NOT EXISTS idx_inventory_product ON inventory(product_id);
CREATE INDEX IF NOT EXISTS idx_approvals_status ON approvals(status);
"""


def connect() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=5)
    conn.row_factory = sqlite3.Row
    # 读写并发：WAL 允许"一写多读"，演示时前端轮询不会把写操作锁住
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=3000")
    return conn


def init_db(reset: bool = False) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if reset and DB_PATH.exists():
        DB_PATH.unlink()
    conn = connect()
    try:
        conn.executescript(SCHEMA)
        conn.commit()
    finally:
        conn.close()


def rows_to_dicts(rows) -> list[dict]:
    return [dict(r) for r in rows]
