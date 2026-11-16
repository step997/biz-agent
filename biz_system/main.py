"""Mock 企业业务系统 —— HTTP 服务（模拟"企业内网 ERP/商城后台"）。

启动：uvicorn biz_system.main:app --port 8001
鉴权：所有业务接口需要请求头 X-Api-Key: <BIZ_API_KEY>（默认 demo-key）
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, Field

from biz_system.db import connect, rows_to_dicts

API_KEY = os.getenv("BIZ_API_KEY", "demo-key")
app = FastAPI(title="BizAgent Mock 业务系统", version="1.0.0",
              description="模拟中小电商的订单/库存/工单/审批/制度知识库")


def require_key(x_api_key: str | None = Header(default=None)) -> None:
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-Api-Key")


BIZ = [Depends(require_key)]


def _now() -> datetime:
    return datetime.now()


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────── 健康检查 / 概览 ───────────────────────────


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "biz_system", "ts": _iso(_now())}


@app.get("/stats/overview", dependencies=BIZ)
def stats_overview() -> dict:
    conn = connect()
    try:
        by_status = {r["status"]: r["n"] for r in conn.execute(
            "SELECT status, COUNT(*) n FROM orders GROUP BY status")}
        total = sum(by_status.values())
        amount = conn.execute("SELECT COALESCE(SUM(amount),0) a FROM orders").fetchone()["a"]
        low = conn.execute(
            "SELECT COUNT(*) n FROM inventory WHERE qty < safety_stock").fetchone()["n"]
        overdue = conn.execute(
            "SELECT COUNT(*) n FROM tickets WHERE status != '已解决' AND sla_deadline < ?",
            (_iso(_now()),)).fetchone()["n"]
        pending_approval = conn.execute(
            "SELECT COUNT(*) n FROM approvals WHERE status = 'pending'").fetchone()["n"]
        return {
            "orders_total": total,
            "orders_by_status": by_status,
            "orders_amount": round(amount, 2),
            "low_stock_skus": low,
            "overdue_tickets": overdue,
            "pending_approvals": pending_approval,
        }
    finally:
        conn.close()


# ─────────────────────────────── 订单 ───────────────────────────────


@app.get("/orders", dependencies=BIZ)
def list_orders(
    customer_name: str | None = Query(default=None, description="客户姓名，支持模糊匹配"),
    status: str | None = Query(default=None, description="待付款/待发货/已发货/已完成/退款中/已退款"),
    date_from: str | None = Query(default=None, description="YYYY-MM-DD"),
    date_to: str | None = Query(default=None, description="YYYY-MM-DD"),
    limit: int = Query(default=20, ge=1, le=50),
    offset: int = Query(default=0, ge=0),
) -> dict:
    where, params = [], []
    if customer_name:
        where.append("c.name LIKE ?")
        params.append(f"%{customer_name}%")
    if status:
        where.append("o.status = ?")
        params.append(status)
    if date_from:
        where.append("o.created_at >= ?")
        params.append(f"{date_from} 00:00:00")
    if date_to:
        where.append("o.created_at <= ?")
        params.append(f"{date_to} 23:59:59")
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    conn = connect()
    try:
        rows = conn.execute(
            f"""SELECT o.order_no, c.name AS customer, o.status, o.amount, o.created_at,
                       o.logistics_no, o.carrier
                FROM orders o JOIN customers c ON c.id = o.customer_id
                {sql_where} ORDER BY o.created_at DESC LIMIT ? OFFSET ?""",
            (*params, limit, offset),
        ).fetchall()
        total = conn.execute(
            f"SELECT COUNT(*) n FROM orders o JOIN customers c ON c.id = o.customer_id {sql_where}",
            params,
        ).fetchone()["n"]
        return {"total": total, "returned": len(rows), "items": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/orders/{order_no}", dependencies=BIZ)
def get_order(order_no: str) -> dict:
    conn = connect()
    try:
        row = conn.execute(
            """SELECT o.*, c.name AS customer, c.level, c.phone
               FROM orders o JOIN customers c ON c.id = o.customer_id
               WHERE o.order_no = ?""", (order_no,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"订单 {order_no} 不存在")
        data = dict(row)
        data.pop("id", None)
        data.pop("customer_id", None)
        return data
    finally:
        conn.close()


@app.get("/orders/{order_no}/items", dependencies=BIZ)
def get_order_items(order_no: str) -> dict:
    conn = connect()
    try:
        o = conn.execute("SELECT id FROM orders WHERE order_no = ?", (order_no,)).fetchone()
        if not o:
            raise HTTPException(status_code=404, detail=f"订单 {order_no} 不存在")
        rows = conn.execute(
            """SELECT p.sku, p.name, i.qty, i.unit_price, ROUND(i.qty * i.unit_price, 2) AS line_amount
               FROM order_items i JOIN products p ON p.id = i.product_id
               WHERE i.order_id = ?""", (o["id"],)).fetchall()
        return {"order_no": order_no, "items": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/logistics/{order_no}", dependencies=BIZ)
def get_logistics(order_no: str) -> dict:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT order_no, status, shipped_at, logistics_no, carrier FROM orders WHERE order_no = ?",
            (order_no,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"订单 {order_no} 不存在")
        if not row["logistics_no"]:
            return {"order_no": order_no, "status": row["status"], "shipped": False,
                    "note": "该订单尚未发货，暂无物流信息"}
        shipped = datetime.strptime(row["shipped_at"], "%Y-%m-%d %H:%M:%S")
        elapsed_days = max(0, (_now() - shipped).days)
        nodes = [
            {"time": row["shipped_at"], "desc": f"{row['carrier']}已揽收，运单号 {row['logistics_no']}"},
        ]
        if elapsed_days >= 1:
            nodes.append({"time": _iso(shipped + timedelta(days=1)), "desc": "到达始发分拨中心"})
        if elapsed_days >= 2:
            nodes.append({"time": _iso(shipped + timedelta(days=2)), "desc": "运输中，预计次日到达"})
        if elapsed_days >= 3:
            nodes.append({"time": _iso(shipped + timedelta(days=3)), "desc": "派送中，请保持电话畅通"})
        return {
            "order_no": order_no,
            "carrier": row["carrier"],
            "logistics_no": row["logistics_no"],
            "shipped": True,
            "elapsed_days": elapsed_days,
            "nodes": nodes,
            "latest": nodes[-1]["desc"],
        }
    finally:
        conn.close()


# ───────────────────────────── 库存 / 商品 ─────────────────────────────


@app.get("/inventory/{key}", dependencies=BIZ)
def get_inventory(key: str) -> dict:
    """key 先按 SKU 精确匹配，匹配不到再按商品名模糊匹配。"""
    conn = connect()
    try:
        prod = conn.execute("SELECT * FROM products WHERE sku = ?", (key,)).fetchone()
        if not prod:
            prod = conn.execute(
                "SELECT * FROM products WHERE name LIKE ? LIMIT 1", (f"%{key}%",)).fetchone()
        if not prod:
            raise HTTPException(status_code=404, detail="商品不存在（sku 或名称不匹配）")
        rows = conn.execute(
            "SELECT warehouse, qty, safety_stock, updated_at FROM inventory WHERE product_id = ?",
            (prod["id"],)).fetchall()
        total = sum(r["qty"] for r in rows)
        below = any(r["qty"] < r["safety_stock"] for r in rows)
        return {
            "sku": prod["sku"], "name": prod["name"], "category": prod["category"],
            "price": prod["price"], "total_qty": total, "below_safety": below,
            "warehouses": rows_to_dicts(rows),
        }
    finally:
        conn.close()


@app.get("/inventory-low", dependencies=BIZ)
def low_stock(limit: int = Query(default=20, ge=1, le=50)) -> dict:
    conn = connect()
    try:
        rows = conn.execute(
            """SELECT p.sku, p.name, i.warehouse, i.qty, i.safety_stock,
                      (i.safety_stock - i.qty) AS shortage
               FROM inventory i JOIN products p ON p.id = i.product_id
               WHERE i.qty < i.safety_stock
               ORDER BY shortage DESC LIMIT ?""", (limit,)).fetchall()
        return {"below_safety_count": len(rows), "items": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/customers/{customer_id}", dependencies=BIZ)
def get_customer(customer_id: int) -> dict:
    conn = connect()
    try:
        row = conn.execute("SELECT * FROM customers WHERE id = ?", (customer_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="客户不存在")
        agg = conn.execute(
            """SELECT COUNT(*) n, COALESCE(SUM(amount),0) amount FROM orders
               WHERE customer_id = ?""", (customer_id,)).fetchone()
        data = dict(row)
        data["orders_count"] = agg["n"]
        data["orders_amount"] = round(agg["amount"], 2)
        return data
    finally:
        conn.close()


# ─────────────────────────────── 工单 ───────────────────────────────

SLA_HOURS = {"物流": 4, "咨询": 8, "质量": 24, "退款": 24}


class TicketIn(BaseModel):
    order_no: str | None = None
    type: Literal["物流", "质量", "退款", "咨询"] = Field(default="咨询")
    content: str = Field(min_length=2, max_length=500)


@app.get("/tickets", dependencies=BIZ)
def list_tickets(
    status: str | None = Query(default=None, description="待处理/处理中/已解决"),
    overdue_only: bool = Query(default=False),
) -> dict:
    where, params = [], []
    if status:
        where.append("t.status = ?")
        params.append(status)
    if overdue_only:
        where.append("t.status != '已解决' AND t.sla_deadline < ?")
        params.append(_iso(_now()))
    sql_where = ("WHERE " + " AND ".join(where)) if where else ""
    conn = connect()
    try:
        rows = conn.execute(
            f"""SELECT t.ticket_no, t.type, t.status, t.sla_deadline, t.created_at,
                       o.order_no,
                       CASE WHEN t.status != '已解决' AND t.sla_deadline < ? THEN 1 ELSE 0 END AS overdue
                FROM tickets t LEFT JOIN orders o ON o.id = t.order_id
                {sql_where} ORDER BY t.created_at DESC LIMIT 50""",
            (_iso(_now()), *params),
        ).fetchall()
        return {"count": len(rows), "items": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.post("/tickets", dependencies=BIZ, status_code=201)
def create_ticket(payload: TicketIn) -> dict:
    conn = connect()
    try:
        order_id = None
        if payload.order_no:
            o = conn.execute("SELECT id FROM orders WHERE order_no = ?", (payload.order_no,)).fetchone()
            if not o:
                raise HTTPException(status_code=404, detail=f"订单 {payload.order_no} 不存在，无法建单")
            order_id = o["id"]
        seq = conn.execute("SELECT COALESCE(MAX(id),0)+1 n FROM tickets").fetchone()["n"]
        while conn.execute("SELECT 1 FROM tickets WHERE ticket_no = ?", (f"TK{20000 + seq}",)).fetchone():
            seq += 1
        ticket_no = f"TK{20000 + seq}"
        created = _now()
        deadline = created + timedelta(hours=SLA_HOURS.get(payload.type, 8))
        conn.execute(
            """INSERT INTO tickets (id, ticket_no, order_id, type, content, status, sla_deadline, created_at, resolved_at)
               VALUES (?,?,?,?,?,?,?,?,NULL)""",
            (seq, ticket_no, order_id, payload.type, payload.content, "待处理", _iso(deadline), _iso(created)),
        )
        conn.commit()
        return {"ticket_no": ticket_no, "type": payload.type, "status": "待处理",
                "sla_deadline": _iso(deadline), "sla_hours": SLA_HOURS.get(payload.type, 8)}
    finally:
        conn.close()


@app.patch("/tickets/{ticket_no}", dependencies=BIZ)
def update_ticket(ticket_no: str, status: str = Query()) -> dict:
    if status not in ("待处理", "处理中", "已解决"):
        raise HTTPException(status_code=400, detail="status 仅支持 待处理/处理中/已解决")
    conn = connect()
    try:
        row = conn.execute("SELECT id FROM tickets WHERE ticket_no = ?", (ticket_no,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="工单不存在")
        conn.execute(
            "UPDATE tickets SET status = ?, resolved_at = ? WHERE id = ?",
            (status, _iso(_now()) if status == "已解决" else None, row["id"]),
        )
        conn.commit()
        return {"ticket_no": ticket_no, "status": status}
    finally:
        conn.close()


# ─────────────────────────────── 审批 ───────────────────────────────


class ApprovalIn(BaseModel):
    kind: Literal["退款", "补发", "调价"] = "退款"
    payload: dict[str, Any] = Field(default_factory=dict)
    requested_by: str = "agent"


@app.get("/approvals", dependencies=BIZ)
def list_approvals(status: str | None = Query(default=None)) -> dict:
    conn = connect()
    try:
        if status:
            rows = conn.execute("SELECT * FROM approvals WHERE status = ? ORDER BY created_at DESC",
                                (status,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM approvals ORDER BY created_at DESC").fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["payload"] = json.loads(d.pop("payload_json"))
            out.append(d)
        return {"count": len(out), "items": out}
    finally:
        conn.close()


@app.post("/approvals", dependencies=BIZ, status_code=201)
def create_approval(payload: ApprovalIn) -> dict:
    conn = connect()
    try:
        seq = conn.execute("SELECT COALESCE(MAX(id),0)+1 n FROM approvals").fetchone()["n"]
        approval_no = f"AP{datetime.now():%Y%m%d}{seq:03d}"
        while conn.execute("SELECT 1 FROM approvals WHERE approval_no = ?", (approval_no,)).fetchone():
            seq += 1
            approval_no = f"AP{datetime.now():%Y%m%d}{seq:03d}"
        conn.execute(
            """INSERT INTO approvals (id, approval_no, kind, payload_json, status, requested_by,
                                      decided_by, decided_at, created_at)
               VALUES (?,?,?,?,?,?,NULL,NULL,?)""",
            (seq, approval_no, payload.kind, json.dumps(payload.payload, ensure_ascii=False),
             "pending", payload.requested_by, _iso(_now())),
        )
        conn.commit()
        return {"approval_no": approval_no, "kind": payload.kind, "status": "pending",
                "note": "已提交审批，需人工确认后执行"}
    finally:
        conn.close()


class DecisionIn(BaseModel):
    decision: Literal["approved", "rejected"]
    decided_by: str = "human"


@app.patch("/approvals/{approval_no}", dependencies=BIZ)
def decide_approval(approval_no: str, payload: DecisionIn) -> dict:
    conn = connect()
    try:
        row = conn.execute("SELECT id, status FROM approvals WHERE approval_no = ?",
                           (approval_no,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="审批单不存在")
        if row["status"] != "pending":
            raise HTTPException(status_code=409, detail=f"审批单已是 {row['status']} 状态")
        conn.execute("UPDATE approvals SET status = ?, decided_by = ?, decided_at = ? WHERE id = ?",
                     (payload.decision, payload.decided_by, _iso(_now()), row["id"]))
        conn.commit()
        return {"approval_no": approval_no, "status": payload.decision}
    finally:
        conn.close()


# ─────────────────────────── 制度知识库 ───────────────────────────


@app.get("/policies/search", dependencies=BIZ)
def search_policies(q: str = Query(min_length=1), limit: int = Query(default=5, ge=1, le=20)) -> dict:
    """业务系统侧的朴素关键词检索（供对照：智能体侧走自写混合检索）。"""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT chunk_id, title, content FROM policies WHERE content LIKE ? LIMIT ?",
            (f"%{q}%", limit)).fetchall()
        return {"query": q, "hits": rows_to_dicts(rows)}
    finally:
        conn.close()


@app.get("/policies/chunks", dependencies=BIZ)
def policy_chunks(limit: int = Query(default=500, ge=1, le=2000)) -> dict:
    """供智能体的 RAG 索引拉取全量 chunks。"""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT chunk_id, doc_id, title, seq, content FROM policies ORDER BY doc_id, seq LIMIT ?",
            (limit,)).fetchall()
        return {"count": len(rows), "items": rows_to_dicts(rows)}
    finally:
        conn.close()
