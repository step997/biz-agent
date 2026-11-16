"""9 个业务工具：智能体与"企业业务系统"之间的唯一通道。

每个工具 = 一个 Pydantic 参数模型 + 一个 handler（HTTP 调 biz_system）。
handler 不依赖 LLM，因此可以纯 pytest 断言。
"""

from __future__ import annotations

import os
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from tools.base import Tool, ToolError, ToolRegistry

BIZ_BASE_URL = os.getenv("BIZ_BASE_URL", "http://127.0.0.1:8001").rstrip("/")
BIZ_API_KEY = os.getenv("BIZ_API_KEY", "demo-key")
TIMEOUT = float(os.getenv("BIZ_TIMEOUT", "10"))
RETRIES = int(os.getenv("BIZ_RETRIES", "1"))

# 工具对象与 schema 只构造一次（见 build_registry）
_TOOLS: list[Any] | None = None


def _client() -> httpx.Client:
    return httpx.Client(base_url=BIZ_BASE_URL, timeout=TIMEOUT,
                        headers={"X-Api-Key": BIZ_API_KEY})


def _request(method: str, path: str, **kw) -> httpx.Response:
    """唯一的出网口：连接级失败（超时/断开）重试一次，4xx/5xx 不重试（重试也没用）。"""
    last: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            with _client() as c:
                return c.request(method, path, **kw)
        except httpx.TransportError as exc:  # 超时、连接被拒、中途断开
            last = exc
            if attempt < RETRIES:
                time.sleep(0.2 * (attempt + 1))
    raise ToolError(f"连不上业务系统（{method} {path}）：{last}")


def _get(path: str, params: dict | None = None) -> Any:
    r = _request("GET", path, params={k: v for k, v in (params or {}).items() if v is not None})
    if r.status_code >= 400:
        raise ToolError(f"业务系统返回 {r.status_code}：{_detail(r)}")
    return r.json()


def _post(path: str, payload: dict) -> Any:
    r = _request("POST", path, json=payload)
    if r.status_code >= 400:
        raise ToolError(f"业务系统返回 {r.status_code}：{_detail(r)}")
    return r.json()


def _patch(path: str, params: dict | None = None, payload: dict | None = None) -> Any:
    r = _request("PATCH", path, params=params, json=payload)
    if r.status_code >= 400:
        raise ToolError(f"业务系统返回 {r.status_code}：{_detail(r)}")
    return r.json()


def _detail(r: httpx.Response) -> str:
    try:
        return str(r.json().get("detail", r.text[:200]))
    except Exception:  # noqa: BLE001
        return r.text[:200]


# ───────────────────────────── 参数模型 ─────────────────────────────

OrderStatus = Literal["待付款", "待发货", "已发货", "已完成", "退款中", "已退款"]
TicketType = Literal["物流", "质量", "退款", "咨询"]
TicketStatus = Literal["待处理", "处理中", "已解决"]


class QueryOrdersArgs(BaseModel):
    customer_name: str | None = Field(default=None, description="客户姓名，支持模糊匹配，如“王芳”")
    status: OrderStatus | None = Field(default=None, description="按订单状态过滤")
    since_days: int | None = Field(default=None, ge=1, le=365, description="只看最近 N 天下单的订单，如 7 表示最近一周")
    limit: int = Field(default=10, ge=1, le=50, description="返回条数")


class OrderNoArgs(BaseModel):
    order_no: str = Field(description="订单号，形如 SO10001")


class CheckInventoryArgs(BaseModel):
    product: str = Field(description="商品名或 SKU，如“充电器”或“SKU0031”")


class LowStockArgs(BaseModel):
    limit: int = Field(default=20, ge=1, le=50, description="返回条数")


class QueryTicketsArgs(BaseModel):
    status: TicketStatus | None = Field(default=None, description="工单状态")
    overdue_only: bool = Field(default=False, description="是否只看已超 SLA 的工单")


class CreateTicketArgs(BaseModel):
    content: str = Field(min_length=2, max_length=500, description="工单内容，写清客户诉求")
    type: TicketType = Field(default="咨询", description="工单类型，决定 SLA 时长")
    order_no: str | None = Field(default=None, description="关联订单号，可选")


class CreateApprovalArgs(BaseModel):
    kind: Literal["退款", "补发", "调价"] = Field(description="审批类型")
    order_no: str | None = Field(default=None, description="关联订单号")
    amount: float | None = Field(default=None, ge=0, description="涉及金额（退款/补发必填）")
    reason: str = Field(default="", description="申请理由")


class QueryPolicyArgs(BaseModel):
    question: str = Field(min_length=2, description="制度类问题，如“退款几天内可以无理由退”")
    top_k: int = Field(default=3, ge=1, le=8, description="返回段落数")


# ───────────────────────────── handler ─────────────────────────────


def query_orders(a: QueryOrdersArgs) -> Any:
    params: dict[str, Any] = {"customer_name": a.customer_name, "status": a.status, "limit": a.limit}
    if a.since_days:
        import datetime as dt
        since = dt.date.today() - dt.timedelta(days=a.since_days)
        params["date_from"] = since.isoformat()
    data = _get("/orders", params)
    return {"total": data["total"], "orders": data["items"]}


def get_order_detail(a: OrderNoArgs) -> Any:
    order = _get(f"/orders/{a.order_no}")
    items = _get(f"/orders/{a.order_no}/items")["items"]
    logistics = _get(f"/logistics/{a.order_no}")
    return {"order": order, "items": items, "logistics": logistics}


def get_logistics(a: OrderNoArgs) -> Any:
    return _get(f"/logistics/{a.order_no}")


def check_inventory(a: CheckInventoryArgs) -> Any:
    return _get(f"/inventory/{a.product}")


def low_stock_list(a: LowStockArgs) -> Any:
    return _get("/inventory-low", {"limit": a.limit})


def query_tickets(a: QueryTicketsArgs) -> Any:
    return _get("/tickets", {"status": a.status, "overdue_only": a.overdue_only})


def create_ticket(a: CreateTicketArgs) -> Any:
    return _post("/tickets", {"order_no": a.order_no, "type": a.type, "content": a.content})


def create_approval(a: CreateApprovalArgs) -> Any:
    payload = {k: v for k, v in {"order_no": a.order_no, "amount": a.amount, "reason": a.reason}.items()
               if v is not None}
    return _post("/approvals", {"kind": a.kind, "payload": payload, "requested_by": "agent"})


def query_policy(a: QueryPolicyArgs) -> Any:
    """制度类问答：本地混合检索（BM25 + 向量 + RRF），命中不足时明确说"没查到"。"""
    try:
        from rag.index import get_index
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"制度索引不可用：{exc}") from exc
    idx = get_index()
    return idx.answer_context(a.question, top_k=a.top_k)


# ───────────────────────────── 注册 ─────────────────────────────


def build_registry(on_call=None) -> ToolRegistry:
    """返回工具注册表。

    工具对象（含 Pydantic 参数模型与 schema）与调用上下文无关，模块级只构造一次：
    每次 /chat 都重建 9 个 Tool 会白白多出对象分配与 schema 重算。
    """
    global _TOOLS
    if _TOOLS is None:
        _TOOLS = list(_build_tools().tools.values())
    return ToolRegistry(tools={t.name: t for t in _TOOLS}, on_call=on_call)


def _build_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(Tool(
        name="query_orders",
        description="按客户姓名/状态/时间范围查询订单列表（只读）。用户问“某人最近买了什么”“有哪些待发货订单”时用。",
        args_model=QueryOrdersArgs, handler=query_orders, kind="read"))
    reg.register(Tool(
        name="get_order_detail",
        description="查单个订单的完整信息：订单主体 + 商品明细 + 物流进度（只读）。已知订单号时优先用这个。",
        args_model=OrderNoArgs, handler=get_order_detail, kind="read"))
    reg.register(Tool(
        name="get_logistics",
        description="查订单物流轨迹与签收状态（只读）。用户问“到哪了”“为什么还没到”时用。",
        args_model=OrderNoArgs, handler=get_logistics, kind="read"))
    reg.register(Tool(
        name="check_inventory",
        description="查某商品的库存与安全库存水位（只读）。支持商品名模糊匹配或 SKU。",
        args_model=CheckInventoryArgs, handler=check_inventory, kind="read"))
    reg.register(Tool(
        name="low_stock_list",
        description="列出所有低于安全库存的商品（只读）。用于补货提醒、巡检。",
        args_model=LowStockArgs, handler=low_stock_list, kind="read"))
    reg.register(Tool(
        name="query_tickets",
        description="查工单列表，可按状态或“是否超 SLA”过滤（只读）。用于超时工单巡检、客服进度查询。",
        args_model=QueryTicketsArgs, handler=query_tickets, kind="read"))
    reg.register(Tool(
        name="create_ticket",
        description="创建客服工单（写操作，可撤销）。类型决定 SLA：物流4h/咨询8h/质量24h/退款24h。",
        args_model=CreateTicketArgs, handler=create_ticket, kind="write"))
    reg.register(Tool(
        name="create_approval",
        description="发起退款/补发/调价审批单（写操作，涉及资金，必须人工审批后才执行）。",
        args_model=CreateApprovalArgs, handler=create_approval, kind="write", requires_approval=True))
    reg.register(Tool(
        name="query_policy",
        description="检索公司制度知识库（退换货、发票、发货时效、SLA、会员、批发、破损、催付），只答制度类问题。",
        args_model=QueryPolicyArgs, handler=query_policy, kind="read"))
    return reg


REGISTRY = build_registry()
