"""把 9 个业务工具暴露成 MCP Server（stdio），让 Claude Desktop / 其他 MCP 客户端直接调。

    python mcp_server.py            # stdio 模式，供 MCP 客户端拉起
    python mcp_server.py --list     # 只打印工具清单，自检用

设计要点：MCP 只是"另一条入口"，工具实现只有 tools/biz_tools.py 一份，
所以权限分类、参数校验、审计回调全部复用，不会出现两套行为。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mcp.server.mcpserver import MCPServer  # noqa: E402

from agent import audit  # noqa: E402
from tools.biz_tools import build_registry  # noqa: E402

REG = build_registry(on_call=lambda rec: audit.log_tool("mcp", rec))

mcp = MCPServer(
    name="biz-agent",
    title="BizAgent 企业业务工具集",
    version="1.0.0",
    instructions=(
        "电商企业内部业务工具：查订单/物流/库存/工单、查公司制度（带 citations）、"
        "建工单、提交退款/补发/调价审批。写操作只生成待审批单据，不会直接改业务数据。"
    ),
)


def _call(tool: str, **args: Any) -> dict:
    res = REG.call(tool, args)
    if not res.ok:
        return {"ok": False, "tool": tool, "error": res.error}
    return {"ok": True, "tool": tool, "data": res.data, "latency_ms": res.latency_ms}


@mcp.tool(description="按客户姓名/状态/最近天数查询订单列表（只读）")
def query_orders(customer_name: str | None = None, status: str | None = None,
                 since_days: int | None = None, limit: int = 10) -> dict:
    args: dict[str, Any] = {"limit": limit}
    if customer_name:
        args["customer_name"] = customer_name
    if status:
        args["status"] = status
    if since_days:
        args["since_days"] = since_days
    return _call("query_orders", **args)


@mcp.tool(description="查单个订单的完整信息：主体 + 商品明细 + 物流（只读）")
def get_order_detail(order_no: str) -> dict:
    return _call("get_order_detail", order_no=order_no)


@mcp.tool(description="查订单物流轨迹与签收状态（只读）")
def get_logistics(order_no: str) -> dict:
    return _call("get_logistics", order_no=order_no)


@mcp.tool(description="查某商品库存与安全库存水位，支持 SKU 或名称（只读）")
def check_inventory(product: str) -> dict:
    return _call("check_inventory", product=product)


@mcp.tool(description="列出所有低于安全库存的商品（只读）")
def low_stock_list(limit: int = 10) -> dict:
    return _call("low_stock_list", limit=limit)


@mcp.tool(description="查工单列表，可按状态或“是否超 SLA”过滤（只读）")
def query_tickets(status: str | None = None, overdue_only: bool = False, limit: int = 10) -> dict:
    args: dict[str, Any] = {"overdue_only": overdue_only, "limit": limit}
    if status:
        args["status"] = status
    return _call("query_tickets", **args)


@mcp.tool(description="创建客服工单（写操作，可撤销；类型决定 SLA 时长）")
def create_ticket(content: str, type: str = "咨询", order_no: str | None = None) -> dict:
    args: dict[str, Any] = {"content": content, "type": type}
    if order_no:
        args["order_no"] = order_no
    return _call("create_ticket", **args)


@mcp.tool(description="提交退款/补发/调价审批单（写操作，必须人工审批后才执行）")
def create_approval(kind: str, amount: float | None = None, order_no: str | None = None,
                    reason: str = "") -> dict:
    payload: dict[str, Any] = {}
    if amount is not None:
        payload["amount"] = amount
    if order_no:
        payload["order_no"] = order_no
    if reason:
        payload["reason"] = reason
    return _call("create_approval", kind=kind, payload=payload)


@mcp.tool(description="检索公司制度知识库（退换货/发票/发货/SLA/会员/批发/破损/催收），返回带 citations")
def query_policy(question: str, top_k: int = 3) -> dict:
    return _call("query_policy", question=question, top_k=top_k)


if __name__ == "__main__":
    if "--list" in sys.argv:
        tools = REG.describe()
        print(json.dumps(tools, ensure_ascii=False, indent=2))
    else:
        mcp.run(transport="stdio")
