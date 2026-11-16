"""工具层测试：不依赖大模型，纯断言，证明"业务能力可测"。"""

from __future__ import annotations

from tools.base import ToolRegistry
from tools.biz_tools import build_registry

EXPECTED = {
    "query_orders", "get_order_detail", "get_logistics", "check_inventory", "low_stock_list",
    "query_tickets", "create_ticket", "create_approval", "query_policy",
}


def test_registry_has_nine_tools():
    reg = build_registry()
    assert isinstance(reg, ToolRegistry)
    assert set(reg.tools) == EXPECTED


def test_schemas_are_llm_ready():
    reg = build_registry()
    schemas = reg.schemas()
    assert len(schemas) == len(EXPECTED)
    for s in schemas:
        fn = s["function"] if "function" in s else s
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"
    write_tools = {t.name for t in reg.tools.values() if t.kind == "write"}
    assert write_tools == {"create_ticket", "create_approval"}


def test_only_read_tools_can_be_filtered():
    reg = build_registry()
    read_schemas = reg.schemas(kinds=("read",))
    names = {(s["function"] if "function" in s else s)["name"] for s in read_schemas}
    assert names == EXPECTED - {"create_ticket", "create_approval"}


def test_unknown_tool_is_soft_error():
    res = build_registry().call("drop_database", {})
    assert res.ok is False and "未注册" in res.error


def test_bad_args_are_rejected_before_http():
    res = build_registry().call("get_logistics", {})  # 缺必填 order_no
    assert res.ok is False and res.error


def test_read_tool_hits_business_system(base):
    res = build_registry().call("low_stock_list", {"limit": 3})
    assert res.ok, res.error
    assert 1 <= len(res.data["items"]) <= 3


def test_ticket_write_tool(base):
    res = build_registry().call("create_ticket", {"type": "咨询", "content": "工具层直调测试"})
    assert res.ok, res.error
    assert res.data["ticket_no"].startswith("TK")


def test_audit_callback_receives_records():
    seen: list[dict] = []
    reg = build_registry(on_call=seen.append)
    reg.call("query_policy", {"question": "退货运费谁承担", "top_k": 2})
    assert seen and seen[0]["tool"] == "query_policy" and "latency_ms" in seen[0]


def test_write_tools_are_marked_for_approval():
    reg = build_registry()
    assert reg.tools["create_approval"].requires_approval is True
    assert reg.tools["create_ticket"].requires_approval is False
