"""业务系统（mock ERP）接口测试：认证、查数、写单、状态机。"""

from __future__ import annotations

import httpx

KEY = {"X-Api-Key": "demo-key"}


def test_health_needs_no_key(base):
    r = httpx.get(f"{base}/health", timeout=5)
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_orders_requires_api_key(base):
    assert httpx.get(f"{base}/orders", timeout=5).status_code == 401
    assert httpx.get(f"{base}/orders", headers={"X-Api-Key": "wrong"}, timeout=5).status_code == 401


def test_orders_list_and_filter(base):
    r = httpx.get(f"{base}/orders", headers=KEY, timeout=5).json()
    assert r["total"] >= 20 and len(r["items"]) == r["returned"] <= 50
    page = httpx.get(f"{base}/orders?status=待发货&limit=5", headers=KEY, timeout=5).json()
    assert page["returned"] <= 5
    assert all(i["status"] == "待发货" for i in page["items"])


def test_orders_by_customer_and_date(base):
    r = httpx.get(f"{base}/orders?customer_name=王伟&limit=50", headers=KEY, timeout=5).json()
    assert r["total"] >= 1
    assert {i["customer"] for i in r["items"]} == {"王伟"}


def test_order_detail_and_404(base):
    no = httpx.get(f"{base}/orders?limit=1", headers=KEY, timeout=5).json()["items"][0]["order_no"]
    ok = httpx.get(f"{base}/orders/{no}", headers=KEY, timeout=5)
    assert ok.status_code == 200 and ok.json()["order_no"] == no
    assert httpx.get(f"{base}/orders/SO99999", headers=KEY, timeout=5).status_code == 404
    items = httpx.get(f"{base}/orders/{no}/items", headers=KEY, timeout=5).json()
    assert len(items["items"]) >= 1
    assert abs(sum(i["line_amount"] for i in items["items"]) - ok.json()["amount"]) < 0.01


def test_logistics_timeline(base):
    shipped = httpx.get(f"{base}/orders?status=已发货&limit=1", headers=KEY, timeout=5).json()["items"][0]
    body = httpx.get(f"{base}/logistics/{shipped['order_no']}", headers=KEY, timeout=5).json()
    assert body["carrier"] and body["logistics_no"]
    assert body["shipped"] is True and len(body["nodes"]) >= 1
    assert "揽收" in body["nodes"][0]["desc"]


def test_inventory_exact_and_fuzzy(base):
    exact = httpx.get(f"{base}/inventory/手机0001", headers=KEY, timeout=5).json()
    assert exact["sku"] == "手机0001" and exact["total_qty"] > 0
    assert sum(w["qty"] for w in exact["warehouses"]) == exact["total_qty"]
    fuzzy = httpx.get(f"{base}/inventory/充电器", headers=KEY, timeout=5).json()
    assert fuzzy.get("count", 1) >= 1


def test_low_stock_list(base):
    r = httpx.get(f"{base}/inventory-low", headers=KEY, timeout=5).json()
    assert r["below_safety_count"] >= 1
    assert all(i["qty"] < i["safety_stock"] and i["shortage"] > 0 for i in r["items"])


def test_tickets_overdue_filter(base):
    all_t = httpx.get(f"{base}/tickets", headers=KEY, timeout=5).json()
    late = httpx.get(f"{base}/tickets?overdue_only=true", headers=KEY, timeout=5).json()
    assert all_t["count"] >= 20
    assert 1 <= late["count"] < all_t["count"]
    assert all(i["overdue"] for i in late["items"])


def test_ticket_create_and_patch(base):
    no = httpx.get(f"{base}/orders?limit=1", headers=KEY, timeout=5).json()["items"][0]["order_no"]
    created = httpx.post(f"{base}/tickets", headers=KEY, timeout=5,
                         json={"order_no": no, "type": "质量", "content": "外包装破损，需换货"})
    assert created.status_code == 201
    body = created.json()
    assert body["ticket_no"].startswith("TK") and body["sla_hours"] == 24
    patched = httpx.patch(f"{base}/tickets/{body['ticket_no']}?status=已解决", headers=KEY, timeout=5)
    assert patched.status_code == 200 and patched.json()["status"] == "已解决"


def test_ticket_validation(base):
    bad_type = httpx.post(f"{base}/tickets", headers=KEY, timeout=5,
                          json={"type": "投诉", "content": "类型不在枚举内"})
    assert bad_type.status_code == 422
    too_short = httpx.post(f"{base}/tickets", headers=KEY, timeout=5,
                           json={"type": "咨询", "content": "x"})
    assert too_short.status_code == 422


def test_approval_flow(base):
    created = httpx.post(f"{base}/approvals", headers=KEY, timeout=5,
                         json={"kind": "退款", "payload": {"amount": 88.5, "reason": "测试"}})
    assert created.status_code == 201
    no = created.json()["approval_no"]
    pending = httpx.get(f"{base}/approvals?status=pending", headers=KEY, timeout=5).json()
    assert no in {i["approval_no"] for i in pending["items"]}
    decided = httpx.patch(f"{base}/approvals/{no}", headers=KEY, timeout=5,
                          json={"decision": "approved", "decided_by": "tester"})
    assert decided.status_code == 200 and decided.json()["status"] == "approved"
    assert httpx.patch(f"{base}/approvals/{no}", headers=KEY, timeout=5,
                       json={"decision": "maybe"}).status_code == 422


def test_overview_stats(base):
    r = httpx.get(f"{base}/stats/overview", headers=KEY, timeout=5).json()
    assert r["orders_total"] == sum(r["orders_by_status"].values())
    assert r["low_stock_skus"] >= 1 and r["overdue_tickets"] >= 1
