"""定时巡检：每天早上把"超时工单 / 低库存 / 待审批"推到企业微信群。

没有配 WECOM_WEBHOOK 时只打印（并把结果写进审计库），所以本地 demo 不需要企业微信也能跑。
用法：
    python -m agent.jobs          # 常驻，按 .env 里的 PATROL_CRON_* 每天跑
    python -m agent.jobs --now    # 立刻跑一次（评测/演示用）
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from agent import audit  # noqa: E402
from tools.biz_tools import BIZ_API_KEY, BIZ_BASE_URL, build_registry  # noqa: E402

WEBHOOK = os.getenv("WECOM_WEBHOOK", "").strip()
CRON_HOUR = int(os.getenv("PATROL_CRON_HOUR", "8"))
CRON_MINUTE = int(os.getenv("PATROL_CRON_MINUTE", "0"))


def collect() -> dict:
    """四个只读工具跑一遍，得到今日需要人关注的清单。"""
    reg = build_registry()
    out: dict = {"ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

    tickets = reg.call("query_tickets", {"overdue_only": True, "limit": 20})
    out["overdue_tickets"] = (tickets.data or {}).get("items", []) if tickets.ok else []
    out["tickets_error"] = None if tickets.ok else tickets.error

    low = reg.call("low_stock_list", {"limit": 20})
    out["low_stock"] = (low.data or {}).get("items", []) if low.ok else []
    out["low_error"] = None if low.ok else low.error

    try:
        r = httpx.get(f"{BIZ_BASE_URL}/approvals", params={"status": "pending"},
                      headers={"X-Api-Key": BIZ_API_KEY}, timeout=10)
        r.raise_for_status()
        out["pending_approvals"] = r.json().get("items", [])
    except Exception as exc:  # noqa: BLE001
        out["pending_approvals"] = []
        out["approvals_error"] = str(exc)
    return out


def render_markdown(data: dict) -> str:
    lines = [f"**BizAgent 每日巡检**（{data['ts']}）", ""]

    tickets = data.get("overdue_tickets") or []
    lines.append(f"> 超时工单：{len(tickets)} 张" if tickets else "> 超时工单：0 张，SLA 全部在期内")
    for t in tickets[:5]:
        lines.append(f"> - {t['ticket_no']}　{t['type']}　截止 {t['sla_deadline']}")

    low = data.get("low_stock") or []
    lines.append("")
    lines.append(f"> 低库存 SKU：{len(low)} 个" if low else "> 低库存 SKU：0 个")
    for i in low[:5]:
        lines.append(f"> - {i['name']}（{i['sku']}）{i['warehouse']} 剩 {i['qty']}，缺 {i['shortage']}")

    appr = data.get("pending_approvals") or []
    lines.append("")
    lines.append(f"> 待审批：{len(appr)} 张" if appr else "> 待审批：0 张")
    for a in appr[:5]:
        lines.append(f"> - {a['approval_no']}　{a['kind']}　{a.get('payload') or ''}")

    errors = [v for k, v in data.items() if k.endswith("_error") and v]
    if errors:
        lines.append("")
        lines.append("> 采集异常：" + "；".join(str(e) for e in errors))
    return "\n".join(lines)


def push(text: str) -> dict:
    if not WEBHOOK:
        print("[patrol] 未配置 WECOM_WEBHOOK，仅打印：")
        print(text)
        return {"sent": False, "reason": "no webhook"}
    r = httpx.post(WEBHOOK, json={"msgtype": "markdown", "markdown": {"content": text}}, timeout=10)
    return {"sent": r.status_code == 200, "status": r.status_code, "body": r.text[:200]}


def patrol() -> dict:
    data = collect()
    text = render_markdown(data)
    result = push(text)
    audit.log_conversation("patrol", "[每日巡检]", text,
                           [{"tool": "query_tickets"}, {"tool": "low_stock_list"},
                            {"tool": "GET /approvals"}], "ok", False)
    return {"data": data, "text": text, "push": result}


def serve() -> None:
    from apscheduler.schedulers.blocking import BlockingScheduler

    scheduler = BlockingScheduler(timezone=os.getenv("TZ", "Asia/Shanghai"))
    scheduler.add_job(patrol, "cron", hour=CRON_HOUR, minute=CRON_MINUTE,
                      id="daily_patrol", replace_existing=True)
    print(f"[patrol] 已启动，每天 {CRON_HOUR:02d}:{CRON_MINUTE:02d} 巡检一次（Ctrl+C 退出）")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("[patrol] stopped")


if __name__ == "__main__":
    if "--now" in sys.argv:
        r = patrol()
        if r["push"].get("sent"):
            print(r["text"])
        print("push:", r["push"])
    else:
        serve()
