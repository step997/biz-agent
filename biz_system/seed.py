"""Mock 企业业务系统 —— 种子数据。

一次运行造出：
- 30 个客户（普通/VIP/批发）
- 50 个商品（5 个类目）+ 各仓库存（含低于安全线的）
- 200 张订单（状态分布刻意覆盖各环节）+ 订单明细 + 物流单号
- 20 张售后工单（含 3 张已超 SLA）
- 2 张待审批单（退款/调价各一）
- 制度知识库 chunks（来自 docs/policies/*.md）

用法：python -m biz_system.seed [--reset]
"""

from __future__ import annotations

import json
import random
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path

from biz_system.db import DB_PATH, connect, init_db

ROOT = Path(__file__).resolve().parent.parent
POLICY_DIR = ROOT / "docs" / "policies"

SURNAMES = "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
GIVEN = ["伟", "芳", "娜", "敏", "静", "强", "磊", "洋", "艳", "勇", "军", "杰", "娟", "涛", "明", "超", "秀英", "桂英", "建国", "雪"]

CATALOG = {
    "手机配件": [("快充充电器 65W", 89, 42), ("编织数据线 1.5m", 29, 9), ("磁吸充电宝 10000mAh", 159, 88),
             ("无线充电板", 119, 60), ("手机支架 铝合金", 45, 18), ("钢化膜 两片装", 25, 6),
             ("车载支架 出风口", 39, 15), ("蓝牙耳机 半入耳", 199, 110), ("耳机保护套", 19, 4),
             ("多口氮化镓 100W", 219, 120)],
    "小家电": [("桌面风扇 USB", 59, 24), ("迷你加湿器", 79, 33), ("电动牙刷 声波", 149, 70),
            ("电热水杯 300ml", 129, 58), ("折叠电煮锅", 189, 95), ("空气炸锅 3L", 299, 168),
            ("养生壶 1.5L", 199, 105), ("手持挂烫机", 259, 140), ("除螨仪 有线", 329, 190),
            ("电吹风 负离子", 179, 92)],
    "日用百货": [("抽取式纸巾 24 包", 49, 22), ("洗衣凝珠 50 颗", 59, 26), ("垃圾袋 加厚 300 只", 35, 14),
             ("收纳箱 60L", 79, 34), ("厨房抹布 10 条", 25, 8), ("一次性手套 200 只", 19, 6),
             ("拖把 免手洗", 99, 44), ("衣架 20 个装", 29, 10), ("密封罐 三件套", 45, 17),
             ("洗漱杯 硅胶", 22, 7)],
    "食品饮料": [("挂耳咖啡 20 包", 69, 32), ("冻干柠檬片", 39, 16), ("坚果礼盒 750g", 129, 68),
             ("自热米饭 4 盒", 55, 26), ("气泡水 24 罐", 79, 40), ("燕麦片 1kg", 45, 19),
             ("黑巧克力 礼盒", 89, 42), ("蜂蜜 500g", 65, 28), ("茶叶 铁观音 250g", 158, 85),
             ("果干混合装", 42, 18)],
    "文具办公": [("中性笔 20 支", 29, 10), ("A5 笔记本 3 本", 39, 14), ("桌面收纳盒", 55, 22),
             ("标签打印机", 269, 150), ("文件夹 10 个", 25, 8), ("无线鼠标 静音", 79, 35),
             ("键鼠套装", 129, 62), ("USB 集线器 4 口", 49, 20), ("显示器支架", 189, 98),
             ("白板 60x90", 139, 66)],
}

WAREHOUSES = ["华东仓", "华南仓", "华北仓"]
STATUS_WEIGHTS = [("待付款", 20), ("待发货", 60), ("已发货", 80), ("已完成", 30), ("退款中", 6), ("已退款", 4)]
CARRIERS = ["顺丰速运", "中通快递", "德邦物流"]


def _iso(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _chunk_markdown(text: str, doc_id: str, title: str, max_len: int = 300, overlap: int = 50) -> list[dict]:
    """按二级标题切分，超长再按定长+重叠切开。"""
    chunks: list[dict] = []
    seq = 0
    parts = re.split(r"\n(?=## )", text.strip())
    for part in parts:
        part = part.strip()
        if not part or part.startswith("# ") and len(part) < 40:
            continue
        heading = part.splitlines()[0].lstrip("# ").strip()
        body = part[len(part.splitlines()[0]):].strip()
        section = f"{title} · {heading}" if heading else title
        if len(body) <= max_len:
            pieces = [body]
        else:
            pieces, start = [], 0
            while start < len(body):
                pieces.append(body[start:start + max_len])
                start += max_len - overlap
        for piece in pieces:
            if not piece.strip():
                continue
            seq += 1
            chunks.append({
                "doc_id": doc_id,
                "title": title,
                "chunk_id": f"{doc_id}#{seq:02d}",
                "seq": seq,
                "content": f"{section}\n{piece.strip()}",
            })
    return chunks


def load_policy_chunks() -> list[dict]:
    chunks: list[dict] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        first_line = text.splitlines()[0]
        title = first_line.lstrip("# ").strip() if first_line.startswith("# ") else path.stem
        chunks.extend(_chunk_markdown(text, path.stem, title))
    return chunks


def seed(reset: bool = True) -> dict:
    init_db(reset=reset)
    rng = random.Random(42)
    now = datetime.now()
    conn = connect()
    stats: dict[str, int] = {}
    try:
        cur = conn.cursor()

        # ---- 客户 ----
        customers = []
        for i in range(1, 31):
            name = rng.choice(SURNAMES)[0] + rng.choice(GIVEN)
            level = "普通" if i % 5 else ("VIP" if i % 2 else "批发")
            customers.append((i, name, f"138{rng.randint(10000000, 99999999)}", level,
                              _iso(now - timedelta(days=rng.randint(30, 700)))))
        cur.executemany("INSERT INTO customers VALUES (?,?,?,?,?)", customers)
        stats["customers"] = len(customers)

        # ---- 商品 ----
        products, pid = [], 1
        for category, items in CATALOG.items():
            for name, price, cost in items:
                sku = f"{category[:2]}{pid:04d}"
                products.append((pid, sku, name, category, float(price), float(cost)))
                pid += 1
        cur.executemany("INSERT INTO products VALUES (?,?,?,?,?,?)", products)
        stats["products"] = len(products)

        # ---- 库存（刻意留 6 个低于安全线）----
        inv = []
        low_seed = {3, 11, 18, 24, 37, 44}
        for p in products:
            for wh in WAREHOUSES[: rng.randint(1, 3)]:
                safety = rng.choice([10, 15, 20, 30])
                qty = rng.randint(1, 12) if p[0] in low_seed and wh == WAREHOUSES[0] else rng.randint(30, 400)
                inv.append((p[0], wh, qty, safety, _iso(now - timedelta(hours=rng.randint(1, 72)))))
        cur.executemany("INSERT INTO inventory VALUES (?,?,?,?,?)", inv)
        stats["inventory_rows"] = len(inv)

        # ---- 订单 ----
        status_pool: list[str] = []
        for status, weight in STATUS_WEIGHTS:
            status_pool.extend([status] * weight)
        orders, items, ticket_seed = [], [], []
        order_no_by_id: dict[int, str] = {}
        for i in range(200):
            oid = i + 1
            order_no = f"SO{10001 + i}"
            order_no_by_id[oid] = order_no
            customer = rng.choice(customers)
            created = now - timedelta(days=rng.randint(0, 60), hours=rng.randint(0, 23),
                                      minutes=rng.randint(0, 59))
            status = status_pool[i % len(status_pool)]
            picked = rng.sample(products, rng.randint(1, 3))
            line = [(oid, p[0], rng.randint(1, 4), p[4]) for p in picked]
            amount = round(sum(q * up for _, _, q, up in line), 2)
            paid_at = None if status == "待付款" else _iso(created + timedelta(minutes=rng.randint(1, 120)))
            shipped_at = _iso(created + timedelta(hours=rng.randint(12, 60))) if status in ("已发货", "已完成") else None
            logistics_no = f"SF{rng.randint(10**11, 10**12 - 1)}" if shipped_at else None
            carrier = rng.choice(CARRIERS) if shipped_at else None
            orders.append((oid, order_no, customer[0], status, amount, _iso(created),
                           paid_at, shipped_at, logistics_no, carrier))
            items.extend(line)
            if status in ("已发货", "已完成"):
                ticket_seed.append(oid)
        # 给 3 个"主角客户"补近 7 天的订单，保证"查某人上周订单"这类演示有数据
        hero_extra = [
            (customers[0], "待发货", 3), (customers[0], "已发货", 2),
            (customers[1], "已发货", 2), (customers[1], "已完成", 1),
            (customers[2], "退款中", 1), (customers[2], "待发货", 2),
        ]
        oid = 200
        for customer, status, n in hero_extra:
            for _ in range(n):
                oid += 1
                order_no = f"SO{10001 + oid - 1}"
                order_no_by_id[oid] = order_no
                created = now - timedelta(days=rng.randint(0, 6), hours=rng.randint(0, 20))
                picked = rng.sample(products, rng.randint(1, 2))
                line = [(oid, p[0], rng.randint(1, 3), p[4]) for p in picked]
                amount = round(sum(q * up for _, _, q, up in line), 2)
                paid_at = _iso(created + timedelta(minutes=rng.randint(2, 90)))
                shipped_at = _iso(created + timedelta(hours=rng.randint(6, 30))) if status in ("已发货", "已完成") else None
                logistics_no = f"SF{rng.randint(10**11, 10**12 - 1)}" if shipped_at else None
                orders.append((oid, order_no, customer[0], status, amount, _iso(created),
                               paid_at, shipped_at, logistics_no,
                               rng.choice(CARRIERS) if shipped_at else None))
                items.extend(line)

        cur.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?,?,?,?,?)", orders)
        cur.executemany("INSERT INTO order_items (order_id, product_id, qty, unit_price) VALUES (?,?,?,?)", items)
        stats["orders"] = len(orders)
        stats["order_items"] = len(items)

        # ---- 工单（定额 20 张，其中 3 张超时）----
        tickets = []
        types = ["物流", "质量", "退款", "咨询"]
        sla_hours = {"物流": 4, "咨询": 8, "质量": 24, "退款": 24}
        pool = ticket_seed or [o[0] for o in orders]
        step = max(1, len(pool) // 20)
        chosen = pool[::step][:20]
        for idx, oid in enumerate(chosen, start=1):
            ttype = types[idx % len(types)]
            sla = sla_hours[ttype]
            if idx <= 3:
                # 前 3 张：SLA 已超时且仍未解决（巡检脚本要能抓出来）
                status = "处理中"
                created = now - timedelta(hours=sla + rng.randint(2, 12))
                resolved = None
            else:
                status = "已解决" if idx % 4 == 0 else ("处理中" if idx % 3 else "待处理")
                # 未超时：创建时间落在 "距截止还有 10%~70% SLA" 的区间内
                created = now - timedelta(hours=round(sla * rng.uniform(0.3, 0.9), 1))
                resolved = _iso(created + timedelta(hours=round(sla * 0.5, 1))) if status == "已解决" else None
            deadline = created + timedelta(hours=sla)
            tickets.append((idx, f"TK{20000 + idx}", oid, ttype,
                            f"客户反馈{ttype}问题：订单 {order_no_by_id.get(oid, '未知')} 需要处理", status,
                            _iso(deadline), _iso(created), resolved))
        cur.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)", tickets)
        stats["tickets"] = len(tickets)

        # ---- 待审批单 2 张 ----
        approvals = [
            (1, "AP20261116001", "退款",
             json.dumps({"order_no": orders[5][1], "amount": orders[5][4], "reason": "客户称商品破损"}, ensure_ascii=False),
             "pending", "agent", None, None, _iso(now - timedelta(hours=30))),
            (2, "AP20261116002", "调价",
             json.dumps({"sku": products[7][1], "from": products[7][4], "to": round(products[7][4] * 0.9, 2),
                         "reason": "清库存"}, ensure_ascii=False),
             "pending", "agent", None, None, _iso(now - timedelta(hours=52))),
        ]
        cur.executemany("INSERT INTO approvals VALUES (?,?,?,?,?,?,?,?,?)", approvals)
        stats["approvals"] = len(approvals)

        # ---- 知识库 chunks ----
        chunks = load_policy_chunks()
        cur.executemany(
            "INSERT INTO policies (doc_id, title, chunk_id, seq, content) VALUES (?,?,?,?,?)",
            [(c["doc_id"], c["title"], c["chunk_id"], c["seq"], c["content"]) for c in chunks],
        )
        stats["policy_chunks"] = len(chunks)

        conn.commit()
    finally:
        conn.close()
    stats["db"] = str(DB_PATH)
    return stats


if __name__ == "__main__":
    result = seed(reset="--keep" not in sys.argv)
    for k, v in result.items():
        print(f"{k:>16}: {v}")
