"""规划器：把自然语言问题拆成工具调用序列。

两种实现共用同一套工具与同一套输出结构：
- offline：正则/关键词规则（确定性、可 pytest、无 key 也能演示）
- llm：DeepSeek function calling，多轮（最多 3 轮）直到能回答
"""

from __future__ import annotations

import json
import re
from typing import Any

from agent import llm
from tools.base import ToolRegistry

ORDER_NO = re.compile(r"SO\d{5,}")
CN_NUM = {"一": 1, "二": 2, "三": 3, "两": 2, "五": 5, "七": 7, "十": 10, "半个月": 15}
POLICY_HINTS = ("制度", "政策", "规定", "流程", "几天", "多久", "多少天", "能不能退", "可不可以退",
                "运费", "发票", "时效", "标准", "会员", "批发", "破损", "催付", "规则", "怎么算")
WRITE_HINTS = {"退款": "退款", "补发": "补发", "调价": "调价", "改价": "调价"}
# 写操作只在这些"人真的在让我办事"的说法下才允许，光提"退款"两个字不算
WRITE_ACTIONS = ("提交", "申请", "发起", "走审批", "走个审批", "报批", "帮我退", "给我退", "给我办",
                 "帮我建", "帮我开", "建个", "开个", "登记一下", "处理一下", "批一下", "开单", "建单")
STOP_PROD = {"多少", "还有", "一下", "什么", "这个", "那个", "现在", "我们", "商品", "东西", "所有"}

SYSTEM_PROMPT = """你是「BizAgent」，某中小电商企业内部业务智能体，服务于客服与运营。
你可以调用工具查询订单、物流、库存、工单，检索公司制度，创建工单与审批单。

硬性规则：
1. 任何业务数据都必须来自工具返回，绝不能凭常识推测订单号、金额、库存数字。
2. 制度类问题只用 query_policy 的结果回答，并附 citations。
3. 退款/补发/调价只能通过 create_approval 提交审批，不能承诺"已经给你退了"。
3.1 用户明确提出"提交退款/补发/调价审批"（通常带金额或订单号）时，直接调用 create_approval，
    把用户给的金额、订单号、原因填进去；字段缺失也要先建单（order_no 可留空），不要先反问。
4. 工具返回为空就如实说明没查到，并建议下一步（换关键词、转人工、建单跟进）。
5. 回答简洁，用中文，先给结论再给依据。涉及金额、单号、时间要写具体数字。
6. 只调用真正需要的那一个工具，**同一个工具不要重复调用**（参数相同更是禁止重复）。
   制度/规则/流程类问题（退货、发票、运费、时效、会员、批发、赔付、催收）只调 query_policy 一次，
   不要顺手再查库存或订单。
7. 已经拿到足够数据就直接停，不要为了"多确认一下"再加调用。
8. 先判断这是「问规则」还是「问具体数据」，再选工具：
   - 问规则 = 问条件/时效/费用/流程/标准，句子里没有具体单号（例如"寄过来大概要几天"、
     "老客户有优惠吗"、"钱多久退回来"）→ 这是 query_policy，**不要反问订单号**。
   - 问具体数据 = 带单号/客户名/商品名，问"这单到哪了""某人下了几单"→ 查数工具。
9. 用户只问规则时，不要因为缺少订单号而要求补充信息，直接查制度回答。
10. 上面的历史对话只用来理解指代（"那这个单呢""就那个"），**不能当数据源**：
   涉及订单/库存/工单的当前状态，即使历史里刚查过，也要重新调用工具拿最新值。
"""


# ─────────────────────── offline 规则规划 ───────────────────────


def _extract_customer(q: str) -> str | None:
    m = re.search(r"(?:查|看|找|查询)(?:一下|下)?\s*([\u4e00-\u9fff]{2,3}?)"
                  r"(?=的|最近|上周|这周|本周|近|今天|昨天|下单|买了|下过|一|二|三|四|五|六|七|八|九|十|\d|$|，|,|\s)", q)
    if m:
        cand = m.group(1)
        if cand not in ("订单", "客户", "所有", "全部", "库存", "工单", "数据"):
            return cand
    m = re.search(r"客户\s*(?:叫)?([\u4e00-\u9fff]{2,3})", q)
    return m.group(1) if m else None


def _extract_days(q: str) -> int | None:
    if m := re.search(r"(\d+)\s*天", q):
        return int(m.group(1))
    if "上周" in q or "最近一周" in q or "这周" in q or "本周" in q:
        return 7
    if "今天" in q or "今日" in q:
        return 1
    for cn, n in CN_NUM.items():
        if cn in q and ("天" in q or "日" in q):
            return n
    return None


def _extract_amount(q: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:元|块)", q)
    return float(m.group(1)) if m else None


def _product(q: str) -> str | None:
    if m := re.search(r"(SKU\d{3,6})", q, re.I):
        return m.group(1)
    head = re.split(r"(库存|有货|没货|缺货|够不够|还剩多少)", q)[0]
    head = re.sub(r"^(帮我|请|麻烦|你|我想|我要)?(查一下|查查|查询|查|看看|看|问一下|问)?", "", head)
    prev = None
    while prev != head:
        prev = head
        head = re.sub(r"(还有|还剩|剩下|剩余|有|的|多少|剩|一下|吗|呢|？|\?|，|,|\s)+$", "", head)
    head = head.strip("，,。. 的")
    if len(head) >= 2 and head not in STOP_PROD:
        return head[-8:]
    return None


def offline_plan(question: str, loop: int, trace: list[dict]) -> list[dict]:
    q = question
    order_no = ORDER_NO.search(q)
    order = order_no.group(0) if order_no else None
    calls: list[dict] = []

    if loop == 0:
        # 1) 建工单（明确动作，优先级最高）
        ticket_action = ("工单" in q and any(k in q for k in ("建", "开", "报", "登记", "记录", "新建"))) or \
                        any(k in q for k in ("建个单", "建一个单", "建单", "报单", "记录一下", "投诉", "登记"))
        if ticket_action:
            ttype = "物流" if "物流" in q else ("质量" if any(k in q for k in ("破损", "坏了", "坏了", "质量问题", "少发", "错发")) else
                                               ("退款" if "退款" in q else "咨询"))
            return [{"tool": "create_ticket", "args": {"content": q[:200], "type": ttype, "order_no": order}}]
        # 2) 审批类写操作（含"提交/申请"等动作词）
        write_intent = any(k in q for k in ("提交", "申请", "走审批", "发起", "帮我退", "给我退", "处理一下"))
        if write_intent:
            for key, kind in WRITE_HINTS.items():
                if key in q:
                    args: dict[str, Any] = {"kind": kind, "order_no": order, "reason": q[:80]}
                    if (amt := _extract_amount(q)) is not None:
                        args["amount"] = amt
                    return [{"tool": "create_approval", "args": args}]
        # 3) 制度类（问"规定/几天/标准"这类，且不是在让我执行）
        if any(h in q for h in POLICY_HINTS) and not (order and ("物流" in q or "到哪" in q)):
            return [{"tool": "query_policy", "args": {"question": q, "top_k": 3}}]
        # 4) 工单/超时
        if any(k in q for k in ("工单", "超时", "SLA", "sla", "超期")):
            return [{"tool": "query_tickets",
                     "args": {"overdue_only": ("超时" in q or "超期" in q or "SLA" in q)}}]
        # 5) 缺货/补货
        if any(k in q for k in ("缺货", "低于安全库存", "补货", "库存预警")):
            return [{"tool": "low_stock_list", "args": {"limit": 10}}]
        # 6) 单品库存
        if "库存" in q or "有货" in q or "没货" in q:
            if prod := _product(q):
                return [{"tool": "check_inventory", "args": {"product": prod}}]
        # 7) 物流
        if any(k in q for k in ("物流", "到哪", "签收", "快递", "配送")):
            if order:
                return [{"tool": "get_logistics", "args": {"order_no": order}}]
            if name := _extract_customer(q):
                return [{"tool": "query_orders", "args": {"customer_name": name, "limit": 5}}]
        # 8) 指定订单详情
        if order:
            return [{"tool": "get_order_detail", "args": {"order_no": order}}]
        # 9) 按客户查订单
        if name := _extract_customer(q):
            args = {"customer_name": name, "limit": 10}
            if days := _extract_days(q):
                args["since_days"] = days
            if "待发货" in q:
                args["status"] = "待发货"
            elif "已发货" in q:
                args["status"] = "已发货"
            return [{"tool": "query_orders", "args": args}]
        if "待发货" in q or "退款中" in q or "已发货" in q:
            status = "退款中" if "退款中" in q else ("已发货" if "已发货" in q else "待发货")
            return [{"tool": "query_orders", "args": {"status": status, "limit": 10}}]
        # 10) 兜底：先查制度（最可能的意图），并让回答里说明怎么问更准
        return [{"tool": "query_policy", "args": {"question": q, "top_k": 3}}]

    # 第二轮：把"订单+物流"这种复合问题补齐
    if trace:
        last = trace[-1]
        if last["tool"] == "query_orders" and ("物流" in q or "到哪" in q):
            orders = (last.get("data") or {}).get("orders") or []
            if len(orders) == 1:
                return [{"tool": "get_logistics", "args": {"order_no": orders[0]["order_no"]}}]
        if last["tool"] == "query_orders" and ("详情" in q or "明细" in q):
            orders = (last.get("data") or {}).get("orders") or []
            if len(orders) == 1:
                return [{"tool": "get_order_detail", "args": {"order_no": orders[0]["order_no"]}}]
        if last["tool"] == "query_orders" and ("建单" in q or "报单" in q):
            orders = (last.get("data") or {}).get("orders") or []
            if orders:
                return [{"tool": "create_ticket", "args": {"content": q[:200], "type": "物流",
                                                           "order_no": orders[0]["order_no"]}}]
    return []


# ─────────────────────── LLM 规划 ───────────────────────


def llm_plan(question: str, loop: int, trace: list[dict], registry: ToolRegistry,
             history: list[dict] | None = None) -> tuple[list[dict], str | None]:
    """返回 (待执行工具调用, 若已可直接给答案则该答案为字符串)。"""
    model = llm.get_llm().bind_tools(registry.schemas())
    msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    # 带上最近几轮问答：用户经常用「1」「就那个」「建吧」这种省略句，没有上文就无从判断
    for turn in (history or [])[-3:]:
        q = (turn.get("question") or "").strip()
        a = (turn.get("answer") or "").strip()[:300]
        if not q:
            continue
        msgs.append({"role": "user", "content": q})
        if a:
            msgs.append({"role": "assistant", "content": a})
    msgs.append({"role": "user", "content": question})
    if trace:
        msgs.append({"role": "user", "content": "已获得的工具结果（JSON）：\n" +
                     json.dumps(trace, ensure_ascii=False, default=str)[:6000]})
    ai = model.invoke(msgs)
    calls = [
        {"tool": tc["name"], "args": tc.get("args") or {}}
        for tc in (getattr(ai, "tool_calls", None) or [])
    ]
    if calls:
        return calls[:2], None
    return [], (ai.content or "").strip()


def is_policy_question(question: str) -> bool:
    """问的是"规则本身"（怎么规定、几天、能不能），而不是"帮我办这件事"。"""
    q = question or ""
    if any(h in q for h in POLICY_HINTS):
        return not any(a in q for a in WRITE_ACTIONS)
    return False


def wants_write(question: str) -> bool:
    q = question or ""
    return any(a in q for a in WRITE_ACTIONS) or any(k in q for k in ("破损", "坏了", "少发", "错发", "漏发"))


def guard_plan(plan: list[dict], question: str, registry: ToolRegistry) -> tuple[list[dict], list[str]]:
    """代码级护栏：模型不该在"只问规则"时动写操作。

    多轮记忆会把上一轮的"提交退款审批"带到下一轮制度问答上，光靠 prompt 约束不可靠，
    所以在这里按工具类型硬性过滤，返回 (保留的调用, 被拦下的工具名)。
    """
    if not plan:
        return plan, []
    kinds = {t.name: t.kind for t in registry.tools.values()}
    policy_only = is_policy_question(question) and not wants_write(question)
    if not policy_only:
        return plan, []
    kept, dropped = [], []
    for call in plan:
        if kinds.get(call.get("tool")) == "write":
            dropped.append(call.get("tool", ""))
        else:
            kept.append(call)
    return kept, dropped


def make_plan(question: str, loop: int, trace: list[dict], registry: ToolRegistry,
              history: list[dict] | None = None) -> tuple[list[dict], str | None]:
    if llm.mode() == "llm" and llm.available():
        try:
            return llm_plan(question, loop, trace, registry, history)
        except Exception:  # noqa: BLE001 —— 大模型抖动时退回规则规划，保证 demo 可用
            return offline_plan(question, loop, trace), None
    return offline_plan(question, loop, trace), None
