"""RAG 评测（写进 CI：指标退化就是回归）。"""

from __future__ import annotations

import pytest

from rag.index import answer_context, get_index

POSITIVES = [
    ("退款几天内可以无理由退货", "return_policy"),
    ("退货运费谁承担", "return_policy"),
    ("能开增值税专用发票吗", "invoice_policy"),
    ("发票抬头写错了怎么改", "invoice_policy"),
    ("默认发什么快递", "shipping_policy"),
    ("哪些工单算超时", "sla_policy"),
    ("VIP 会员有什么权益", "member_policy"),
    ("批发客户起订量和价格怎么算", "wholesale_policy"),
    ("运输途中商品破损怎么赔付", "damage_policy"),
    ("客户一直不付款怎么催收", "urge_policy"),
]
NEGATIVES = ["今天杭州天气怎么样", "帮我写一首关于秋天的诗", "公司的股票代码是多少",
             "SO10204 物流到哪了", "王伟最近一周下了几单"]


def test_index_is_built_from_business_db():
    idx = get_index()
    assert len(idx.chunks) >= 30
    assert len({c["doc_id"] for c in idx.chunks}) >= 8


@pytest.mark.parametrize("question,doc", POSITIVES)
def test_policy_top3_recall(question, doc):
    ctx = answer_context(question, top_k=3)
    docs = [h["chunk_id"].split("#")[0] for h in ctx["hits"]]
    assert doc in docs, f"{question} 未召回 {doc}"
    assert ctx["confidence"] != "low"


@pytest.mark.parametrize("question", NEGATIVES)
def test_negative_is_rejected(question):
    ctx = answer_context(question)
    assert ctx["confidence"] == "low", f"{question} 不该被判为可信"
    assert ctx["covered"] is False


def test_hits_are_ranked_and_scored():
    ctx = answer_context("退货规则", top_k=3)
    scores = [h["score"] for h in ctx["hits"]]
    assert scores == sorted(scores, reverse=True)
    assert all(h["chunk_id"] and h["citation"] for h in ctx["hits"])
