"""RAG 评测：正样本命中 + 负样本拒答 + 阈值扫描，支持三种后端横向对比。

用法：
    python scripts/eval_rag.py                  # 单后端（RAG_BACKEND，默认 auto）
    python scripts/eval_rag.py --compare        # tfidf / embedding / hybrid 三档对比
    python scripts/eval_rag.py --backend hybrid --sweep

输出可直接粘进 README / 面试材料：Top1、Top3、负样本拒答率、分数分布、阈值扫描。
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.corpus import load_chunks  # noqa: E402
from rag.index import HybridIndex  # noqa: E402

# 每条：(问句, 期望命中的文档)
POSITIVES = [
    ("退款几天内可以无理由退货", "return_policy"),
    ("退款多久能到账", "return_policy"),
    ("退货运费谁承担", "return_policy"),
    ("能开增值税专用发票吗", "invoice_policy"),
    ("发票抬头写错了怎么改", "invoice_policy"),
    ("下单后多久发货", "shipping_policy"),
    ("默认发什么快递", "shipping_policy"),
    ("物流工单多长时间必须处理完", "sla_policy"),
    ("哪些工单算超时", "sla_policy"),
    ("VIP 会员有什么权益", "member_policy"),
    ("批发客户起订量和价格怎么算", "wholesale_policy"),
    ("运输途中商品破损怎么赔付", "damage_policy"),
    ("客户一直不付款怎么催收", "urge_policy"),
]

# 同义改写：词面几乎不含原始术语 —— 这一组专门用来暴露「纯词法检索」的短板
# 期望值可以是 tuple：制度边界本来就有交叠（"东西坏了想退"退货与理赔两篇都对）
PARAPHRASES = [
    ("买的东西坏了想退怎么办", ("damage_policy", "return_policy")),
    ("钱多久能退回我卡上", "return_policy"),
    ("寄到我这儿大概要几天", "shipping_policy"),
    ("能不能给开个公司抬头的票", "invoice_policy"),
    ("老客户打折吗", "member_policy"),
    ("催了好几次还不打款", "urge_policy"),
]

NEGATIVES = [
    "今天杭州天气怎么样",
    "帮我写一首关于秋天的诗",
    "公司的股票代码是多少",
    "SO10204 物流到哪了",
    "王伟最近一周下了几单",
]


def run(idx: HybridIndex, positives, negatives, *, show: bool) -> dict:
    top1 = top3 = 0
    dense_pos = []
    t0 = time.perf_counter()
    for q, expected in positives:
        exp = (expected,) if isinstance(expected, str) else tuple(expected)
        hits, conf = idx.search(q, top_k=3)
        docs = [h.doc_id for h in hits]
        ok1 = bool(docs) and docs[0] in exp
        ok3 = any(d in exp for d in docs)
        top1 += ok1
        top3 += ok3
        dense_pos.append(hits[0].dense if hits else 0.0)
        if show:
            print(f"{'OK ' if ok3 else 'MISS'} top1={'OK ' if ok1 else 'NO '} {q[:22]:<24} "
                  f"-> {docs[:3]} conf={conf} dense={(hits[0].dense if hits else 0):.3f}")
    elapsed = (time.perf_counter() - t0) * 1000 / max(1, len(positives))

    neg_rej = 0
    dense_neg = []
    for q in negatives:
        hits, conf = idx.search(q, top_k=3)
        rej = conf == "low"
        neg_rej += rej
        top_dense = max((h.dense for h in hits), default=0.0)
        dense_neg.append(top_dense)
        if show:
            print(f"{'REJ' if rej else 'PASS'} {q[:24]:<26} conf={conf} "
                  f"top={hits[0].doc_id if hits else '-'} dense={top_dense:.3f}")

    n, m = len(positives), len(negatives)
    return {
        "backend": idx.backend,
        "top1": top1 / n, "top3": top3 / n, "neg_rej": neg_rej / m,
        "pos_min": min(dense_pos), "pos_max": max(dense_pos),
        "neg_min": min(dense_neg), "neg_max": max(dense_neg),
        "ms_per_query": elapsed,
    }


def sweep(idx: HybridIndex, positives, negatives, thresholds) -> None:
    dense_pos, dense_neg = [], []
    for q, _ in positives:
        hits, _c = idx.search(q, top_k=3)
        dense_pos.append(hits[0].dense if hits else 0.0)
    for q in negatives:
        hits, _c = idx.search(q, top_k=3)
        dense_neg.append(max((h.dense for h in hits), default=0.0))
    print(f"\n── 阈值扫描 [{idx.backend}] ──")
    print(f"{'thr':>6} {'正样本接受':>10} {'负样本误收':>10}")
    for thr in thresholds:
        pos_acc = sum(1 for s in dense_pos if s >= thr) / len(dense_pos)
        neg_fp = sum(1 for s in dense_neg if s >= thr) / len(dense_neg)
        print(f"{thr:>6.2f} {pos_acc:>10.0%} {neg_fp:>10.0%}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=os.getenv("RAG_BACKEND", "auto"),
                    choices=["auto", "tfidf", "embedding", "hybrid"])
    ap.add_argument("--compare", action="store_true", help="跑 tfidf/embedding/hybrid 三档对比")
    ap.add_argument("--sweep", action="store_true", help="打印阈值扫描表")
    args = ap.parse_args()

    chunks = load_chunks()
    print(f"chunks={len(chunks)} docs={len({c['doc_id'] for c in chunks})}")

    def build(backend: str) -> HybridIndex:
        t = time.perf_counter()
        idx = HybridIndex(chunks, backend=backend)
        print(f"[{backend}] 建索引 {1000 * (time.perf_counter() - t):.1f}ms")
        return idx

    if args.compare:
        rows = []
        for be in ("tfidf", "embedding", "hybrid"):
            if be != "tfidf":
                from rag import embedder

                if not embedder.available():
                    print(f"[{be}] 跳过：{embedder.failure_reason()}")
                    continue
            idx = build(be)
            print(f"\n═══ {be} ═══")
            print("— 原始正样本 —")
            rows.append({"set": "原始", **run(idx, POSITIVES, NEGATIVES, show=True)})
            print("— 同义改写 —")
            rows.append({"set": "改写", **run(idx, PARAPHRASES, NEGATIVES, show=True)})
            if args.sweep:
                sweep(idx, POSITIVES + PARAPHRASES, NEGATIVES,
                      (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70))

        print("\n\n═══ 对比汇总 ═══")
        print(f"{'backend':<10}{'样本集':<8}{'Top1':>7}{'Top3':>7}{'负样本拒答':>11}"
              f"{'正样本dense区间':>18}{'负样本dense上限':>16}{'ms/query':>10}")
        for r in rows:
            pos = f"{r['pos_min']:.2f}~{r['pos_max']:.2f}"
            print(f"{r['backend']:<10}{r['set']:<8}{r['top1']:>7.0%}{r['top3']:>7.0%}"
                  f"{r['neg_rej']:>11.0%}{pos:>18}{r['neg_max']:>16.2f}{r['ms_per_query']:>10.1f}")
        return 0

    from rag.index import resolve_backend

    be = resolve_backend(args.backend)
    idx = build(be)
    print("\n── 原始正样本 ──")
    a = run(idx, POSITIVES, NEGATIVES, show=True)
    print("\n── 同义改写（词法检索的短板）──")
    b = run(idx, PARAPHRASES, NEGATIVES, show=False)
    print(f"改写集 Top1 {b['top1']:.0%}  Top3 {b['top3']:.0%}  负样本拒答 {b['neg_rej']:.0%}")
    print(f"\n[{be}] Top1 {a['top1']:.0%}　Top3 {a['top3']:.0%}　负样本拒答 {a['neg_rej']:.0%}"
          f"　正样本 dense {a['pos_min']:.3f}~{a['pos_max']:.3f}"
          f"　负样本 ≤{a['neg_max']:.3f}　{a['ms_per_query']:.1f}ms/query")
    if args.sweep:
        sweep(idx, POSITIVES + PARAPHRASES, NEGATIVES,
              (0.20, 0.30, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
