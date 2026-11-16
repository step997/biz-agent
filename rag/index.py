"""混合检索：词法（BM25 over 字 bigram）+ 语义（TF-IDF 余弦 / bge-small-zh 向量）+ RRF 融合 + 两档置信阈值。

三种后端，同一套接口（`search()` / `answer_context()`），由 `RAG_BACKEND` 选择：

- `tfidf`     BM25 + TF-IDF 余弦 RRF —— 零额外依赖，34MB 环境也能跑（pytest 默认走这条）
- `embedding` 纯语义向量（bge-small-zh-v1.5 + FAISS 内积索引），命中间距最宽
- `hybrid`    BM25 + 语义向量 RRF —— 生产推荐，词面精确与同义改写都兜得住

为什么默认 `auto` = 有模型就走 hybrid、没模型退回 tfidf：demo 要能在无 GPU、无 torch 的机器上
一键起来，但也不能因此放弃语义召回。降级路径不写在文档里，写在代码里。
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from rag import embedder
from rag.corpus import load_chunks

RRF_K = 60
RRF_DENSE_W = 1.2   # 向量序比词法序略可信，融合时加权

# 三档后端的置信阈值不一样：TF-IDF 余弦天然偏低（正样本 0.19~0.48），
# bge 归一化向量的正样本余弦在 0.55~0.85，混合同理。用同一组阈值会误判。
THRESHOLDS = {
    "tfidf": (0.42, 0.20),       # (STRONG, WEAK)
    "embedding": (0.62, 0.45),
    "hybrid": (0.58, 0.40),
}
STRONG, WEAK = THRESHOLDS["tfidf"]   # 兼容旧引用

# 领域门：问句里连一个业务/制度词都没有，就不该在制度库里"硬答"
DOMAIN_TERMS = ("退", "换货", "售后", "发货", "快递", "物流", "运费", "发票", "开票", "税",
                "工单", "超时", "时效", "标准", "会员", "VIP", "vip", "积分", "批发", "起订",
                "破损", "赔付", "理赔", "催收", "催付", "付款", "账期", "库存", "订单", "SLA", "sla")

# 结构化查询门：带单号/工单号/审批号的问句属于「查具体数据」，知识库不该接，转工具。
# 语义检索对这类句子分数会偏高（"SO10204 物流到哪了" 与发货制度语义很近），必须用规则拦。
STRUCTURED = re.compile(r"(SO\d{4,}|TK\d{4,}|AP\d{6,}|订单号|工单号|审批号|几单|下了几|这一单|这个单)")

# 域外门：语义后端对短句普遍给 0.45~0.55 的"礼貌分"，光靠阈值分不开
# （"今天杭州天气怎么样"=0.507，"寄到我这儿大概要几天"=0.529，就差 0.02）。
# 所以补一张域外词表：命中即判定越域，成本几乎为零，专治这类误收。
OFF_DOMAIN_TERMS = ("天气", "气温", "股票", "股价", "汇率", "写一首", "写首诗", "作诗", "作文",
                    "笑话", "讲个故事", "新闻", "几点", "星期几", "翻译成", "帮我写代码", "热搜")

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """中文按字 bigram（不依赖分词器）+ 单字 + 英文/数字词。"""
    text = text.lower()
    chars = _CJK.findall(text)
    bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    words = _LATIN.findall(text)
    return bigrams + chars + words


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    title: str
    seq: int
    content: str
    bm25: float = 0.0
    dense: float = 0.0
    rrf: float = 0.0

    @property
    def citation(self) -> str:
        return f"[{self.doc_id}#{self.seq:02d}]"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id, "title": self.title, "citation": self.citation,
            "content": self.content, "score": round(self.rrf, 5),
            "bm25": round(self.bm25, 3), "dense": round(self.dense, 3),
        }


@dataclass
class HybridIndex:
    chunks: list[dict]
    backend: str = "tfidf"       # tfidf | embedding | hybrid
    k1: float = 1.5
    b: float = 0.75
    _tokens: list[list[str]] = field(default_factory=list)   # bigram+字+词，用于 BM25
    _df: Counter = field(default_factory=Counter)
    _avgdl: float = 0.0
    _idf: dict[str, float] = field(default_factory=dict)
    _tfs: list[Counter] = field(default_factory=list)        # 用于 TF-IDF 余弦
    _norms: list[float] = field(default_factory=list)
    _emb: np.ndarray | None = None                            # (n, dim) 归一化语义向量
    _faiss: object | None = None

    def __post_init__(self) -> None:
        n = max(1, len(self.chunks))
        self._tfs = []
        for c in self.chunks:
            toks = tokenize(f"{c['title']} {c['content']}")
            self._tfs.append(Counter(toks))
            self._df.update(set(toks))
        self._tokens = [list(t.elements()) for t in self._tfs]
        self._avgdl = sum(len(t) for t in self._tokens) / max(1, len(self._tokens))
        self._idf = {term: math.log(1 + n / (df + 0.5)) for term, df in self._df.items()}
        self._norms = []
        for tf in self._tfs:
            norm = math.sqrt(sum((c * self._idf.get(t, 0.0)) ** 2 for t, c in tf.items())) or 1.0
            self._norms.append(norm)
        if self.backend in ("embedding", "hybrid"):
            self._emb = embedder.chunk_matrix(self.chunks)
            self._faiss = _build_faiss(self._emb)

    # ---- 词法分 ----
    def _bm25(self, query: str) -> list[float]:
        q = tokenize(query)
        n = max(1, len(self._tokens))
        scores = []
        for tf, toks in zip(self._tfs, self._tokens):
            dl = len(toks) or 1
            s = 0.0
            for term in q:
                f = tf.get(term, 0)
                if not f:
                    continue
                df = self._df.get(term, 0) or n
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / self._avgdl))
            scores.append(s)
        return scores

    # ---- 语义分 ----
    def _dense(self, query: str) -> list[float]:
        if self.backend == "tfidf":
            return self._dense_tfidf(query)
        return self._dense_embedding(query)

    def _dense_tfidf(self, query: str) -> list[float]:
        q_tf = Counter(tokenize(query))
        q_w = {t: c * self._idf.get(t, 0.0) for t, c in q_tf.items() if t in self._idf}
        q_norm = math.sqrt(sum(v * v for v in q_w.values())) or 1.0
        out = []
        for tf, norm in zip(self._tfs, self._norms):
            dot = sum(w * (tf.get(t, 0) * self._idf.get(t, 0.0)) for t, w in q_w.items())
            out.append(dot / (q_norm * norm))
        return out

    def _dense_embedding(self, query: str) -> list[float]:
        """bge 语义向量与 chunk 矩阵的内积（向量已 L2 归一化 → 内积即余弦）。"""
        q = embedder.encode_query(query).astype(np.float32).reshape(1, -1)
        if self._faiss is not None:
            scores, idxs = self._faiss.search(q, len(self.chunks))  # type: ignore[attr-defined]
            out = [0.0] * len(self.chunks)
            for score, i in zip(scores[0], idxs[0]):
                if i >= 0:
                    out[int(i)] = float(score)
            return out
        return [float(x) for x in (self._emb @ q[0])]  # type: ignore[union-attr]

    def search(self, query: str, top_k: int = 3) -> tuple[list[Hit], str]:
        if not self.chunks or not query.strip():
            return [], "low"
        bm25 = self._bm25(query) if self.backend != "embedding" else [0.0] * len(self.chunks)
        dense = self._dense(query)
        order_den = sorted(range(len(self.chunks)), key=lambda i: dense[i], reverse=True)
        rrf: dict[int, float] = {}
        if self.backend != "embedding":
            order_lex = sorted(range(len(self.chunks)), key=lambda i: bm25[i], reverse=True)
            for rank, i in enumerate(order_lex):
                rrf[i] = rrf.get(i, 0) + 1.0 / (RRF_K + rank + 1)
        for rank, i in enumerate(order_den):
            rrf[i] = rrf.get(i, 0) + RRF_DENSE_W / (RRF_K + rank + 1)

        ranked = sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        hits = []
        for i, score in ranked:
            c = self.chunks[i]
            hits.append(Hit(chunk_id=c["chunk_id"], doc_id=c["doc_id"], title=c["title"],
                            seq=c.get("seq", 0), content=c["content"],
                            bm25=bm25[i], dense=dense[i], rrf=score))
        if not hits:
            return [], "low"
        strong, weak = THRESHOLDS.get(self.backend, THRESHOLDS["tfidf"])
        top_dense = max(h.dense for h in hits)
        off_domain = any(t in query for t in OFF_DOMAIN_TERMS)
        if STRUCTURED.search(query) or off_domain:
            # 具体单号/工单号 → 交给 query_orders 这类工具；域外闲聊 → 知识库不接
            in_domain = False
        elif self.backend == "tfidf":
            # 词法后端没有语义能力，只能靠字面词表兜底
            in_domain = any(t in query for t in DOMAIN_TERMS)
        else:
            # 语义后端：字面词表只用来"兜住"改写句（"寄到我这儿大概要几天"没有业务词，
            # 但语义分 0.529 说明它就在域内），所以词表命中 OR 分数够 都算在域。
            in_domain = any(t in query for t in DOMAIN_TERMS) or top_dense >= weak
        if not in_domain or top_dense < weak:
            confidence = "low"
        elif top_dense >= strong:
            confidence = "high"
        else:
            confidence = "medium"
        return hits, confidence

    def answer_context(self, question: str, top_k: int = 3) -> dict:
        hits, confidence = self.search(question, top_k)
        if not hits or confidence == "low":
            return {
                "question": question, "hits": [h.to_dict() for h in hits],
                "confidence": "low", "covered": False,
                "guidance": "知识库没有可信依据，不要编造制度条款；建议转人工或补充制度文档。",
            }
        return {
            "question": question,
            "hits": [h.to_dict() for h in hits],
            "confidence": confidence,
            "covered": True,
            "guidance": ("依据充分，可直接引用并给出 citations。" if confidence == "high"
                         else "依据偏弱，回答时用“根据现有制度…，具体情况建议人工确认”的措辞。"),
        }


_INDEX: HybridIndex | None = None


def resolve_backend(name: str | None = None) -> str:
    """把 RAG_BACKEND 的 auto 解析成实际后端：有模型走 hybrid，没模型退回 tfidf。"""
    want = (name or os.getenv("RAG_BACKEND", "auto")).strip().lower()
    if want in ("tfidf", "embedding", "hybrid"):
        if want != "tfidf" and not embedder.available():
            return "tfidf"
        return want
    # auto
    return "hybrid" if embedder.available() else "tfidf"


def get_index(rebuild: bool = False, backend: str | None = None) -> HybridIndex:
    global _INDEX
    be = resolve_backend(backend)
    if _INDEX is None or rebuild or _INDEX.backend != be:
        _INDEX = HybridIndex(load_chunks(), backend=be)
    return _INDEX


def _build_faiss(mat: np.ndarray):
    """能用 faiss 就用它的内积索引（IndexFlatIP），装不上就退回 numpy 点积。

    34 条语料两者耗时都在 0.02ms 级，这里选 faiss 是为了把「真实工程路径」跑通：
    语料涨到几万条时，改 `IndexFlatIP` → `IndexIVFFlat` 即可，接口不动。
    """
    if os.getenv("RAG_USE_FAISS", "1") != "1":
        return None
    try:
        import faiss  # noqa: PLC0415 - 可选依赖，装不上就走 numpy

        index = faiss.IndexFlatIP(int(mat.shape[1]))
        index.add(mat)
        return index
    except Exception:  # noqa: BLE001
        return None


@lru_cache(maxsize=256)
def _answer_cached(question: str, top_k: int) -> str:
    """同一句话反复问（demo 里点示例按钮很常见）直接命中，省掉一次 BM25 + TF-IDF 计算。"""
    return json.dumps(get_index().answer_context(question, top_k=top_k), ensure_ascii=False)


def answer_context(question: str, top_k: int = 3) -> dict:
    # 返回副本：调用方（工具层）可能会往结果里塞字段，别污染缓存
    return json.loads(_answer_cached(question, top_k))
