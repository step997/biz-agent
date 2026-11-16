"""Embedding 编码器：默认 BAAI/bge-small-zh-v1.5（sentence-transformers，CPU 推理）。

三个设计约束：
1. **懒加载**：只有 `RAG_BACKEND` 需要向量时才 import torch/sentence-transformers。
   默认路径（tfidf）不背 400MB 依赖，pytest 在没有模型的环境里也能跑。
2. **失败降级**：依赖没装 / 模型下载不到 / 显存内存不够 → `available()` 返回 False，
   调用方回退词法检索，不让 demo 起不来。
3. **磁盘缓存**：34 个 chunk 的向量算一次存 `data/emb.npy`，用「语料指纹」（chunk_id + 内容 sha1）
   判断是否过期。改了制度文档会自动重建，没改就 0 成本加载。
"""

from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "data"
EMB_CACHE = CACHE_DIR / "emb.npy"
EMB_META = CACHE_DIR / "emb_meta.json"

# 读 .env（与 agent/llm.py 用同一份配置），并给 huggingface 兜几个默认值：
# 国内直连 huggingface.co 的 etag 检查会挂几分钟，镜像 + 不复检是最省事的组合。
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except Exception:  # noqa: BLE001 - 没装 python-dotenv 也要能跑
    pass
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MODEL_NAME = os.getenv("EMB_MODEL", "BAAI/bge-small-zh-v1.5")
# bge 官方建议：检索任务里 query 侧加指令前缀，document 侧不加
QUERY_PREFIX = os.getenv("EMB_QUERY_PREFIX", "为这个句子生成表示以用于检索相关文章：")

_MODEL = None
_MODEL_FAILED: str | None = None


def available() -> bool:
    """模型能不能用（只加载一次，失败记住原因）。"""
    global _MODEL, _MODEL_FAILED
    if _MODEL is not None:
        return True
    if _MODEL_FAILED is not None:
        return False
    try:
        from sentence_transformers import SentenceTransformer  # 懒导入：不带 torch 的环境不会走到这

        _MODEL = SentenceTransformer(MODEL_NAME, device="cpu")
        return True
    except Exception as exc:  # noqa: BLE001 - 依赖缺失/下载失败都要降级
        _MODEL_FAILED = f"{type(exc).__name__}: {exc}"
        return False


def failure_reason() -> str | None:
    return _MODEL_FAILED


def encode(texts: Sequence[str], *, is_query: bool = False, batch_size: int = 32) -> np.ndarray:
    """返回 L2 归一化后的 float32 矩阵 (n, dim)。"""
    if not available():
        raise RuntimeError(f"embedding 不可用：{_MODEL_FAILED}")
    payload = [(QUERY_PREFIX + t) if is_query else t for t in texts]
    vecs = _MODEL.encode(  # type: ignore[union-attr]
        payload,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vecs, dtype=np.float32)


def encode_query(text: str) -> np.ndarray:
    return encode([text], is_query=True)[0]


def _fingerprint(chunks: list[dict]) -> str:
    h = hashlib.sha1()
    for c in chunks:
        h.update(c["chunk_id"].encode("utf-8"))
        h.update(hashlib.sha1(c["content"].encode("utf-8")).digest())
    h.update(MODEL_NAME.encode("utf-8"))
    return h.hexdigest()


def chunk_matrix(chunks: list[dict], *, rebuild: bool = False) -> np.ndarray:
    """chunk 向量矩阵（带磁盘缓存）。返回 (n_chunks, dim) 的归一化 float32。"""
    fp = _fingerprint(chunks)
    if not rebuild and EMB_CACHE.exists() and EMB_META.exists():
        try:
            meta = json.loads(EMB_META.read_text(encoding="utf-8"))
            if meta.get("fingerprint") == fp:
                mat = np.load(EMB_CACHE)
                if mat.shape[0] == len(chunks):
                    return mat.astype(np.float32)
        except Exception:  # noqa: BLE001 - 缓存坏了就重建
            pass
    mat = encode([c["content"] for c in chunks])
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_CACHE, mat)
    EMB_META.write_text(
        json.dumps(
            {"fingerprint": fp, "model": MODEL_NAME, "dim": int(mat.shape[1]),
             "n_chunks": int(mat.shape[0])},
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return mat


@lru_cache(maxsize=1)
def dim() -> int:
    if not available():
        return 0
    return int(encode(["探测维度"])[0].shape[0])
