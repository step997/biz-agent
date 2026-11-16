"""离线建向量库：把制度 chunk 用 bge-small-zh-v1.5 编码成向量矩阵并落盘。

用途：
- 部署时预热（避免第一个用户问句承担 20s 模型加载）
- 改完 `docs/policies/*.md` 后重建（指纹变了会自动重建，也可以手动跑一次）

产物：`data/emb.npy`（L2 归一化 float32 矩阵）+ `data/emb_meta.json`（模型名/维度/chunk 数/指纹）
"""

from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag import embedder  # noqa: E402
from rag.corpus import load_chunks  # noqa: E402


def main() -> int:
    print(f"模型: {embedder.MODEL_NAME}")
    t = time.perf_counter()
    if not embedder.available():
        print(f"不可用：{embedder.failure_reason()}")
        print("提示：pip install sentence-transformers，或设置 EMB_MODEL / HF_ENDPOINT 指向可用镜像")
        return 1
    load_ms = 1000 * (time.perf_counter() - t)
    print(f"模型加载 {load_ms:.0f}ms")

    chunks = load_chunks()
    print(f"语料: {len(chunks)} chunks / {len({c['doc_id'] for c in chunks})} docs")

    t = time.perf_counter()
    mat = embedder.chunk_matrix(chunks, rebuild=True)
    norms = (mat ** 2).sum(axis=1) ** 0.5
    print(f"编码完成 {1000 * (time.perf_counter() - t):.0f}ms → shape={mat.shape} "
          f"dtype={mat.dtype} 范数 {norms.min():.4f}~{norms.max():.4f}")
    print(f"落盘: {embedder.EMB_CACHE}（{embedder.EMB_CACHE.stat().st_size / 1024:.0f} KB）")

    t = time.perf_counter()
    embedder.chunk_matrix(chunks)
    print(f"缓存命中重载 {1000 * (time.perf_counter() - t):.1f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
