FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Shanghai \
    HF_HOME=/app/.hf \
    HF_ENDPOINT=https://hf-mirror.com \
    RAG_BACKEND=auto

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 语义检索依赖（约 250MB）。要镜像更小就删掉这两行 + 下面 build_embeddings，
# 运行时 RAG_BACKEND=auto 会自动回退到词法 BM25+TF-IDF，接口不变。
COPY requirements-embed.txt .
RUN pip install --no-cache-dir -r requirements-embed.txt

COPY . .

# 首次启动：建库 + 灌种子数据（数据目录挂卷，已存在则跳过）
RUN python -m biz_system.seed || true

# 构建期把制度语料编码成向量并烧进镜像，容器启动即用（不再依赖运行时下载模型）
RUN python scripts/build_embeddings.py || echo "embedding 预热失败，运行时将回退词法检索"

EXPOSE 8000 8001

CMD ["bash", "-lc", "uvicorn biz_system.main:app --host 0.0.0.0 --port 8001 & uvicorn agent.main:app --host 0.0.0.0 --port 8000"]
