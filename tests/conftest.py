"""pytest 公共夹具：确保测试用的是临时端口上的业务系统实例，且不污染开发库。"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
TEST_PORT = 8011
os.environ.setdefault("BIZ_BASE_URL", f"http://127.0.0.1:{TEST_PORT}")
os.environ.setdefault("BIZ_API_KEY", "demo-key")
os.environ.setdefault("AGENT_MODE", "offline")
# 测试固定走词法后端：不加载 400MB 的 embedding 模型，保证 CI/离线机器上确定性、秒级跑完
os.environ.setdefault("RAG_BACKEND", "tfidf")
sys.path.insert(0, str(ROOT))


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.4)
        return s.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="session", autouse=True)
def biz_server():
    """测试期起一个独立业务系统进程（端口 8011），结束后关闭。"""
    proc = None
    if not _port_open(TEST_PORT):
        proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "biz_system.main:app",
             "--host", "127.0.0.1", "--port", str(TEST_PORT), "--log-level", "warning"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        for _ in range(60):
            if _port_open(TEST_PORT):
                break
            time.sleep(0.3)
    base = f"http://127.0.0.1:{TEST_PORT}"
    for _ in range(30):
        try:
            if httpx.get(f"{base}/health", timeout=2).status_code == 200:
                break
        except Exception:  # noqa: BLE001
            time.sleep(0.3)
    yield base
    if proc is not None:
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture()
def base(biz_server) -> str:
    return biz_server
