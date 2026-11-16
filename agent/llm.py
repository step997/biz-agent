"""LLM 接入（OpenAI 兼容协议，默认 DeepSeek）。没有 key 时自动降级为 offline 规则规划器。"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
BASE_URL = (os.getenv("LLM_BASE_URL") or os.getenv("DEEPSEEK_BASE_URL")
            or "https://api.deepseek.com/v1")
API_KEY = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY") or ""


def mode() -> str:
    """offline：规则规划器（无 key 也能跑）；llm：大模型规划 + 生成。"""
    explicit = os.getenv("AGENT_MODE", "").strip().lower()
    if explicit in ("offline", "llm"):
        return explicit
    return "llm" if API_KEY else "offline"


def available() -> bool:
    return bool(API_KEY)


def get_llm(temperature: float = 0.0):
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=MODEL, base_url=BASE_URL, api_key=API_KEY, temperature=temperature)
