"""工具层基础设施：把"业务系统的查询/操作能力"封装成可复用、可审计、可测试的工具。

设计要点（面试口径）：
1. 智能体永远不直连数据库，只通过这里声明的工具访问业务系统 —— 工具即权限边界。
2. 工具用 Pydantic 声明参数，schema 自动导出给 LLM 做 function calling，不手写 JSON。
3. read / write 两类：write 一律标记 requires_approval，由上层（Graph 的 HITL 节点）拦。
4. 所有调用经过 `on_call` 回调打点，落 audit_log（谁、什么参数、多久、成功与否）。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import httpx
from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """工具级错误（参数不合法、业务系统报错、网络不可达）。"""


class ToolResult(BaseModel):
    ok: bool
    tool: str
    data: Any = None
    error: str | None = None
    latency_ms: int = 0

    def brief(self, limit: int = 600) -> str:
        if not self.ok:
            return f"[{self.tool}] 失败：{self.error}"
        text = str(self.data)
        return f"[{self.tool}] {text[:limit]}{'…' if len(text) > limit else ''}"


@dataclass
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]
    handler: Callable[..., Any]
    kind: Literal["read", "write"] = "read"
    requires_approval: bool = False

    def schema(self) -> dict:
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": params},
        }


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)
    on_call: Callable[[dict], None] | None = None

    def register(self, tool: Tool) -> None:
        if tool.name in self.tools:
            raise ValueError(f"tool {tool.name} already registered")
        self.tools[tool.name] = tool

    # ---- 供 LLM 使用 ----
    def schemas(self, kinds: tuple[str, ...] = ("read", "write")) -> list[dict]:
        return [t.schema() for t in self.tools.values() if t.kind in kinds]

    def describe(self) -> list[dict]:
        return [{"name": t.name, "kind": t.kind, "requires_approval": t.requires_approval,
                 "description": t.description, "args": list(t.args_model.model_fields)}
                for t in self.tools.values()]

    def describe_text(self) -> str:
        lines = []
        for t in self.tools.values():
            tag = "写/需审批" if t.requires_approval else "读"
            lines.append(f"- {t.name}（{tag}）：{t.description}")
        return "\n".join(lines)

    # ---- 执行 ----
    def call(self, name: str, args: dict | None = None) -> ToolResult:
        started = time.perf_counter()
        args = args or {}
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult(ok=False, tool=name, error=f"未注册的工具：{name}")

        try:
            parsed = tool.args_model(**args)
        except ValidationError as exc:
            errs = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            res = ToolResult(ok=False, tool=name, error=f"参数校验失败：{errs}")
            self._emit(name, args, res, started)
            return res

        try:
            data = tool.handler(parsed)
            res = ToolResult(ok=True, tool=name, data=data)
        except ToolError as exc:
            res = ToolResult(ok=False, tool=name, error=str(exc))
        except httpx.HTTPError as exc:
            res = ToolResult(ok=False, tool=name, error=f"业务系统不可达：{exc}")
        except Exception as exc:  # noqa: BLE001 —— 工具不让异常穿透到图节点
            res = ToolResult(ok=False, tool=name, error=f"{type(exc).__name__}: {exc}")
        res.latency_ms = int((time.perf_counter() - started) * 1000)
        self._emit(name, args, res, started)
        return res

    def _emit(self, name: str, args: dict, res: ToolResult, started: float) -> None:
        if self.on_call is None:
            return
        self.on_call({
            "tool": name,
            "kind": self.tools[name].kind if name in self.tools else "unknown",
            "args": args,
            "ok": res.ok,
            "error": res.error,
            "latency_ms": res.latency_ms or int((time.perf_counter() - started) * 1000),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        })
