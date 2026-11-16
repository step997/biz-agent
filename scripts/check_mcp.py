"""用官方 MCP 客户端（stdio）连一遍自己的 server，确认工具能被发现、能调通。"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mcp import ClientSession, StdioServerParameters  # noqa: E402
from mcp.client.stdio import stdio_client  # noqa: E402


def _payload(res) -> str:
    data = getattr(res, "structured_content", None)
    if data is None:
        data = res.content[0].text if res.content else None
    return json.dumps(data, ensure_ascii=False)[:240]


async def main() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")],
                                   cwd=str(ROOT))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = [t.name for t in tools.tools]
            print(f"发现 {len(names)} 个工具: {names}")

            r1 = await session.call_tool("query_orders", {"customer_name": "王伟", "limit": 3})
            print("query_orders ->", _payload(r1))

            r2 = await session.call_tool("query_policy", {"question": "退货运费谁承担", "top_k": 2})
            print("query_policy ->", _payload(r2))

            r3 = await session.call_tool("low_stock_list", {"limit": 2})
            print("low_stock_list ->", _payload(r3))


if __name__ == "__main__":
    asyncio.run(main())
