"""Tool discovery: connect to every MCP server and turn its tools into LangChain tools.

Agents never import tool functions directly. They receive whatever the servers advertise at
runtime (name, description, input schema, annotations). Adding a tool to a server makes it
available to every agent with no agent code change; removing a server removes its tools.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

# server name -> python module that runs it over stdio
MCP_SERVERS = {
    "ehr": "mediscribe.mcp_servers.ehr_server",
    "terminology": "mediscribe.mcp_servers.terminology_server",
    "drugs": "mediscribe.mcp_servers.drug_server",
}


def mcp_client(servers: list[str] | None = None) -> MultiServerMCPClient:
    return MultiServerMCPClient({
        name: {"transport": "stdio", "command": sys.executable, "args": ["-m", module],
               "env": dict(os.environ)}
        for name, module in MCP_SERVERS.items() if servers is None or name in servers
    })


def is_read_only(tool: BaseTool) -> bool:
    """MCP annotations travel with the tool (the adapter copies them into tool.metadata).
    Write tools are reserved for the physician-approved EHR step."""
    return bool((tool.metadata or {}).get("readOnlyHint"))


async def discover_tools(servers: list[str] | None = None, read_only: bool = False) -> dict[str, list[BaseTool]]:
    """Returns {server_name: [tools]} discovered at runtime."""
    client = mcp_client(servers)
    found: dict[str, list[BaseTool]] = {}
    for name in (servers or MCP_SERVERS):
        tools = await client.get_tools(server_name=name)
        found[name] = [t for t in tools if is_read_only(t)] if read_only else tools
    return found


# ---------- Calling a tool from workflow code (no LLM involved) ----------

class ToolCallError(RuntimeError):
    pass


async def acall_tool(server: str, tool: str, args: dict[str, Any]) -> Any:
    """Call one MCP tool and return its result as plain Python data."""
    async with mcp_client([server]).session(server) as session:
        result = await session.call_tool(tool, args)
    if result.isError:
        raise ToolCallError(f"{server}.{tool} failed: {result.content}")
    data = result.structuredContent
    if data is not None:
        # FastMCP wraps non-dict return values as {"result": value}
        return data["result"] if set(data) == {"result"} else data
    texts = [c.text for c in result.content if getattr(c, "type", "") == "text"]
    parsed = [json.loads(t) for t in texts]
    return parsed[0] if len(parsed) == 1 else parsed


def call_tool(server: str, tool: str, args: dict[str, Any]) -> Any:
    """Synchronous wrapper used by the LangGraph workflow nodes.

    The workflow runs synchronously (the Postgres checkpointer's async driver cannot share
    Windows' event loop with MCP's subprocess transport), so each tool call gets its own loop.
    """
    return asyncio.run(acall_tool(server, tool, args))
