"""Tool discovery: connect to every MCP server and turn its tools into LangChain tools.

Agents never import tool functions directly. They receive whatever the servers advertise at
runtime (name, description, input schema, annotations). Adding a tool to a server makes it
available to every agent with no agent code change; removing a server removes its tools.
"""

from __future__ import annotations

import os
import sys

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
