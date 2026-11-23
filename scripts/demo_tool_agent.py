"""Tool discovery + dynamic tool calling demo.

1. Starts the three MCP servers and lists the tools they advertise (discovery).
2. Gives the READ-ONLY tools to an agent. The agent decides by itself which tools to call,
   in which order, with which arguments (dynamic tool calling, ReAct loop).
3. Prints every tool call and result, then the agent's answer.

Run:  python scripts/demo_tool_agent.py
      python scripts/demo_tool_agent.py --question "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from rich.console import Console
from rich.table import Table

from mediscribe.llm import agent_middleware, chat_model
from mediscribe.mcp_client import discover_tools, is_read_only

console = Console()

DEFAULT_QUESTION = (
    "The doctor seeing the patient from encounter D2N001 wants to prescribe "
    "Bactrim (sulfamethoxazole-trimethoprim) for an uncomplicated urinary tract infection. "
    "Is this safe for this patient? Also give the ICD-10-CM code for the diagnosis."
)
SYSTEM = (
    "You are a clinical decision-support assistant helping a physician. Use the available tools "
    "to look up the patient's record and check drug safety before answering. Never guess "
    "patient facts: look them up. Finish with a short recommendation, the evidence for it "
    "(cite which check found what) and a reminder that the physician makes the final decision."
)


def tool_text(content) -> str:
    """MCP tool results arrive as a list of content blocks; keep just their text."""
    if isinstance(content, list):
        return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def short(value, limit=350) -> str:
    text = value if isinstance(value, str) else json.dumps(value)
    return text if len(text) <= limit else text[:limit] + " …"


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--provider", default="gemini", choices=["gemini", "claude"])
    ap.add_argument("--timeout", type=int, default=180, help="seconds before giving up")
    args = ap.parse_args()

    # 1. Discovery
    servers = await discover_tools()
    table = Table(title="Tools discovered from MCP servers at runtime")
    for col in ("Server", "Tool", "Access", "Description"):
        table.add_column(col, overflow="fold")
    for server, tools in servers.items():
        for t in tools:
            table.add_row(server, t.name, "read" if is_read_only(t) else "[red]write[/red]",
                          t.description.split("\n")[0])
    console.print(table)

    # 2. Agent gets read-only tools; writes need physician approval (multi-agent flow)
    tools = [t for ts in servers.values() for t in ts if is_read_only(t)]
    console.print(f"\nAgent receives {len(tools)} read-only tools. Write tools are withheld.\n")
    agent = create_agent(chat_model(args.provider), tools=tools, system_prompt=SYSTEM,
                         middleware=agent_middleware(args.provider))

    console.print(f"[bold]Question:[/bold] {args.question}\n")

    # 3. Stream the reasoning trace live: each model turn and each tool result as it happens
    step, final = 0, None
    console.print("[dim]Thinking… (the model may request several tools at once; "
                  "their results then follow, labelled ←)[/dim]")
    stream = agent.astream({"messages": [{"role": "user", "content": args.question}]},
                           stream_mode="updates")
    try:
        async with asyncio.timeout(args.timeout):
            async for update in stream:
                for data in update.values():
                    for msg in (data.get("messages", []) if isinstance(data, dict) else []):
                        if isinstance(msg, AIMessage) and msg.tool_calls:
                            for call in msg.tool_calls:
                                step += 1
                                console.print(f"[cyan]Step {step}: {call['name']}[/cyan]"
                                              f"({short(call['args'], 200)})")
                        elif isinstance(msg, AIMessage):
                            final = msg
                        elif isinstance(msg, ToolMessage):
                            console.print(f"   [dim]← {msg.name}: {short(tool_text(msg.content))}[/dim]")
    except TimeoutError:
        console.print(f"[red]Stopped after {args.timeout}s without a final answer.[/red]")
        return

    if final is None:
        console.print("[red]The agent finished without a text answer.[/red]")
        return
    answer = final.content if isinstance(final.content, str) else "".join(
        part.get("text", "") for part in final.content if isinstance(part, dict))
    console.print(f"\n[bold green]Agent answer ({step} tool calls):[/bold green]\n{answer}")


if __name__ == "__main__":
    asyncio.run(main())
