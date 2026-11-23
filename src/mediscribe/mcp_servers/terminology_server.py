"""MCP server: medical terminology (ICD-10-CM codes from NLM).

Run standalone (stdio):  python -m mediscribe.mcp_servers.terminology_server
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mediscribe.tools.terminology import search_icd10

mcp = FastMCP("terminology", instructions="Look up official ICD-10-CM diagnosis codes.")


@mcp.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=True))
def lookup_icd10_code(diagnosis: str, max_results: int = 7) -> list[dict]:
    """Search ICD-10-CM codes for a diagnosis in plain words, e.g. 'radiation pneumonitis'
    or 'type 2 diabetes with neuropathy'. Use short clinical terms; if nothing matches,
    retry with a broader term. Returns [{code, name}]. Pick the most specific match."""
    return search_icd10(diagnosis, max_results)


if __name__ == "__main__":
    mcp.run()
