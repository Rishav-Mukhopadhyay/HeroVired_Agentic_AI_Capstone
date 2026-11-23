"""MCP server: drug safety (openFDA labels + allergy class checks).

Run standalone (stdio):  python -m mediscribe.mcp_servers.drug_server
"""

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from mediscribe.tools import drugs

mcp = FastMCP("drugs", instructions="Drug safety checks for prescriptions: FDA label warnings, "
                                     "interactions with current medications, allergy conflicts.")
READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


@mcp.tool(annotations=READ)
def get_drug_label(drug: str) -> dict:
    """FDA label safety sections for a drug (generic or brand name): boxed warning,
    contraindications, drug interactions, warnings. Text is truncated."""
    return drugs.drug_label_info(drug)


@mcp.tool(annotations=READ)
def check_drug_interactions(new_drug: str, current_medications: list[str]) -> dict:
    """Check a drug you plan to prescribe against the patient's current medications using
    FDA label text. Pass medication names, e.g. ['lisinopril 10 mg', 'warfarin']."""
    return drugs.check_interactions(new_drug, current_medications)


@mcp.tool(annotations=READ)
def check_drug_allergy(drug: str, allergies: list[str]) -> dict:
    """Check a drug against the patient's recorded allergies, including drug-class
    cross-reactivity (e.g. ibuprofen vs an NSAID allergy). Returns safe true/false + conflicts."""
    return drugs.check_allergy(drug, allergies)


if __name__ == "__main__":
    mcp.run()
