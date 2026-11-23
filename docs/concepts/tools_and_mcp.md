# Tools, tool discovery and dynamic tool calling (MCP)

An LLM on its own can only produce text. **Tools** let an agent act: read a patient record, look up
a code, check a drug label. MediScribe exposes all its tools through the
**Model Context Protocol (MCP)**, an open standard for connecting AI applications to tools and data.

## Three MCP servers

| Server | Module | Tools | Backed by |
|---|---|---|---|
| `ehr` | `mcp_servers/ehr_server.py` | `find_patient_by_encounter`, `search_patients`, `get_patient_summary` (read); `write_clinical_note`, `add_diagnosis`, `create_prescription` (write) | HAPI FHIR server |
| `terminology` | `mcp_servers/terminology_server.py` | `lookup_icd10_code` | NLM Clinical Tables ICD-10-CM API |
| `drugs` | `mcp_servers/drug_server.py` | `get_drug_label`, `check_drug_interactions`, `check_drug_allergy` | openFDA drug labels + built-in allergy class table |

Splitting by domain mirrors real hospitals, where the EHR, terminology service and drug database are
separate systems, often owned by different teams. Each server runs as its own process over stdio.

## Tool discovery

`mediscribe/mcp_client.py` starts the servers and asks each one "what tools do you have?"
(the MCP `tools/list` request). Each answer includes the tool name, description, JSON input schema and
**annotations**. Agents receive whatever is discovered: adding a tool to a server makes it available to
every agent without changing agent code.

**Annotations as a safety layer.** Read tools are marked `readOnlyHint`; EHR write tools are marked
`destructiveHint`. The client filters on these, so research and safety-check agents only ever get
read tools. Write tools are reserved for the step that runs after the physician approves.

## Dynamic tool calling

`scripts/demo_tool_agent.py` gives the read-only tools to a LangChain agent (a ReAct loop: the model
reasons, picks a tool, reads the result, repeats). Nothing in the code says which tools to call or in
what order: the model decides from the question and the tool descriptions. **Tool docstrings are
prompts**: "Always call this before suggesting a diagnosis or prescription" changes agent behaviour.

Demo scenario: prescribing Bactrim (a sulfonamide antibiotic) to a patient whose record lists a
sulfonamide allergy and who takes lisinopril. The agent has to find the patient, read the record, run
the allergy and interaction checks, and look up the ICD-10 code, then recommend against it.

## Safety design of the drug tools

- The allergy check is a **deterministic** drug-class table (penicillins, cephalosporins, sulfonamides,
  NSAIDs, opioids) with penicillin–cephalosporin cross-reactivity. Rules are auditable; an LLM guess is not.
- The interaction check searches FDA label text for each current medication and returns the exact
  excerpt as evidence, so the physician can verify it.
- Every result says these are decision support: absence of a finding is not proof of safety.

## Inspecting a server

The MCP SDK ships an inspector UI: `mcp dev src/mediscribe/mcp_servers/drug_server.py` opens a browser
page where each tool can be called by hand.
