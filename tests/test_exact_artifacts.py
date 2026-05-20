"""
Verify deterministic exact artifact extraction.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_exact_artifacts.py
"""
from ragmemory.memory import ExactArtifactExtractor

CONFIG_BLOCK = """```json
{
  "mcp": {
    "rag-memory": {
      "command": ["uv", "run", "ragm_mcp/server.py"],
      "enabled": true
    }
  }
}
```"""

TABLE_BLOCK = """| File | Purpose |
|---|---|
| memory.py | reference engine |
| ragm_mcp/server.py | stable MCP server |"""

MERMAID_BLOCK = """```mermaid
flowchart TD
  user --> recall
  recall --> context
```"""

CODE_BLOCK = """```python
from ragmemory.memory import MemoryStore
store = MemoryStore()
```"""

MESSAGE = f"""
Save these exact artifacts:

{CONFIG_BLOCK}

{TABLE_BLOCK}

{MERMAID_BLOCK}

{CODE_BLOCK}
""".strip()

objects = ExactArtifactExtractor().extract("user", MESSAGE, message_id=7)
by_type = {}
for obj in objects:
    by_type.setdefault(obj.type, []).append(obj)

configs = by_type.get("config", [])
tables = by_type.get("table", [])
charts = by_type.get("chart", [])
code_refs = by_type.get("code_reference", [])

assert len(configs) == 1
assert configs[0].source_text == CONFIG_BLOCK
assert configs[0].summary == "Exact json config block"
assert configs[0].tags == ["json", "config"]
assert configs[0].message_id == 7
assert configs[0].role == "user"

assert len(tables) == 1
assert tables[0].source_text == TABLE_BLOCK
assert tables[0].summary == "Exact Markdown table"
assert tables[0].tags == ["markdown", "table"]

assert len(charts) == 1
assert charts[0].source_text == MERMAID_BLOCK
assert charts[0].summary == "Exact Mermaid chart block"
assert charts[0].tags == ["mermaid", "chart"]

assert len(code_refs) == 1
assert code_refs[0].source_text == CODE_BLOCK
assert code_refs[0].summary == "Exact python block"
assert code_refs[0].tags == ["python", "code_reference"]

print("Exact artifact extraction test passed.")
