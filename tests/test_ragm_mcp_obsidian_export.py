"""
Verify MCP mutating tools update the Obsidian mirror.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_ragm_mcp_obsidian_export.py
"""
import contextlib
import importlib
import io
import os
import shutil
import sys
from pathlib import Path

DB_PATH = Path("./.data/ragm_mcp_obsidian_export_test")
OUT_PATH = Path("./.data/ragm_mcp_obsidian_export_vault_test")
ROOT = Path.cwd()

for path in (DB_PATH, OUT_PATH):
    if path.exists():
        shutil.rmtree(path)

os.environ["RAGMEMORY_DB_PATH"] = str(DB_PATH)
os.environ["RAGMEMORY_OBSIDIAN_PATH"] = str(OUT_PATH)
sys.path.insert(0, str(ROOT))

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    server = importlib.import_module("ragm_mcp.server")

assert stdout.getvalue() == ""
assert server.OBSIDIAN_PATH == OUT_PATH

saved = server.save("MCP_OBSIDIAN_SAVE should appear in the mirror.")
assert "Saved." in saved
assert "Obsidian mirror updated" in saved
assert (OUT_PATH / "index.md").exists()
assert (OUT_PATH / "active/messages/msg-000000.md").exists()
assert "MCP_OBSIDIAN_SAVE" in (OUT_PATH / "active/messages/msg-000000.md").read_text(encoding="utf-8")

document = server.remember_document("MCP_OBSIDIAN_DOCUMENT should also export.")
assert "Stored" in document
assert "Obsidian mirror updated" in document
assert (OUT_PATH / "active/messages/msg-000001.md").exists()

preview = server.forget_preview(message_ids="0")
assert "Forget preview:" in preview
assert (OUT_PATH / "active/messages/msg-000000.md").exists()

confirm = server.forget_confirm(message_ids="0")
assert "Forget confirmed:" in confirm
assert "Obsidian mirror updated" in confirm
assert not (OUT_PATH / "active/messages/msg-000000.md").exists()
forgotten = OUT_PATH / "forgotten/messages/msg-000000.md"
assert forgotten.exists()
assert "status: \"forgotten\"" in forgotten.read_text(encoding="utf-8")

stats = server.memory_stats()
assert str(OUT_PATH) in stats

print("RagM MCP Obsidian export test passed.")
