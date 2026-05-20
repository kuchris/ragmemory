"""
Verify MCP forget tools wrap the root forget API safely.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_ragm_mcp_forget_tools.py
"""
import contextlib
import importlib
import io
import os
import shutil
import sys
from pathlib import Path

DB_PATH = Path("./.data/ragm_mcp_forget_tools_test")
ROOT = Path.cwd()

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

os.environ["RAGMEMORY_DB_PATH"] = str(DB_PATH)
sys.path.insert(0, str(ROOT))

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    server = importlib.import_module("ragm_mcp.server")

assert stdout.getvalue() == ""

target = server.save_user_message(
    "MCP_FORGET_TARGET should disappear when forgotten through the MCP adapter.",
    extract_structured=False,
)
keep = server.save_user_message(
    "MCP_FORGET_KEEP should remain active after another message is forgotten.",
    extract_structured=False,
)
assert target.message_id is not None
assert keep.message_id is not None

preview = server.forget_preview(message_ids=str(target.message_id))
assert "Forget preview:" in preview
assert "Messages: 1" in preview
assert "Run forget_confirm" in preview
assert "MCP_FORGET_TARGET" in preview

multi_preview = server.forget_preview(message_ids=f"{target.message_id}, {keep.message_id}", sample_limit=1)
assert "Messages: 2" in multi_preview
assert "Truncated: True" in multi_preview

bad_preview = server.forget_preview()
assert bad_preview.startswith("Forget preview failed:")

confirm = server.forget_confirm(message_ids=str(target.message_id))
assert "Forget confirmed:" in confirm
assert "Tombstoned messages: 1" in confirm
assert "MCP_FORGET_TARGET" in confirm

assert all(
    chunk.message_id != target.message_id
    for chunk in server.store.retrieve("MCP_FORGET_TARGET")
)
assert any(
    chunk.message_id == keep.message_id
    for chunk in server.store.retrieve("MCP_FORGET_KEEP")
)

print("RagM MCP forget tools test passed.")
