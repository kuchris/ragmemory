"""
Verify MCP forget tools wrap the root forget API safely.

Run:
    $env:PYTHONIOENCODING='utf-8'
    uv run python tests/test_ragm_mcp_forget_tools.py
"""
import contextlib
import importlib
import io
import json
import os
import shutil
import sys
from pathlib import Path

DB_PATH = Path("./.data/ragm_mcp_forget_tools_test")
ROOT = Path.cwd()

if DB_PATH.exists():
    shutil.rmtree(DB_PATH)

os.environ["RAGMEMORY_DB_PATH"] = str(DB_PATH)
os.environ["NVIDIA_API_KEY"] = ""
os.environ.pop("RAGMEMORY_MCP_TOOLS_ENABLE_RECALL", None)
os.environ.pop("RAGMEMORY_MCP_TOOLS_ENABLE_SAVE", None)
os.environ["RAGMEMORY_MCP_TOOLS_ENABLE_TOMBSTONE"] = "false"
sys.path.insert(0, str(ROOT))

stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    server = importlib.import_module("ragm_mcp.server")

assert stdout.getvalue() == ""

disabled_save = server.save("MCP save should be disabled by default.")
assert disabled_save.startswith("MCP save disabled:")
disabled_document = server.remember_document("MCP document save should be disabled by default.")
assert disabled_document.startswith("MCP save disabled:")
readonly_recall = server.recall("MCP recall should not save by default.")
assert readonly_recall.startswith("MCP recall disabled:")
assert server.stats()["messages"] == 0

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

preview = server.remove_memory_preview(message_ids=str(target.message_id))
assert "Remove preview:" in preview
assert "Messages: 1" in preview
assert "Run remove_memory_confirm" in preview
assert "MCP_FORGET_TARGET" in preview

before_preview = server.remove_memory_preview(before="2999-01-01T00:00:00Z", sample_limit=1)
assert "Remove preview:" in before_preview

multi_preview = server.remove_memory_preview(message_ids=f"{target.message_id}, {keep.message_id}", sample_limit=1)
assert "Messages: 2" in multi_preview
assert "Truncated: True" in multi_preview

bad_preview = server.remove_memory_preview()
assert bad_preview.startswith("Remove preview failed:")

disabled = server.remove_memory_confirm(
    message_ids=str(target.message_id),
    reason="User explicitly requested removal of the stale MCP test memory.",
)
assert disabled.startswith("Remove confirm disabled:")

server.os.environ["RAGMEMORY_MCP_TOOLS_ENABLE_TOMBSTONE"] = "true"

bad_before_confirm = server.forget_confirm(before="2999-01-01T00:00:00Z")
assert "before= is preview-only" in bad_before_confirm

bad_reason = server.remove_memory_confirm(message_ids=str(target.message_id), reason="too short")
assert bad_reason.startswith("Remove confirm failed:")
assert "reason must be" in bad_reason

too_many = server.remove_memory_confirm(
    message_ids=f"{target.message_id}, {keep.message_id}, 99, 100",
    reason="User explicitly requested too many test IDs at once.",
)
assert too_many.startswith("Remove confirm failed:")
assert "at most 3" in too_many

confirm = server.remove_memory_confirm(
    message_ids=str(target.message_id),
    reason="User explicitly requested removal of the MCP target test memory.",
)
assert "Remove confirmed:" in confirm
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

deprecated = server.forget_preview(message_ids=str(keep.message_id))
assert "Remove preview:" in deprecated

events = []
for line in (DB_PATH / "events.jsonl").read_text(encoding="utf-8").splitlines():
    events.append(json.loads(line))
assert any(event["event"] == "memory_tombstoned_via_mcp" for event in events)
assert any(
    event["event"] == "mcp_tool_deprecated_call" and event["tool"] == "forget_preview"
    for event in events
)

print("RagM MCP forget tools test passed.")
